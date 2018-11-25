# encoding=utf-8

import os
import fire
import numpy as np

from scipy.sparse.csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .common.similarities import SIM_FUNCTIONS
from .common.dataset import prepare_shuffled_dataset
from .common.scorers import map_scorer, trr_scorer, nr_scorer

from .executor import BaseExecutor

SCORING = {
            'MAP': map_scorer,
            'TRR': trr_scorer,
            'NR': nr_scorer
          }

def do_nothing_tokenizer(tokens):
    return tokens


class PerRecCBR(BaseEstimator):
    """CBR component for recommending permission lists

    Input: A list of used apis.
    Output: The ranked permission list of the app.
    """
    def __init__(self, sim_func="cosine"):
        if callable(sim_func):
            self.sim_func = sim_func
        else:
            self.sim_func = SIM_FUNCTIONS.get(sim_func, None)
        if not self.sim_func:
            raise ValueError("Error sim_func" + str(sim_func))

    @staticmethod
    def build_perm_docs(perm_vectors, api_vectors):
        """Build permission profiles

        Args:
            perm_vectors (Matrix): app perm vectors
            api_vectors (Matrix): app api vectors
            perm_list (List): list of permissions
        """
        perm_docs = []
        # for each column of permission vectors (e.g., each permission)
        for col in perm_vectors.T:
            # find the apps which require this permissions
            if isinstance(col, csr_matrix):
                col = col.toarray().reshape(-1, )
            apps = np.where(col == 1)
            # find the api vectors of such apps
            cur_api_vectors = api_vectors[apps].toarray()
            # construct permission doc
            cur_perm_doc = cur_api_vectors.sum(axis=0)
            perm_docs.append(cur_perm_doc)
        return np.array(perm_docs)

    def fit(self, X, y):
        """Build the profiles for training permissions

        Args:
            X (List(List(API))): The api lists of the training apps.
            y (List(List(Perm))): The permission lists of all apps

        Returns:
            self object: return self
        """
        # Steps:
        # 1. build permission doc
        # 2. calculate the tfidf vector for each permission doc as the profiles of permissions
        # 3. build API CountVectorizer
        self.api_vectorizer_ = CountVectorizer(binary=True, tokenizer=do_nothing_tokenizer,
                                               preprocessor=None, lowercase=False)
        self.train_api_vectors_ = self.api_vectorizer_.fit_transform(X)
        self.perm_vectorizer_ = CountVectorizer(binary=True, tokenizer=do_nothing_tokenizer,
                                                preprocessor=None, lowercase=False)
        self.train_perm_vectors_ = self.perm_vectorizer_.fit_transform(y)
        self.perm_list_ = self.perm_vectorizer_.get_feature_names()

        # build permission doc
        self.perm_docs_ = self.build_perm_docs(self.train_perm_vectors_, self.train_api_vectors_)

        # idf = log(total_num / num) + 1
        self.tfidf_transformer_ = TfidfTransformer(norm="l1", use_idf=True, smooth_idf=False)
        tfidf_matrix = self.tfidf_transformer_.fit_transform(self.perm_docs_)

        self.perm_profiles_ = normalize(tfidf_matrix, norm='l2', axis=1)

    def transform(self, X, *fit_params):
        """Recommend permissions for new apps

        Args:
            X (List(List(API))): A list of apps for testing.

        Returns:
            Perms (List(List(Permission))): The ranked permission lists recommended for input apps
        """
        # ranked the permissions
        # construct app profiles (api vectors)
        test_api_vectors = self.api_vectorizer_.transform(X)
        # calculate the similarities between API vector and permission profiles
        # test_num * perm_num
        similarities = self.sim_func(test_api_vectors, self.perm_profiles_)
        perm_scores = normalize(similarities, norm="l1", axis=1)
        # for fusion
        self.perm_scores_ = perm_scores
        sorted_perm_index = np.argsort(-1.0 * perm_scores, 1)
        # each row: perm_i, perm_j, per_k (sorted)
        return np.take(self.perm_list_, sorted_perm_index)

    def predict(self, X):
        return self.transform(X)

class CBR(BaseExecutor):
    def __init__(self, dataset, scoring, **kwargs):
        super().__init__("CBR", dataset, scoring)
        self.sim_func = kwargs.get("sim_func", "cosine")
        self.smooth_idf = kwargs.get("smooth_idf", True)

    def get_result_file(self, data_dir):
        file_name = "_".join([self.name, self.sim_func, str(self.smooth_idf)])
        return os.path.join(data_dir, file_name + ".json")

    def construct_estimator(self):
        return PerRecCBR(sim_func=self.sim_func)

    def run(self):
        api_lists = self.dataset.extract_api_lists()
        perm_lists = self.dataset.extract_perm_lists()
        estimator = self.construct_estimator()
        scores = cross_validate(estimator, api_lists, perm_lists, scoring=self.scoring, cv=10,
                                n_jobs=-1, verbose=1, return_train_score=False)
        return scores

def main(sim_func="cosine"):
    dataset = prepare_shuffled_dataset()
    scoring = SCORING
    executor = CBR(dataset, scoring, sim_func=sim_func)
    scores = executor.run()
    print(scores['test_MAP'].mean())

if __name__ == "__main__":
    fire.Fire({
        'main': main
    })
