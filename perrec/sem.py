# encoding=utf-8

import os
import fire
import torch

from tqdm import tqdm
import numpy as np
from scipy.sparse.csr import csr_matrix

from nltk.stem.porter import PorterStemmer

from sklearn.base import BaseEstimator, clone
from sklearn.utils import safe_indexing
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .common.similarities import SIM_FUNCTIONS, calculate_doc_partial_similarities
from .common import nearest_neighbor as nn
from .common.embedding import load_embeddings_by_type
from .common.dataset import load_json_file, prepare_shuffled_dataset
from .common.scorers import map_scorer, trr_scorer, nr_scorer

from .executor import BaseExecutor
from .config import CONFIG

SCORING = {
            'MAP': map_scorer,
            'TRR': trr_scorer,
            'NR': nr_scorer
          }

def do_nothing_tokenizer(tokens):
    return tokens

def sem_cross_validate(estimator, X, y, scoring, n_splits=10, similarities=None):
    # similarities: doc-doc partial similarities, according to shuffled indexes
    cv = KFold(n_splits=n_splits)
    scores = {}
    result_sims = []
    for key, scorer in scoring.items():
        scores["test_"+key] = []
    for i, (train_idxes, test_idxes) in enumerate(cv.split(X)):
        # create new estimator
        cur_estimator = clone(estimator)
        train_X = safe_indexing(X, train_idxes)
        train_y = safe_indexing(y, train_idxes)
        test_X = safe_indexing(X, test_idxes)
        test_y = safe_indexing(y, test_idxes)
        cur_estimator.fit(train_X, train_y)
        if similarities is None:
            rec_perm_lists = cur_estimator.transform(test_X)
            result_sims.append(cur_estimator.sims_)
        else:
            rec_perm_lists = cur_estimator.transform(test_X, similarities[i])
        for key, scorer in scoring.items():
            cur_score = scorer._sign * scorer._score_func(test_y, rec_perm_lists, **scorer._kwargs)
            scores["test_"+key].append(cur_score)
    result_sims = np.array(result_sims)
    return scores, result_sims

class PerRecSEM(BaseEstimator):
    """SEM component for recommend permission lists

    Initialize: require API-API similarity matrix

    Input: a list of apis
    Output: a list of permissions
    """
    def __init__(self, api_list, api_similarities, sim_func="cosine", nn_num=10):
        """
        :param api_similarities: api-api similarity matrix
        :param nn_num: num of nearest neighbors
        """
        self.api_list = api_list
        self.api_similarities = api_similarities
        self.nn_num = nn_num
        if callable(sim_func):
            self.sim_func = sim_func
        else:
            self.sim_func = SIM_FUNCTIONS.get(sim_func, None)
        if not self.sim_func:
            raise ValueError("Error sim_func" + str(sim_func))

    def fit(self, X, y):
        """Build api vectors for the trianing apps

        Args:
            X (List(List(API))): The api lists of all apps
            y (List(List(Perm))): The permission lists of all apps

        Returns:
            self object: return self
        """
        self.api_vectorizer_ = CountVectorizer(binary=True, tokenizer=do_nothing_tokenizer,
                                               preprocessor=None, lowercase=False,
                                               vocabulary=self.api_list)
        self.train_api_vectors_ = self.api_vectorizer_.fit_transform(X)
        self.perm_vectorizer_ = CountVectorizer(binary=True, tokenizer=do_nothing_tokenizer,
                                                preprocessor=None, lowercase=False)
        self.train_perm_vectors_ = self.perm_vectorizer_.fit_transform(y)
        self.perm_list_ = self.perm_vectorizer_.get_feature_names()
        return self

    def transform(self, X, similarities=None):
        """Transform function

        1. for each testing app, build a new api vector according to current training app
        2. calculate the similarities between testing and training apps, and do collaborative filtering

        Args:
            X (List(List(API))): The api lists of all apps

        Returns:
            Perms (List(List(Permission))): The ranked permission lists recommended for input apps
        """
        test_api_vectors = self.api_vectorizer_.transform(X)
        if isinstance(test_api_vectors, csr_matrix):
            test_api_vectors = test_api_vectors.toarray()
        train_api_vectors = self.train_api_vectors_
        if isinstance(train_api_vectors, csr_matrix):
            train_api_vectors = train_api_vectors.toarray()

        if similarities is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
            simi_matrix = []
            api_similarities = torch.DoubleTensor(self.api_similarities).to(device)
            train_api_vectors = torch.DoubleTensor(train_api_vectors).to(device)
            norm_train_vectors = train_api_vectors / train_api_vectors.norm(p=2, dim=-1)[:, None]
            test_api_vectors = torch.DoubleTensor(test_api_vectors).to(device)
            for train_vector, norm_train_vector in tqdm(zip(train_api_vectors, norm_train_vectors)):
                new_test_matrix = []
                for test_vector in test_api_vectors:
                    api_simi = api_similarities * train_vector[:, None] * test_vector[None, :]
                    new_train_vector = api_simi.max(dim=1)[0]
                    cur_norm = new_train_vector.norm(p=2)
                    new_train_vector = new_train_vector / cur_norm

                    new_test_vector = torch.max(test_vector, new_train_vector)
                    new_test_matrix.append(new_test_vector)
                new_test_matrix = torch.stack(new_test_matrix, dim=0)
                # normalize to calculate cosinine similarities
                norm_test_matrix = new_test_matrix / new_test_matrix.norm(p=2, dim=-1)[:, None]
                # shape = (test_num, )
                simi_col = torch.matmul(norm_test_matrix, norm_train_vector)
                simi_matrix.append(simi_col)
            # (test_num, train_num)
            similarities = torch.stack(simi_matrix, dim=1).cpu().numpy()
            assert (similarities.shape == (len(test_api_vectors), len(train_api_vectors)))
            self.sims_ = similarities

        # test_num * train_num, only reserver top-k nearest neighbors
        nn_scores = nn.reserve_nn_scores(similarities, self.nn_num)
        # test_num * perm_list, calculate the score of each permission for each test sample
        perm_scores = nn.cal_perm_scores(nn_scores, self.train_perm_vectors_)
        perm_scores = normalize(perm_scores, norm='l1', axis=1)
        # store perm_scores for fusion
        self.perm_scores_ = perm_scores
        # test_num * perm_list, from high score to low score
        sorted_perm_index = np.argsort(-1.0 * perm_scores, 1)
        # each row: perm_i, perm_j, per_k (sorted)
        return np.take(self.perm_list_, sorted_perm_index)

    def predict(self, X):
        return self.transform(X)

class SEM(BaseExecutor):
    def __init__(self, dataset, scoring, **kwargs):
        super().__init__("SEM", dataset, scoring)
        self.e_type = kwargs.get("e_type", "glove")
        self.nn_num = kwargs.get("nn_num", 10)
        self.dump_api_sim = kwargs.get("dump_api_sim", True)
        self.load_api_sim = kwargs.get("load_api_sim", False)
        self.dump_app_sim = kwargs.get("dump_app_sim", True)
        self.load_app_sim = kwargs.get("load_app_sim", False)
        self.min_idf = kwargs.get("min_idf", 1.0)
        self.device = kwargs.get("device", "cuda:0")

    def get_api_desc_token_file(with_name):
        file_name = "api_desc_tokens"
        return os.path.join(CONFIG.data_dir, file_name + ".json")

    def get_result_file(self, data_dir):
        file_name = "_".join([self.name, self.e_type, str(self.nn_num)])
        return os.path.join(data_dir, file_name + ".json")

    def get_sim_file_raw(self, s_type):
        if s_type.lower() not in ['api', 'app']:
            raise ValueError("Error similarity type", s_type)
        prefix = "_".join([self.name, self.e_type.lower(), s_type.lower()])
        file_name = prefix + "_" + CONFIG.sim_dump_file
        return os.path.join(CONFIG.data_dir, file_name)

    def get_api_sim_file(self):
        return self.get_sim_file_raw('api')

    def get_app_sim_file(self):
        return self.get_sim_file_raw('app')

    @staticmethod
    def dump_similarities(similarities, sim_file):
        np.save(sim_file, similarities)

    @staticmethod
    def load_similarities(sim_file):
        return np.load(sim_file)

    def _get_embedding(self, embeddings, token):
        if token not in embeddings:
            porter = PorterStemmer()
            token = porter.stem(token)
            return embeddings.get(token, None)
        else:
            return embeddings[token]

    def _build_api_doc_matrixes(self, embeddings, api_desc_list, idf_dict=None):
        api_doc_matrixes = []
        api_doc_idf_vectors = []
        for api_doc in api_desc_list:
            cur_api_matrix = []
            cur_idf_vector = []
            # remove the tokens which is not in embeddings
            for token in api_doc:
                vector = self._get_embedding(embeddings, token)
                if vector is not None:
                    cur_api_matrix.append(vector)
                    if idf_dict:
                        cur_idf_vector.append(idf_dict.get(token, self.min_idf))
            # do not normalize here, since we will normalize it when we calculate the sims
            cur_api_matrix = np.array(cur_api_matrix)
            api_doc_matrixes.append(cur_api_matrix)
            if idf_dict:
                api_doc_idf_vectors.append(np.array(cur_idf_vector))
        return api_doc_matrixes, api_doc_idf_vectors

    @staticmethod
    def _get_api_list(api_desc_tokens):
        return list(api_desc_tokens.keys())

    def cal_api_similarities(self, api_desc_tokens, embeddings):
        """
        calculate api->api partial similarities

        :param api_desc_tokens: list(list(tokens)), tokens of each api descriptions
        :param embeddings: Embedding dict
        :return: api_similarity matrix
        """
        api_desc_list = list(api_desc_tokens.values())
        api_doc_counter = TfidfVectorizer(norm=None, tokenizer=do_nothing_tokenizer,
                                             preprocessor=None, lowercase=False)
        api_doc_counter.fit(api_desc_list)
        token_idf_dict = dict( zip(api_doc_counter.get_feature_names(), api_doc_counter.idf_) )
        api_doc_matrixes, api_doc_idf_vectors = self._build_api_doc_matrixes(embeddings, api_desc_list, token_idf_dict)
        api_similarities = calculate_doc_partial_similarities(api_doc_matrixes, api_doc_matrixes,
                                                              api_doc_idf_vectors, device=self.device)
        return api_similarities

    def prepare_api_desc_tokens(self):
        print("[SEM] Load api descriptions")
        api_desc_token_file = self.get_api_desc_token_file()
        api_desc_tokens = load_json_file(api_desc_token_file)
        return api_desc_tokens

    def prepare_api_similarities(self, api_desc_tokens):
        api_sim_file = self.get_api_sim_file()
        if self.load_api_sim:
            api_similarities = SEM.load_similarities(api_sim_file)
        else:
            print("[SEM] Load embeddings")
            embeddings = load_embeddings_by_type(self.e_type)
            # api->api similairites, api order is the same as api_desc_tokens' api order
            api_similarities = self.cal_api_similarities(api_desc_tokens, embeddings)
        if self.dump_api_sim:
            SEM.dump_similarities(api_similarities, api_sim_file)
        np.fill_diagonal(api_similarities, 1)
        return api_similarities

    def construct_estimator(self):
        # different api_desc token -> different api list -> different simialrities
        # load api_desc tokens
        api_desc_tokens = self.prepare_api_desc_tokens()
        api_list = SEM._get_api_list(api_desc_tokens)
        api_similarities = self.prepare_api_similarities(api_desc_tokens)
        estimator = PerRecSEM(api_list, api_similarities, nn_num=self.nn_num)
        return estimator

    def run(self):
        estimator = self.construct_estimator()
        input_api_lists = self.dataset.extract_api_lists()
        output_perm_lists = self.dataset.extract_perm_lists()
        app_sim_file = self.get_app_sim_file()
        if self.load_app_sim:
            print("[SEM] Load app similarities")
            app_similarities = SEM.load_similarities(app_sim_file)
            scores, _ = sem_cross_validate(estimator, input_api_lists, output_perm_lists, scoring=self.scoring,
                                           n_splits=10, similarities=app_similarities)
        else:
            scores, app_similarities = sem_cross_validate(estimator, input_api_lists, output_perm_lists,
                                           scoring=self.scoring, n_splits=10, similarities=None)
        if self.dump_app_sim:
            print("[SEM] Dump app similarities")
            self.dump_similarities(app_similarities, app_sim_file)
        return scores

def main(e_type="glove", nn_num=10, dump_api_sim=True, load_api_sim=False, dump_app_sim=True, load_app_sim=False):
    """

    :param e_type: glove | api
    :param nn_num: the number of nearest neighbors
    :param dump_api_sim: whether dump computed api-api similarities
    :param load_api_sim: whether load pre-computed api-api similarities
    :param dump_app_sim: whether dump computed app-app similarities
    :param load_app_sim: whether load pre-computed app-app similarities
    :return:
    """
    dataset = prepare_shuffled_dataset()
    scoring = SCORING
    executor = SEM(dataset, scoring, e_type=e_type, nn_num=nn_num, load_api_sim=load_api_sim,
                   dump_api_sim=dump_api_sim, load_app_sim=load_app_sim, dump_app_sim=dump_app_sim)
    scores = executor.run()
    print(scores)
    print(np.array(scores['test_MAP']).mean())

if __name__ == '__main__':
    fire.Fire({
        'main': main
    })
