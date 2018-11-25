# encoding=utf-8

import os
import fire
import json

import numpy as np
from sklearn.base import clone, BaseEstimator
from sklearn.utils import safe_indexing
from sklearn.model_selection import KFold
from numpy.testing import assert_array_equal

from . import fusion
from .cbr import PerRecCBR
from .sem import SEM, PerRecSEM
from .common.dataset import prepare_shuffled_dataset
from .common.scorers import map_scorer, trr_scorer, nr_scorer

from .executor import BaseExecutor
from .config import CONFIG

SCORING = {
            'MAP': map_scorer,
            'TRR': trr_scorer,
            'NR': nr_scorer
          }

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

FUSION_TYPES = ["max", "min", "sum", "anz", "mnz", "borda_count"]

def fusion_evaluate(estimators, scoring, train_X, train_y, test_X, test_y, similarity, ftypes):
    perm_score_sets = []
    perm_lists_ = []
    for est in estimators:
        est.fit(train_X, train_y)
        rec_perm_lists = est.transform(test_X, similarity)
        perm_score_sets.append(est.perm_scores_)
        # sem and cbr should have the same permission list (since they use the same
        # training set)
        perm_lists_.append(est.perm_list_)
    for p_list in perm_lists_[1:]:
        assert_array_equal(perm_lists_[0], p_list)
    perm_score_sets = np.array(perm_score_sets)
    cur_scores = {}
    for key in scoring:
        cur_scores["test_" + key] = []
    for ftype in ftypes:
        fusion_rec_perm_lists = fusion.fusion(perm_score_sets, perm_lists_[0], ftype)
        for key, scorer in scoring.items():
            cur_score = scorer._sign * scorer._score_func(test_y, fusion_rec_perm_lists, **scorer._kwargs)
            cur_scores["test_" + key].append(cur_score)
    return cur_scores

def fusion_cross_validate(estimators, X, y, similarities, scoring, n_splits=10, ftypes=FUSION_TYPES):
    """Conduct cross validation for fusion methods
    
    Args:
        estimators (Estimator): Estimators combined
        X (Input): List of api lists
        y (Labels): List of permission lists
        n_splits (int, optional): Defaults to 10. # of folds
        similarities (ndarray, optional): Defaults to None. Pre-trained similarities
        ftypes (List(String)): Fusion types
    """
    # similarities: doc-doc partial similarities, according to shuffled indexes
    cv = KFold(n_splits=n_splits)
    # n_split * len(fusion_types) * len(scorers)
    scores = {}
    for key in scoring:
        scores["test_"+key] = []
    app_sims = []
    for i, (train_idxes, test_idxes) in enumerate(cv.split(X)):
        train_X = safe_indexing(X, train_idxes)
        train_y = safe_indexing(y, train_idxes)
        test_X = safe_indexing(X, test_idxes)
        test_y = safe_indexing(y, test_idxes)
        cur_estimators = [clone(est) for est in estimators]
        cur_sim = None
        if similarities is None:
            for est in cur_estimators:
                if isinstance(est, PerRecSEM):
                    est.fit(train_X, train_y)
                    rec_perm_lists = est.transform(test_X)
                    cur_sim = est.sims_
                    app_sims.append(cur_sim)
                    break
        else:
            cur_sim = similarities[i]
        cur_scores = fusion_evaluate(cur_estimators, scoring, train_X, train_y, test_X, test_y,
                                        cur_sim, ftypes)
        for key, scorer in scoring.items():
            scores["test_" + key].append(cur_scores["test_" + key])
    # dict, each value is 10 * 6
    app_sims = np.array(app_sims)
    return scores, app_sims

class PerRec(BaseEstimator):
    def __init__(self, **kwargs):
        self.methods = kwargs.get("methods", ["SEM", "CBR"])
        self.sem_e_type = kwargs.get("sem_e_type", "glove")
        print("SEM embedding type", self.sem_e_type)
        self.sem_nn_num = kwargs.get("sem_nn_num", 10)
        self.sem_load_api_sim = kwargs.get("sem_load_api_sim", True)
        self.sem_dump_api_sim = kwargs.get("sem_dump_api_sim", False)
        self.sem_load_app_sim = kwargs.get("sem_load_app_sim", True)
        self.sem_dump_app_sim = kwargs.get("sem_dump_app_sim", False)
        self.cbr_sim_func = kwargs.get("cbr_sim_func", "cosine")
        print("CBR sim func", self.cbr_sim_func)

    def get_sem_executor(self):
        return SEM(None, None, e_type=self.sem_e_type, desc_with_name=True,nn_num=self.sem_nn_num,
                   api_sim_norm=2,dump_api_sim=self.sem_dump_api_sim,load_api_sim=self.sem_load_api_sim,
                   dump_app_sim=self.sem_dump_app_sim, load_app_sim=self.sem_load_app_sim)

    def construct_estimators_and_similarities(self):
        similarities = None
        estimators = []
        for method in self.methods:
            if method.lower() == "cbr":
                cur_est = PerRecCBR(sim_func=self.cbr_sim_func)
            elif method.lower() == "sem":
                cur_executor = self.get_sem_executor()
                cur_est = cur_executor.construct_estimator()
                if self.sem_load_app_sim:
                    print("[SEM] Load app similarities")
                    app_sim_file = cur_executor.get_app_sim_file()
                    similarities = SEM.load_similarities(app_sim_file)
            else:
                raise ValueError("Error method name!")
            estimators.append(cur_est)
        return estimators, similarities

    def fit(self, X, y):
        """
        Fit function
        :param X: api lists
        :param y: permission lists
        :return: self
        """
        self.estimators, self.transform_simi= self.construct_estimators_and_similarities()
        for est in self.estimators:
            est.fit(X, y)
        return self

    def transform(self, X):
        """
        Transform function
        :param X: api lists
        :return: recommended permissions
        """
        perm_score_sets = []
        perm_lists_ = []
        for est in self.estimators:
            rec_perm_lists = est.transform(X, self.transform_simi)
            perm_score_sets.append(est.perm_scores_)
            perm_lists_.append(est.perm_list_)
        for p_list in perm_lists_[1:]:
            assert_array_equal(perm_lists_[0], p_list)
        perm_score_sets = np.array(perm_score_sets)
        fusion_rec_perm_lists = fusion.fusion(perm_score_sets, perm_lists_[0], self.ftype)
        return fusion_rec_perm_lists

class PerRecExecutor(BaseExecutor):
    def __init__(self, dataset, scoring, **kwargs):
        super().__init__("PerRec", dataset, scoring)
        self.ftypes = kwargs.get("ftypes", FUSION_TYPES)
        self.estimator = PerRec(**kwargs)

    def construct_estimators_and_similarities(self):
        return self.estimator.construct_estimators_and_similarities()

    def get_result_file(self, data_dir):
        # do not consider methods and ftypes
        file_name = "_".join([self.name, self.estimator.sem_e_type, str(self.estimator.sem_nn_num),
                              self.estimator.cbr_sim_func])
        return os.path.join(data_dir, file_name + ".json")

    def run(self):
        api_lists = self.dataset.extract_api_lists()
        perm_lists = self.dataset.extract_perm_lists()
        estimators, similarities = self.construct_estimators_and_similarities()
        scores, similarities = fusion_cross_validate(estimators, api_lists, perm_lists, similarities, self.scoring,
                                       n_splits=10, ftypes=self.ftypes)
        if self.estimator.sem_dump_app_sim:
            sim_file = self.estimator.get_sem_executor().get_app_sim_file()
            SEM.dump_similarities(similarities, sim_file)
        return scores

def main(sem_e_type="glove", sem_nn_num=10, cbr_sim_func="cosine", load_api_sim=False, dump_api_sim=True,
         load_app_sim=False, dump_app_sim=True):
    dataset = prepare_shuffled_dataset()
    scoring = SCORING
    executor = PerRecExecutor(dataset, scoring, ftypes=FUSION_TYPES, sem_e_type=sem_e_type,
                              sem_nn_num=sem_nn_num, cbr_sim_func=cbr_sim_func, sem_load_api_sim=load_api_sim,
                              sem_dump_api_sim=dump_api_sim, sem_load_app_sim=load_app_sim,
                              sem_dump_app_sim=dump_app_sim)
    scores = executor.run()
    print(scores)
    result_file = os.path.join(CONFIG.data_dir, "perrec_scores.json")
    with open(result_file, 'w') as f:
        json.dump(scores, f)

if __name__ == '__main__':
    fire.Fire({
        'main': main
    })
