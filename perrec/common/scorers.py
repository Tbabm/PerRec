# encoding=utf-8

import numpy as np

from sklearn.metrics.scorer import make_scorer

from perrec.common.dataset import get_perm_num

def average_precision(y_true, y_pred):
    """Calculate the mean average precision score
    
    Args:
        y_true (List(Permission)): A list of permissions used.
        y_pred (List(Permission)): A Ranked list of permissions recommended
    
    Returns:
        ap: Average precision
    """
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if min(len(y_true), len(y_pred)) == 0:
        return 0.0
    return score / len(y_true)

def mean_average_precision(y_true, y_pred):
    """Calculate the mean average precision score
    
    Args:
        y_true (List(List(Permission))): Lists of permissions used.
        y_pred (List(List(Permission))): Ranked lists of permissions recommended
    
    Returns:
        map: Mean average precision
    """
    return np.mean([average_precision(a,p) for a,p in zip(y_true, y_pred)])

def total_recall_ratio(y_true, y_pred, candidate_num):
    """Calculate the mean average precision score

        Args:
            y_true (List(Permission)): A list of permissions used.
            y_pred (List(Permission)): A Ranked list of permissions recommended

        Returns:
            complete: whether y_pred contains all permission in y_true
            ttr: the Total-Recall Ratio
    """
    # 如果包含所有：正确的ratio
    # 如果有缺失：candidate_num / true_num
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) == 0:
        return 1
    max_index = -1
    for p in y_true:
        try:
            cur_index = y_pred.index(p)
        except ValueError:
            return candidate_num / len(y_true)
        max_index = max(max_index, cur_index)
    return (max_index + 1) / len(y_true)

def average_total_recall_ratio(y_true, y_pred, candidate_num):
    """Calculate the Total-Recall Ratio

    Args:
        y_true (List(List(Permission))): Lists of permissions used.
        y_pred (List(List(Permission))): Ranked lists of permissions recommended

    Returns:
        ttr: the average Total-Recall Ratio
    """
    return np.mean([total_recall_ratio(a, p, candidate_num) for a, p in zip(y_true, y_pred)])

def necessary_recall(y_true, y_pred):
    y_true, y_pred = list(y_true), list(y_pred)
    y_necessary = y_pred[:len(y_true)]
    correct = 0
    for p in y_necessary:
        if p in y_true:
            correct += 1
    return correct / len(y_true)

def average_necessary_recall(y_true, y_pred):
    cps = []
    for accurate, pred, in zip(y_true, y_pred):
        accurate, pred = list(accurate), list(pred)
        cur_cp = necessary_recall(accurate, pred)
        cps.append(cur_cp)
    return np.mean(cps)

map_scorer = make_scorer(mean_average_precision, greater_is_better=True)

# the lower the better, but we do not want the -1
trr_scorer = make_scorer(average_total_recall_ratio, greater_is_better=True, candidate_num=get_perm_num())

nr_scorer = make_scorer(average_necessary_recall, greater_is_better=True)
