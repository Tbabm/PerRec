# encoding=utf-8

import numpy as np

def sum_fusion(perm_score_sets):
    # test_num * len(perm_list)
    score_matrix_sum = perm_score_sets.sum(axis=0)
    return score_matrix_sum

def anz_fusion(perm_score_sets):
    # test_num * len(perm_list)
    score_matrix_sum = perm_score_sets.sum(axis=0)
    # test_num * len(perm_list)
    perm_count_matrix = (perm_score_sets != 0).astype(int).sum(axis=0)
    return np.nan_to_num(score_matrix_sum / perm_count_matrix)

def mnz_fusion(perm_score_sets):
    # test_num * len(perm_list)
    score_matrix_sum = perm_score_sets.sum(axis=0)
    # test_num * len(perm_list)
    perm_count_matrix = (perm_score_sets != 0).astype(int).sum(axis=0)
    return score_matrix_sum * perm_count_matrix

def max_fusion(perm_score_sets):
    # test_num * len(perm_list)
    return perm_score_sets.max(axis=0)

def min_fusion(perm_score_sets):
    return perm_score_sets.min(axis=0)

def borda_count_fusion(perm_score_sets):
    # convert score matrix to matrix of rank_point
    perm_rp_sets = []
    for score_matrix in perm_score_sets:
        rp_matrix = []
        for score_array in score_matrix:
            # len(perm_list)
            sort_idx = score_array.argsort()
            cur_rank = np.empty_like(sort_idx)
            cur_rank[sort_idx] = np.arange(len(score_array))
            rp_matrix.append(cur_rank)
        rp_matrix = np.stack(rp_matrix)
        perm_rp_sets.append(rp_matrix)
    perm_rp_sets = np.stack(perm_rp_sets)
    assert(perm_score_sets.shape == perm_rp_sets.shape)
    return perm_rp_sets.sum(axis=0)

def fusion(perm_score_sets, perm_list, ftype):
    """fusion method for combining scores output by different components
    
    Args:
        perm_score_sets (3d-array): List of permission scores. Each element
    is the score set output by one method. The size of the element is 
    len(test_set) x len(perm_list). Each score array is listed according to perm_list.
        perm_list (1d-array of string): List of training permissions.
        type (String): The type of fusion functions, sum | anz | mnz | max | min | 
        borda_count
    """
    type_list = ["max", "min", "sum", "anz", "mnz", "borda_count"]
    ftype = ftype.lower()
    if ftype not in type_list:
        raise ValueError("Error fusion type: ", ftype)
    perm_score_sets = np.array(perm_score_sets)
    # test_num * len(perm_list)
    new_scores = globals()[ftype+"_fusion"](perm_score_sets)
    sorted_perm_index = np.argsort(-1.0 * new_scores, 1)
    # each row: perm_i, perm_j, per_k (sorted)
    return np.take(perm_list, sorted_perm_index)
