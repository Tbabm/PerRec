# encoding=utf-8

import torch
from scipy.sparse.csr import csr_matrix

def reserve_nn_scores(similarities, nn_num):
    """Reserver top-k nearest neighbors' similarity scores
    
    Args:
        similarities (Matrix): test_num * train_num
    
    Returns:
        nn_scores (Matrix): only keep the scores of the top-k nearest neighbors
    """
    scores = torch.FloatTensor(similarities)
    # sorted each row, and get the index
    # ascending order
    sorted_scores, sorted_index = torch.sort(scores, 1, descending=True)
    nn_scores = sorted_scores[:, 0:nn_num]
    nn_index = sorted_index[:, 0:nn_num]
    # only accept float32
    nn_scores = torch.zeros(scores.size()).scatter_(1, nn_index, nn_scores)
    # convert float32 to float64
    return nn_scores.numpy().astype(float)

def cal_perm_scores(nn_scores, train_perm_vectors):
    """Calculate scores of permissions for each test sample
    
    Args:
        nn_scores (Matrix): test_num * train_num, only reserve the scores of top-k 
    nearest neighbors
    """
    # test_num * train_num * 1 x 1 * train_num * perm_list = test_num * train_num * perm_list
    # using numpy's broadcast
    dense_train_perm_vectors = train_perm_vectors
    if isinstance(dense_train_perm_vectors, csr_matrix):
        dense_train_perm_vectors = dense_train_perm_vectors.toarray()
    test_train_perm_scores = nn_scores[:, :, None] * dense_train_perm_vectors[None, :, :]
    # find the max score of each permission for each test sample
    # test_num * perm_lis
    perm_scores = test_train_perm_scores.sum(axis=1)
    return perm_scores