# encoding=utf-8

"""Functions for calculate similarities
"""

import torch
import numpy as np

from tqdm import tqdm
from scipy.spatial.distance import correlation, jaccard
from sklearn.metrics.pairwise import cosine_similarity as cosine 
from sklearn.metrics.pairwise import euclidean_distances as euclidean_dis

def _compute_similarities(simi_func, X, Y):
    """Only for distance function whose output is between 0 and 1!
    """
    similarities = []
    for x in X:
        simi = []
        for y in Y:
            simi.append(simi_func(x, y))
        similarities.append(simi)
    return np.array(similarities)

def jaccard_similarity_1D(x, y):
    return 1 - jaccard(x, y)

def correlation_similarity_1D(x, y):
    return 1 - correlation(x, y) / 2

def cosine_similarity(X, Y):
    return cosine(X, Y)

def euclidean_similarity(X, Y):
    # euclidean distances may be larger than 1 and may also be 0
    similarities = 1 / euclidean_dis(X, Y)
    return similarities

def jaccard_similarity(X, Y):
    return _compute_similarities(jaccard_similarity_1D, X, Y)

def correlation_similarity(X, Y):
    """calculate the Pearson correlation similarities between two matrix
    
    Args:
        X (ndarray or sparse matrix): operator 1
        Y (ndarray or sparse matrix): operator 2
    
    Returns:
        ndarray : correlation simialrities
    """
    # correlation_similarity(x, y) = cosine_similarity(x-x.mean(), y-y.mean())
    new_X = X - X.mean(axis=1).reshape(X.shape[0], -1)
    new_Y = Y - Y.mean(axis=1).reshape(Y.shape[0], -1)
    return 0.5 + cosine_similarity(new_X, new_Y)/2

def torch_cosine_similarity(matrix1, matrix2, device=torch.device('cpu')):
    assert(2 == len(matrix1.shape))
    assert(2 == len(matrix2.shape))
    assert(matrix1.shape[1] == matrix2.shape[1])
    matrix1 = torch.DoubleTensor(matrix1).to(device)
    matrix2 = torch.DoubleTensor(matrix2).to(device)
    matrix1 = matrix1 / matrix1.norm(p=2, dim=-1)[:, None]
    matrix2 = matrix2 / matrix2.norm(p=2, dim=-1)[:, None]
    sim = matrix1.mm(matrix2.t())
    return sim

def torch_cosine_similarity_with_max(matrix1, matrix2, device=torch.device('cpu')):
    torch.cuda.empty_cache()
    assert(2 == len(matrix1.shape))
    assert(2 == len(matrix2.shape))
    assert(matrix1.shape[1] == matrix2.shape[1])
    matrix1 = torch.DoubleTensor(matrix1).to(device)
    matrix2 = torch.DoubleTensor(matrix2).to(device)
    matrix1 = matrix1 / matrix1.norm(p=2, dim=1)[:, None]
    matrix2 = matrix2 / matrix2.norm(p=2, dim=1)[:, None]
    # deal with out of memory!
    try:
        max_sim = matrix1.mm(matrix2.t()).max(-1)[0]
    except:
        matrix1 = matrix1.cpu()
        matrix2 = matrix2.cpu()
        max_sim = matrix1.mm(matrix2.t()).max(-1)[0].to(device)
    return max_sim

def doc_partial_similarity_with_max(max_sim, idf_vector, device):
    # calculate doc similarity
    # 1. calculate the max
    # 2. calculate the idf-weighted average
    word_doc_sim_vector = max_sim.to(device)
    idf_vector = torch.DoubleTensor(idf_vector).to(device)
    doc_doc_sim = (idf_vector.dot(word_doc_sim_vector) / idf_vector.sum()).cpu().item()
    return doc_doc_sim

def calculate_doc_partial_similarities(test_matrixes, train_matrixes, test_app_idf_vectors, device):
    """Calculate the partial similarities between two set of docs
    
    Args:
        test_matrixes (List of 2D-array): The doc matrixes of test docs
        test_app_idf_vectors (List of 1D-array): The idf vectors of test docs
    
    Returns:
        2D-array: similarities between test docs and train docs
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
    else:
        dev = torch.device('cpu')
    # calculate cosine similarities between tokens
    similarities = []
    for test_m, idf_vector in tqdm(zip(test_matrixes, test_app_idf_vectors)):
        cur_sim = []
        for count, train_m in tqdm(enumerate(train_matrixes)):
            if len(test_m) == 0 or len(train_m) == 0:
                cur_sim.append(0)
                continue
            max_sim = torch_cosine_similarity_with_max(test_m, train_m, dev)
            doc_sim = doc_partial_similarity_with_max(max_sim, idf_vector, dev)
            cur_sim.append(doc_sim)
        similarities.append(cur_sim)
    return np.array(similarities)

SIM_FUNCTIONS = {
            "cosine": cosine_similarity,
            "euclidean": euclidean_similarity,
            "jaccard": jaccard_similarity,
            "correlation": correlation_similarity 
        }
