# encoding=utf-8

import numpy as np

from perrec.config import CONFIG

def load_embeddings(embedding_file):
    with open(embedding_file, 'r') as f:
        vecs = f.readlines()
    embeddings = {}
    for vec in vecs:
        values = vec.strip().split()
        embeddings[values[0]] = np.array([ float(weight) for weight in values[1:] ])
    return embeddings

def load_embeddings_by_type(e_type):
    if e_type.lower() == "glove":
        embedding_file = CONFIG.glove_embedding_file
    elif e_type.lower() == "api":
        embedding_file = CONFIG.api_embedding_file
    else:
        raise ValueError("Error embedding type", e_type)
    return load_embeddings(embedding_file)
