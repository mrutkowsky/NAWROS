import hdbscan
import umap.umap_ as UMAP
import torch
import pandas as pd
import numpy as np
import faiss
import os
import logging

from utils.data_processing import get_embedded_files, VALID_FILES, FAISS_VECTORS_PATH

def dimension_reduction(sentence_embeddings, dimension=5):
    clusterable_embeddings = UMAP.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=5,
        random_state=42,
        metric='cosine'
    ).fit_transform(sentence_embeddings)
    return clusterable_embeddings


def cluster_sentences(clusterable_embeddings, threshold=0.5):
    cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          min_samples=10,
                          metric='euclidean',                      
                          cluster_selection_method='leaf').fit(clusterable_embeddings)

    return cluster.labels_


def get_clusters_for_choosen_files(choosen_files: list) -> pd.DataFrame:
    """
    Functions that calculates clusters based on vectors for choosen files
    returns a dataframe with labels and 2d coordinates for each sentence.
    """
    embeddings_files = get_embedded_files()
    all_tensors = []
    for file_ in choosen_files:
        logging.info(f'Path to index {embeddings_files[file_]}')
        index = faiss.read_index(os.path.join(FAISS_VECTORS_PATH, embeddings_files[file_]))
        
        n = index.ntotal
        vectors = np.zeros((n, index.d), dtype=np.float32)
        index.reconstruct_n(0, n, vectors)

        all_tensors.append(vectors)


    all_tensors = np.vstack(all_tensors)
    print(type(all_tensors[0]))
    print(all_tensors.shape)
    
    clusterable_embeddings = dimension_reduction(all_tensors)
    dimensions_2d = dimension_reduction(all_tensors, dimension=2)
    labels = cluster_sentences(clusterable_embeddings)
    logging.info(f"Clusters calculated successfully")

    df = pd.DataFrame({
        'labels': labels,
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1]
        })

    return df


if __name__ == "__main__":
    VALID_FILES = os.path.join('validated_files')
    all_files = os.listdir(VALID_FILES)
    df = get_clusters_for_choosen_files(all_files)
    n_clusters = len(df['labels'].unique())
    print(f'Number of clusters: {n_clusters}')