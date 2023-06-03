import hdbscan
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import umap.umap_ as UMAP
import torch
import pandas as pd
import numpy as np
import faiss
import os
import logging

from utils.data_processing import get_embedded_files, VALID_FILES, FAISS_VECTORS_PATH, read_file, CLEARED_DATA_FOLDER


def dimension_reduction(
        sentence_embeddings: np.ndarray, 
        dimension: int = 5):
    
    clusterable_embeddings = UMAP.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=dimension,
        random_state=42,
        metric='cosine').fit_transform(sentence_embeddings)

    return clusterable_embeddings


def cluster_sentences(
        clusterable_embeddings, 
        threshold=0.5):
    
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=10,
        metric='euclidean',                      
        cluster_selection_method='leaf') \
            .fit(clusterable_embeddings)

    return cluster.labels_


def get_clusters_for_choosen_files(
        choosen_files: list,
        content_column: str = 'content') -> pd.DataFrame:
    """
    Functions that calculates clusters based on vectors for choosen files
    returns a dataframe with labels and 2d coordinates for each sentence.
    """
    embeddings_files = get_embedded_files()
    
    all_vectors = None
    result_df = None

    for file_ in choosen_files:
        logging.info(f'Path to index {embeddings_files[file_]}')
        index = faiss.read_index(os.path.join(FAISS_VECTORS_PATH, embeddings_files[file_]))
        
        n = index.ntotal
        vectors = np.zeros((n, index.d), dtype=np.float32)
        index.reconstruct_n(0, n, vectors)

        all_vectors = np.vstack((all_vectors, vectors)) if all_vectors is not None else vectors

        filename, ext = os.path.splitext(file_)
        current_file_df = read_file(os.path.join(CLEARED_DATA_FOLDER, f'{filename}.csv'), columns=[content_column])
        result_df = pd.concat([result_df, current_file_df])
        result_df = result_df.reset_index(drop=True)
    
    clusterable_embeddings = dimension_reduction(all_vectors)
    dimensions_2d = dimension_reduction(all_vectors, dimension=2)
    labels = cluster_sentences(clusterable_embeddings)
    logging.info(f"Clusters calculated successfully")

    df = pd.DataFrame({
        'labels': labels,
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1]
        })

    logging.info(f'result_df: {result_df}')
    df['content'] = result_df[content_column].copy()
    logging.info(f'result_df: {result_df}')

    return df


if __name__ == "__main__":
    VALID_FILES = os.path.join('validated_files')
    all_files = os.listdir(VALID_FILES)
    df = get_clusters_for_choosen_files(all_files)
    n_clusters = len(df['labels'].unique())
