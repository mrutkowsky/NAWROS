import hdbscan
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import umap.umap_ as UMAP
import pandas as pd
import numpy as np
import faiss
import os
import logging

from utils.data_processing import get_embedded_files, read_file

logger = logging.getLogger(__file__)

def dimension_reduction(
        sentence_embeddings: np.ndarray, 
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        n_components: int = 5,
        random_state: int = 42):
    
    clusterable_embeddings = UMAP.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='cosine').fit_transform(sentence_embeddings)

    return clusterable_embeddings


def cluster_sentences(
        clusterable_embeddings,
        min_cluster_size: int = 15,
        min_samples: int = 15,
        metric: str = 'euclidean',                      
        cluster_selection_method: str = 'eom'):
    
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,                      
        cluster_selection_method=cluster_selection_method) \
            .fit(clusterable_embeddings)

    return cluster.labels_


def get_clusters_for_choosen_files(
        chosen_files: list,
        path_to_cleared_files: str,
        path_to_embeddings_dir: str,
        faiss_vectors_dirname: str,
        embedded_files_filename: str,
        cleared_files_ext: str = '.parquet.gzip',
        labels_column: str = 'labels',
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int = 15,
        metric: str = 'euclidean',                      
        cluster_selection_method: str = 'eom') -> pd.DataFrame:
    """
    Functions that calculates clusters based on vectors for choosen files
    returns a dataframe with labels and 2d coordinates for each sentence.
    """

    PATH_TO_JSON_EMBEDDED_FILES = os.path.join(path_to_embeddings_dir, embedded_files_filename)
    PATH_TO_FAISS_VECTORS = os.path.join(path_to_embeddings_dir, faiss_vectors_dirname)

    embeddings_files = get_embedded_files(
        path_to_embeddings_file=PATH_TO_JSON_EMBEDDED_FILES
    )
    
    all_vectors = None
    result_df = None

    for file_ in chosen_files:

        logging.info(f'Path to index {embeddings_files[file_]}')
        index = faiss.read_index(os.path.join(PATH_TO_FAISS_VECTORS, embeddings_files[file_]))
        
        n = index.ntotal
        vectors = np.zeros((n, index.d), dtype=np.float32)
        index.reconstruct_n(0, n, vectors)

        all_vectors = np.vstack((all_vectors, vectors)) if all_vectors is not None else vectors

        filename, _ = os.path.splitext(file_)

        current_file_df = read_file(
            os.path.join(path_to_cleared_files, f'{filename}{cleared_files_ext}'), 
            columns=None)
        
        result_df = pd.concat([result_df, current_file_df])
        result_df = result_df.reset_index(drop=True)
    
    clusterable_embeddings = dimension_reduction(
        all_vectors,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state)
    
    logger.info(f"Applied UMAP to reduce dimensions of embeddings to {n_components}")
    
    dimensions_2d = dimension_reduction(
        all_vectors, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state)
    
    logger.info(f"Applied UMAP to reduce dimensions for visualization")

    labels = cluster_sentences(
        clusterable_embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,                      
        cluster_selection_method=cluster_selection_method)
    
    logger.info(f"Clusters calculated successfully, number of clusters: {len(set(labels))}")

    df = pd.DataFrame({
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1],
        labels_column: labels,
    })

    logger.info("Succesfully created dataframe for topic visuzalization")
    logger.debug(f'result_df: {result_df}')

    df = pd.concat([result_df, df], axis=1)

    logger.info("Succesfully merged dataframes into result df")
    logger.debug(f'result_df: {result_df}')

    return df