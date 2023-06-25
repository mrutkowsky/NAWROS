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
import pickle

from utils.data_processing import get_embedded_files, read_file, set_rows_cardinalities

logger = logging.getLogger(__file__)

def dimension_reduction(
        sentence_embeddings: np.ndarray, 
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        n_components: int = 5,
        random_state: int = 42):
    
    """
    Perform dimension reduction on sentence embeddings using UMAP.

    Args:
        sentence_embeddings (np.ndarray): Array of sentence embeddings.
        n_neighbors (int): The number of nearest neighbors to consider during UMAP dimension reduction.
        min_dist (float): The minimum distance between points in the UMAP embedding.
        n_components (int): The number of components (dimensions) in the UMAP embedding.
        random_state (int): Random seed for UMAP.

    Returns:
        np.ndarray: The UMAP embedding of the sentence embeddings.
    """
    
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
    
    """
    Perform clustering on the clusterable embeddings using HDBSCAN.

    Args:
        clusterable_embeddings: Embeddings to be clustered.
        min_cluster_size (int): The minimum size of a cluster.
        min_samples (int): The minimum number of samples in a neighborhood to be considered a core point.
        metric (str): The metric to use for distance calculations.
        cluster_selection_method (str): The method used to select clusters from the condensed tree.

    Returns:
        np.ndarray: The cluster labels for each embedding.
    """
    
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,                      
        cluster_selection_method=cluster_selection_method,
        prediction_data=True) \
            .fit(clusterable_embeddings)

    return cluster

def save_hdbscan_model(
        model: hdbscan.HDBSCAN,
        path_to_current_df_dir: str,
        model_name: str = 'clusterer.pkl'): 
    
    with open(os.path.join(path_to_current_df_dir, model_name), 'wb') as file_:
        pickle.dump(model, file_)

def load_hdbscan_model(
        path_to_current_df_dir: str,
        model_name: str = 'clusterer.pkl'): 
    
    with open(os.path.join(path_to_current_df_dir, model_name), 'rb') as file_:
        model = pickle.load(file_)

    return model

def load_embeddings_from_index(path_to_faiss_vector: str):
    
    index = faiss.read_index(path_to_faiss_vector)
        
    n = index.ntotal
    vectors = np.zeros((n, index.d), dtype=np.float32)
    index.reconstruct_n(0, n, vectors)

    return vectors

def load_embeddings_get_result_df(
        embeddings_files: list,
        chosen_files: list,
        path_to_faiss_vectors: str,
        path_to_cleared_files: str,
        cleared_files_ext: str = '.parquet.gzip',
        filename_column: str = 'filename',
        used_as_base_key: str = 'used_as_base',
        only_classified_key: str = 'only_classified'):
    
    rows_cardinalities_dict = {
        used_as_base_key: {},
        only_classified_key: {}
    }
    
    all_vectors = None
    result_df = None

    for file_ in chosen_files:

        logging.info(f'Path to index {embeddings_files[file_]}')
        path_to_faiss_vector_file = os.path.join(path_to_faiss_vectors, embeddings_files[file_])

        vectors = load_embeddings_from_index(path_to_faiss_vector_file)
       
        all_vectors = np.vstack((all_vectors, vectors)) if all_vectors is not None else vectors

        filename, _ = os.path.splitext(file_)

        current_file_df = read_file(
            os.path.join(path_to_cleared_files, f'{filename}{cleared_files_ext}'), 
            columns=None)
        
        current_file_df[filename_column] = file_

        rows_cardinalities_dict[used_as_base_key][file_] = len(current_file_df)
        
        result_df = pd.concat([result_df, current_file_df])
        result_df = result_df.reset_index(drop=True)

    return all_vectors, result_df, rows_cardinalities_dict


def get_clusters_for_choosen_files(
        chosen_files: list,
        path_to_cleared_files: str,
        path_to_embeddings_dir: str,
        path_to_current_df_dir: str,
        rows_cardinalities_file: str,
        faiss_vectors_dirname: str,
        embedded_files_filename: str,
        used_as_base_key: str = 'used_as_base',
        only_classified_key: str = 'only_classified',
        cleared_files_ext: str = '.parquet.gzip',
        labels_column: str = 'labels',
        filename_column: str = 'filename',
        clusterer_model_name: str = 'clusterer.pkl',
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int = 15,
        metric: str = 'euclidean',                      
        cluster_selection_method: str = 'eom') -> pd.DataFrame:
    """
    Calculate clusters based on vectors for chosen files and return a dataframe with labels and 2D coordinates for each sentence.

    Args:
       chosen_files (list): List of file names to process.
       path_to_cleared_files (str): Path to the directory containing cleared files.
       path_to_embeddings_dir (str): Path to the directory containing embeddings.
       faiss_vectors_dirname (str): Directory name where the Faiss vectors are stored.
       embedded_files_filename (str): Filename of the embedded files.
       cleared_files_ext (str, optional): Extension of the cleared files. Defaults to '.gzip.parquet'.
       labels_column (str, optional): Column name for the cluster labels. Defaults to 'labels'.
       random_state (int, optional): Random seed. Defaults to 42.
       n_neighbors (int, optional): The number of nearest neighbors to consider during UMAP dimension reduction. Defaults to 15.
       min_dist (float, optional): The minimum distance between points in the UMAP embedding. Defaults to 0.0.
       n_components (int, optional): The number of components (dimensions) in the UMAP embedding. Defaults to 5.
       min_cluster_size (int, optional): The minimum size of a cluster. Defaults to 15.
       min_samples (int, optional): The minimum number of samples in a neighborhood to be considered a core point. Defaults to 10.
       metric (str, optional): The metric to use for distance calculations. Defaults to 'euclidean'.
       cluster_selection_method (str, optional): The method used to select clusters from the condensed tree. Defaults to 'leaf'.

    Returns:
       pd.DataFrame: Dataframe with labels and 2D coordinates for each sentence.
    """

    PATH_TO_JSON_EMBEDDED_FILES = os.path.join(path_to_embeddings_dir, embedded_files_filename)
    PATH_TO_FAISS_VECTORS = os.path.join(path_to_embeddings_dir, faiss_vectors_dirname)

    embeddings_files = get_embedded_files(
        path_to_embeddings_file=PATH_TO_JSON_EMBEDDED_FILES
    )

    all_vectors, result_df, rows_cardinalities_dict = load_embeddings_get_result_df(
        embeddings_files=embeddings_files,
        chosen_files=chosen_files,
        path_to_faiss_vectors=PATH_TO_FAISS_VECTORS,
        path_to_cleared_files=path_to_cleared_files,
        cleared_files_ext=cleared_files_ext,
        filename_column=filename_column,
        used_as_base_key=used_as_base_key)
    
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

    clusterer = cluster_sentences(
        clusterable_embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,                      
        cluster_selection_method=cluster_selection_method)

    save_hdbscan_model(
        model=clusterer,
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=clusterer_model_name
    )
    
    logger.info(f"Clusters calculated successfully, number of clusters: {len(set(clusterer.labels_))}")

    df = pd.DataFrame({
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1],
        labels_column: clusterer.labels_,
    })

    logger.info("Succesfully created dataframe for topic visuzalization")
    logger.debug(f'result_df: {result_df}')

    df = pd.concat([result_df, df], axis=1)

    logger.info("Succesfully merged dataframes into result df")
    logger.debug(f'result_df: {result_df}')

    saved_cardinalities = set_rows_cardinalities(
        path_to_cardinalities_file=os.path.join(path_to_current_df_dir, rows_cardinalities_file),
        updated_cardinalities=rows_cardinalities_dict
    )

    if isinstance(saved_cardinalities, bool):
        logger.info('Rows cardinalities of each file which is a part of current_df saved successfully')
    else:
        logger.error('Failed to save rows cardinalities for current_df')

    return df