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
from datetime import datetime

from utils.data_processing import get_embedded_files, read_file, set_rows_cardinalities, save_df_to_file
from utils.c_tf_idf_module import get_topics_from_texts

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
    
    dim_reducer = UMAP.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='cosine').fit(sentence_embeddings)
    
    clusterable_embeddings = dim_reducer.transform(sentence_embeddings)

    return dim_reducer, clusterable_embeddings


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

def save_model(
        model,
        path_to_current_df_dir: str,
        model_name: str = 'clusterer.pkl'):
    """
        Save a machine learning model to a file.

        Args:
            model: The machine learning model to be saved.
            path_to_current_df_dir (str): The path to the directory where the model file will be saved.
            model_name (str, optional): The name of the model file. Defaults to 'clusterer.pkl'.

        Returns:
            None
    """

    with open(os.path.join(path_to_current_df_dir, model_name), 'wb') as file_:
        pickle.dump(model, file_)

def load_model(
        path_to_current_df_dir: str,
        model_name: str = 'clusterer.pkl'):
    """
        Save a machine learning model to a file.

        Args:
            model: The machine learning model to be saved.
            path_to_current_df_dir (str): The path to the directory where the model file will be saved.
            model_name (str, optional): The name of the model file. Defaults to 'clusterer.pkl'.

        Returns:
            None
    """

    with open(os.path.join(path_to_current_df_dir, model_name), 'rb') as file_:
        model = pickle.load(file_)

    return model

def load_embeddings_from_index(
        path_to_faiss_vector: str):
    """
       Save a machine learning model to a file.

       Args:
           model: The machine learning model to be saved.
           path_to_current_df_dir (str): The path to the directory where the model file will be saved.
           model_name (str, optional): The name of the model file. Defaults to 'clusterer.pkl'.

       Returns:
           None
    """

    index = faiss.read_index(path_to_faiss_vector)
        
    n = index.ntotal
    vectors = np.zeros((n, index.d), dtype=np.float32)
    index.reconstruct_n(0, n, vectors)

    return vectors

def concat_new_df_to_current_df(
    new_file_df: pd.DataFrame,
    current_df: pd.DataFrame,
    current_df_path: str) -> pd.DataFrame:
    """
        Save a machine learning model to a file.

        Args:
            model: The machine learning model to be saved.
            path_to_current_df_dir (str): The path to the directory where the model file will be saved.
            model_name (str, optional): The name of the model file. Defaults to 'clusterer.pkl'.

        Returns:
            None
    """

    new_current_df = pd.concat([current_df, new_file_df])
    new_current_df = new_current_df.reset_index(drop=True)

    new_current_df.to_parquet(
        path=current_df_path,
        index=False
    )

    return new_current_df

def load_embeddings_get_result_df(
        embeddings_files: list,
        chosen_files: list,
        path_to_faiss_vectors: str,
        path_to_cleared_files: str,
        cleared_files_ext: str = '.parquet.gzip',
        filename_column: str = 'filename',
        used_as_base_key: str = 'used_as_base',
        only_classified_key: str = 'only_classified'):
    """
       Load embeddings from Faiss vectors and generate a result DataFrame.

       Args:
           embeddings_files (list): A list of filenames for the embeddings files.
           chosen_files (list): A list of filenames for the chosen files.
           path_to_faiss_vectors (str): The path to the directory containing the Faiss vectors.
           path_to_cleared_files (str): The path to the directory containing the cleared files.
           cleared_files_ext (str, optional): The extension of the cleared files. Defaults to '.parquet.gzip'.
           filename_column (str, optional): The column name to store the filename in the result DataFrame. Defaults to 'filename'.
           used_as_base_key (str, optional): The key for the rows cardinalities of files used as base. Defaults to 'used_as_base'.
           only_classified_key (str, optional): The key for the rows cardinalities of files only classified. Defaults to 'only_classified'.

       Returns:
           tuple: A tuple containing the following elements:
               - all_vectors (np.ndarray): The concatenated embeddings vectors.
               - result_df (pd.DataFrame): The result DataFrame with the combined data from the chosen files.
               - rows_cardinalities_dict (dict): A dictionary containing the rows cardinalities of the files used as base and only classified.
    """

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
        umap_model_name: str = 'umap_reducer.pkl',
        reducer_2d_model_name: str = 'dim_reducer_2d.pkl',
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
    
    umap_dim_reducer, clusterable_embeddings = dimension_reduction(
        all_vectors,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state)
    
    save_model(
        model=umap_dim_reducer,
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=umap_model_name
    )
    
    logger.info(f"Applied UMAP to reduce dimensions of embeddings to {n_components}")
    
    dim_reducer_2d, dimensions_2d = dimension_reduction(
        all_vectors, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state)
    
    save_model(
        model=dim_reducer_2d,
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=reducer_2d_model_name
    )
    
    logger.info(f"Applied UMAP to reduce dimensions for visualization")

    clusterer = cluster_sentences(
        clusterable_embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,                      
        cluster_selection_method=cluster_selection_method)

    save_model(
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

def get_cluster_labels_for_new_file(
        filename: str,
        path_to_current_df: str,
        path_to_current_df_dir: str,
        path_to_cleared_files_dir: str,
        path_to_faiss_vetors_dir: str,
        required_columns: list,
        clusterer_model_name: str = 'clusterer.pkl',
        umap_model_name: str = 'umap_reducer.pkl',
        reducer_2d_model_name: str = 'dim_reducer_2d.pkl',
        cleared_files_ext: str = '.parquet.gzip',
        index_ext: str = '.index',
        filename_column: str = 'filename',
        label_column: str = 'labels'):
    """
       Get cluster labels for a new file and update the current DataFrame.

       Args:
           filename (str): The name of the new file.
           path_to_current_df (str): The path to the current DataFrame file.
           path_to_current_df_dir (str): The directory path of the current DataFrame file.
           path_to_cleared_files_dir (str): The directory path of the cleared files.
           path_to_faiss_vectors_dir (str): The directory path of the Faiss vectors.
           required_columns (list): A list of required column names in the new file DataFrame.
           clusterer_model_name (str, optional): The filename of the clusterer model. Defaults to 'clusterer.pkl'.
           umap_model_name (str, optional): The filename of the UMAP reducer model. Defaults to 'umap_reducer.pkl'.
           reducer_2d_model_name (str, optional): The filename of the 2D reducer model. Defaults to 'dim_reducer_2d.pkl'.
           cleared_files_ext (str, optional): The extension of the cleared files. Defaults to '.parquet.gzip'.
           index_ext (str, optional): The extension of the Faiss index file. Defaults to '.index'.
           filename_column (str, optional): The column name to store the filename in the new file DataFrame. Defaults to 'filename'.
           label_column (str, optional): The column name to store the cluster labels in the new file DataFrame. Defaults to 'labels'.

       Returns:
           pd.DataFrame: The updated current DataFrame with the new file data.
    """

    hdbscan_loaded_model = load_model(
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=clusterer_model_name
    )

    logger.debug(f'Loaded HDBSCAN model {clusterer_model_name}')

    umap_reducer = load_model(
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=umap_model_name
    )

    logger.debug(f'Loaded UMAP model {umap_model_name}')

    reducer_2d = load_model(
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=reducer_2d_model_name
    )

    logger.debug(f'Loaded UMAP model for 2D reduction {reducer_2d_model_name}')

    vector_embeddings = load_embeddings_from_index(
        os.path.join(path_to_faiss_vetors_dir, f"{os.path.splitext(filename)[0]}{index_ext}")
    )

    logger.debug(f'Successuffly laoded embeddings for {filename}')

    clusterable_embeddings = umap_reducer.transform(vector_embeddings)
    logger.info(f"Applied UMAP model for getting clusterable_embeddings")

    dimensions_2d = reducer_2d.transform(vector_embeddings)
    logger.info(f"Applied UMAP model for getting 2D coridinates for vizualization")

    labels_for_new_file, strenghts = hdbscan.approximate_predict(
        hdbscan_loaded_model, clusterable_embeddings)
    
    logger.info(f'Successfully calculated labels for new file {filename}')
    
    new_file_df = read_file(
        os.path.join(path_to_cleared_files_dir, f'{os.path.splitext(filename)[0]}{cleared_files_ext}'), 
        columns=None)
        
    new_file_df[filename_column] = filename

    cords_and_labels_df = pd.DataFrame({
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1],
        label_column: labels_for_new_file,
    })

    new_file_df = pd.concat([new_file_df, cords_and_labels_df], axis=1)

    logger.info(f'Prepared df {filename} for concatenation with current_df')

    current_df = read_file(
        file_path=path_to_current_df
    )

    new_current_df = concat_new_df_to_current_df(
        new_file_df=new_file_df.astype({col: str for col in required_columns}),
        current_df=current_df,
        current_df_path=path_to_current_df
    )

    logger.info(f'Updated current_df with new rows from {filename}')

    return new_current_df

def cluster_recalculation_needed(
    n_of_rows: int,
    rows_cardinalities_current_df: dict,
    recalculate_treshold: float,
    used_as_base_key: str = 'used_as_base',
    only_classified_key: str = 'only_classified') -> bool:
    """
        Check if cluster recalculation is needed based on the number of rows and cardinalities.

        Args:
            n_of_rows (int): The number of rows to be added.
            rows_cardinalities_current_df (dict): The dictionary of rows cardinalities in the current DataFrame.
            recalculate_threshold (float): The threshold value for determining if recalculation is needed.
            used_as_base_key (str, optional): The key for base cluster cardinalities in the dictionary. Defaults to 'used_as_base'.
            only_classified_key (str, optional): The key for only classified cluster cardinalities in the dictionary. Defaults to 'only_classified'.

        Returns:
            bool: True if cluster recalculation is needed, False otherwise.
    """

    try:
        n_of_rows_for_base = sum(rows_cardinalities_current_df.get(used_as_base_key).values())
    except ValueError:
        logger.error('Can not calculate number of rows used for base clusterization')
        return None
    else:

        only_classified_dict = rows_cardinalities_current_df.get(only_classified_key)

        if not only_classified_dict:
            n_of_only_classified = 0
        else:
            n_of_only_classified = sum(only_classified_dict.values())

        logger.debug(f'Sum: {n_of_only_classified + n_of_rows}')
        logger.debug(f'Threshold: {n_of_rows_for_base * recalculate_treshold}')

        if n_of_only_classified + n_of_rows \
            <= n_of_rows_for_base * recalculate_treshold:

            return False
        
        return True
    
def join_topics_to_df(
        detailed_df: pd.DataFrame,
        topics_df: pd.DataFrame,
        joining_column: str = 'labels') -> pd.DataFrame:
        """
        Join topics to the detailed DataFrame based on a joining column.

        Args:
            detailed_df (pd.DataFrame): The detailed DataFrame.
            topics_df (pd.DataFrame): The DataFrame containing topics.
            joining_column (str, optional): The column used for joining the DataFrames. Defaults to 'labels'.

        Returns:
            pd.DataFrame: The joined DataFrame.
        """

        joined_topics_df = detailed_df.join(
            topics_df.set_index(joining_column),
            on=joining_column,
            how='left'
        )

        return joined_topics_df

def save_cluster_exec_report(
        df: pd.DataFrame, 
        filename: str,
        path_to_cluster_exec_reports_dir: str,
        clusters_topics: pd.DataFrame,
        filename_ext: str = '.gzip.parquet',
        labels_column_name: str = 'labels',
        cardinalities_column_name: str = 'counts'):
    
    """
    Saves a report to a CSV file.

    Args:
       df (pd.DataFrame): The DataFrame containing the report data.
       filename (str): The filename of the CSV file.
       path_to_raports_dir (str): The directory path to save the CSV file.
       clusters_topics (pd.DataFrame): The DataFrame containing cluster topics.
       classes_column_name (str, optional): The column name for the classes. Defaults to 'labels'.

   Returns:
       None
   """
    df = df.groupby(df[labels_column_name]).size().reset_index(name=cardinalities_column_name)
    df = pd.concat([df, clusters_topics.drop(columns=[labels_column_name])], axis=1) 

    save_df_to_file(
        df=df,
        filename=filename,
        path_to_dir=path_to_cluster_exec_reports_dir,
        file_ext=filename_ext
    )

def cns_after_clusterization(
        new_current_df: pd.DataFrame,
        path_to_current_df_dir: str,
        path_to_cluster_exec_dir: str,
        stop_words: list = None,
        only_update: bool = False,
        topic_df_file_name: str = None,
        current_df_filename: str = 'current_df',
        content_column_name: str = 'content',
        labels_column: str = 'labels',
        cardinalities_column: str = 'counts',
        no_topic_token: str = '-',
        cluster_exec_filename_prefix: str = 'cluster_exec',
        cluster_exec_filename_ext: str = '.parquet.gzip'):
    """
        Perform post-processing steps after clusterization.

        Args:
            new_current_df (pd.DataFrame): The new current DataFrame after clusterization.
            path_to_current_df_dir (str): The path to the directory where the current DataFrame is stored.
            path_to_cluster_exec_dir (str): The path to the directory where cluster execution reports will be saved.
            stop_words (list, optional): List of stop words to be removed. Defaults to None.
            only_update (bool, optional): Flag indicating whether to only update the existing data without re-computing topics. Defaults to False.
            topic_df_file_name (str, optional): The filename for the topic DataFrame if only_update is True. Defaults to None.
            current_df_filename (str, optional): The filename for the updated current DataFrame. Defaults to 'current_df'.
            content_column_name (str, optional): The name of the column in the DataFrame that contains the content. Defaults to 'content'.
            labels_column (str, optional): The name of the column in the DataFrame that contains the cluster labels. Defaults to 'labels'.
            cardinalities_column (str, optional): The name of the column in the clusterization execution report that contains the cluster cardinalities. Defaults to 'counts'.
            no_topic_token (str, optional): The token to represent clusters without assigned topics. Defaults to '-'.
            cluster_exec_filename_prefix (str, optional): The prefix for the clusterization execution report filename. Defaults to 'cluster_exec'.
            cluster_exec_filename_ext (str, optional): The file extension for the clusterization execution report. Defaults to '.parquet.gzip'.

        Returns:
            None
    """

    TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"


    if not only_update:

        clusters_topics_df = get_topics_from_texts(
            df=new_current_df,
            stop_words=stop_words,
            content_column_name=content_column_name,
            label_column_name=labels_column,
            no_topic_token=no_topic_token
        )

        logger.debug('Extracted topics from DataFrame')

        new_current_df = join_topics_to_df(
            detailed_df=new_current_df,
            topics_df=clusters_topics_df,
            joining_column=labels_column
        )

        logger.debug('Joined topics to new current_df')

        new_current_df.to_parquet(
            index=False, 
            path=os.path.join(path_to_current_df_dir, current_df_filename))
        
        logger.debug('Saved current_df on disk')

        clusters_topics_df.to_csv(
            os.path.join(path_to_current_df_dir, topic_df_file_name),
            index=False
        )

        logger.debug('Saved topic df on disk')

    else:

        clusters_topics_df = read_file(
            file_path=os.path.join(path_to_current_df_dir, topic_df_file_name)
        )

    clusterization_exec_filename = f"""{cluster_exec_filename_prefix}_{datetime.now().strftime(TIMESTAMP_FORMAT)}"""

    save_cluster_exec_report(
        df=new_current_df,
        filename=clusterization_exec_filename,
        filename_ext=cluster_exec_filename_ext,
        path_to_cluster_exec_reports_dir=path_to_cluster_exec_dir,
        clusters_topics=clusters_topics_df,
        labels_column_name=labels_column,
        cardinalities_column_name=cardinalities_column
    )

    logger.info(f'Report from clusterization execution: {clusterization_exec_filename}{cluster_exec_filename_ext} has been successfully saved on disk')

    return None




