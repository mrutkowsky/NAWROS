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
from itertools import product
import scipy

from utils.data_processing import get_embedded_files, read_file, set_rows_cardinalities, save_df_to_file, get_report_name_with_timestamp
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
        coverage_with_best: float = 0.87,
        min_cluster_size: list = [5, 10, 15, 30, 45, 60],
        min_samples: list = [5, 10, 15, 20, 30],
        metric: list = ['euclidean', 'manhattan'],                      
        cluster_selection_method: list = ['eom']):
    
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

    param_dist = {
        'min_samples': min_samples,
        'min_cluster_size': min_cluster_size,
        'metric': metric,
        'cluster_selection_method': cluster_selection_method 
    }

    possible_combinations_parameters = list(product(
        param_dist.get('min_samples'), 
        param_dist.get('min_cluster_size'), 
        param_dist.get('metric'),
        param_dist.get('cluster_selection_method')))
    
    scores = {}

    for min_samples, min_cluster_size, metric, cluster_selection_method in possible_combinations_parameters:

        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric=metric,                
                                    cluster_selection_method=cluster_selection_method,
                                    gen_min_span_tree=True,
                                    prediction_data=True).fit(clusterable_embeddings)
        
        logger.debug(f'Execution: {min_samples=}, {min_cluster_size=}, {metric=}, {cluster_selection_method=} - validity: {cluster.relative_validity_} - num labels: {len(set(cluster.labels_))}')
        
        scores[(min_samples, min_cluster_size, metric, cluster_selection_method)] = {
            'validity': cluster.relative_validity_, 
            'num_labels': len(set(cluster.labels_))
        }
    
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1]['validity'], reverse=True))
    best_validity = max(sorted_scores.values(), key=lambda x: x['validity'])['validity']

    to_choose = {params: metrics for params, metrics in sorted_scores.items() if metrics.get('validity') / best_validity >= coverage_with_best}
    sorted_by_num_labels = dict(sorted(to_choose.items(), key=lambda x: x[1]['num_labels'], reverse=True))

    best_min_samples, best_min_cluster_size, best_metric, best_cluster_selection_method = list(sorted_by_num_labels.keys())[0]

    logger.info(f'Chosen params: {list(sorted_by_num_labels.keys())[0]}')
    logger.info(f'Scores for chosen params: {sorted_by_num_labels.get(list(sorted_by_num_labels.keys())[0])}')

    cluster = hdbscan.HDBSCAN(
        min_samples=best_min_samples,
        min_cluster_size=best_min_cluster_size,
        metric=best_metric,                      
        cluster_selection_method=best_cluster_selection_method,
        prediction_data=True) \
            .fit(clusterable_embeddings)

    return cluster

def perform_soft_clustering(
    clusterer: hdbscan.HDBSCAN,
    original_labels: np.array = None,
    new_points: np.array = None,
    outlier_treshold: float = 0.1) -> list:

    logger.debug(f'Start of execution of soft_clustering')

    soft_clusters = hdbscan.all_points_membership_vectors(clusterer) \
        if new_points is None else hdbscan.membership_vector(clusterer, new_points)
    
    if soft_clusters.ndim == 1:
        return original_labels
    
    logger.debug(f'Type of soft_clusters: {type(soft_clusters)}')

    logger.debug(f'soft_clusters: {soft_clusters}')
    
    closest_cluster_label = [np.argmax(x) for x in soft_clusters]
    closest_cluster_prob = [max(x) for x in soft_clusters]

    if original_labels is not None:

        new_labels = [
            label if (original_labels[i] == -1) and (closest_cluster_prob[i] > outlier_treshold)
            else original_labels[i] 
            for i, label in enumerate(closest_cluster_label)
        ]

        logger.debug(f'Number of outliers before soft clustering: {list(original_labels).count(-1)}')
        logger.debug(f'Number of outliers after soft clustering: {list(new_labels).count(-1)}')
        
    else:

        new_labels = [
            label if prob > outlier_treshold else -1 
            for label, prob in zip(closest_cluster_label, closest_cluster_prob)
        ]

    logger.debug(f'End of execution soft_clustering')

    return new_labels, closest_cluster_prob

def save_model(
        model,
        path_to_current_df_dir: str,
        model_name: str = 'clusterer.pkl'): 
    
    with open(os.path.join(path_to_current_df_dir, model_name), 'wb') as file_:
        pickle.dump(model, file_)

def load_model(
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

def concat_new_df_to_current_df(
    new_file_df: pd.DataFrame,
    current_df: pd.DataFrame,
    current_df_path: str) -> pd.DataFrame:

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

def get_most_representantive_tickets(
    df: pd.DataFrame,
    one_prob_indexes: list,
    clusterable_embeddings: np.array,
    label_column_name: str = 'labels',
    content_column_name: str = 'preprocessed_content',
    cluster_summary_column: str = 'cluster_summary'):

    only_1_prob = df[[label_column_name, content_column_name]].iloc[one_prob_indexes]

    representative_contents = {}

    labels_to_iterate_over = sorted(list(set(df[label_column_name].unique()).difference(set([-1]))))

    if not labels_to_iterate_over:
        return pd.DataFrame(columns=[label_column_name, cluster_summary_column])

    for label in labels_to_iterate_over:

        label_indexes = list(only_1_prob.loc[only_1_prob[label_column_name] == label].index)

        dist = scipy.spatial.distance.cdist(
            clusterable_embeddings[label_indexes], 
            clusterable_embeddings[label_indexes])
        
        representative_contents[label] = np.argsort(np.sum(dist, axis=1))[:5]

    logger.debug(f'Calculated candidates for most representantive samples')

    needed_for_comparison_indexes = []

    for label, idx in representative_contents.items():

        needed_for_comparison_indexes.extend(
            list(only_1_prob.loc[only_1_prob[label_column_name] == label] \
                 .iloc[list(idx)][content_column_name].index))
        
    logger.debug(f'Got indexes for most most representantive samples')

    if len(labels_to_iterate_over) == 1:
        return pd.DataFrame(
            {label_column_name: labels_to_iterate_over[0],
             cluster_summary_column: list(df.iloc[needed_for_comparison_indexes[0]][content_column_name])})

    reshaped_candidates_indexes = np.array(needed_for_comparison_indexes).reshape(-1, 5)

    logger.debug(f'Reshaped array with candidates')

    dist_cands = scipy.spatial.distance.cdist(
        clusterable_embeddings[needed_for_comparison_indexes], 
        clusterable_embeddings[needed_for_comparison_indexes])
    
    logger.debug(f'Calculated distances between most representantive samples among clusters')

    adder = 0
    to_sub = []

    for i, elem in enumerate(dist_cands):

        if (i % 5 == 0) and (i != 0):
            adder += 1

        to_sub.append(np.sum(elem[5 * adder : 5 * (adder + 1)]))

    to_sub = np.array(to_sub)

    logger.debug(f'Calculated distances to subtract (within cluster)')

    chosen_candidates_indexes_in_group = np.argmax((np.sum(dist_cands, axis=1) - to_sub).reshape(-1, 5), axis=1)

    logger.debug(f'Exctracted indexes of best representantive samples')

    chosen_for_labels = []

    for row_label, chosen_can in zip(reshaped_candidates_indexes, chosen_candidates_indexes_in_group):
        chosen_for_labels.append(row_label[chosen_can])
    
    most_representantive_df = pd.DataFrame({
        label_column_name: labels_to_iterate_over,
        cluster_summary_column: list(df.iloc[chosen_for_labels][content_column_name])})

    logger.debug(f'Successfully extracted most representantive contents for each cluster')

    return most_representantive_df

def get_clusters_for_choosen_files(
        chosen_files: list,
        path_to_cleared_files: str,
        path_to_embeddings_dir: str,
        path_to_current_df_dir: str,
        rows_cardinalities_file: str,
        faiss_vectors_dirname: str,
        embedded_files_filename: str,
        cluster_summary_filename: str = 'topics_df',
        cluster_summary_ext: str = '.csv',
        used_as_base_key: str = 'used_as_base',
        only_classified_key: str = 'only_classified',
        cleared_files_ext: str = '.parquet.gzip',
        content_column_name: str = 'preprocessed_content',
        labels_column: str = 'labels',
        filename_column: str = 'filename',
        cluster_summary_column: str = 'cluster_summary',
        clusterer_model_name: str = 'clusterer.pkl',
        umap_model_name: str = 'umap_reducer.pkl',
        reducer_2d_model_name: str = 'dim_reducer_2d.pkl',
        outlier_treshold: float = 0.1,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        n_components: int = 5,
        coverage_with_best: float = 0.87,
        min_cluster_size: list = [5, 10, 15, 30, 45, 60],
        min_samples: list = [5, 10, 15, 20, 30],
        metric: list = ['euclidean', 'manhattan'],                      
        cluster_selection_method: list = ['eom']) -> pd.DataFrame:
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

    MIN_SAMPLES = 15
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
        used_as_base_key=used_as_base_key,
        only_classified_key=only_classified_key)
    
    n_of_samples = sum(rows_cardinalities_dict.get(used_as_base_key, {}).values())
    if n_of_samples < MIN_SAMPLES:
        return f'Number of samples should be at least {MIN_SAMPLES} to perform clusterization.'
    
    try:
    
        umap_dim_reducer, clusterable_embeddings = dimension_reduction(
            all_vectors,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state)
        
    except (TypeError, ValueError):
        return f'Number of samples is to small ({n_of_samples}) to use it with current settings.'
    
    else:
    
        save_model(
            model=umap_dim_reducer,
            path_to_current_df_dir=path_to_current_df_dir,
            model_name=umap_model_name
        )
    
    logger.info(f"Applied UMAP to reduce dimensions of embeddings to {n_components}")

    try:

        clusterer = cluster_sentences(
            clusterable_embeddings,
            coverage_with_best=coverage_with_best,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,                      
            cluster_selection_method=cluster_selection_method)
        
    except (TypeError, ValueError):
        return f'Number of samples is to small ({n_of_samples}) to use it with current settings.'
    
    else:

        logger.info(f"Successfully clustered embeddings")

        save_model(
            model=clusterer,
            path_to_current_df_dir=path_to_current_df_dir,
            model_name=clusterer_model_name
        )
    
    cluster_labels, closest_cluster_prob = perform_soft_clustering(
        clusterer=clusterer,
        original_labels=clusterer.labels_,
        new_points=None,
        outlier_treshold=outlier_treshold
    )

    logger.info(f"Clusters calculated successfully, number of clusters: {len(set(cluster_labels))}")
    
    dim_reducer_2d, dimensions_2d = dimension_reduction(
        all_vectors, 
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state)
    
    dimensions_2d = np.round(dimensions_2d, decimals=2)
    
    save_model(
        model=dim_reducer_2d,
        path_to_current_df_dir=path_to_current_df_dir,
        model_name=reducer_2d_model_name
    )
    
    logger.info(f"Applied UMAP to reduce dimensions for visualization")

    df = pd.DataFrame({
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1],
        labels_column: cluster_labels,
    })

    logger.info("Succesfully created dataframe for topic visuzalization")
    logger.debug(f'result_df: {result_df}')

    df = pd.concat([result_df, df], axis=1)

    logger.info("Succesfully merged dataframes into result df")
    logger.debug(f'result_df: {result_df}')

    most_represent_df = get_most_representantive_tickets(
        df=df,
        one_prob_indexes=np.where(np.array(closest_cluster_prob) == 1)[0],
        clusterable_embeddings=clusterable_embeddings,
        label_column_name=labels_column,
        content_column_name=content_column_name,
        cluster_summary_column=cluster_summary_column)
    
    logger.info("Succesfully exctracted summary clusters df")

    save_df_to_file(
        df=most_represent_df,
        filename=cluster_summary_filename,
        path_to_dir=path_to_current_df_dir,
        file_ext=cluster_summary_ext
    )

    logger.info("Succesfully saved summary clusters df")

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
        outlier_treshold: float = 0.1,
        topic_df_filename: str = 'topics_df.csv',
        clusterer_model_name: str = 'clusterer.pkl',
        umap_model_name: str = 'umap_reducer.pkl',
        reducer_2d_model_name: str = 'dim_reducer_2d.pkl',
        cleared_files_ext: str = '.parquet.gzip',
        index_ext: str = '.index',
        filename_column: str = 'filename',
        label_column: str = 'labels'):
    
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

    try:
        clusterable_embeddings = umap_reducer.transform(vector_embeddings)
    except (TypeError, ValueError):
        return f'Number of samples is to small to use it with current settings.'
    else:
        logger.info(f"Applied UMAP model for getting clusterable_embeddings")

    dimensions_2d = np.round(reducer_2d.transform(vector_embeddings), decimals=2)
    logger.info(f"Applied UMAP model for getting 2D coridinates for vizualization")

    try:

        labels_for_new_file, _ = perform_soft_clustering(
            clusterer=hdbscan_loaded_model,
            original_labels=None,
            new_points=clusterable_embeddings,
            outlier_treshold=outlier_treshold
        )
    
    except (TypeError, ValueError):
        return f'Number of samples is to small to use it with current settings.'
    
    else:
        logger.info(f'Successfully calculated labels for new file {filename}')
    
    new_file_df = read_file(
        os.path.join(path_to_cleared_files_dir, f'{os.path.splitext(filename)[0]}{cleared_files_ext}'))
        
    new_file_df[filename_column] = filename

    cords_and_labels_df = pd.DataFrame({
        'x': dimensions_2d[:, 0],
        'y': dimensions_2d[:, 1],
        label_column: labels_for_new_file,
    })

    new_file_df = pd.concat([new_file_df, cords_and_labels_df], axis=1)

    logger.debug(f'Concateneted new_file_df with cords_and_labels_df')

    topics_df = read_file(os.path.join(path_to_current_df_dir, topic_df_filename))

    new_file_df = join_topics_to_df(
        detailed_df=new_file_df,
        topics_df=topics_df,
        joining_column=label_column
    )

    logger.debug(f'Joined topics df to {filename}')

    current_df = read_file(
        file_path=path_to_current_df
    )

    logger.debug(f'Prepared df {filename} for concatenation with current_df')

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

        operate_topics_df = topics_df.copy()

        logger.debug(f'Detailed df Columns: {list(detailed_df.columns)}')

        if joining_column not in operate_topics_df.columns:
            operate_topics_df[joining_column] = np.arange(-1, len(operate_topics_df) - 1)

        joined_topics_df = detailed_df.join(
            operate_topics_df.set_index(joining_column),
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
    
    if len(df) == len(clusters_topics):
        df = pd.concat([df, clusters_topics.drop(columns=[labels_column_name])], axis=1) 
    else:
        df = df.merge(clusters_topics, on=labels_column_name, how='left')

    path_to_exec_report, destination_filename = save_df_to_file(
        df=df,
        filename=filename,
        path_to_dir=path_to_cluster_exec_reports_dir,
        file_ext=filename_ext
    )

    return df, path_to_exec_report, destination_filename

def cns_after_clusterization(
        new_current_df: pd.DataFrame,
        path_to_current_df_dir: str,
        path_to_cluster_exec_dir: str,
        stop_words: list = None,
        only_update: bool = False,
        topic_df_file_name: str = None,
        topic_preffix_name: str = 'Word',
        current_df_filename: str = 'current_df',
        content_column_name: str = 'preprocessed_content',
        labels_column: str = 'labels',
        cardinalities_column: str = 'counts',
        topics_concat_viz_col: str = 'topics',
        no_topic_token: str = '-',
        cluster_exec_filename_prefix: str = 'cluster_exec',
        cluster_exec_filename_ext: str = '.parquet.gzip'):

    TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"

    if not only_update:

        clusters_topics_df = get_topics_from_texts(
            df=new_current_df,
            topic_preffix_name=topic_preffix_name,
            stop_words=stop_words,
            content_column_name=content_column_name,
            label_column_name=labels_column,
            no_topic_token=no_topic_token
        )

        logger.debug('Extracted topics from DataFrame')

        only_summaries_df = read_file(
            file_path=os.path.join(path_to_current_df_dir, topic_df_file_name)
        )

        clusters_topics_df = clusters_topics_df.merge(
            only_summaries_df,
            on=labels_column,
            how='left'
        ).sort_values(by=labels_column)

        logger.debug(f'Joined cluster summaries to topic_df')

        clusters_topics_df.to_csv(
            os.path.join(path_to_current_df_dir, topic_df_file_name),
            index=False
        )

        logger.debug('Saved topic df on disk')


        new_current_df = join_topics_to_df(
            detailed_df=new_current_df,
            topics_df=clusters_topics_df,
            joining_column=labels_column
        )

        logger.debug('Joined topics to new current_df')

        new_current_df[topics_concat_viz_col] = \
            new_current_df[[f"{topic_preffix_name}_{i}" for i in range(1, 4)]].apply(lambda x: ', '.join([str(v) for v in x]), axis=1)

        new_current_df.to_parquet(
            index=False, 
            path=os.path.join(path_to_current_df_dir, current_df_filename))
        
        logger.debug('Saved current_df on disk')

    else:

        clusters_topics_df = read_file(
            file_path=os.path.join(path_to_current_df_dir, topic_df_file_name)
        )

    clusterization_exec_filename = get_report_name_with_timestamp(
        filename_prefix=cluster_exec_filename_prefix,
        timestamp_format=TIMESTAMP_FORMAT
    )

    _, path_to_exec_report, destination_filename = save_cluster_exec_report(
        df=new_current_df,
        filename=clusterization_exec_filename,
        filename_ext=cluster_exec_filename_ext,
        path_to_cluster_exec_reports_dir=path_to_cluster_exec_dir,
        clusters_topics=clusters_topics_df,
        labels_column_name=labels_column,
        cardinalities_column_name=cardinalities_column
    )

    logger.info(f'Report from clusterization execution: {clusterization_exec_filename}{cluster_exec_filename_ext} has been successfully saved on disk')

    return path_to_exec_report, destination_filename




