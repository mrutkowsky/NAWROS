import pandas as pd
import numpy as np
import logging
import lemminflect

import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

from utils.data_processing import preprocess_pipeline, load_lang_detector, load_translation_model, create_dataloader, detect_lang, translate_text

logger = logging.getLogger(__file__)

class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)
        self._idf_diag = None

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights)

        Args:
        X : sparse matrix of shape (n_samples, n_features)
            A matrix of term/token counts.
        n_samples : int -  number of samples in the dataset

        Returns:
        self : CTFIDFVectorizer
        """

        # Prepare input
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        # Calculate IDF scores
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        avg_nr_samples = int(X.sum(axis=1).mean())
        idf = np.log(avg_nr_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)
        return self

    def transform(self, X: sp.csr_matrix, copy=True) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF

        Args:
            X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts
            copy : bool, default True

        Returns:
            vectors : sparse matrix of shape (n_samples, n_features)

        """

        # Prepare input
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        # idf_ being a property, the automatic attributes detection
        # does not work as usual and we need to specify the attribute
        # name:
        check_is_fitted(self, attributes=["idf_"],
                        msg='idf vector is not fitted')

        # Check if expected nr features is found
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        X = X * self._idf_diag

        if self.norm:
            X = normalize(X, axis=1, norm='l1', copy=False)

        return X


def prepare_df_for_ctfidf(
        df: pd.DataFrame,
        stopwords: list,
        content_column_name: str = 'preprocessed_content',
        label_column_name: str = 'labels') -> pd.DataFrame:
    """Prepare a DataFrame for c-TF-IDF transformation by grouping documents per class.

    Args:
    df : pd.DataFrame
        Input DataFrame containing documents and labels.
    stopwords : list
        List of stopwords.
    content_column_name : str, optional
        Name of the column containing the document content, by default 'content'.
    label_column_name : str, optional
        Name of the column containing the document labels, by default 'labels'.

    Returns:
    pd.DataFrame
        DataFrame with documents per class.
    """
    docs_per_class = df.groupby(
        by=label_column_name, 
        as_index=False).agg({content_column_name: ' '.join})
    
    docs_per_class[content_column_name] = \
        docs_per_class[content_column_name].apply(lambda x: preprocess_pipeline(x, stopwords=stopwords))

    return docs_per_class


def perform_ctfidf(
        joined_texts: list or pd.Series,
        clusters_labels: list or pd.Series,
        df_number_of_rows: int,
        stop_words: list = None,
        outliers_topic_name: str = '<NO_TOPIC>',
        no_topic_token: str = '-') -> np.array:
    
    """Perform c-TF-IDF transformation on joined texts.

    Args:
    joined_texts : list or pd.Series
        List or Series of joined texts.
    clusters_labels : list or pd.Series
        List or Series of cluster labels.
    df_number_of_rows : int
        Number of rows in the original DataFrame.
    stop_words : list, optional
        List of stopwords, by default None.
    outliers_topic_name : str, optional
        Name of the topic for outliers, by default '<NO_TOPIC>'.
    no_topic_token : str, optional
        Token for no topic, by default '-'.

    Returns:
    np.array
        Array of c-TF-IDF values.

    """

    count_vectorizer = CountVectorizer().fit(joined_texts)
    count = count_vectorizer.transform(joined_texts)

    words = count_vectorizer.get_feature_names_out()

    ctfidf = CTFIDFVectorizer().fit_transform(
        count, 
        n_samples=df_number_of_rows).toarray()
    
    words_per_class = []

    if stop_words is None:
        stop_words = []
    
    words_per_class = []

    logger.debug(f'ct-idf: {ctfidf}')

    logger.debug(f'Cluster labels: {clusters_labels}')

    index_adder = 0 if -1 not in list(clusters_labels) else 1

    for label in clusters_labels:

        if label == -1:
            words_per_class.append(5 * [outliers_topic_name])
            continue

        current = []

        tf_idf_scores = sorted(ctfidf[label + index_adder], reverse=True)
        best_topics_idx = ctfidf[label + index_adder].argsort()[::-1]

        for score, idx in zip(tf_idf_scores, best_topics_idx):

            if score <= 0:
                while len(current) != 5:
                    current.append(no_topic_token)

            if len(current) == 5:
                 break

            if score > 0 and words[idx] not in stop_words:
                current.append(words[idx])
            else:
                continue

        words_per_class.append(current)
    
    return np.array(words_per_class)


def transform_topic_vec_to_df(
        topics_array: np.ndarray,
        topic_preffix_name: str = 'Word'):
    
    """Transform a topic vector to a DataFrame.

    Parameters:
    topics_array : np.ndarray
        Array of topics.
    topic_preffix_name : str, optional
        Prefix name for the topic columns, by default 'Word'.

    Returns:
    pd.DataFrame
        DataFrame of topics.

    """

    topics_df = pd.DataFrame(
        topics_array, 
        columns=[f'{topic_preffix_name}_{i}' for i in range(1, topics_array.shape[-1] + 1)])

    return topics_df

def get_topics_from_texts(
        df: pd.DataFrame,
        topic_preffix_name: str = 'Word',
        stop_words: list = None,
        content_column_name: str = 'preprocessed_content',
        label_column_name: str = 'labels',
        no_topic_token: str = '-') -> tuple[list]:
    
    """Get topics from texts using c-TF-IDF.

    Args:
    df : pd.DataFrame
        Input DataFrame containing documents and labels.
    topic_preffix_name : str, optional
        Prefix name for the topic columns, by default 'Word'.
    stop_words : list, optional
        List of stopwords, by default None.
    content_column_name : str, optional
        Name of the column containing the document content, by default 'content'.
    label_column_name : str, optional
        Name of the column containing the document labels, by default 'labels'.
    no_topic_token : str, optional
        Token for no topic, by default '-'.

    Returns:
    tuple[list]
        Tuple containing a list of topics.
    """

    df_for_topics = df.copy()
    
    logging.info('Start get_topics_from_texts() execution')

    n_of_rows = len(df_for_topics)

    docs_per_class = prepare_df_for_ctfidf(
        df=df_for_topics,
        stopwords=stop_words,
        content_column_name=content_column_name,
        label_column_name=label_column_name
    )

    logger.debug(f'Length docs_per_class: {len(docs_per_class)}')
    logger.debug(f'Number of classes: {len(docs_per_class[label_column_name])}')
    logger.debug(f'Classes: {docs_per_class[label_column_name]}')

    logging.info('Properly prepared df for c-tf-idf')

    topics_array = perform_ctfidf(
        joined_texts=docs_per_class[content_column_name],
        clusters_labels=docs_per_class[label_column_name],
        df_number_of_rows=n_of_rows,
        stop_words=stop_words,
        no_topic_token=no_topic_token)
    
    logging.info('Properly exctracted topics from clusters')
    
    topics_df = transform_topic_vec_to_df(
        topics_array=topics_array,
        topic_preffix_name=topic_preffix_name
    )

    topics_df[label_column_name] = sorted(docs_per_class[label_column_name])

    logging.info('End of executing - topic_df created succesfully')
    
    return topics_df