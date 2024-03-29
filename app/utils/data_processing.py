import os
import re
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter
from datetime import datetime
import io
from flask import make_response, Response
import lemminflect

from utils.embeddings_module import load_transformer_model, get_embeddings
from utils.etl import preprocess_text
from utils.sentiment_analysis import load_sentiment_model, predict_sentiment, offensive_language
from utils.translation import load_lang_detector, load_translation_model, detect_lang, translate_text
from dateutil import parser

logger = logging.getLogger(__file__)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data.values

    def __getitem__(self, index):
        return self.data[index][0]

    def __len__(self):
        return len(self.data)
    
def read_file(
        file_path: str, 
        columns: list = None) -> pd.DataFrame:
    
    """
    Read a file and return its content as a DataFrame.

    Args:
        file_path (str): The path to the file.
        columns (list, optional): A list of columns to read from the file. Defaults to None.

    Returns:
        pd.DataFrame: The content of the file as a DataFrame.
    """

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, usecols=columns)
    elif file_path.endswith('.parquet.gzip'):
        df = pd.read_parquet(file_path, columns=columns)
    else:

        for sep in [',', ';']:

            try:

                df_one_col = pd.read_csv(
                    file_path, 
                    usecols=columns,
                    sep=sep,
                    nrows=1)
        
            except (ValueError, pd.errors.ParserError):
                continue

            else:

                df = pd.read_csv(
                    filepath_or_buffer=file_path, 
                    usecols=columns,
                    sep=sep)

                break

    logger.debug(f'df: {df}')

    return df


def cleanup_data(
        df: pd.DataFrame, 
        content_column_name: str = 'preprocessed_content',
        dropped_indexes_column_name: str = 'dropped_indexes') -> pd.DataFrame:
    
    """
    Remove rows with empty contents, save them to a separate file, and preprocess the remaining contents.

    Args:
        df (pd.DataFrame): The DataFrame to clean up.
        filename (str): The name of the file being cleaned.
        path_to_empty_content_dir (str): The directory path to save the file with empty contents.
        empty_contents_suffix (str): The suffix to append to the filename for the file with empty contents.
        empty_content_ext (str): The file extension for the file with empty contents.
        content_column_name (str, optional): The name of the column containing the contents.
            Defaults to 'preprocessed_content'.
        dropped_indexes_column_name (str, optional): The name of the column containing the dropped indexes.
            Defaults to 'dropped_indexes'.

    Returns:
        pd.DataFrame: The cleaned-up DataFrame.
    """

    logger.debug(f'Columns: {df.columns}')
    logger.debug(f"{type(df[content_column_name])}")

    float_contents_indexes = np.where(df[content_column_name].apply(lambda x: not isinstance(x, str)))[0]

    df.drop(index=float_contents_indexes, inplace=True)

    preprocessed_contents = list(map(preprocess_text, df[content_column_name].values))
    preprocessed_contents = [
        " ".join(
            [preprocess_text(sentence.strip()) for sentence in re.split(r'[.!?]', text) if (sentence.strip())]) for text in preprocessed_contents
    ]

    df[content_column_name] = preprocessed_contents

    short_contents_indexes = df.loc[df[content_column_name].str.split(" ").str.len() < 3].index

    indexes_to_drop = np.concatenate((float_contents_indexes, short_contents_indexes))

    df_dropped_indexes = pd.DataFrame({dropped_indexes_column_name: indexes_to_drop})
    # df_dropped_indexes.to_csv(save_path)
    
    df.drop(index=short_contents_indexes, inplace=True)

    logger.debug(f'Preprocessed: {preprocessed_contents}')

    return df, df_dropped_indexes


def get_embedded_files(
        path_to_embeddings_file: str):
    
    """
    Get the embedded files from a JSON file.

    Args:
        path_to_embeddings_file (str): The path to the JSON file containing the embedded files.

    Returns:
        dict: The embedded files as a dictionary.
    """

    try:
        with open(path_to_embeddings_file, 'r', encoding='utf-8') as embedded_json:
            return json.load(embedded_json)
    except FileNotFoundError:
        return {}
        

def save_file_as_embeded(
        filename: str, 
        vectors_filename: str,
        path_to_embeddings_file: str):
    
    """
    Save the filename and vectors filename as an embedded file in a JSON file.

    Args:
        filename (str): The original filename.
        vectors_filename (str): The filename for the vectors.
        path_to_embeddings_file (str): The path to the JSON file containing the embedded files.

    Returns:
        None
    """

    embeded_files = get_embedded_files(
        path_to_embeddings_file=path_to_embeddings_file
    )

    embeded_files[filename] = vectors_filename

    with open(path_to_embeddings_file, 'w', encoding='utf-8') as embedded_json:
        json.dump(embeded_files, embedded_json)


def del_file_from_embeded(
    filename_to_del: str,
    path_to_embeddings_file: str) -> bool:
    """
    Delets a file from the embedded files in a JSON file.

    Args:
        filename_to_del (str): The filename to delete.
        path_to_embeddings_file (str): The path to the JSON file containing the embedded files.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """

    logger.debug(path_to_embeddings_file)

    with open(path_to_embeddings_file, 'r', encoding='utf-8') as embedded_json:
        embedded_files_dict = json.load(embedded_json)
    
    try:
        del embedded_files_dict[filename_to_del]
    except KeyError:
        return False
    else:
        with open(path_to_embeddings_file, 'w', encoding='utf-8') as embedded_json:
            json.dump(embedded_files_dict, embedded_json)
            return True


def save_df_to_file(
        df: pd.DataFrame,
        filename: str,
        path_to_dir: str,
        file_ext: str = '.csv') -> str:
    
    """
    Save a DataFrame to a file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The filename of the file.
        path_to_dir (str): The directory path to save the file.
        file_ext (str, optional): The file extension. Defaults to '.csv'.

    Returns:
        str: The path to the saved file or 'Unallowed file extension' if the
            file extension is not allowed.
    """
    
    saving_path = os.path.join(path_to_dir, f'{filename}{file_ext}')
    
    if file_ext == '.parquet.gzip':
        df.to_parquet(
            saving_path,
            index=False)
        
    elif file_ext == '.xlsx':
        df.to_excel(
            saving_path,
            index=False
        )

    elif file_ext == '.csv':
        df.to_csv(
            saving_path,
            index=False
        )

    elif file_ext == '.html':

        formatters = None

        df.to_html(
            saving_path,
            index=False
        )
    
    else:
        return 'Unallowed file extension'
    
    return saving_path, f'{filename}{file_ext}'
    
def create_dataloader(
    df: pd.DataFrame,
    target_column: str or list,
    batch_size: int,
    shuffle: bool = False):

    base_data = df[target_column].to_frame() \
        if isinstance(target_column, str) \
        else df[target_column]
    
    dataset = MyDataset(base_data)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )

    return dataloader

def find_filename_in_dir(
    path_to_dir: str) -> dict:

    lookup_dir = {
        file_.split('.')[0]: file_ for file_ in os.listdir(path_to_dir)
    }

    return lookup_dir

def process_data_from_choosen_files(
        chosen_files: list,
        path_to_valid_files: str,
        path_to_cleared_files: str,
        path_to_empty_content_dir: str,
        path_to_embeddings_dir: str,
        faiss_vectors_dirname: str,
        embedded_files_filename: str,
        embeddings_model_name: str,
        sentiment_model_name: str,
        lang_detection_model_name: str,
        swearwords: list,
        currently_serviced_langs: dict,
        required_columns: list,
        get_sentiment: bool = True,
        translate_content: bool = True,
        original_content_column: str = 'content',
        content_column_name: str = 'preprocessed_content',
        translation_column_name: str = 'translation',
        sentiment_column_name: str = 'sentiment',
        detected_language_column_name: str = 'detected_language',
        dropped_indexes_column_name: str = 'dropped_indexes',
        offensive_label: str = 'offensive',
        faiss_vector_ext: str = '.index',
        cleread_file_ext: str = '.gzip.parquet',
        empty_contents_suffix: str = '_EMPTY_CONTENT',
        empty_content_ext: str = '.csv',
        en_code: str = 'en',
        batch_size: int = 32,
        translation_batch_size: int = 8,
        device: str = "cpu",
        seed: int = 42) -> list:
    
    """
    Process data from files chosen by a user, save embedding for the ones that are not already embedded.

    Args:
        chosen_files (list): A list of filenames chosen by the user.
        path_to_valid_files (str): The directory path to the valid files.
        path_to_cleared_files (str): The directory path to save the cleaned files.
        path_to_empty_content_dir (str): The directory path to save the files with empty contents.
        path_to_embeddings_dir (str): The directory path to save the embeddings.
        faiss_vectors_dirname (str): The name of the directory to save the Faiss vectors.
        embedded_files_filename (str): The filename of the JSON file containing the embedded files.
        embeddings_model_name (str): The name of the embeddings model.
        sentiment_model_name (str): The name of the sentiment model.
        lang_detection_model_name (str): The name of the language detection model.
        swearwords (list): A list of swearwords.
        currently_serviced_langs (dict): A dictionary of currently serviced languages.
        required_columns (list): A list of required columns.
        detect_languages (bool, optional): Whether to detect languages. Defaults to True.
        get_sentiment (bool, optional): Whether to get sentiment. Defaults to True.
        translate_content (bool, optional): Whether to translate content. Defaults to True.
        original_content_column (str, optional): The name of the column containing the original content.
            Defaults to 'content'.
        content_column_name (str, optional): The name of the column containing the preprocessed content.
            Defaults to 'preprocessed_content'.
        sentiment_column_name (str, optional): The name of the column containing the sentiment. Defaults to 'sentiment'.
        detected_language_column_name (str, optional): The name of the column containing the detected language.
            Defaults to 'detected_language'.
        dropped_indexes_column_name (str, optional): The name of the column containing the dropped indexes.
            Defaults to 'dropped_indexes'.
        offensive_label (str, optional): The label for offensive content. Defaults to 'offensive'.
        faiss_vector_ext (str, optional): The file extension for the Faiss vectors. Defaults to '.index'.
        cleread_file_ext (str, optional): The file extension for the cleaned files. Defaults to '.gzip.parquet'.
        empty_contents_suffix (str, optional): The suffix to append to the filename for the files with empty contents.
            Defaults to '_EMPTY_CONTENT'.
        empty_content_ext (str, optional): The file extension for the files with empty contents. Defaults to '.csv'.
        batch_size (int, optional): The batch size for processing the data. Defaults to 32.
        device (str, optional): The device to use for processing the data. Defaults to "cpu".
        seed (int, optional): The random seed for the embeddings model. Defaults to 42.

    Returns:
        list: A list of filenames of the files with empty contents.
    """

    PATH_TO_JSON_EMBEDDED_FILES = os.path.join(path_to_embeddings_dir, embedded_files_filename)
    PATH_TO_FAISS_VECTORS = os.path.join(path_to_embeddings_dir, faiss_vectors_dirname)

    already_embedded = get_embedded_files(
       PATH_TO_JSON_EMBEDDED_FILES 
    )

    cleared_files_names = find_filename_in_dir(path_to_cleared_files)

    only_filenames = [os.path.splitext(file_)[0] for file_ in chosen_files]

    logger.debug(f'Only filenames: {set(only_filenames)}')
    logger.debug(f'Already embedded: {set(already_embedded.keys())}')
    logger.debug(f'Already cleared files: {set(cleared_files_names.keys())}')

    if set(chosen_files).issubset(set(already_embedded.keys())) \
        and set(only_filenames).issubset(set(cleared_files_names.keys())):

        logger.info(f'All selected files from {only_filenames} have been already cleared and embedded')

        return None
    
    embeddings_model, embeddings_tokenizer, vec_size = load_transformer_model(
        model_name=embeddings_model_name,
        seed=seed,
        device=device
    )

    logger.info(f'Embeddings model {embeddings_model_name} loaded successfully')

    if translate_content:

        lang_detect_dict = load_lang_detector(
            model_name=lang_detection_model_name,
            device=device
        )

        lang_detection_model = lang_detect_dict.get('model')
        lang_detection_tokenizer = lang_detect_dict.get('tokenizer')

        logger.info(f'Language detection model {lang_detection_model_name} loaded successfully')

    logger.info(
        f"""Loading data from chosen files:{chosen_files}""")
    
    if get_sentiment:
        # loading sentiment model setup
        sent_tokenizer, sent_models, sent_cofnig = load_sentiment_model(
            sentiment_model_name, 
            device=device)
        
        logger.info(f'Sentiment setup loaded successfully.')

    zero_length_after_processing = []

    for file_ in chosen_files:

        df = None

        if file_ in os.listdir(path_to_valid_files):

            filename, ext = os.path.splitext(file_)

            if filename not in cleared_files_names.keys():

                df = read_file(
                    file_path=os.path.join(path_to_valid_files, file_),
                    columns=required_columns)
                    
                logger.debug(f'{filename=}, {ext=}')
                logger.debug(f'Loaded {file_}')
                logger.debug(df)

                df[content_column_name] = df[original_content_column].copy()

                df, df_dropped_indexes = cleanup_data(
                    df=df,
                    content_column_name=content_column_name,
                    dropped_indexes_column_name=dropped_indexes_column_name)
                
                if len(df) == 0:
                    zero_length_after_processing.append(file_)
                    continue
                
                logger.info(f'Cleaned {file_}')

                dataloader = create_dataloader(
                    df=df,
                    target_column=content_column_name,
                    batch_size=batch_size,
                    shuffle=False
                )

                if translate_content:

                    lang_detection_labels = detect_lang(
                        dataloader=dataloader,
                        detection_model=lang_detection_model,
                        tokenizer=lang_detection_tokenizer,
                        device=device
                    )

                    logger.info(f'Successfully detected languages for {filename}')

                    df[detected_language_column_name] = lang_detection_labels
                    known_langs = [en_code]

                    translation_models_dict = {}

                    langs_to_service = list(df[detected_language_column_name].unique())

                    for lang, model_name in currently_serviced_langs.items():

                        if lang not in langs_to_service:
                            continue

                        current_trans_model_dict = load_translation_model(
                            model_name=model_name,
                            device=device
                        )

                        translation_models_dict[lang] = current_trans_model_dict

                        logger.info(f'Translation model {model_name} for {lang.upper()} loaded successfully')

                    df[translation_column_name] = df[content_column_name].copy()

                    for lang, lang_model_dict in translation_models_dict.items():

                        to_translate_dataloader = create_dataloader(
                            df.loc[df[detected_language_column_name] == lang],
                            target_column=content_column_name,
                            batch_size=translation_batch_size,
                            shuffle=False
                        )

                        translated_tickets = translate_text(
                            dataloader=to_translate_dataloader,
                            trans_model=lang_model_dict.get('model'),
                            trans_tokenizer=lang_model_dict.get('tokenizer'),
                            device=device
                        )

                        logger.info(f'Successfully translated {lang} tickets for {filename}')

                        try:
                            df.loc[df[detected_language_column_name] == lang, 
                                translation_column_name] = translated_tickets
                        except Exception:
                            logger.error(f'Failed to assign translated tickets for {filename}')
                            return None
                        else:
                            logger.info(f'Successfully assigned translated tickets for {filename}')

                    known_langs = list(translation_models_dict.keys()) + known_langs
                    unknown_lang_contents_indexes = df.loc[~df[detected_language_column_name].isin(known_langs)].index

                    df = df.drop(unknown_lang_contents_indexes)

                    df_dropped_indexes = pd.concat(
                        [df_dropped_indexes, 
                            pd.DataFrame({dropped_indexes_column_name: unknown_lang_contents_indexes})])
                    
                df_dropped_indexes = df_dropped_indexes.sort_values(by=dropped_indexes_column_name)

                save_df_to_file(
                    df=df_dropped_indexes,
                    filename=f"{filename}{empty_contents_suffix}",
                    path_to_dir=path_to_empty_content_dir,
                    file_ext=empty_content_ext
                )

                if len(df) == 0:
                    zero_length_after_processing.append(file_)
                    continue

                if get_sentiment:

                    dataloader = create_dataloader(
                        df=df,
                        target_column=content_column_name,
                        batch_size=batch_size,
                        shuffle=False
                    )

                    sentiment_labels = predict_sentiment(
                        data=dataloader,
                        tokenizer=sent_tokenizer, 
                        model=sent_models, 
                        config=sent_cofnig,
                        device=device
                    ) 

                    df[sentiment_column_name] = sentiment_labels
                    df[sentiment_column_name] = np.where(
                        df[content_column_name].apply(lambda x: offensive_language(x, swearwords)), 
                        offensive_label, 
                        df[sentiment_column_name])

                    logger.info(f'Sentiment predicted successfully for {filename}')

                save_df_to_file(
                    df=df,
                    filename=filename,
                    path_to_dir=path_to_cleared_files,
                    file_ext=cleread_file_ext
                )

                logger.info(f'Successfully saved df to {filename}{cleread_file_ext}')

            if file_ not in already_embedded.keys():

                logger.debug(f'before embeddings df: {df}')

                if df is None:

                    logger.debug(f'File had been cleared before - loading DataFrame for {file_}')

                    cleared_filename = cleared_files_names.get(filename)

                    df = read_file(
                        file_path=os.path.join(path_to_cleared_files, cleared_filename),
                        columns=[content_column_name]
                    )

                logger.debug(f"Content column: {df[content_column_name]}")
                    
                dataloader = create_dataloader(
                    df=df,
                    target_column=content_column_name,
                    batch_size=batch_size,
                    shuffle=False
                )
            
                logger.info('Successfully converted df to Torch Dataset')
                
                save_name = f'{filename}{faiss_vector_ext}'
                save_path = os.path.join(PATH_TO_FAISS_VECTORS, save_name)

                get_embeddings(
                    dataloader=dataloader,
                    model=embeddings_model,
                    tokenizer=embeddings_tokenizer,
                    save_path=save_path,
                    vec_size=vec_size,
                    device=device)
                
                logger.info('Embeddings exctracted successfully')

                save_file_as_embeded(
                    filename=file_,
                    vectors_filename=save_name,
                    path_to_embeddings_file=PATH_TO_JSON_EMBEDDED_FILES
                )

                logger.info(f'File with embeddings saved for {file_}')

    return zero_length_after_processing


def get_stopwords(
        path_to_dir_with_stopwords: str) -> list:
    """
    Returns list of stopwords from all files in directory
    """
    
    all_stopwords = []

    for lang_file in os.listdir(path_to_dir_with_stopwords):
        with open(os.path.join(path_to_dir_with_stopwords, lang_file), 'r', encoding='utf-8') as lang_stopwords:
            all_stopwords.extend([stopword.strip("\n") for stopword in lang_stopwords])

    return all_stopwords


def get_swearwords(
        path_to_dir_with_swearwords: str) -> list:
    """
    Returns list of swearwords from all files in directory
    """
    all_swearwords = []

    for lang_file in os.listdir(path_to_dir_with_swearwords):
        with open(os.path.join(path_to_dir_with_swearwords, lang_file), 'r', encoding='utf-8') as lang_stopwords:
            all_swearwords.extend([stopword.strip("\n") for stopword in lang_stopwords])

    return all_swearwords


def get_rows_cardinalities(path_to_cardinalities_file: str) -> dict:
    """
    Returns dict with cardinalities of rows for each file
    """
    try:
        with open(path_to_cardinalities_file, 'r', encoding='utf-8') as cards_json:
            cards = json.load(cards_json)
    except FileNotFoundError:
        return {}
    else:
        return cards


def set_rows_cardinalities(
    path_to_cardinalities_file: str,
    updated_cardinalities) -> str or True:
    """
    Sets cardinalities of rows for a give file.
    """
    try:
        with open(path_to_cardinalities_file, 'w', encoding='utf-8') as cards_json:
            json.dump(updated_cardinalities, cards_json)
    except Exception as e:
        return str(e)
    else:
        return True
    

def remove_stopwords(text: str, stopwords: list) -> str:
    """
    Removes stopwords from text.
    """
    words = re.findall(r'\b\w+\b', text.lower())

    filtered_words = [word for word in words if word not in stopwords]

    filtered_text = ' '.join(filtered_words)

    return filtered_text


def remove_single_occurrence_words(text: str, min_n_of_occurence: int = 3) -> str:
    """
    Removes words that occur only once in text.
    """
    # Split the text into words
    words = text.split()

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Filter out words that appear only once
    filtered_words = [
        word for word in words if word_counts[word] >= min_n_of_occurence]

    # Join the remaining words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def preprocess_pipeline(
    text: str,
    stopwords: list,
    min_n_of_occurence: int = 3) -> str: 
    """
    Function runs the pipeline of preprocessing the text.
    """

    nlp = spacy.load('en_core_web_sm')

    text = re.sub(r'[^a-z ]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)

    removed_before_lem = remove_stopwords(
        text,
        stopwords=stopwords
    )

    lemmatized = " ".join([word._.lemma() for word in nlp(removed_before_lem)])

    removed_after_lemm = remove_stopwords(
        lemmatized,
        stopwords=stopwords
    )

    fully_preprocessed = remove_single_occurrence_words(
        text=removed_after_lemm,
        min_n_of_occurence=min_n_of_occurence)

    return fully_preprocessed

def get_n_of_rows_df(
    file_path: str,
    loaded_column: str = 'OS') -> int:
    """
    Return number of rows in a given file.
    """

    df = read_file(
        file_path=file_path,
        columns=[loaded_column]
    )

    return len(df)


def get_report_name_with_timestamp(
    filename_prefix: str,
    timestamp_format: str = r"%Y_%m_%d_%H_%M_%S") -> str: 
    """
    Returns filename with timestamp as s string.
    """
    timestamp_filename = f"""{filename_prefix}_{datetime.now().strftime(timestamp_format)}"""

    return timestamp_filename


def create_response_report(
        df: pd.DataFrame,
        filename: str,
        ext: str,
        mimetype: str,
        file_format: str = 'csv') -> Response or None:
    """
    Creates response with report in a given format.
    """
    
    CSV_EXT, EXCEL_EXT, HTML_EXT = 'csv', 'excel', 'html'
    
    buffer = io.BytesIO() if file_format in (CSV_EXT, EXCEL_EXT) else io.StringIO()

    if file_format == CSV_EXT:

        df.to_csv(buffer, index=False)

    elif file_format == EXCEL_EXT:

        df.to_excel(buffer, index=False)
    
    elif file_format == HTML_EXT:

        df.to_html(buffer, index=False)
    
    else:
        return None
    
    resp = make_response(buffer.getvalue())
    resp.headers["Content-Disposition"] = \
        f"attachment; filename={filename}.{ext}"
    resp.headers["Content-type"] = mimetype

    return resp


def validate_date_format(
        date_str: str,
        date_format: str) -> str or False:
    """
    Validates date format.
    """
    try:

        parsed_date = parser.parse(date_str)
        return parsed_date.strftime(date_format) == date_str
    
    except ValueError:
        return False


def prepare_filters(
        df: pd.DataFrame,
        date_column: str,
        date_filter_format: str,
        filename_column: str,
        topic_colum_prefix: str,
        topics_range: range,
        sentiment_column: str = None) -> dict[str, list]:
    """
    Function prepares filters for a given dataframe.

    Args:
        df (pd.DataFrame): dataframe to prepare filters for
        date_column (str): name of the column with dates
        date_filter_format (str): format of the date filter
        filename_column (str): name of the column with filenames
        topic_colum_prefix (str): prefix of the column with topics
        topics_range (range): range of topics
        sentiment_column (str, optional): name of the column with sentiment. Defaults to None.

    Returns:
        dict[str, list]: dictionary with filters
    """

    files_for_filtering = list(df[filename_column].unique())

    dates_for_filtering = list(
        set([datetime.strftime(date, date_filter_format) 
            for date in df[date_column].unique()]))
    
    if sentiment_column is not None:
        try:
            sentiment_for_filtering = list(set(df[sentiment_column].unique()))
        except KeyError:
            sentiment_for_filtering = None
    else:
        sentiment_for_filtering = None

    topics_for_filtering = []

    for topic_col in [f'{topic_colum_prefix}_{i}' for i in topics_range]:
        topics_for_filtering.extend(list(set(df[topic_col].unique())))

    topics_for_filtering = list(set(topics_for_filtering)) 

    return {
        'files_filter': files_for_filtering,
        'dates_filter': dates_for_filtering,
        'sentiment_filter': sentiment_for_filtering,
        'topics_filter': topics_for_filtering
    }


def prepare_reports_to_chose(
    path_to_cluster_exec_reports_dir: str,
    path_to_valid_files: str,
    files_for_filtering: list,
    gitkeep_file: str = '.gitkeep',
    exec_report_ext: str = '.gzip') -> dict[str, list]:
    """
    Args:
        path_to_cluster_exec_reports_dir (str): path to the directory with cluster execution reports
        path_to_valid_files (str): path to the directory with valid files
        files_for_filtering (list): list of files for filtering
        gitkeep_file (str, optional): name of the gitkeep file. Defaults to '.gitkeep'.
        exec_report_ext (str, optional): extension of the execution report. Defaults to '.gzip'.
    
    Returns:
        dict[str, list]: dictionary with reports to show and files available for update
    """
    
    reports = os.listdir(path_to_cluster_exec_reports_dir)

    reports_to_show = [
        report.split('.')[0] for report in reports 
        if report.endswith(exec_report_ext) 
    ]

    logger.debug(f'Report to show: {reports_to_show}')

    available_for_update = list(
        set(os.listdir(path_to_valid_files)).difference(
            set(files_for_filtering)))
    
    available_for_update = list(filter(lambda x: x != gitkeep_file, available_for_update))
    logger.debug(f'available_for_update {available_for_update}')

    return {
        'exec_reports_to_show': reports_to_show,
        'available_for_update': available_for_update
    }


def create_filter_query(
    date_column: str,
    filters_dict: dict[str, list],
    date_format: str) -> str:
    """
    Creates a query for filtering.

    Args:
        date_column (str): name of the column with dates
        filters_dict (dict[str, list]): dictionary with filters
        date_format (str): format of the date
    
    Returns:
        str: query for filtering
    """

    filter_query = []

    for column_name, filter_values in filters_dict.items():

        if not filter_values:
            continue

        current_query = ''

        if isinstance(column_name, str):

            if column_name == date_column:

                try:
                    start_date, end_date = filter_values
                except ValueError:
                    logger.error(f'Can not unpack dates for filtering, skipping operation')
                    continue
                except TypeError:
                    continue
                else:

                    current_query = []

                    if (start_date) and (validate_date_format(start_date, date_format=date_format)):

                         current_query.append(f"({column_name}.dt.strftime('{date_format}') >= '{start_date}')")

                    if (end_date) and (validate_date_format(end_date, date_format=date_format)):

                        current_query.append(f"({column_name}.dt.strftime('{date_format}') <= '{end_date}')")

                    if current_query:
                        current_query = " & ".join(current_query)
                    else:
                        continue
                    
            else:

                current_query = f"({column_name} in {filter_values})"
        
        elif isinstance(column_name, tuple):

            current_query = ' | '.join([f"{sub_col} in {filter_values}" for sub_col in column_name])
            current_query = f"({current_query})"

        else:
            continue

        filter_query.append(current_query)

    if not filter_query:

        logger.debug('No chosen values for filtering')
        return None

    logger.debug(f'Filter query: {" & ".join(filter_query)}')

    return " & ".join(filter_query)


def apply_filters_on_df(
    df: pd.DataFrame,
    filters_dict: dict[str, list],
    date_column: str,
    date_format: str) -> pd.DataFrame:
    """
    Applies filters on DataFrame.
    """

    filter_query = create_filter_query(
        date_column=date_column,
        filters_dict=filters_dict,
        date_format=date_format
    )

    if not filter_query:
        return df

    try:

        df = df.query(
            filter_query
        )
    
    except Exception as e:

        logger.error(f'Failed to apply filters on DataFrame: {e}')

    else:

        logger.info('Successfully aplied filters on DataFrame')
        return df
