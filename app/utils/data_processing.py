import os
import re
import json
import yaml
import logging
import numpy as np
import pandas as pd
import faiss
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import spacy
import lemminflect
from collections import Counter
from datetime import datetime
import io
from flask import make_response

from utils.embeddings_module import load_transformer_model, get_embeddings
from utils.etl import preprocess_text
from utils.sentiment_analysis import load_sentiment_model, predict_sentiment, offensive_language
from utils.translation import load_lang_detector, load_translation_model, detect_lang, translate_text

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
        
            except ValueError:
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
        filename: str,
        path_to_empty_content_dir: str,
        empty_contents_suffix: str,
        empty_content_ext: str,
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

    save_path = os.path.join(path_to_empty_content_dir,
         f'{filename.split(".")[-2]}{empty_contents_suffix}{empty_content_ext}')

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
    path_to_embeddings_file: str) -> None:

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
        None
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
        os.path.splitext(file_)[0]: file_ for file_ in os.listdir(path_to_dir)
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
        get_sentiment: bool = True,
        translate_content: bool = True,
        original_content_column: str = 'content',
        content_column_name: str = 'preprocessed_content',
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
        seed: int = 42):
    
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
        faiss_vector_ext (str, optional): The file extension for the Faiss vectors. Defaults to '.index'.
        cleread_file_ext (str, optional): The file extension for the cleaned files. Defaults to '.gzip.parquet'.
        empty_contents_suffix (str, optional): The suffix to append to the filename for the files with empty contents.
            Defaults to '_EMPTY_CONTENT'.
        empty_content_ext (str, optional): The file extension for the files with empty contents. Defaults to '.csv'.
        batch_size (int, optional): The batch size for processing the data. Defaults to 32.
        seed (int, optional): The random seed for the embeddings model. Defaults to 42.

    Returns:
        None
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
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    embeddings_model, vec_size = load_transformer_model(
        model_name=embeddings_model_name,
        seed=seed
    )

    logger.info(f'Embeddings model {embeddings_model_name} loaded successfully')

    lang_detect_dict = load_lang_detector(
        model_name=lang_detection_model_name,
        device=DEVICE
    )

    lang_detection_model = lang_detect_dict.get('model')
    lang_detection_tokenizer = lang_detect_dict.get('tokenizer')

    logger.info(f'Language detection model {lang_detection_model_name} loaded successfully')

    if translate_content:

        translation_models_dict = {}

        for lang, model_name in currently_serviced_langs.items():

            current_trans_model_dict = load_translation_model(
                model_name=model_name,
                device=DEVICE
            )

            translation_models_dict[lang] = current_trans_model_dict

            # translation_model = current_trans_model_dict.get('model')
            # translation_tokenizer = current_trans_model_dict.get('tokenizer')

            logger.info(f'Translation model {model_name} for {lang.upper()} loaded successfully')

    logger.info(
        f"""Loading data from chosen files:{chosen_files}""")
    
    if get_sentiment:
        # loading sentiment model setup
        sent_tokenizer, sent_models, sent_cofnig = load_sentiment_model(
            sentiment_model_name, 
            device=DEVICE)
        
        logger.info(f'Sentiment setup loaded successfully.')

    zero_length_after_processing = []

    for file_ in chosen_files:

        df = None

        if file_ in os.listdir(path_to_valid_files):

            filename, ext = os.path.splitext(file_)

            if filename not in cleared_files_names.keys():

                df = read_file(
                    file_path=os.path.join(path_to_valid_files, file_))
                    
                logger.debug(f'{filename=}, {ext=}')
                logger.debug(f'Loaded {file_}')
                logger.debug(df)

                df[content_column_name] = df[original_content_column].copy()

                df, df_dropped_indexes = cleanup_data(
                    df=df, 
                    filename=file_,
                    path_to_empty_content_dir=path_to_empty_content_dir,
                    empty_contents_suffix=empty_contents_suffix,
                    empty_content_ext=empty_content_ext,
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

                lang_detection_labels = detect_lang(
                    dataloader=dataloader,
                    detection_model=lang_detection_model,
                    tokenizer=lang_detection_tokenizer,
                    device=DEVICE
                )

                logger.info(f'Successfully detected languages for {filename}')

                df[detected_language_column_name] = lang_detection_labels
                known_langs = [en_code]

                if translate_content:

                    for lang, lang_model_dict in translation_models_dict.items():

                        to_translate_dataloader = create_dataloader(
                            df.loc[df[detected_language_column_name] == lang],
                            target_column=content_column_name,
                            batch_size=8,
                            shuffle=False
                        )

                        translated_tickets = translate_text(
                            dataloader=to_translate_dataloader,
                            trans_model=lang_model_dict.get('model'),
                            trans_tokenizer=lang_model_dict.get('tokenizer'),
                            device=DEVICE
                        )

                        logger.info(f'Successfully translated {lang} tickets for {filename}')

                        try:
                            df.loc[df[detected_language_column_name] == lang, 
                                content_column_name] = translated_tickets
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
                        device=DEVICE
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
                    save_path=save_path,
                    vec_size=vec_size)
                
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
    
    all_stopwords = []

    for lang_file in os.listdir(path_to_dir_with_stopwords):
        with open(os.path.join(path_to_dir_with_stopwords, lang_file), 'r', encoding='utf-8') as lang_stopwords:
            all_stopwords.extend([stopword.strip("\n") for stopword in lang_stopwords])

    return all_stopwords

def get_swearwords(
        path_to_dir_with_swearwords: str) -> list:
    
    all_swearwords = []

    for lang_file in os.listdir(path_to_dir_with_swearwords):
        with open(os.path.join(path_to_dir_with_swearwords, lang_file), 'r', encoding='utf-8') as lang_stopwords:
            all_swearwords.extend([stopword.strip("\n") for stopword in lang_stopwords])

    return all_swearwords

def get_rows_cardinalities(path_to_cardinalities_file: str) -> dict:

    with open(path_to_cardinalities_file, 'r', encoding='utf-8') as cards_json:
        return json.load(cards_json)

def set_rows_cardinalities(
    path_to_cardinalities_file: str,
    updated_cardinalities) -> str or True:

    try:
        with open(path_to_cardinalities_file, 'w', encoding='utf-8') as cards_json:
            json.dump(updated_cardinalities, cards_json)
    except Exception as e:
        return str(e)
    else:
        return True
    
def remove_stopwords(text, stopwords):
    
    words = re.findall(r'\b\w+\b', text.lower())

    filtered_words = [word for word in words if word not in stopwords]

    filtered_text = ' '.join(filtered_words)

    return filtered_text

def remove_single_occurrence_words(text, min_n_of_occurence: int = 3):
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
    min_n_of_occurence: int = 3): 

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

    df = read_file(
        file_path=file_path,
        columns=[loaded_column]
    )

    return len(df)

def get_report_name_with_timestamp(
    filename_prefix: str,
    timestamp_format: str = r"%Y_%m_%d_%H_%M_%S"): 

    timestamp_filename = f"""{filename_prefix}_{datetime.now().strftime(timestamp_format)}"""

    return timestamp_filename

def create_response_report(
        df: pd.DataFrame,
        filename: str,
        ext: str,
        mimetype: str,
        file_format: str = 'csv'):
    
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

