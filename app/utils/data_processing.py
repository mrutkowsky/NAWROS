import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import faiss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.embeddings_module import load_model, get_embeddings
from utils.etl import preprocess_text

logger = logging.getLogger(__file__)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data.values

    def __getitem__(self, index):
        return self.data[index][0]

    def __len__(self):
        return len(self.data)


def save_raport_to_csv(
        df: pd.DataFrame, 
        filename: str,
        path_to_raports_dir: str,
        clusters_topics: pd.DataFrame,
        classes_column_name: str = 'labels'):
    """
    Saves raport to csv file.
    """
    df = df.groupby(df[classes_column_name]).size().reset_index(name='counts')
    df = pd.concat([df, clusters_topics], axis=1) 
    save_path = os.path.join(path_to_raports_dir, filename)
    df.to_csv(save_path, index=False)


def read_file(
        file_path: str, 
        columns: list = None) -> pd.DataFrame:

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    logger.debug(f'df: {df}')

    return df


def cleanup_data(
        df: pd.DataFrame, 
        filename: str,
        path_to_empty_content_dir: str,
        empty_contents_suffix: str,
        empty_content_ext: str) -> pd.DataFrame:
    """
    Remove rows with empty contents and save them to a separate file.
    """

    logger.debug(f'Columns: {df.columns}')
    logger.debug(f"{type(df['content'])}")

    float_contents_indexes = np.where(df['content'].apply(lambda x: not isinstance(x, str)))[0]

    df.drop(index=float_contents_indexes, inplace=True)

    preprocessed_contents = list(map(preprocess_text, df['content'].values))
    preprocessed_contents = [
        " ".join(
            [preprocess_text(sentence.strip()) for sentence in re.split(r'[.!?]', text) if (sentence.strip())]) for text in preprocessed_contents
    ]

    df['content'] = preprocessed_contents

    short_contents_indexes = df.loc[df['content'].str.split(" ").str.len() < 3].index

    save_path = os.path.join(path_to_empty_content_dir,
         f'{filename.split(".")[-2]}{empty_contents_suffix}{empty_content_ext}')

    indexes_to_drop = np.concatenate((float_contents_indexes, short_contents_indexes))

    df_dropped_indexes = pd.DataFrame({'Dropped_indexes': indexes_to_drop})
    df_dropped_indexes.to_csv(save_path)
    
    df.drop(index=short_contents_indexes, inplace=True)

    logger.debug(f'Preprocessed: {preprocessed_contents}')

    

    logger.debug(f"{df['content']}")

    return df

def get_embedded_files(
        path_to_embeddings_file: str
):

    with open(path_to_embeddings_file, 'r', encoding='utf-8') as embedded_json:
        return json.load(embedded_json)

def save_file_as_embeded(
        filename: str, 
        vectors_filename: str,
        path_to_embeddings_file: str):

    embeded_files = get_embedded_files(
        path_to_embeddings_file=path_to_embeddings_file
    )

    embeded_files[filename] = vectors_filename

    with open(path_to_embeddings_file, 'w', encoding='utf-8') as embedded_json:
        json.dump(embeded_files, embedded_json)

def save_df_to_file(
        df: pd.DataFrame,
        filename: str,
        path_to_dir: str,
        file_ext: str = '.csv'

) -> None:
    
    saving_path = os.path.join(path_to_dir, f'{filename}{file_ext}')
    
    if file_ext == '.gzip.parquet':
        df.to_parquet(
            saving_path,
            index=False)
        
    elif file_ext == '.excel':
        df.to_excel(
            saving_path,
            index=False
        )

    elif file_ext == '.csv':
        df.to_csv(
            saving_path,
            index=False
        )
    
    else:
        return 'Unallowed file extension'

def process_data_from_choosen_files(
        chosen_files: list,
        path_to_valid_files: str,
        path_to_cleared_files: str,
        path_to_empty_content_dir: str,
        path_to_embeddings_dir: str,
        faiss_vectors_dirname: str,
        embedded_files_filename: str,
        embeddings_model_name: str,
        faiss_vector_ext: str = '.index',
        cleread_file_ext: str = '.gzip.parquet',
        empty_contents_suffix: str = '_EMPTY_CONTENT',
        empty_content_ext: str = '.csv',
        batch_size: int = 32,
        seed: int = 42):
    """
    Process data from files choosen by a user, save embedding for the ones that 
    are not already embedded.
    """

    PATH_TO_JSON_EMBEDDED_FILES = os.path.join(path_to_embeddings_dir, embedded_files_filename)
    PATH_TO_FAISS_VECTORS = os.path.join(path_to_embeddings_dir, faiss_vectors_dirname)

    already_embedded = get_embedded_files(
       PATH_TO_JSON_EMBEDDED_FILES 
    )

    cleared_files_names = {
        os.path.splitext(file_)[0]: file_ for file_ in os.listdir(path_to_cleared_files)
    }

    model, vec_size = load_model(
        model_name=embeddings_model_name,
        seed=seed
    )

    logger.info(f'Model {embeddings_model_name} loaded successfully')
    
    logger.info(
        f"""Loading data from chosen files:{chosen_files}""")

    for file_ in chosen_files:

        if file_ in os.listdir(path_to_valid_files):

            df = None
            filename, ext = os.path.splitext(file_)

            if filename not in cleared_files_names.keys():

                df = read_file(
                    file_path=os.path.join(path_to_valid_files, file_))
                    
                logger.info(f'{filename=}, {ext=}')
                logger.info(f'Loaded {file_}')
                logger.debug(df)

                df = cleanup_data(
                    df=df, 
                    filename=file_,
                    path_to_empty_content_dir=path_to_empty_content_dir,
                    empty_contents_suffix=empty_contents_suffix,
                    empty_content_ext=empty_content_ext)
                
                logger.info(f'Cleaned {file_}')

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

                    logger.debug('df was None')

                    cleared_filename = cleared_files_names.get(filename)

                    df = read_file(
                        file_path=os.path.join(path_to_cleared_files, cleared_filename),
                        columns=['content']
                    )

                logger.debug(f"Content column: {df['content']}")
            
                dataset = MyDataset(df['content'].to_frame())
                
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False
                )
                
                logger.info('Successfully converted df to Torch Dataset')
                
                save_name = f'{filename}{faiss_vector_ext}'
                save_path = os.path.join(PATH_TO_FAISS_VECTORS, save_name)

                get_embeddings(
                    dataloader=dataloader,
                    model=model,
                    save_path=save_path,
                    vec_size=vec_size)
                
                logger.info('Embeddings exctracted successfully')

                save_file_as_embeded(
                    filename=file_,
                    vectors_filename=save_name,
                    path_to_embeddings_file=PATH_TO_JSON_EMBEDDED_FILES
                )

                logger.info(f'File with embeddings saved for {file_}')

def get_stopwords(
        path_to_dir_with_stopwords: str) -> list:
    
    all_stopwords = []

    for lang_file in os.listdir(path_to_dir_with_stopwords):
        with open(os.path.join(path_to_dir_with_stopwords, lang_file), 'r', encoding='utf-8') as lang_stopwords:
            all_stopwords.extend([stopword.strip("\n") for stopword in lang_stopwords])

    return all_stopwords