import os
import json
import yaml
import logging
import numpy as np
import pandas as pd
import faiss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.embeddings import embed_sentence, VEC_SIZE
from utils.etl import preprocess_text

CONFIG_PATH = os.path.join('utils', 'config.yaml')

with open(CONFIG_PATH) as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


DATA_FOLDER = os.path.join(*config['pipeline']['data_folder'])
CLEARED_DATA_FOLDER = os.path.join(config.get('pipeline').get('cleared_data_folder'))
VALID_FILES = os.path.join(*config['pipeline']['valid_files'])
EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, config['pipeline']['embeddings_folder'])
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_FOLDER, config['pipeline']['embeddings_file'])
EMPTY_CONTENTS_EXT = config['pipeline']['empty_content_ext']
EMPTY_CONTENTS_SUFFIX = config['pipeline']['empty_content_suffix']
EMPTY_CONTENT_FOLDER = os.path.join(DATA_FOLDER, config['pipeline']['empty_content_folder'])
FAISS_VECTORS_PATH = os.path.join(DATA_FOLDER, config['pipeline']['faiss_vec_folder'])
COLUMNS = config['pipeline']['columns']
BATCH_SIZE = config['pipeline']['batch_size']

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
        clusters_topics: pd.DataFrame,
        classes_column_name: str = 'labels'):
    """
    Saves raport to csv file.
    """
    df = df.groupby(df[classes_column_name]).size().reset_index(name='counts')
    df = pd.concat([df, clusters_topics], axis=1) 
    save_path = os.path.join(DATA_FOLDER, filename)
    df.to_csv(save_path, index=False)


def read_file(file_path: str, columns=COLUMNS) -> pd.DataFrame:
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, usecols=columns)
    else:
        df = pd.read_csv(file_path, usecols=columns)
    return df


def cleanup_data(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Remove rows with empty contents and save them to a separate file.
    """
    empty_contents_indexes = np.where(df['content'].apply(lambda x: not isinstance(x, str)))[0]

    save_path = os.path.join(EMPTY_CONTENT_FOLDER,
         f'{filename.split(".")[-2]}{EMPTY_CONTENTS_SUFFIX}{EMPTY_CONTENTS_EXT}')

    df.iloc[empty_contents_indexes].to_csv(
        path_or_buf=save_path,
        index=True)
  
    df.drop(index=empty_contents_indexes, inplace=True)
    df['content'] = df['content'].apply(preprocess_text)

    return df


def save_embeddings(dataloader: DataLoader, save_path: str):

    index = faiss.IndexFlatL2(VEC_SIZE)
    logger.info(f'Saving embeddings to {save_path}')

    for batch in dataloader:
        index.add(embed_sentence(batch))

    assert save_path.endswith('.index')
    faiss.write_index(index, save_path)


def get_embedded_files():
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as embedded_json:
        return json.load(embedded_json)

def save_file_as_embeded(filename, vectors_filename):
    embeded_files = get_embedded_files()
    embeded_files[filename] = vectors_filename
    json.dump(embeded_files, open(EMBEDDINGS_FILE, 'w'))


def process_data_from_choosen_files(
        chosen_files: list,
        columns: list = None,
        content_column: str = 'content'):
    """
    Process data from files choosen by a user, save embedding for the ones that 
    are not already embedded.
    """

    already_embedded = get_embedded_files()
    logger.info(
        f"""Loading data from chosen files:{chosen_files}""")

    for file_ in chosen_files:

        if file_ in os.listdir(VALID_FILES) and file_ not in already_embedded.keys():

            df = read_file(
                file_path=os.path.join(VALID_FILES, file_),
                columns=None)
            
            filename, ext = os.path.splitext(file_)
            
            logger.info(f'{filename=}, {ext=}')
            
            logger.info(f'Loaded {file_}')
            df = cleanup_data(df, file_)
            logger.info(f'Cleaned {file_}')
            
            dataset = MyDataset(df[content_column].to_frame())
            
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            logger.info('Successfully converted df to Torch Dataset')
            
            save_name = f'{filename}.index'
            save_path = os.path.join(FAISS_VECTORS_PATH, save_name)

            save_embeddings(dataloader, save_path)
            logger.info('Embeddings exctracted successfully')
            save_file_as_embeded(file_, save_name)
            logger.info(f'File with embeddings saved for {file_}')

            df.to_csv(
                index=False, 
                path_or_buf=os.path.join(CLEARED_DATA_FOLDER, f'{filename}.csv'))


if __name__ == "__main__":
    all_files = os.listdir(VALID_FILES)
    process_data_from_choosen_files(all_files)