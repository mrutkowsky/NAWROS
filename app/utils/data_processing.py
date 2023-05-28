import os
import sys
import json
import logging
import numpy as np
import pandas as pd

import faiss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, COLUMNS, VECTOR_SIZE, VALID_FILES,
                    EMBEDDINGS_FILE, FAISS_VECTORS_PATH)
from embeddings import embed_sentence
from etl import preprocess_text


logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data.values

    def __getitem__(self, index):
        return self.data[index][0]

    def __len__(self):
        return len(self.data)


def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, names=COLUMNS)
    else:
        df = pd.read_csv(file_path, names=COLUMNS)
    return df


def cleanup_data(df: pd.DataFrame) -> pd.DataFrame:
    sentences = list(df['content'])
    indexes_to_del = [i for i, sent in enumerate(sentences) if not isinstance(sent, str)]
    sentences = list(filter(lambda x: sentences.index(x) not in indexes_to_del, sentences))
    df = df.drop(index=indexes_to_del).dropna()
    sentences = list(map(preprocess_text, sentences))
    df['content'] = sentences
    return df


def create_dataloaders(valid_files: str) -> list:
    dataloaders = []
    for f in valid_files:
        df = read_file(os.path.join(VALID_FILES, f))
        dataset = MyDataset(cleanup_data(df))
        dataloaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False))        

    return dataloaders


def get_embeddings(dataloader: DataLoader):
    vectors = []
    for batch in tqdm(dataloader):
        vectors.append(embed_sentence(batch))
        logger.info(f"Embedding stoped after one batch for testing purposes")
        #WARNING: REMOVE THIS BREAK IF YOU WANT TO EMBED ALL THE DATA
        break

    return vectors

def save_embeddings(embeddings: list, path: str):
    logger.info(f'Saving embeddings to {path}')
    index = faiss.IndexFlatL2(VECTOR_SIZE)
    index.add(embeddings)
    assert path.endswith('.index')
    faiss.write_index(index, path)


if __name__ == "__main__":
    all_files = os.listdir(VALID_FILES)

    files_to_embed = []
    embeded_files = json.load(open(EMBEDDINGS_FILE, 'r'))
    for f in all_files:
        if f not in embeded_files.keys():
            files_to_embed.append(f)

    dls = create_dataloaders(files_to_embed)

    
    
    for n, filename in enumerate(files_to_embed):
        vecs = get_embeddings(dls[n])
        save_name = filename.replace('.', '_') + '.index'
        save_path = os.path.join(FAISS_VECTORS_PATH, save_name)
        save_embeddings(vecs[0], save_path)
        embeded_files[filename] = save_name
        

    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeded_files, f)