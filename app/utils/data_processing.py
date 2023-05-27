import os
import sys
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from config import BATCH_SIZE, FOLDER_PATH, COLUMNS



logging.basicConfig(level=logging.DEBUG,
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


def create_dataloaders(valid_files: str) -> list:
    dataloaders = []
    for f in valid_files:
        dataset = MyDataset(read_file(os.path.join(FOLDER_PATH, f)))
        dataloaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False))        

    return dataloaders


def get_embeddings(dataloader: DataLoader):
    for batch in tqdm(dataloader):
        #TODO: call embdeddings model and potentially save embeddings
        pass


if __name__ == "__main__":
    valid_files = os.listdir(FOLDER_PATH)
    logger.info(f"The following files will be processed: {valid_files}")
    dls = create_dataloaders(valid_files)

    for dl in dls:
        get_embeddings(dl)
