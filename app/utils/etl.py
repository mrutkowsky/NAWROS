import pandas as pd
import os
import re
import numpy as np

def load_files_into_df(
    data_folder_path: str,
    use_cols: list = None,
    skip_files: list = None) -> pd.DataFrame:

    df_result = None

    for dataset in sorted(os.listdir(data_folder_path), reverse=True):

        if skip_files is not None:
            if dataset in skip_files:
                continue

        dataset_filepath = os.path.join(data_folder_path, dataset)

        filename, f_extension = os.path.splitext(
            dataset_filepath
        )

        if f_extension in ['.xlsx']:
            df = pd.read_excel(dataset_filepath, usecols=use_cols)
        elif f_extension in ['.csv']:
            df = pd.read_csv(dataset_filepath, usecols=use_cols)
        else:
            print(f'Can not read {dataset_filepath}!')
            continue

        if df_result is not None:
            df_result = pd.concat([df_result, df])
        else:
            df_result = df.copy()

    if df_result is not None:
        df_result = df_result.reset_index(drop=True)

    return df_result

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove newlines
    text = re.sub(r'\n+', ' ', text)
    
    # Remove unknown signs, but keep dots, commas, question marks, exclamation marks and hashtags
    text = re.sub(r'[^a-ząćęłńóśźż\s.,!?()\'\"]|\-(?!\w)|(?<!\w)\-', '', text)
    text = re.sub(r'([?!.,])\1+', r'\1', text)
    
    text = re.sub(r'(?<![a-z0-9])-|-(?![a-z0-9])', '', text)
    text = re.sub(r'([\'"])([^\1]*)\1', r'\2', text)
    text = re.sub(r'(?<!\w)([\'"])([^\1]*)\b\1', r'\2', text)
    text = re.sub(r'\((?!\s*\))|(?<!\()\s*\)', '', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def layer_normalize(vector: np.array):
    return (vector - np.mean(vector)) / np.std(vector)