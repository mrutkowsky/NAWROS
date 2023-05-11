import re
import nltk
import logging
import torch
import tensorflow as tf

import pandas as pd
from nltk.corpus import stopwords
from transformers import RobertaTokenizer, RobertaModel, TFRobertaModel
from typing import Iterable

from src.config import POLISH_STOPWORDS_PATH, MODEL_NAME, VECTOR_LEN, DATASET_PATH


nltk.download("stopwords")
POLISH_STOPWORDS = open(POLISH_STOPWORDS_PATH, "r").read().split("\n")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
embedding_model = RobertaModel.from_pretrained(MODEL_NAME)
tf_embedding_model = TFRobertaModel.from_pretrained(MODEL_NAME)


def clean_up_text(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r'[^\w]', ' ', sentence)  # romove punctuation
    sentence = re.sub(r'[0-9]', '', sentence)  # remove numbers
    sentence = re.sub(r'\s[a-z]\s', ' ', sentence)  # remove single characters
    sentence = re.sub(r'^[a-z]\s', '', sentence)  # remove single characters from the start
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # remove extra spaces
  
    return sentence


def remove_stop_words(text):
    text = [word for word in str(text).split() 
            if word not in stopwords.words("english")
            and word not in POLISH_STOPWORDS]
    text = ' '.join(text)
    return text


def tokenize_text(text) -> list:
    try:
        text = tokenizer.encode(text, add_special_tokens=True, max_length=VECTOR_LEN,
                                return_token_type_ids=True, padding="max_length",
                                truncation=True)
    except:
        text = None

    return text


def embed(input: str, tensor_type="torch"):

    tokens = tokenizer.encode(input, add_special_tokens=True, max_length=VECTOR_LEN,
                                return_token_type_ids=True, padding="max_length",
                                truncation=True)
  
    if tensor_type == "torch":
        input_ids = torch.tensor(tokens).unsqueeze(dim=0)
        outputs = embedding_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
    else:
        input_ids = tf.constant(tokens, shape=(1, 64))
        outputs = tf_embedding_model(input_ids)
        embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    
    return embeddings


if __name__ == "__main__":
    df = pd.read_excel(DATASET_PATH)

    df = pd.read_excel(DATASET_PATH)
    logger.info("Dataset loaded successfully.")

    df_clean_data = pd.DataFrame(df["content"].apply(clean_up_text))
    logger.info("Text cleaned up successfully.")

    df_clean_data["content"] = df_clean_data["content"].apply(remove_stop_words)
    logger.info("Stop words removed successfully.")

    df_clean_data["tokens"] = df_clean_data["content"].apply(tokenize_text)
    logger.info("Text tokenized successfully.")


    b = 0
    e_column = []
    for i in range(64, len(df_clean_data), 64):
        embd = [embed(s, tensor_type="torch") for s in df_clean_data["content"][b:i]]
        b += 64
        e_column.extend(embd)
    
    df_clean_data["torch"] = e_column
    



    # df_clean_data["torch"] = df_clean_data["tokens"].apply(embed, tensor_type="torch")
    # logger.info("Text embedded successfully.")
    # df_clean_data["tensorflow"] = df_clean_data["tokens"].apply(embed, tensor_type="tf")
    logger.info("Text embedded successfully.")
    print(df_clean_data["content"].head())
