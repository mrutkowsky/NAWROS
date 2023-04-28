import os
import re
import nltk
import logging
import pandas as pd

from typing import Iterable

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import DATASET_PATH, POLISH_STOPWORDS_PATH

nltk.download("stopwords")
POLISH_STOPWORDS = open(POLISH_STOPWORDS_PATH, "r").read().split("\n")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_up_text(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r'[^\w]', ' ', sentence)  # romove punctuation
    sentence = re.sub(r'[0-9]', '', sentence)  # remove numbers
    sentence = re.sub(r'\s[a-z]\s', ' ', sentence)  # remove single characters
    sentence = re.sub(r'^[a-z]\s', '', sentence)  # remove single characters from the start
    sentence = re.sub(r'\s+', ' ', sentence).strip()  # remove extra spaces

    return sentence


def remove_stop_words(text):
    text = [word for word in text.split() 
            if word not in stopwords.words("english")
            and word not in POLISH_STOPWORDS]
    text = ' '.join(text)
    return text


def count_words(tokenizer) -> list:
    word_count = tokenizer.word_counts
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return word_count


def create_tokenizer(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    return tokenizer


def tokenize_text(text: Iterable[str], tokenizer: object) -> list:
    text = tokenizer.texts_to_sequences(text)
    return text


if __name__ == "__main__":
    df = pd.read_excel(DATASET_PATH)

    df = pd.read_excel(DATASET_PATH)
    logger.info("Dataset loaded successfully.")

    df_clean_data = pd.DataFrame(df["content"].apply(clean_up_text))
    logger.info("Text cleaned up successfully.")

    df_clean_data["content"] = df_clean_data["content"].apply(remove_stop_words)
    logger.info("Stop words removed successfully.")

    tokenizer = create_tokenizer(df_clean_data['content'])
    logger.info("Tokenizer created successfully.")
    
    word_count = count_words(tokenizer)
    logger.info("Word count calculated successfully.")
    
    most_popular_words = word_count[:20]
    logger.info("Most popular words are: %s", most_popular_words)

