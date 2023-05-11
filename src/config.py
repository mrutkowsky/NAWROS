import os
import sys

sys.path.append('../')


MODEL_NAME = "roberta-base"
DATASET_PATH = os.path.join("..", "data", "VOC_for_NLP.xlsx")
POLISH_STOPWORDS_PATH = os.path.join("..", "data", "polish_stopwords.txt")
VECTOR_LEN = 64
