import tensorflow as tf
from transformers import TFRobertaModel

from config import POLISH_STOPWORDS_PATH, MODEL_NAME, DATASET_PATH

embedding_model = TFRobertaModel.from_pretrained(MODEL_NAME)


def embed(tokens: list) -> tf.Tensor:
    input_ids = tf.constant(tokens, shape=(1, 64))
    outputs = embedding_model(input_ids)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1)
