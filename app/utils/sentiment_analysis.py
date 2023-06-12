import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import logging
from scipy.special import softmax
import numpy as np

logger = logging.getLogger(__file__)


def load_sentiment_model(model_name: str) -> tuple:
    """
    Initialize tokenizer, model and config from pretrained model.

    Args:
        model_name (str): name of pretrained model
    
    Returns:
        tuple: tokenizer, model, config
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    return tokenizer, model, config


def offensive_language(text, swear_words):
    text = str(text).lower()
    if any(word in text for word in swear_words):
        return True
    else:
        return False


def predict_sentiment(data, tokenizer, model, config, swear_words):
    encoded_input = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=64)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    label = config.id2label[np.argmax(scores)]

    if label != 'negative':
        if offensive_language(data, swear_words):
            label = 'negative'

    return label
