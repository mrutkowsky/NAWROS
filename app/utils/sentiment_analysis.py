import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import logging
from scipy.special import softmax
import numpy as np
import torch

logger = logging.getLogger(__file__)


def load_sentiment_model(model_name: str) -> tuple:
    """
    Initialize tokenizer, model and config from pretrained model.

    Args:
        model_name (str): name of pretrained model
    
    Returns:
        tuple: tokenizer, model, config
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == 'cuda:0':
        logger.info('Cuda avaiable for sentiment model')
    else:
        logger.warning('Can not load CUDA for sentiment model')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    config = AutoConfig.from_pretrained(model_name)

    return tokenizer, model, config


def offensive_language(text, swear_words):

    text = str(text).lower()

    if any(word in text for word in swear_words):
        return True
    else:
        return False


def predict_sentiment(
        data, 
        tokenizer, 
        model, 
        config):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    labels = []
    
    for i, batch in enumerate(data):

        encoded_input = tokenizer(
            batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=64).to(device)

        output = model(**encoded_input)

        sentiment_batch_logits = output.get('logits').detach().cpu().numpy()
        sentiment_batch_softmaxed = np.apply_along_axis(softmax, arr=sentiment_batch_logits, axis=0)

        batch_labels = [config.id2label[np.argmax(scores)] for scores in sentiment_batch_softmaxed]

        labels.extend(batch_labels)

    return labels



