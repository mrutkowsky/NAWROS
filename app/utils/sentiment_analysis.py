import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import logging
from scipy.special import softmax
import numpy as np
import torch
import operator as op

logger = logging.getLogger(__file__)


def load_sentiment_model(
        model_name: str,
        device: str = 'cpu') -> tuple:
    """
    Initialize tokenizer, model and config from pretrained model.

    Args:
        model_name (str): name of pretrained model
    
    Returns:
        tuple: tokenizer, model, config
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    config = AutoConfig.from_pretrained(model_name)

    return tokenizer, model, config


def offensive_language(text, swear_words):
    """
       Translates text using a pre-trained translation model and tokenizer.

       Args:
           dataloader (DataLoader): The data loader containing the input texts to translate.
           trans_model: The pre-trained translation model.
           trans_tokenizer: The tokenizer associated with the translation model.
           device (str, optional): The device to use for translation. Defaults to 'cpu'.

       Returns:
           List[str]: A list of translated texts.
       """

    text = text.lower().split()

    if any(word in text for word in swear_words):
        return True
    else:
        return False

def predict_sentiment(
        data, 
        tokenizer, 
        model, 
        config,
        device: str = 'cpu'):
    """
        Translates text using a pre-trained translation model and tokenizer.

        Args:
            dataloader (DataLoader): The data loader containing the input texts to translate.
            trans_model: The pre-trained translation model.
            trans_tokenizer: The tokenizer associated with the translation model.
            device (str, optional): The device to use for translation. Defaults to 'cpu'.

        Returns:
            List[str]: A list of translated texts.
    """

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



