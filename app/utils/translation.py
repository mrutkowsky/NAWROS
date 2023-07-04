from sentence_transformers import SentenceTransformer
import random
import pandas as pd
from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    pipeline, \
    AutoModelForSeq2SeqLM
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__file__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_lang_detector(
    model_name: str,
    device: str = "cpu"):
    """
        Performs language detection using the provided data loader, detection model, tokenizer, and device.

        Args:
            dataloader (DataLoader): The data loader object providing the data for language detection.
            detection_model: The language detection model.
            tokenizer: The tokenizer associated with the detection model.
            device (str, optional): The device to use for language detection. Defaults to "cpu".

        Returns:
            List[str]: A list of the language labels extracted from the language detections.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return {'tokenizer': tokenizer, 'model': model}

def detect_lang(
    dataloader: DataLoader,
    detection_model,
    tokenizer,
    device: str = "cpu"):
    """
        Performs language detection using the provided data loader, detection model, tokenizer, and device.

        Args:
            dataloader (DataLoader): The data loader object providing the data for language detection.
            detection_model: The language detection model.
            tokenizer: The tokenizer associated with the detection model.
            device (str, optional): The device to use for language detection. Defaults to "cpu".

        Returns:
            List[str]: A list of the language labels extracted from the language detections.
    """

    TASK = "text-classification"
    RESULT_LABEL_KEY = "label"
    CUDA_DEVICE = "cuda:0"

    all_lang_detections = []

    lang_detect_pipe = pipeline(
        TASK,
        model=detection_model,
        tokenizer=tokenizer,
        device=device
    )

    for i, batch in enumerate(dataloader):

        batch_detections = lang_detect_pipe(
            batch
        )

        logger.debug(f"Batch detections for {i} batch: {batch_detections}")

        all_lang_detections.extend(batch_detections)

        logger.debug(f"Current all lang detections: {all_lang_detections}")

    if device == CUDA_DEVICE:
        torch.cuda.empty_cache()

    return [l.get(RESULT_LABEL_KEY) for l in all_lang_detections]

def load_translation_model(
    model_name: str,
    device: str = "cpu"):
    """
        Loads a translation model and tokenizer based on the specified model name and device.

        Args:
            model_name (str): The name or path of the pre-trained translation model.
            device (str, optional): The device to use for loading the model and tokenizer. Defaults to "cpu".

        Returns:
            Dict[str, Union[PreTrainedTokenizer, PreTrainedModel]]: A dictionary containing the loaded tokenizer and model.
        """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    return {'tokenizer': tokenizer, 'model': model}

def translate_text(
    dataloader: DataLoader,
    trans_model,
    trans_tokenizer,
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

    CUDA_DEVICE = "cuda:0"

    all_translated = []
    
    for i, batch in enumerate(dataloader):

        encoded_inputs = trans_tokenizer.batch_encode_plus(
            batch, 
            padding=True, 
            truncation=True,
            max_length=64,
            return_tensors="pt").to(device)

        with torch.no_grad():
            translations = trans_model.generate(
                encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask
            )

        translated_texts = trans_tokenizer.batch_decode(
            translations, 
            skip_special_tokens=True)
        
        all_translated.extend(translated_texts)

    if device == CUDA_DEVICE:
        torch.cuda.empty_cache()

    return all_translated
    




