from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification, \
    pipeline, \
    AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__file__)


def load_lang_detector(
    model_name: str,
    device: str = "cpu") -> dict:
    """
    Load a pre-trained language detection model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return {'tokenizer': tokenizer, 'model': model}


def detect_lang(
    dataloader: DataLoader,
    detection_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str = "cpu") -> list:
    """
    Detect the language of a text using a pre-trained language detection model.

    Args:
        dataloader (DataLoader): The dataloader containing the text to detect the language of.
        detection_model (AutoModelForSequenceClassification): The pre-trained language detection model.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the text.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        list: A list of the detected languages.
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
    device: str = "cpu") -> dict:
    """
    Load a pre-trained translation model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    return {'tokenizer': tokenizer, 'model': model}


def translate_text(
    dataloader: DataLoader,
    trans_model: AutoModelForSeq2SeqLM,
    trans_tokenizer: AutoTokenizer,
    device: str = 'cpu'):
    """
    Translate a text using a pre-trained translation model.
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
