import torch
from transformers import RobertaModel

from config import POLISH_STOPWORDS_PATH, MODEL_NAME, DATASET_PATH

embedding_model = RobertaModel.from_pretrained(MODEL_NAME)


def embed(tokens: list):
    input_ids = torch.tensor(tokens).unsqueeze(dim=0)

    outputs = embedding_model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence-level embeddings
    
    return embeddings
