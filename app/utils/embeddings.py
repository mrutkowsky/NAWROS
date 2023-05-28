import torch
from sentence_transformers import SentenceTransformer

from config import SEED, ST_MODEL_NAME

torch.manual_seed(SEED)

model = SentenceTransformer(ST_MODEL_NAME)


def embed_sentence(text: str) -> torch.Tensor:
    return model.encode(
        sentences=text,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True)
