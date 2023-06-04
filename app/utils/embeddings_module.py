import torch
import faiss
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

def load_model(
        model_name: str,
        seed: int = 42,

) -> SentenceTransformer:
    
    torch.manual_seed(seed)

    model = SentenceTransformer(model_name)
    vec_size = model.get_sentence_embedding_dimension()

    return model, vec_size
    
def embed_sentence(
        model: SentenceTransformer,
        text) -> torch.Tensor:
    
    return model.encode(
        sentences=text,
        show_progress_bar=True,
        convert_to_numpy=True)

def get_embeddings(
        dataloader: DataLoader, 
        model: SentenceTransformer,
        save_path: str,
        vec_size: int):

    index = faiss.IndexFlatL2(vec_size)

    for batch in dataloader:
        index.add(embed_sentence(model, batch))

    assert save_path.endswith('.index')
    faiss.write_index(index, save_path)
