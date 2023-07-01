import torch
import faiss
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

def load_transformer_model(
        model_name: str,
        seed: int = 42,

) -> SentenceTransformer:
    
    """
    Load a pre-trained SentenceTransformer model.

    Args:
        model_name (str): The name or path of the pre-trained model to load.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        SentenceTransformer: The loaded SentenceTransformer model.
    """
    
    torch.manual_seed(seed)

    model = SentenceTransformer(model_name)
    vec_size = model.get_sentence_embedding_dimension()

    return model, vec_size
    
def embed_sentence(
        model: SentenceTransformer,
        text) -> torch.Tensor:
    
    """
    Embed a single sentence using a pre-trained SentenceTransformer model.

    Args:
        model (SentenceTransformer): The SentenceTransformer model used for sentence embedding.
        text (str or List[str]): The input sentence(s) to embed.

    Returns:
        torch.Tensor: The embedded sentence(s) as a torch tensor.
    """
    
    return model.encode(
        sentences=text,
        show_progress_bar=True,
        convert_to_numpy=True)

def get_embeddings(
        dataloader: DataLoader, 
        model: SentenceTransformer,
        save_path: str,
        vec_size: int):
    
    """
    Compute sentence embeddings for a dataset using a pre-trained SentenceTransformer model
    and store them in a FAISS index.

    Args:
        dataloader (DataLoader): The data loader for the dataset.
        model (SentenceTransformer): The SentenceTransformer model used for sentence embedding.
        save_path (str): The path to save the FAISS index file.
        vec_size (int): The dimensionality of the sentence embeddings.

    Returns:
        None
    """

    index = faiss.IndexFlatL2(int(vec_size))

    for batch in dataloader:
        index.add(embed_sentence(model, batch))

    assert save_path.endswith('.index')
    faiss.write_index(index, save_path)
