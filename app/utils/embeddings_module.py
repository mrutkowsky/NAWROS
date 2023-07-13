import torch
import faiss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__file__)

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_transformer_model(
        model_name: str,
        seed: int = 42,
        device: str = 'cpu') -> AutoModel:
    
    """
    Load a pre-trained SentenceTransformer model.

    Args:
        model_name (str): The name or path of the pre-trained model to load.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        SentenceTransformer: The loaded SentenceTransformer model.
    """
    
    torch.manual_seed(seed)

    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    vec_size = model.config.hidden_size

    return model, tokenizer, vec_size
    
def embed_sentence(
        model: AutoModel,
        tokenizer: AutoTokenizer,
        text,
        device: str = 'cpu') -> torch.Tensor:
    
    """
    Embed a single sentence using a pre-trained SentenceTransformer model.

    Args:
        model (SentenceTransformer): The SentenceTransformer model used for sentence embedding.
        text (str or List[str]): The input sentence(s) to embed.

    Returns:
        torch.Tensor: The embedded sentence(s) as a torch tensor.
    """

    encoded_input = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)
    
    return mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().numpy()

def get_embeddings(
        dataloader: DataLoader, 
        model: AutoModel,
        tokenizer: AutoTokenizer,
        save_path: str,
        vec_size: int,
        device: str = 'cpu'):
    
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

    logger.debug(f'Device for calculating embeddings: {device}')
    logger.debug(f'Start getting embeddings for {save_path}')

    for batch in dataloader:
        index.add(embed_sentence(model, tokenizer, batch, device=device))

    assert save_path.endswith('.index')
    faiss.write_index(index, save_path)
