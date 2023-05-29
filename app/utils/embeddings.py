import os
import torch
import yaml
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.path.join('utils', 'config.yaml')

with open(CONFIG_PATH) as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

ST_MODEL_NAME = config['pipeline']['st_model_name']
SEED = config['pipeline']['seed']


torch.manual_seed(SEED)

model = SentenceTransformer(ST_MODEL_NAME)

VEC_SIZE = model.get_sentence_embedding_dimension()

def embed_sentence(text: str) -> torch.Tensor:
    return model.encode(
        sentences=text,
        show_progress_bar=True,
        convert_to_numpy=True)
