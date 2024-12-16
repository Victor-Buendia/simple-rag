import ollama
import torch

from typing import List, Union
from log import logger

def generate_embeddings(text: Union[str, List[str]], model: str) -> List:
    return ollama.embed(
        model=model,
        input=text
    ).get("embeddings", [[]])

def query_topk_embeddings_indices(input_embeddings_tensor: List[str], vault_embeddings_tensor: List[str], topk: int) -> List[int]:
    try:
        cosine_similarity = torch.cosine_similarity(
            torch.tensor(input_embeddings_tensor),
            torch.tensor(vault_embeddings_tensor),
        )
    except Exception as e:
        logger.error(f"An error occurred while calculating cosine similarity: {e}")
        return []
        
    real_topk = min(topk, len(cosine_similarity))
    return torch.topk(input=cosine_similarity, k=real_topk).indices.tolist()