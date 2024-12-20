import ollama
import torch

from typing import List, Union
from interfaces.log import logger
from interfaces.ollama import ollama_client

CHUNK_SIZE = 250
OVERLAP_SIZE = 50


def generate_text_chunks(text: str, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    words = text.split()
    for i in range(overlap_size, len(words), chunk_size):
        yield " ".join(words[(i - overlap_size) : (i + chunk_size + overlap_size)])


def generate_embeddings(text: Union[str, List[str]], model: str) -> List:
    embedding = ollama_client.embed(model=model, input=text).get("embeddings", [[]])
    logger.debug(
        f"Embedding Generated for <{text[:50]}>...: {torch.tensor(embedding).dim()}"
    )
    return embedding


def query_topk_embeddings_indices(
    input_embeddings_tensor: List[float],
    vault_embeddings_tensor: List[float],
    topk: int,
) -> List[float]:
    input_tensor = torch.tensor(input_embeddings_tensor)
    logger.debug(f"Input tensor: {input_tensor}")

    vault_tensor = torch.tensor(vault_embeddings_tensor).squeeze()
    logger.debug(f"Vault tensor: {vault_tensor}")

    try:
        cosine_similarity = torch.cosine_similarity(
            input_tensor,
            vault_tensor,
        )
        logger.debug(f"Cosine similarity: {cosine_similarity}")
    except Exception as e:
        logger.error(f"An error occurred while calculating cosine similarity: {e}")
        return []

    real_topk = min(topk, len(cosine_similarity))
    logger.debug(f"Real topk: {real_topk}")

    indices = torch.topk(input=cosine_similarity, k=real_topk).indices.tolist()
    logger.debug(f"Indices: {indices}")

    return indices
