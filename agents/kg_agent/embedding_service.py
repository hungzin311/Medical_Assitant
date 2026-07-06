from .kg_manager import get_kg_manager
from typing import List
import numpy as np

kg_manager = get_kg_manager()

def embed_text(text: str) -> List[float]:
    try:
        return kg_manager.embedding_model.embed_query(text)
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))