from kg_manager import get_kg_manager
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

def find_similar_embeddings(query_embedding: List[float], candidate_embeddings: List[List[float]], top_k: int = 5) -> List[int]:
    similarities = []
    for i, candidate in enumerate(candidate_embeddings):
        sim = cosine_similarity(query_embedding, candidate)
        similarities.append((i, sim))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [idx for idx, _ in similarities[:top_k]]
