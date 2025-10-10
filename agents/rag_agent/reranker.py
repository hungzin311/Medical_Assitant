import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, config):
        
        self.logger = logging.getLogger(__name__)
        # For medical data, specialized models like 'pritamdeka/S-PubMedBert-MS-MARCO'
        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            self.logger.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(self, query: str, documents: Union[List[Dict[str, Any]], List[str]]) -> List[Dict[str, Any]]:
        try:
            if not documents:
                return []
            
            for i, doc in enumerate(documents):
                if "id" not in doc:
                    doc["id"] = i
                if "score" not in doc:
                    doc["score"] = 1.0
                        
            # Create query-document pairs for scoring
            pairs = [(query, doc["content"]) for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            for i, score in enumerate(scores):
                documents[i]["rerank_score"] = float(score)  # Store the new score from reranking
                if "score" not in documents[i]:
                    documents[i]["score"] = 1.0
                documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
            
            # Sort by combined score
            reranked_docs = sorted(documents, key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to top_k if needed
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            # Fallback to original ranking if reranking fails
            self.logger.warning("Falling back to original ranking")
            return documents