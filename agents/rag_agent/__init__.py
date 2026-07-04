import os
import time
import logging
from typing import List, Optional, Dict, Any
from .vectorstore_qdrant_cloud import VectorStoreCloud
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

logging.getLogger("httpx").disabled = True

class MedicalRAG:
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.logger.info("Initializing Medical RAG system")
        self.config = config
        self.vector_store = VectorStoreCloud(config)
        self.reranker = Reranker(config)
        self.query_expander = QueryExpander(config)
        self.response_generator = ResponseGenerator(config)
        self.vectorstore = self.vector_store.load_vectorstore(config.rag.collection_name)
    
    def ingest_file(self, document_path: str, document_chunks: List[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # If document_chunks is provided, use them directly
            if document_chunks:
                self.logger.info(f"Using {len(document_chunks)} pre-processed document chunks")
                
            # Step 5: Create vector store and document store
            self.logger.info("5. Creating vector store knowledge base...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
                )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"Error ingesting file: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def evaluate_mcq(self, question: str, choices: List[str]) -> Dict[str, Any]:
        try:
            expansion_result = self.query_expander.expand_query(question, mode="rag", chat_history=None)
            expanded_query = expansion_result["expanded_query"]
            
            retrieved_documents = self.vectorstore.similarity_search_with_score(
                query=expanded_query,
                k=self.config.rag.top_k
            )

            reranked_documents = self.reranker.rerank(expanded_query, retrieved_documents)

            return self.response_generator.generate_response_benchmark(expanded_query, choices, reranked_documents)
        except Exception as e:
            self.logger.error(f"Error evaluating MCQ: {e}")
            return {
                "answer_index": None,
                "not_enough_info": True,
                "confidence": 0.0
            }
