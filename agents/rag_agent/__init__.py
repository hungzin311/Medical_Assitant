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
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Ingesting files from directory: {directory_path}")
        
        try:
            # Check if directory exists
            if not os.path.isdir(directory_path):
                raise ValueError(f"Directory not found: {directory_path}")
            
            # Get all files in the directory
            files = [os.path.join(directory_path + '/', f) for f in os.listdir(directory_path) 
                     if os.path.isfile(os.path.join(directory_path, f))]
            
            if not files:
                self.logger.warning(f"No files found in directory: {directory_path}")
                return {
                    "success": True,
                    "documents_ingested": 0,
                    "chunks_processed": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Track statistics
            total_chunks_processed = 0
            successful_ingestions = 0
            failed_ingestions = 0
            failed_files = []
            
            # Process each file
            for file_path in files:
                self.logger.info(f"Processing file {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
                
                try:
                    result = self.ingest_file(file_path)
                    if result["success"]:
                        successful_ingestions += 1
                        total_chunks_processed += result.get("chunks_processed", 0)
                    else:
                        failed_ingestions += 1
                        failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": str(e)})
            
            return {
                "success": True,
                "documents_ingested": successful_ingestions,
                "failed_documents": failed_ingestions,
                "failed_files": failed_files,
                "chunks_processed": total_chunks_processed,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error ingesting directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def ingest_file(self, document_path: str, document_chunks: List[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"Ingesting file: {document_path}")

        try:
            # If document_chunks is provided, use them directly
            if document_chunks:
                self.logger.info(f"Using {len(document_chunks)} pre-processed document chunks")
                formatted_document = document_path  # Use document_path as identifier
                
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
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info(f"RAG Agent processing query: {query}")
        
        try:
            # Step 1: Expand query
            expansion_result = self.query_expander.expand_query(query, mode="rag", chat_history=chat_history)
            expanded_query = expansion_result["expanded_query"]
            self.logger.info(f"   Expanded: '{expanded_query}'")
            query = expanded_query

            # Step 2: Retrieval
            vectorstore = self.vector_store.load_vectorstore()
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                )

            self.logger.info(f"Retrieved {len(retrieved_documents)} relevant document chunks")
            
            # Check if we have any documents
            if not retrieved_documents:
                self.logger.warning("No relevant documents found in the knowledge base")
                processing_time = time.time() - start_time
                return {
                    "response": "Tôi không có đủ thông tin để trả lời câu hỏi này dựa trên ngữ cảnh được cung cấp.",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": processing_time
                }

            # Step 3: Rerank the retrieved documents if we have a reranker and enough documents
            if self.reranker and len(retrieved_documents) > 1:
                reranked_documents = self.reranker.rerank(query, retrieved_documents)
                self.logger.info(f"Reranked retrieved documents and chose top {len(reranked_documents)}")
            else:
                self.logger.info(f"Could not rerank the retrieved documents, falling back to original scores")
                reranked_documents = retrieved_documents

            # Step 4: Generate response
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=reranked_documents,
                chat_history=chat_history
                )
            
            # Add timing information
            processing_time = time.time() - start_time
            response["processing_time"] = processing_time
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            # Return error response
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }

    def evaluate_mcq(self, question: str, choices: List[str]) -> Dict[str, Any]:
        try:
            expansion_result = self.query_expander.expand_query(question, mode="rag", chat_history=None)
            expanded_query = expansion_result["expanded_query"]
            
            retrieved_documents = self.vector_store.retrieve_kg_docs_chunks(
                query=expanded_query,
                vectorstore=self.vectorstore,
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
