import os
import re
import logging
from uuid import uuid4
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams, OptimizersConfigDiff
from ..qdrant_client_manager import QdrantClientManager

logging.getLogger("httpx").disabled = True

class VectorStoreCloud:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type

        # Cloud configuration
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        
        # Initialize singleton client manager
        self.client_manager = QdrantClientManager(config)
        self.client = self.client_manager.client

    def _clean_text(self, text: str) -> str:

        # Replace problematic characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control characters
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long words (often garbage or binary data)
        text = re.sub(r'\S{100,}', '[REMOVED_LONG_SEQUENCE]', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Limit line length
        lines = []
        for line in text.split('\n'):
            if len(line) > 300:
                lines.append(line[:300] + '...')
            else:
                lines.append(line)
        
        return '\n'.join(lines).strip()

    def _does_collection_exist(self) -> bool:
        return self.client_manager.does_collection_exist(self.collection_name)

    def _create_collection(self):
        vectors_config = {"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
        optimizers_config = OptimizersConfigDiff(
            indexing_threshold=0,  # Build index immediately
        )
        self.client_manager.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            optimizers_config=optimizers_config
        )
            
    def load_vectorstore(self) -> QdrantVectorStore:
        """
        Load existing cloud vectorstore for retrieval operations without ingesting new documents.
        
        Returns:
            QdrantVectorStore instance
        """
        # Check if collection exists
        if not self._does_collection_exist():
            self.logger.error(f"Cloud collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Cloud collection {self.collection_name} does not exist")
            
        # Fall back to dense-only retrieval
        self.logger.info("Loading vectorstore with dense embeddings")
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )
        
        self.logger.info(f"Successfully loaded existing cloud vectorstore")
        return qdrant_vectorstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, List[str]]:
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents with length limit
        MAX_CHUNK_LENGTH = 8000  # Characters limit for embeddings
        langchain_documents = []
        valid_doc_ids = []
        
        for id_idx, chunk in enumerate(document_chunks):
            # Clean and truncate chunk if too long
            chunk = self._clean_text(chunk)
            
            if len(chunk) > MAX_CHUNK_LENGTH:
                self.logger.warning(f"Chunk {id_idx} exceeds max length. Truncating from {len(chunk)} to {MAX_CHUNK_LENGTH} characters.")
                chunk = chunk[:MAX_CHUNK_LENGTH]
                
            try:
                # Store full content in metadata payload
                langchain_documents.append(
                    Document(
                        page_content=chunk,  # This will be used for embedding
                        metadata={
                            "source": os.path.basename(document_path),
                            "doc_id": doc_ids[id_idx],
                            "source_path": os.path.join("http://localhost:8000/", document_path),
                            "full_content": chunk  # Store full content in payload
                        }
                    )
                )
                valid_doc_ids.append(doc_ids[id_idx])
            except Exception as e:
                self.logger.error(f"Error creating document for chunk {id_idx}: {e}")
                continue
        
        # Check if collection exists, create if it doesn't
        if not self._does_collection_exist():
            self._create_collection()
            self.logger.info(f"Created new cloud collection: {self.collection_name}")
        else:
            self.logger.info(f"Cloud collection {self.collection_name} already exists, will upsert documents")
        
        qdrant_vectorstore = self.load_vectorstore()
        
        # Ingest documents into vector store with error handling
        try:
            # Process documents in smaller batches to avoid API limits
            BATCH_SIZE = 1  # Reduced batch size for better reliability
            for i in range(0, len(langchain_documents), BATCH_SIZE):
                batch_docs = langchain_documents[i:i+BATCH_SIZE]
                batch_ids = valid_doc_ids[i:i+BATCH_SIZE]
                
                self.logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(langchain_documents)-1)//BATCH_SIZE + 1} with {len(batch_docs)} documents")
                
                try:
                    # Process each document individually for maximum reliability
                    for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                        try:
                            # Add document with full content in payload
                            qdrant_vectorstore.add_documents(documents=[doc], ids=[doc_id])
                        except Exception as doc_error:
                            self.logger.error(f"Error adding document {i+j+1}: {doc_error}")
                            continue
                        
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch: {batch_error}")
                    # Continue with next batch instead of failing completely
            
            self.logger.info(f"Completed processing all document batches")
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise e
        
        return qdrant_vectorstore, valid_doc_ids

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
        ) -> List[Dict[str, Any]]:
       
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        
        for chunk, score in results:
            try:
                # Get document content directly from payload
                doc_content = chunk.metadata.get('full_content') or chunk.page_content
                
                if not doc_content:
                    self.logger.warning(f"Missing content in metadata for chunk")
                    continue
                
                # Create document dict in the format expected by reranker
                doc_dict = {
                    "id": chunk.metadata.get('doc_id', 'unknown'),
                    "content": doc_content,
                    "score": score,
                    "source": chunk.metadata.get('source', 'unknown'),
                    "source_path": chunk.metadata.get('source_path', ''),
                }
                
                # Add any additional metadata from the chunk
                for key, value in chunk.metadata.items():
                    if key not in ['doc_id', 'source', 'source_path', 'full_content']:
                        doc_dict[key] = value
                        
                retrieved_docs.append(doc_dict)
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue
            
        return retrieved_docs

    def create_vectorstore_with_metadata(
            self,
            document_chunks: List[str],
            metadatas: List[Dict[str, Any]],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, List[str]]:
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents with length limit
        MAX_CHUNK_LENGTH = 8000  # Characters limit for embeddings
        langchain_documents = []
        valid_doc_ids = []
        
        for id_idx, (chunk, metadata) in enumerate(zip(document_chunks, metadatas)):
            # Clean and truncate chunk if too long
            chunk = self._clean_text(chunk)
            
            if len(chunk) > MAX_CHUNK_LENGTH:
                self.logger.warning(f"Chunk {id_idx} exceeds max length. Truncating from {len(chunk)} to {MAX_CHUNK_LENGTH} characters.")
                chunk = chunk[:MAX_CHUNK_LENGTH]
                
            try:
                # Merge metadata with standard fields
                combined_metadata = {
                    "source": os.path.basename(document_path),
                    "doc_id": doc_ids[id_idx],
                    "source_path": document_path,
                    "full_content": chunk  # Store full content in payload
                }
                # Add custom metadata
                combined_metadata.update(metadata)
                
                langchain_documents.append(
                    Document(
                        page_content=chunk,  # This will be used for embedding
                        metadata=combined_metadata
                    )
                )
                valid_doc_ids.append(doc_ids[id_idx])
            except Exception as e:
                self.logger.error(f"Error creating document for chunk {id_idx}: {e}")
                continue
        
        # Check if collection exists, create if it doesn't
        if not self._does_collection_exist():
            self._create_collection()
            self.logger.info(f"Created new cloud collection: {self.collection_name}")
        else:
            self.logger.info(f"Cloud collection {self.collection_name} already exists, will upsert documents")
        
        qdrant_vectorstore = self.load_vectorstore()
        
        # Ingest documents into vector store with error handling
        try:
            # Process documents in smaller batches to avoid API limits
            BATCH_SIZE = 100  # Reduced batch size for better reliability
            for i in range(0, len(langchain_documents), BATCH_SIZE):
                batch_docs = langchain_documents[i:i+BATCH_SIZE]
                batch_ids = valid_doc_ids[i:i+BATCH_SIZE]
                                
                try:
                    # Process each document individually for maximum reliability
                    for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                        try:
                            # Add document with full content in payload
                            qdrant_vectorstore.add_documents(documents=[doc], ids=[doc_id])
                        except Exception as doc_error:
                            self.logger.error(f"Error adding document {i+j+1}: {doc_error}")
                            continue
                        
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch: {batch_error}")
                    # Continue with next batch instead of failing completely
            
            self.logger.info(f"Completed processing all document batches")
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise e
        
        return qdrant_vectorstore, valid_doc_ids 