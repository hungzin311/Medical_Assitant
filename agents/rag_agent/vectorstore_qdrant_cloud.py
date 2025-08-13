import os
import re
import logging
import importlib.util
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from .cloud_docstore import CloudDocStore
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStoreCloud:
    """
    Create cloud-based vector store, ingest documents, retrieve relevant documents
    """
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
        
        # Initialize cloud client
        if not self.qdrant_url or not self.qdrant_api_key:
            self.logger.error("Qdrant cloud URL or API key not provided. Check your environment variables.")
            raise ValueError("Qdrant cloud URL or API key not provided")
            
        self.logger.info(f"Connecting to Qdrant cloud at {self.qdrant_url}")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

    def _clean_text(self, text: str) -> str:
        """
        Clean text to avoid API validation errors.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
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
        """Check if the collection already exists in Qdrant cloud."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking for collection existence: {e}")
            return False

    def _create_collection(self):
        """Create a new collection with only dense vectors in cloud."""
        try:
            # Delete collection if it exists
            if self._does_collection_exist():
                self.logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)
                
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0,  # Build index immediately
                )
            )
            self.logger.info(f"Created new cloud collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating cloud collection: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, CloudDocStore]:
        """
        Load existing cloud vectorstore and local docstore for retrieval operations without ingesting new documents.
        
        Returns:
            Tuple containing (vectorstore, docstore)
        """
        # Check if collection exists
        if not self._does_collection_exist():
            self.logger.error(f"Cloud collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Cloud collection {self.collection_name} does not exist")
            
        # Fall back to dense-only retrieval
        self.logger.info("Falling back to dense embeddings only")
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )
        
        # Document storage (now cloud-based)
        docstore = CloudDocStore(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=f"{self.collection_name}_docstore"
        )
        
        self.logger.info(f"Successfully loaded existing cloud vectorstore and local docstore")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> Tuple[QdrantVectorStore, CloudDocStore, List[str]]:
        """
        Create a vector store in cloud from document chunks or upsert documents to existing store.
        
        Args:
            document_chunks: List of document chunks
            document_path: Path to the original document
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
        """
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents with length limit
        MAX_CHUNK_LENGTH = 2000  # Characters limit for Together API - further reduced to avoid 400 errors
        langchain_documents = []
        valid_doc_ids = []
        
        for id_idx, chunk in enumerate(document_chunks):
            # Clean and truncate chunk if too long
            # Remove special characters that might cause API issues
            chunk = self._clean_text(chunk)
            
            if len(chunk) > MAX_CHUNK_LENGTH:
                self.logger.warning(f"Chunk {id_idx} exceeds max length. Truncating from {len(chunk)} to {MAX_CHUNK_LENGTH} characters.")
                chunk = chunk[:MAX_CHUNK_LENGTH]
                
            try:
                langchain_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.basename(document_path),
                            "doc_id": doc_ids[id_idx],
                            "source_path": os.path.join("http://localhost:8000/", document_path)
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
        
        qdrant_vectorstore, docstore = self.load_vectorstore()
        
        # Ingest documents into vector and doc stores with error handling
        try:
            # Process documents in smaller batches to avoid API limits
            BATCH_SIZE = 1  # Reduced batch size for better reliability
            for i in range(0, len(langchain_documents), BATCH_SIZE):
                batch_docs = langchain_documents[i:i+BATCH_SIZE]
                batch_ids = valid_doc_ids[i:i+BATCH_SIZE]
                
                self.logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(langchain_documents)-1)//BATCH_SIZE + 1} with {len(batch_docs)} documents")
                
                try:
                    # Process each document individually for maximum reliability
                    successful_docs = []
                    successful_ids = []
                    
                    for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                        try:
                            # Try to add single document
                            qdrant_vectorstore.add_documents(documents=[doc], ids=[doc_id])
                            successful_docs.append(doc)
                            successful_ids.append(doc_id)
                            self.logger.info(f"Successfully added document {i+j+1}/{len(langchain_documents)}")
                        except Exception as doc_error:
                            self.logger.error(f"Error adding document {i+j+1}: {doc_error}")
                            continue
                    
                    # Encode string chunks to bytes before storing in cloud (only successful ones)
                    if successful_docs:
                        encoded_chunks = []
                        for doc in successful_docs:
                            encoded_chunks.append(doc.page_content.encode('utf-8'))
                            
                        docstore.mset(list(zip(successful_ids, encoded_chunks)))
                        
                        self.logger.info(f"Successfully added {len(successful_docs)}/{len(batch_docs)} documents in this batch")
                    else:
                        self.logger.warning(f"No documents were successfully added in batch {i//BATCH_SIZE + 1}")
                        
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch: {batch_error}")
                    # Continue with next batch instead of failing completely
            
            self.logger.info(f"Completed processing all document batches")
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise e
        
        return qdrant_vectorstore, docstore, valid_doc_ids

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: CloudDocStore,
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Retrieve relevant chunks based on a query from cloud vector store.
        
        Args:
            query: User query
            vectorstore: Vector store containing embeddings
            docstore: Document store containing actual content
            
        Returns:
            Tuple containing (retrieved_docs, picture_reference_paths)
            where retrieved_docs is a list of dictionaries with content and score
        """
        # Use similarity_search_with_score to get documents and scores
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.retrieval_top_k
        )
        
        retrieved_docs = []
        
        for chunk, score in results:
            try:
                # Get document ID from metadata
                doc_id = chunk.metadata.get('doc_id')
                if not doc_id:
                    self.logger.warning(f"Missing doc_id in metadata for chunk")
                    continue
                
                # Get full document from doc store as bytes and decode to string
                doc_content_bytes = docstore.mget([doc_id])[0]
                
                # Skip if document content is None
                if doc_content_bytes is None:
                    self.logger.warning(f"Document content is None for doc_id: {doc_id}")
                    continue
                
                doc_content = doc_content_bytes.decode('utf-8')
                
                # Add metadata to the document
                formatted_doc = doc_content
                
                # Create document dict in the format expected by reranker
                doc_dict = {
                    "id": chunk.metadata['doc_id'],
                    "content": formatted_doc,
                    "score": score,  # Use the actual similarity score
                    "source": chunk.metadata['source'],
                    "source_path": chunk.metadata['source_path'],
                }
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
        ) -> Tuple[QdrantVectorStore, CloudDocStore, List[str]]:
        """
        Create a vector store in cloud from document chunks with metadata
        
        Args:
            document_chunks: List of document chunks
            metadatas: List of metadata dictionaries corresponding to each chunk
            document_path: Path or identifier for the document source
            
        Returns:
            Tuple containing (vectorstore, docstore, doc_ids)
        """
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # Create langchain documents with length limit
        MAX_CHUNK_LENGTH = 2000  # Characters limit for Together API - further reduced to avoid 400 errors
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
                    "source_path": document_path
                }
                # Add custom metadata
                combined_metadata.update(metadata)
                
                langchain_documents.append(
                    Document(
                        page_content=chunk,
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
        
        qdrant_vectorstore, docstore = self.load_vectorstore()
        
        # Ingest documents into vector and doc stores with error handling
        try:
            # Process documents in smaller batches to avoid API limits
            BATCH_SIZE = 1  # Reduced batch size for better reliability
            for i in range(0, len(langchain_documents), BATCH_SIZE):
                batch_docs = langchain_documents[i:i+BATCH_SIZE]
                batch_ids = valid_doc_ids[i:i+BATCH_SIZE]
                
                self.logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(langchain_documents)-1)//BATCH_SIZE + 1} with {len(batch_docs)} documents")
                
                try:
                    # Process each document individually for maximum reliability
                    successful_docs = []
                    successful_ids = []
                    
                    for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                        try:
                            # Try to add single document
                            qdrant_vectorstore.add_documents(documents=[doc], ids=[doc_id])
                            successful_docs.append(doc)
                            successful_ids.append(doc_id)
                            self.logger.info(f"Successfully added document {i+j+1}/{len(langchain_documents)}")
                        except Exception as doc_error:
                            self.logger.error(f"Error adding document {i+j+1}: {doc_error}")
                            continue
                    
                    # Encode string chunks to bytes before storing in cloud (only successful ones)
                    if successful_docs:
                        encoded_chunks = []
                        for doc in successful_docs:
                            encoded_chunks.append(doc.page_content.encode('utf-8'))
                            
                        docstore.mset(list(zip(successful_ids, encoded_chunks)))
                        
                        self.logger.info(f"Successfully added {len(successful_docs)}/{len(batch_docs)} documents in this batch")
                    else:
                        self.logger.warning(f"No documents were successfully added in batch {i//BATCH_SIZE + 1}")
                        
                except Exception as batch_error:
                    self.logger.error(f"Error processing batch: {batch_error}")
                    # Continue with next batch instead of failing completely
            
            self.logger.info(f"Completed processing all document batches")
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise e
        
        return qdrant_vectorstore, docstore, valid_doc_ids 