import os
import asyncio
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
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\S{100,}', '[REMOVED_LONG_SEQUENCE]', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Limit line length
        lines = []
        for line in text.split('\n'):
            if len(line) > 300:
                lines.append(line[:300] + '...')
            else:
                lines.append(line)
        
        return '\n'.join(lines).strip()

    def _does_collection_exist(self, collection_name: str) -> bool:
        return self.client_manager.does_collection_exist(collection_name)

    def _create_collection(self, collection_name: str):
        if self._does_collection_exist(collection_name):
            self.logger.info(f"Collection {collection_name} already exists")
            return
        vectors_config = {"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)}
        optimizers_config = OptimizersConfigDiff(
            indexing_threshold=0,  # Build index immediately
        )
        self.client_manager.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            optimizers_config=optimizers_config
        )
            
    def load_vectorstore(self, collection_name: str) -> QdrantVectorStore:
        
        if not self._does_collection_exist(collection_name):
            self._create_collection(collection_name)
            
        qdrant_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acreate_vectorstore(document_chunks, document_path))
        raise RuntimeError("create_vectorstore() was called inside an event loop; use await acreate_vectorstore() instead")

    async def acreate_vectorstore(
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
        if not self._does_collection_exist(self.collection_name):
            self._create_collection(self.collection_name)
            self.logger.info(f"Created new cloud collection: {self.collection_name}")
        else:
            self.logger.info(f"Cloud collection {self.collection_name} already exists, will upsert documents")
        
        qdrant_vectorstore = self.load_vectorstore(self.collection_name)
        
        # Ingest documents in batches so embeddings and Qdrant upserts are not
        # forced into one network request per chunk.
        try:
            batch_size = 100
            for i in range(0, len(langchain_documents), batch_size):
                batch_docs = langchain_documents[i:i+batch_size]
                batch_ids = valid_doc_ids[i:i+batch_size]
                                
                try:
                    await qdrant_vectorstore.aadd_documents(documents=batch_docs, ids=batch_ids)
                    self.logger.info(
                        f"Ingested batch {i // batch_size + 1}: {len(batch_docs)} documents"
                    )
                except Exception as batch_error:
                    self.logger.error(
                        f"Error processing batch starting at document {i+1}: {batch_error}"
                    )
                    self.logger.info("Retrying failed batch document-by-document")
                    for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                        try:
                            await qdrant_vectorstore.aadd_documents(documents=[doc], ids=[doc_id])
                        except Exception as doc_error:
                            self.logger.error(f"Error adding document {i+j+1}: {doc_error}")
                            continue
            
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

    def retrieve_kg_docs_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
        ) -> List[Dict[str, Any]]:
        try:
        
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=self.retrieval_top_k
            )
        
            retrieved_docs = []
            
            for chunk, score in results:
                try:
                    doc_content = chunk.page_content
                    
                    if not doc_content:
                        self.logger.warning(f"Missing content in chunk")
                        continue
                    
                    # Create document dict with KG-specific metadata
                    doc_dict = {
                        "id": chunk.metadata.get('doc_id', 'unknown'),
                        "content": doc_content,
                        "score": float(score),
                        "disease_name": chunk.metadata.get('disease_name', ''),
                        "description": chunk.metadata.get('description', ''),
                        "cause": chunk.metadata.get('cause', ''),
                        "symptom": chunk.metadata.get('symptom', ''),
                    }
                    
                    # Add any additional metadata from the chunk
                    for key, value in chunk.metadata.items():
                        if key not in ['doc_id', 'disease_name', 'description', 'cause', 'symptom']:
                            doc_dict[key] = value
                            
                    retrieved_docs.append(doc_dict)
                except Exception as e:
                    self.logger.error(f"Error processing KG document: {e}")
                    continue
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} KG documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_kg_docs_chunks: {e}")
            return []
