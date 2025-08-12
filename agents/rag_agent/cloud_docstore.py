import os
import json
import logging
from typing import List, Dict, Any, Optional, Sequence, Iterator
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance


class CloudDocStore:
    """
    Cloud-based document store using Qdrant as the backend storage.
    This replaces LocalFileStore to provide fully cloud-based document storage.
    """
    
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str = "medical_docstore"):
        """
        Initialize cloud document store.
        
        Args:
            qdrant_url: Qdrant cloud URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of the collection to store documents
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the document storage collection exists."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.logger.info(f"Creating document store collection: {self.collection_name}")
                
                # Create collection with minimal vector config (we're using it as a document store)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE)  # Minimal vector config
                )
                
                # Create index for the key field to enable filtering
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="key",
                        field_schema="keyword"
                    )
                    self.logger.info(f"Created index for 'key' field in collection: {self.collection_name}")
                except Exception as index_error:
                    self.logger.warning(f"Could not create index for 'key' field: {index_error}")
                    # Continue without index - will use less efficient filtering
                self.logger.info(f"Created document store collection: {self.collection_name}")
            else:
                self.logger.info(f"Document store collection already exists: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error ensuring collection exists: {e}")
            raise e
    
    def mset(self, key_value_pairs: List[tuple]) -> None:
        """
        Store multiple key-value pairs.
        
        Args:
            key_value_pairs: List of (key, value) tuples where value is bytes
        """
        try:
            points = []
            for i, (key, value) in enumerate(key_value_pairs):
                # Convert bytes to string for storage
                if isinstance(value, bytes):
                    value_str = value.decode('utf-8')
                else:
                    value_str = str(value)
                
                point = PointStruct(
                    id=hash(key) % (2**63),  # Convert string key to integer ID
                    vector=[0.0],  # Dummy vector since we're using this as document store
                    payload={
                        "key": key,
                        "content": value_str,
                        "content_type": "document"
                    }
                )
                points.append(point)
            
            # Upload points in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                self.logger.info(f"Uploaded batch {i//batch_size + 1} with {len(batch)} documents")
            
            self.logger.info(f"Successfully stored {len(key_value_pairs)} documents in cloud")
            
        except Exception as e:
            self.logger.error(f"Error storing documents: {e}")
            raise e
    
    def mget(self, keys: List[str]) -> List[Optional[bytes]]:
        """
        Retrieve multiple values by keys.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            List of values (as bytes) or None if key not found
        """
        try:
            results = []
            
            for key in keys:
                try:
                    # Try to search for the document with this key using filter
                    search_result = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter={
                            "must": [
                                {
                                    "key": "key",
                                    "match": {"value": key}
                                }
                            ]
                        },
                        limit=1
                    )
                    
                    if search_result[0]:  # If we found the document
                        content = search_result[0][0].payload.get("content")
                        if content:
                            results.append(content.encode('utf-8'))
                        else:
                            results.append(None)
                    else:
                        results.append(None)
                        
                except Exception as filter_error:
                    # If filtering fails (no index), fall back to scanning all documents
                    self.logger.warning(f"Filter search failed for key '{key}': {filter_error}")
                    try:
                        # Fallback: scroll through all documents and find the key
                        found = False
                        offset = None
                        while not found:
                            scroll_result = self.client.scroll(
                                collection_name=self.collection_name,
                                limit=100,
                                offset=offset
                            )
                            
                            points, next_offset = scroll_result
                            
                            if not points:
                                break
                            
                            for point in points:
                                if point.payload.get("key") == key:
                                    content = point.payload.get("content")
                                    if content:
                                        results.append(content.encode('utf-8'))
                                    else:
                                        results.append(None)
                                    found = True
                                    break
                            
                            if next_offset is None:
                                break
                            offset = next_offset
                        
                        if not found:
                            results.append(None)
                            
                    except Exception as fallback_error:
                        self.logger.error(f"Both filter and fallback search failed for key '{key}': {fallback_error}")
                        results.append(None)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            # Return None for all keys if there's an error
            return [None] * len(keys)
    
    def mdelete(self, keys: List[str]) -> None:
        """
        Delete multiple documents by keys.
        
        Args:
            keys: List of keys to delete
        """
        try:
            for key in keys:
                try:
                    # Try to find and delete the document with this key using filter
                    search_result = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter={
                            "must": [
                                {
                                    "key": "key", 
                                    "match": {"value": key}
                                }
                            ]
                        },
                        limit=1
                    )
                    
                    if search_result[0]:  # If we found the document
                        point_id = search_result[0][0].id
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=[point_id]
                        )
                        
                except Exception as filter_error:
                    # If filtering fails, fall back to scanning all documents
                    self.logger.warning(f"Filter delete failed for key '{key}': {filter_error}")
                    try:
                        # Fallback: scroll through all documents and find the key to delete
                        offset = None
                        while True:
                            scroll_result = self.client.scroll(
                                collection_name=self.collection_name,
                                limit=100,
                                offset=offset
                            )
                            
                            points, next_offset = scroll_result
                            
                            if not points:
                                break
                            
                            for point in points:
                                if point.payload.get("key") == key:
                                    self.client.delete(
                                        collection_name=self.collection_name,
                                        points_selector=[point.id]
                                    )
                                    break
                            
                            if next_offset is None:
                                break
                            offset = next_offset
                            
                    except Exception as fallback_error:
                        self.logger.error(f"Both filter and fallback delete failed for key '{key}': {fallback_error}")
            
            self.logger.info(f"Successfully deleted {len(keys)} documents from cloud")
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            raise e
    
    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """
        Yield all keys in the store, optionally filtered by prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Yields:
            Document keys
        """
        try:
            # Scroll through all documents
            offset = None
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    key = point.payload.get("key")
                    if key:
                        if prefix is None or key.startswith(prefix):
                            yield key
                
                if next_offset is None:
                    break
                offset = next_offset
                
        except Exception as e:
            self.logger.error(f"Error yielding keys: {e}")
            return
