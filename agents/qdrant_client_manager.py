import logging
import threading
from qdrant_client import QdrantClient

logging.getLogger("httpx").disabled = True

class QdrantClientManager:
    _instance = None
    _client = None
    _lock = threading.Lock()
    def __new__(cls, config=None):
        if cls._instance is None:
            with cls._lock: 
                if cls._instance is None:
                    cls._instance = super(QdrantClientManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config=None):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        if config is None:
            raise ValueError("Config must be provided on first initialization")
            
        # Cloud configuration
        self.qdrant_url = config.rag.url
        self.qdrant_api_key = config.rag.api_key
        
        # Initialize cloud client
        if not self.qdrant_url or not self.qdrant_api_key:
            self.logger.error("Qdrant cloud URL or API key not provided. Check your configuration.")
            raise ValueError("Qdrant cloud URL or API key not provided")
            
        self._client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        self._initialized = True
    
    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client instance."""
        if self._client is None:
            raise RuntimeError("QdrantClientManager not properly initialized")
        return self._client
    
    def does_collection_exist(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False
    
    def create_collection(self, collection_name: str, vectors_config, optimizers_config=None):
        """Create a new collection in Qdrant."""
        try:
            # Delete collection if it exists
            if self.does_collection_exist(collection_name):
                self.logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name=collection_name)
                
            # Create collection with provided configuration
            create_params = {
                "collection_name": collection_name,
                "vectors_config": vectors_config
            }
            
            if optimizers_config:
                create_params["optimizers_config"] = optimizers_config
                
            self.client.create_collection(**create_params)
            self.logger.info(f"Created new collection: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            raise e
