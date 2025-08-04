import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# # Set proxy environment variables for the entire application
os.environ['HTTP_PROXY'] = 'http://10.61.11.42:3128'
os.environ['HTTPS_PROXY'] = 'http://10.61.11.42:3128'

load_dotenv()

# Create Qdrant client with port 443 (standard HTTPS port) for proxy compatibility
qdrant_client = QdrantClient(
    url="https://46b5af21-cc96-4663-aa25-efb284f99b44.europe-west3-0.gcp.cloud.qdrant.io:443", 
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=60,  # Increased timeout for proxy connections
    prefer_grpc=False,  # Use HTTP instead of gRPC
    https=True  # Ensure HTTPS is used
)

collections = qdrant_client.get_collections()
print("Successfully connected to Qdrant!")
print(collections)
