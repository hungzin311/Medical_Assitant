import sys
import json
import logging
import os
from pathlib import Path
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent))

# Import components
from agents.rag_agent import MedicalRAG
from config import Config
from agents.rag_agent.vectorstore_qdrant_cloud import VectorStoreCloud

def process_jsonl_with_payload(file_path):
    """
    Process a JSONL file with payload data (question, context, etc.)
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of documents with metadata
    """
    logger.info(f"Processing JSONL file with payload: {file_path}")
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Create document with metadata
                    doc = {
                        "page_content": data.get("context", ""),
                        "metadata": {
                            "question_idx": data.get("question_idx", f"unknown_{i}"),
                            "title": data.get("title", ""),
                            "keyword": data.get("keyword", ""),
                            "topic": data.get("topic", ""),
                            "article_url": data.get("article_url", ""),
                            "author": data.get("author", ""),
                            "source": os.path.basename(file_path)
                        }
                    }
                    
                    # Only add if there's content
                    if doc["page_content"]:
                        documents.append(doc)
                        logger.info(f"Processed document {i+1}: {doc['metadata']['question_idx']} - {doc['metadata']['title'][:30]}...")
                    else:
                        logger.warning(f"Skipping line {i+1} - missing context content")
                        
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON line {i+1}: {line[:50]}...")
                    continue
        
        logger.info(f"Successfully processed {len(documents)} documents from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error processing JSONL file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def ingest_documents_to_qdrant(documents, config, collection_name=None):
    """
    Ingest documents directly to Qdrant
    
    Args:
        documents: List of documents with metadata
        config: Configuration object
        collection_name: Optional custom collection name
    
    Returns:
        Result of ingestion
    """
    try:
        # Use custom collection name if provided
        if collection_name:
            original_collection_name = config.rag.collection_name
            config.rag.collection_name = collection_name
        
        # Initialize vector store
        vector_store = VectorStoreCloud(config)
        
        # Extract content and metadata
        contents = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Create vector store
        start_time = time.time()
        logger.info(f"Creating vector store with {len(contents)} documents...")
        
        # Use a custom document path identifier
        document_path = f"jsonl_payload_{start_time}"
        
        # Create vector store with documents and metadata
        result = vector_store.create_vectorstore_with_metadata(
            document_chunks=contents,
            metadatas=metadatas,
            document_path=document_path
        )
        
        # Restore original collection name if changed
        if collection_name and hasattr(config.rag, 'collection_name'):
            config.rag.collection_name = original_collection_name
            
        return {
            "success": True,
            "documents_ingested": len(contents),
            "document_path": document_path
        }
        
    except Exception as e:
        logger.error(f"Error ingesting documents to Qdrant: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }

def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Ingest JSONL with payload data to Qdrant Cloud.")
    
    # Add arguments
    parser.add_argument("--file", type=str, help="Path to JSONL file to ingest")
    parser.add_argument("--dir", type=str, help="Path to directory containing JSONL files to ingest")
    parser.add_argument("--collection", type=str, help="Custom collection name (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = Config()
    
    # Check if Qdrant cloud credentials are available
    if not config.rag.use_local:
        if not config.rag.url or not config.rag.api_key:
            logger.error("Qdrant cloud URL or API key not provided in environment variables.")
            logger.error("Please set QDRANT_URL and QDRANT_API_KEY in your .env file.")
            sys.exit(1)
        else:
            logger.info(f"Using Qdrant cloud instance at {config.rag.url}")
    
    collection_name = args.collection if args.collection else config.rag.collection_name
    logger.info(f"Using collection name: {collection_name}")
    
    try:
        if args.file:
            # Process a single JSONL file
            file_path = args.file
            if not file_path.endswith('.jsonl'):
                logger.error(f"File {file_path} is not a JSONL file")
                sys.exit(1)
                
            # Process the JSONL file to get documents with metadata
            documents = process_jsonl_with_payload(file_path)
            
            if documents:
                # Ingest the documents
                logger.info(f"Ingesting {len(documents)} documents from {file_path}")
                result = ingest_documents_to_qdrant(documents, config, collection_name)
                print("Ingestion result:", json.dumps(result, indent=2))
            else:
                logger.error(f"No valid documents found in {file_path}")
                
        elif args.dir:
            # Process all JSONL files in a directory
            dir_path = args.dir
            if not os.path.isdir(dir_path):
                logger.error(f"Directory {dir_path} does not exist")
                sys.exit(1)
                
            # Get all JSONL files in the directory
            jsonl_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if f.endswith('.jsonl') and os.path.isfile(os.path.join(dir_path, f))]
            
            if not jsonl_files:
                logger.error(f"No JSONL files found in directory {dir_path}")
                sys.exit(1)
                
            # Process each JSONL file
            all_documents = []
            for file_path in jsonl_files:
                documents = process_jsonl_with_payload(file_path)
                if documents:
                    all_documents.extend(documents)
                    logger.info(f"Added {len(documents)} documents from {file_path}")
                else:
                    logger.error(f"No valid documents found in {file_path}")
            
            # Ingest all documents together
            if all_documents:
                logger.info(f"Ingesting {len(all_documents)} total documents from {len(jsonl_files)} files")
                result = ingest_documents_to_qdrant(all_documents, config, collection_name)
                print("Ingestion result:", json.dumps(result, indent=2))
            else:
                logger.error("No valid documents found in any files")
        else:
            logger.error("No file or directory specified. Use --file or --dir argument.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 