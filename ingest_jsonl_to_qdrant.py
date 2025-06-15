import sys
import json
import logging
import os
from pathlib import Path
import argparse

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

def process_jsonl_file(file_path):
    """
    Process a JSONL file and return a list of documents
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of document chunks ready for ingestion
    """
    logger.info(f"Processing JSONL file: {file_path}")
    document_chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Extract content and metadata
                    if 'contents' in data:
                        content = data['contents']
                        document_id = data.get('id', '')
                        title = data.get('title', '')
                        
                        # Add as a document chunk
                        document_chunks.append(content)
                        logger.info(f"Processed document: {document_id} - {title[:30]}...")
                    else:
                        logger.warning(f"Skipping line - missing 'contents' field")
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON line: {line[:50]}...")
                    continue
        
        logger.info(f"Successfully processed {len(document_chunks)} documents from {file_path}")
        return document_chunks
    except Exception as e:
        logger.error(f"Error processing JSONL file: {e}")
        return []

def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Ingest JSONL data to Qdrant Cloud.")
    
    # Add arguments
    parser.add_argument("--file", type=str, help="Path to JSONL file to ingest")
    parser.add_argument("--dir", type=str, help="Path to directory containing JSONL files to ingest")
    
    # Parse arguments
    args = parser.parse_args()
    
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
    
    try:
        # Initialize RAG system
        rag = MedicalRAG(config)
        logger.info("Successfully initialized MedicalRAG with cloud configuration")
        
        if args.file:
            # Process a single JSONL file
            file_path = args.file
            if not file_path.endswith('.jsonl'):
                logger.error(f"File {file_path} is not a JSONL file")
                sys.exit(1)
                
            # Process the JSONL file to get document chunks
            document_chunks = process_jsonl_file(file_path)
            
            if document_chunks:
                # Ingest the document chunks
                logger.info(f"Ingesting {len(document_chunks)} document chunks from {file_path}")
                result = rag.ingest_file(file_path, document_chunks=document_chunks)
                print("Ingestion result:", json.dumps(result, indent=2))
            else:
                logger.error(f"No valid document chunks found in {file_path}")
                
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
            for file_path in jsonl_files:
                document_chunks = process_jsonl_file(file_path)
                
                if document_chunks:
                    # Ingest the document chunks
                    logger.info(f"Ingesting {len(document_chunks)} document chunks from {file_path}")
                    result = rag.ingest_file(file_path, document_chunks=document_chunks)
                    print(f"Ingestion result for {os.path.basename(file_path)}:", json.dumps(result, indent=2))
                else:
                    logger.error(f"No valid document chunks found in {file_path}")
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