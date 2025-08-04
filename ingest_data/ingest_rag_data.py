import sys
import json
import logging
import os
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import your components
from agents.rag_agent import MedicalRAG
from config import Config

import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")

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
    rag = MedicalRAG(config)
    logger.info("Successfully initialized MedicalRAG with cloud configuration")
except Exception as e:
    logger.error(f"Failed to initialize MedicalRAG: {e}")
    sys.exit(1)

# document ingestion
def data_ingestion():
    try:
        if args.file:
            # Define path to file
            file_path = args.file
            # Process and ingest the file
            result = rag.ingest_file(file_path)
        elif args.dir:
            # Define path to dir
            dir_path = args.dir
            # Process and ingest the files
            result = rag.ingest_directory(dir_path)
        else:
            logger.error("No file or directory specified. Use --file or --dir argument.")
            return False

        print("Ingestion result:", json.dumps(result, indent=2))
        return result["success"]
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        return False

# Run tests
if __name__ == "__main__":
    if not args.file and not args.dir:
        print("\nNo arguments provided. Using default test file...")
        args.file = "D:/Multi-Agent-Medical-Assistant/data/raw_extras/diabetes.pdf"
   
    print("\nIngesting document(s)...")
    ingestion_success = data_ingestion()
    
    if ingestion_success:
        print("\nSuccessfully ingested the documents.")
    else:
        print("\nFailed to ingest the documents. Check logs for details.")