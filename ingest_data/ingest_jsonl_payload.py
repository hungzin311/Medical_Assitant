import sys
import json
import logging
import os
from pathlib import Path
import argparse
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").disabled = True

sys.path.append(str(Path(__file__).parent.parent))

# Import components
from utils.config import Config
from agents.rag_agent.vectorstore_qdrant_cloud import VectorStoreCloud
from utils.proxy_setting import set_proxy
set_proxy()

def process_jsonl_with_payload(file_path):
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
                    else:
                        logger.warning(f"Skipping line {i+1} - missing context content")
                        
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON line {i+1}: {line[:50]}...")
                    continue
        
        return documents
    except Exception as e:
        logger.error(f"Error processing JSONL file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def chunk_large_content(documents, chunk_size=10000, chunk_overlap=100):
    chunked_documents = []
    
    for doc in documents:
        content = doc["page_content"]
        metadata = doc["metadata"].copy()
        title = metadata.get("title", "")
        
        # If content is smaller than chunk size, keep as is
        if len(content) <= chunk_size:
            chunked_documents.append(doc)
            continue
            
        # Split into chunks
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + chunk_size, len(content))
            
            # If not the first chunk and we have overlap available
            if start > 0:
                start = max(0, start - chunk_overlap)
                
            # Extract chunk
            chunk = content[start:end]
            
            # Add title to the first chunk
            if chunk_id == 0 and title:
                chunk_content = f"{title}\n\n{chunk}"
            else:
                chunk_content = chunk
                
            # Create chunk document with updated metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["is_chunked"] = True
            chunk_metadata["original_length"] = len(content)
            
            chunked_documents.append({
                "page_content": chunk_content,
                "metadata": chunk_metadata
            })
            
            # Move to next chunk
            start = end
            chunk_id += 1
            
        logger.info(f"Split document '{title[:30]}...' into {chunk_id + 1} chunks")
        
    return chunked_documents

def process_document_batch(batch_data, qdrant_vectorstore, document_path):
    batch_contents, batch_metadatas, batch_start_idx = batch_data
    
    try:
        from uuid import uuid4
        from langchain_core.documents import Document
        
        # Generate unique IDs for each chunk in this batch
        doc_ids = [str(uuid4()) for _ in range(len(batch_contents))]
        
        # Create langchain documents
        langchain_documents = []
        for idx, (chunk, metadata) in enumerate(zip(batch_contents, batch_metadatas)):
            # Merge metadata with standard fields
            combined_metadata = {
                "source": f"{document_path}_batch_{batch_start_idx}",
                "doc_id": doc_ids[idx],
                "source_path": f"{document_path}_batch_{batch_start_idx}",
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
        
        # Add documents directly to the shared vectorstore
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        
        return {
            "success": True,
            "batch_start_idx": batch_start_idx,
            "documents_processed": len(doc_ids),
            "doc_ids": doc_ids
        }
        
    except Exception as e:
        logger.error(f"Error processing batch starting at {batch_start_idx}: {e}")
        return {
            "success": False,
            "batch_start_idx": batch_start_idx,
            "error": str(e),
            "documents_processed": 0,
            "doc_ids": []
        }

def save_failed_batches_to_jsonl(failed_batches, output_path):
    """Write permanently failed chunks to JSONL for manual re-ingest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_contents, batch_metadatas, batch_start_idx in failed_batches:
            for content, metadata in zip(batch_contents, batch_metadatas):
                record = {"context": content, "batch_start_idx": batch_start_idx, **metadata}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"Saved {count} failed documents to {output_path}")
    return str(output_path)

def run_batches_with_retry(batches, qdrant_vectorstore, document_path, num_workers, max_retries=3, retry_delay=2.0):
    """Process batches concurrently; retry failures with exponential backoff."""
    all_doc_ids = []
    successful_batches = 0
    pending = list(batches)

    for attempt in range(max_retries + 1):
        if not pending:
            break

        if attempt > 0:
            delay = retry_delay * (2 ** (attempt - 1))
            logger.info(
                f"Retry {attempt}/{max_retries}: {len(pending)} failed batches "
                f"(waiting {delay:.1f}s before retry)..."
            )
            time.sleep(delay)
            # Fewer workers on retry to reduce rate-limit / overload errors
            workers = max(1, num_workers // 2)
        else:
            workers = num_workers
            logger.info(f"Processing {len(pending)} batches with {workers} workers")

        failed_this_round = []
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(process_document_batch, batch, qdrant_vectorstore, document_path): batch
                for batch in pending
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_start_idx = batch[2]
                try:
                    result = future.result()
                    if result["success"]:
                        successful_batches += 1
                        all_doc_ids.extend(result["doc_ids"])
                        logger.info(
                            f"Batch {result['batch_start_idx']} completed: "
                            f"{result['documents_processed']} documents"
                        )
                    else:
                        failed_this_round.append(batch)
                        logger.error(
                            f"Batch {result['batch_start_idx']} failed: "
                            f"{result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    failed_this_round.append(batch)
                    logger.error(f"Batch {batch_start_idx} failed with exception: {e}")

                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    logger.info(
                        f"Round {attempt}: {completed}/{len(pending)} batches done "
                        f"({len(failed_this_round)} failed so far)"
                    )

        pending = failed_this_round

    return all_doc_ids, successful_batches, pending

def ingest_documents_to_qdrant(
    documents,
    config,
    collection_name=None,
    batch_size=50,
    num_workers=8,
    max_retries=3,
    retry_delay=2.0,
    failed_output_path=None,
):
    try:
        # Use custom collection name if provided
        if collection_name:
            original_collection_name = config.rag.collection_name
            config.rag.collection_name = collection_name
        
        # Initialize vector store
        vector_store = VectorStoreCloud(config)
        
        # Chunk large documents
        chunked_documents = chunk_large_content(documents, chunk_size=16000, chunk_overlap=100)
        logger.info(f"Processing {len(documents)} documents into {len(chunked_documents)} chunks")
        
        # Extract content and metadata
        contents = [doc["page_content"] for doc in chunked_documents]
        metadatas = [doc["metadata"] for doc in chunked_documents]
        
        # Create vector store
        start_time = time.time()
        logger.info(f"Creating vector store with {len(contents)} document chunks using {num_workers} workers...")
        
        # Use a custom document path identifier
        document_path = f"jsonl_payload_{start_time}"
        
        # Initialize shared vectorstore once
        qdrant_vectorstore = vector_store.load_vectorstore(collection_name)
        logger.info("Shared vectorstore initialized successfully")
        
        # Prepare batches for concurrent processing
        batches = []
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batches.append((batch_contents, batch_metadatas, i))
        
        logger.info(
            f"Processing {len(batches)} batches (batch_size={batch_size}, "
            f"max_retries={max_retries})"
        )

        all_doc_ids, successful_batches, permanently_failed = run_batches_with_retry(
            batches,
            qdrant_vectorstore,
            document_path,
            num_workers,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        failed_batches = len(permanently_failed)
        failed_output_file = None
        if permanently_failed:
            out_path = failed_output_path or f"failed_ingest_{int(start_time)}.jsonl"
            failed_output_file = save_failed_batches_to_jsonl(permanently_failed, out_path)
            logger.warning(
                f"{failed_batches} batch(es) still failed after {max_retries} retries. "
                f"Re-ingest with: --file {failed_output_file}"
            )

        # Restore original collection name if changed
        if collection_name and hasattr(config.rag, 'collection_name'):
            config.rag.collection_name = original_collection_name
        
        total_time = time.time() - start_time
        logger.info(f"Concurrent ingestion completed in {total_time:.2f}s: {len(all_doc_ids)} documents ingested successfully")
            
        return {
            "success": failed_batches == 0 and len(all_doc_ids) > 0,
            "documents_ingested": len(all_doc_ids),
            "failed_batches": failed_batches,
            "total_batches": len(batches),
            "successful_batches": successful_batches,
            "processing_time": total_time,
            "failed_output_file": failed_output_file,
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
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing (default: 100)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers (default: 8)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry rounds for failed batches (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Base delay in seconds between retries (default: 2.0)")
    parser.add_argument("--failed-output", type=str, default=None, help="Path to save permanently failed chunks as JSONL")
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
                logger.info(f"Ingesting {len(documents)} documents from {file_path} with batch_size={args.batch_size}, workers={args.workers}")
                result = ingest_documents_to_qdrant(
                    documents,
                    config,
                    collection_name,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    failed_output_path=args.failed_output,
                )
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
                logger.info(f"Ingesting {len(all_documents)} total documents from {len(jsonl_files)} files with batch_size={args.batch_size}, workers={args.workers}")
                result = ingest_documents_to_qdrant(
                    all_documents,
                    config,
                    collection_name,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    failed_output_path=args.failed_output,
                )
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