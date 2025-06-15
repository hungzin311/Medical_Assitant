import logging
import sys
from agents.rag_agent import MedicalRAG 
from config import Config
import json
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize config and RAG agent
config = Config()
rag_agent = MedicalRAG(config)

# Test query
query = "Một số loại thuốc làm dịu đi phát ban do virus, dị ứng hoặc bệnh ban đỏ?"

try:
    # Load vectorstore and docstore
    print("Loading vector store...")
    vectorstore, docstore = rag_agent.vector_store.load_vectorstore()
    print(f"Vector store loaded successfully. Collection: {rag_agent.vector_store.collection_name}")
    
    # Debug: Check if the collection has documents
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=rag_agent.vector_store.url,
            api_key=rag_agent.vector_store.api_key
        )
        collection_info = client.get_collection(collection_name=rag_agent.vector_store.collection_name)
        print(f"Collection info: {collection_info}")
        count = client.count(collection_name=rag_agent.vector_store.collection_name)
        print(f"Number of vectors in collection: {count.count}")
    except Exception as e:
        print(f"Error checking collection info: {e}")
    
    # Debug: List keys in docstore
    try:
        keys = docstore.keys()
        print(f"Document store contains {len(keys)} keys")
        if len(keys) > 0:
            print(f"Sample keys: {keys[:5]}")
    except Exception as e:
        print(f"Error listing docstore keys: {e}")
    
    # Modified retrieval with error handling
    print("\nTesting query retrieval...")
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=rag_agent.vector_store.retrieval_top_k
    )
    
    print(f"Retrieved {len(results)} results from vector search")
    
    retrieved_docs = []
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Doc ID: {chunk.metadata.get('doc_id', 'Unknown')}")
        print(f"  Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"  Score: {score}")
        
        
        # Get document content with error handling
        doc_id = chunk.metadata.get('doc_id')
        if not doc_id:
            print(f"  Warning: Missing doc_id in metadata")
            continue
            
        try:
            doc_content_bytes = docstore.mget([doc_id])[0]
            if doc_content_bytes is None:
                print(f"  Error: Document content is None for doc_id: {doc_id}")
                continue
                
            doc_content = doc_content_bytes.decode('utf-8')
            with open('doc_content.json', 'w', encoding='utf-8') as f:
                json.dump(doc_content, f, ensure_ascii=False, indent=4)
            
            # Create document dict
            doc_dict = {
                "id": doc_id,
                "content": doc_content,
                "score": score,
                "source": chunk.metadata.get('source', 'Unknown'),
                "source_path": chunk.metadata.get('source_path', 'Unknown'),
            }
            retrieved_docs.append(doc_dict)
            
        except Exception as e:
            print(f"  Error processing document: {e}")
            continue
    
    # Process the query with the RAG agent
    print("\nProcessing full query...")
    response = rag_agent.process_query(query)
    print("\nResponse:")
    print(response)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()








