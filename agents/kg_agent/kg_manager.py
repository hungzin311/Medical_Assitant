from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from llm_config import *
from prompt import *
import threading
from proxy_setting import * 
load_dotenv()

class KGManager:
    """
    Singleton class for managing shared components (connections and models)
    Provides centralized access to graph, LLM, and embedding models
    Does NOT contain business logic - only manages shared resources
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(KGManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._graph = None
        self._llm = None
        self._embedding_model = None
        self._cypher_chain = None
        
        # Initialize components
        self._initialize_components()
        self._initialized = True
    
    def _initialize_components(self):
        """Initialize all core components"""
        try:
            # Initialize models
            self._graph = get_graph_db()
            self._llm = get_gemini_llm(temperature=0.0)
            self._embedding_model = get_fpt_vietnamese_embedding()
            
            # Initialize Cypher chain
            self._setup_cypher_chain()
            
            print("KGManager initialized successfully")
        except Exception as e:
            print(f"Error initializing KGManager: {e}")
            raise
    
    def _setup_cypher_chain(self):
        examples = [
            {
                "question": "Phương pháp điều trị cho bệnh u lympho sau phúc mạc là gì?",
                "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'u lympho sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 5;",
            },
            {
                "question": "Nguyên nhân của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
                "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 5;",
            },
            {
                "question": "Triệu chứng của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
                "query": "MATCH (d:Disease)-[:HAS_SYMPTOM]-(s:Symptom) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN s LIMIT 5;",
            },
            {
                "question": "Những bệnh lý nào có thể xuất hiện khi có triệu chứng khóc và đau?",
                "query": "MATCH (s:Symptom) WHERE s.symptoms CONTAINS 'khóc' AND s.symptoms CONTAINS 'đau' MATCH (s)-[:HAS_SYMPTOM]-(d:Disease) WHERE d.description IS NOT NULL RETURN d LIMIT 5;",
            },
            {
                "question": "Có những loại thuốc phổ biến nào để điều trị bệnh chảy máu khoảng cách sau phúc mạc?",
                "query": "MATCH (m:Medication) WHERE m.disease_name CONTAINS 'chảy máu khoảng cách sau phúc mạc' RETURN m LIMIT 5;",
            },
            {
                "question": "Người bệnh u lympho sau phúc mạc nên ăn thực phẩm gì?",
                "query": "MATCH (a:Advice) WHERE a.disease_name CONTAINS 'u lympho sau phúc mạc' RETURN a LIMIT 5;",
            }
        ]
        
        example_prompt = PromptTemplate(
            input_variables=["question", "query"],
            template="User input: {question}\nCypher query: {query}"
        )
        
        prefix_prompt = cypher_chain_prompt
        
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix_prompt,
            suffix="User input: {question}\nCypher query: ",
            input_variables=["question"],
        )
        
        self._cypher_chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            cypher_prompt=prompt,
            allow_dangerous_requests=True,
            return_direct=True
        )
    
    # Property accessors
    @property
    def graph(self):
        return self._graph
    
    @property
    def llm(self):
        return self._llm
    
    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def cypher_chain(self):
        return self._cypher_chain
    
    def reset_connections(self):
        """Reset all connections if needed"""
        try:
            self._graph = get_graph_db()
            self._llm = get_gemini_llm(temperature=0.0)
            self._embedding_model = get_fpt_vietnamese_embedding()
            self._setup_cypher_chain()
            print("All connections reset successfully")
        except Exception as e:
            print(f"Error resetting connections: {e}")

def get_kg_manager() -> KGManager:
    return KGManager()
