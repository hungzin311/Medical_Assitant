from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from llm_config import get_gemini_llm, get_fpt_vietnamese_embedding, get_graph_db
from prompt import cypher_chain_prompt
import threading
from proxy_setting import set_proxy
load_dotenv()
set_proxy()
class KGManager:
    """
    Singleton class for managing shared components (connections and models)
    Provides centralized access to graph, LLM, and embedding models
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None: # Check outside the lock
            with cls._lock:
                if cls._instance is None: # Check inside the lock (in case 2 thread enter the lock)
                    cls._instance = super(KGManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return 
            
        self._graph = get_graph_db()
        self._llm = get_gemini_llm(temperature=0.0)
        self._embedding_model = get_fpt_vietnamese_embedding()
        self._cypher_chain = None
        self._setup_cypher_chain()

        print("KGManager initialized successfully")
        self._initialized = True
                
    def _setup_cypher_chain(self):
        examples = [
            {
                "question": "Phương pháp điều trị cho bệnh u lympho sau phúc mạc là gì?",
                "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'u lympho sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 30;",
            },
            {
                "question": "Nguyên nhân của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
                "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 30;",
            },
            {
                "question": "Triệu chứng của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
                "query": "MATCH (d:Disease)-[:HAS_SYMPTOM]-(s:Symptom) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN s LIMIT 30;",
            },
            {
                "question": "Những bệnh lý nào có thể xuất hiện khi có triệu chứng khóc và đau?",
                "query": "MATCH (s:Symptom) WHERE s.symptoms CONTAINS 'khóc' AND s.symptoms CONTAINS 'đau' MATCH (s)-[:HAS_SYMPTOM]-(d:Disease) WHERE d.description IS NOT NULL RETURN d LIMIT 30;",
            },
            {
                "question": "Có những loại thuốc phổ biến nào để điều trị bệnh chảy máu khoảng cách sau phúc mạc?",
                "query": "MATCH (m:Medication) WHERE m.disease_name CONTAINS 'chảy máu khoảng cách sau phúc mạc' RETURN m LIMIT 30;",
            },
            {
                "question": "Người bệnh u lympho sau phúc mạc nên ăn thực phẩm gì?",
                "query": "MATCH (a:Advice) WHERE a.disease_name CONTAINS 'u lympho sau phúc mạc' RETURN a LIMIT 30;",
            }
        ]
        
        example_prompt = PromptTemplate(
            input_variables=["question", "query"],
            template="User input: {question}\n{query}"
        )
        
        prefix_prompt = cypher_chain_prompt
        
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix_prompt,
            suffix="User input: {question}\n",
            input_variables=["question"],
        )
        
        self._cypher_chain = GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            cypher_prompt=prompt,
            allow_dangerous_requests=True,
            verbose = True,
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

    def get_llm(self, temperature: float = 0.0):
        return get_gemini_llm(temperature=temperature)
    
    def reset_connections(self):
        """Reset all connections if needed"""
        try:
            self._graph = get_graph_db()
            self._llm = self.get_llm(temperature=0.0)
            self._embedding_model = get_fpt_vietnamese_embedding()
            self._setup_cypher_chain()
            print("All connections reset successfully")
        except Exception as e:
            print(f"Error resetting connections: {e}")

def get_kg_manager() -> KGManager:
    return KGManager()
