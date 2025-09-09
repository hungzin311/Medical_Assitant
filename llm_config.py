from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import TogetherEmbeddings
from langchain_neo4j import Neo4jGraph
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class FPTOpenAIEmbeddings:
    """Simple wrapper to provide an embed_query API compatible with existing usage.

    Uses FPT's embedding service via OpenAI client with configurable base_url and model.
    """
    def __init__(self, base_url: str = None, api_key: str = None, model_name: str = None):
        self.base_url = base_url or os.getenv("FPT_BASE_URL") 
        self.api_key = api_key or os.getenv("FPT_API_KEY")
        self.model_name = model_name or os.getenv("FPT_EMBEDDING_MODEL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_query(self, text: str):
        if text is None:
            return []
        response = self.client.embeddings.create(input=text, model=self.model_name)
        return response.data[0].embedding
def get_gemini_llm(temperature=0.7):
    """Initialize and return a Gemini Pro LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=False
    )

def get_gemini_vision_llm(temperature=0.2):
    """Initialize and return a Gemini Pro Vision LLM instance for image analysis"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Gemini 2.0 supports multimodal inputs
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
        convert_system_message_to_human=False,
        max_output_tokens=4096  # Increase token limit for detailed image analysis
    )

def get_together_embeddings():
    """Initialize and return Together AI embeddings instance"""
    return TogetherEmbeddings(
        model="intfloat/multilingual-e5-large-instruct",
        together_api_key=os.getenv("TOGETHER_API_KEY")
    ) 

def get_fpt_embeddings(): 
    """Alias to obtain the FPT embeddings wrapper."""
    return FPTOpenAIEmbeddings()

def get_graph_db():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return Neo4jGraph(url=uri, username=user, password=password)