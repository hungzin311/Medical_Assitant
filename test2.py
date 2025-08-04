from config import Config 
from agents.rag_agent.response_generator import ResponseGenerator
import os 

os.environ['HTTP_PROXY'] = 'http://10.61.11.42:3128'
os.environ['HTTPS_PROXY'] = 'http://10.61.11.42:3128'

config = Config()
print(config.rag.response_generator_model)

llm = config.rag.response_generator_model 

response = llm.invoke('hello')

print(response)

