from config import Config 
from agents.rag_agent.response_generator import ResponseGenerator
import os 
import time

os.environ['HTTP_PROXY'] = 'http://10.61.11.42:3128'
os.environ['HTTPS_PROXY'] = 'http://10.61.11.42:3128'

start_time = time.time()
config = Config()
print(config.rag.response_generator_model)

llm = config.rag.response_generator_model 

response = llm.invoke('hello')
end_time = time.time()

print(response)
print(f"Time taken: {end_time - start_time} seconds")
