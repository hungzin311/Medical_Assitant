import logging
import json
import re
from typing import List, Dict, Any, Optional
from utils.prompt import rag_agent_mcq_evaluation_prompt
from utils.streaming import invoke_with_streaming

class ResponseGenerator:
    def __init__(self, config):
        
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.llm
        self.include_sources = getattr(config.rag, "include_sources", True)

    def generate_response_benchmark(
            self,
            question: str, 
            choices: List[str],
            retrieved_docs: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
    
        try:
            doc_texts = [f"[{doc['disease_name']}]({doc['description']}) \n {doc['cause']} \n {doc['symptom']}" for doc in retrieved_docs]
            prompt = rag_agent_mcq_evaluation_prompt.format(question=question, choices=choices, context = doc_texts)
            from utils.llm_config import get_qwen_extra_body

            response = self.response_generator_model.bind(extra_body=get_qwen_extra_body()).invoke(prompt)

            try:
                response_text = response.content.strip()
                
                match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)  
                
                response_json = json.loads(response_text)
                
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                self.logger.warning(f"Error parsing JSON response: {e}")
                return {
                    "answer_index": None,
                    "not_enough_info": True,
                    "confidence": 0.0
                }
            
            return response_json
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            return {
                "answer_index": None,
                "not_enough_info": True,
                "confidence": 0.0
            }
