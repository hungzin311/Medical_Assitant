import logging
from typing import List, Dict, Optional, Any
from .cypher_query_llm import CypherQueryService
from .context_filter import ContextFilterEmbedding
from utils.llm_config import get_llm
from agents.rag_agent.query_expander import QueryExpander
from utils.config import Config
from utils.prompt import medical_cot_prompt, medical_direct_kg_prompt, medical_mcq_evaluation_prompt
from utils.streaming import invoke_with_streaming
import json
import re

config = Config()
prompt = medical_cot_prompt 
mcq_prompt = medical_mcq_evaluation_prompt

class ResponseGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cypher_query_llm = CypherQueryService()
        self.context_filter = ContextFilterEmbedding()
        self.llm = get_llm(temperature=0.0)
        self.query_expander = QueryExpander(config)
        self.cached_kg_candidates = None
        self.cached_kg_candidates_by_key = {}

    def _cache_key(self, patient_id: Optional[str] = None) -> str:
        return patient_id or "default"

    def set_cached_kg_candidates(self, candidates: List[Dict[str, Any]], patient_id: Optional[str] = None) -> None:
        kg_candidates_json = json.dumps(candidates, ensure_ascii=False)
        self.cached_kg_candidates = kg_candidates_json
        self.cached_kg_candidates_by_key[self._cache_key(patient_id)] = kg_candidates_json

    def get_cached_kg_candidates(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        cached = self.cached_kg_candidates_by_key.get(self._cache_key(patient_id)) or self.cached_kg_candidates
        if not cached:
            return []
        if isinstance(cached, list):
            return cached
        try:
            parsed = json.loads(cached)
        except (TypeError, json.JSONDecodeError):
            self.logger.warning("Cached KG candidates could not be parsed")
            return []
        return parsed if isinstance(parsed, list) else []
    
    def evaluate_mcq(self, question: str, choices: List[str]) -> Dict:
        
        kg_context = self.cypher_query_llm.retrieve_context_from_kg(question)
        
        if kg_context:
            filtered_context = self.context_filter.filter_context(kg_context, patient_context="", question=question)
            kg_candidates_json = json.dumps(filtered_context, ensure_ascii=False)
        else:
            kg_candidates_json = json.dumps([], ensure_ascii=False)
        
        choices_formatted = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
        
        from utils.llm_config import get_qwen_extra_body

        response = self.llm.bind(extra_body=get_qwen_extra_body()).invoke(mcq_prompt.format(
            kg_candidates=kg_candidates_json,
            question=question,
            choices=choices_formatted
        ))
        
        try:
            response_text = response.content.strip()
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if match:
                response_text = match.group(1)  # chỉ lấy phần { ... }
            
            response_json = json.loads(response_text)
            
            # Extract decision từ JSON đơn giản (không có wrapper)
            answer_index = response_json.get("answer_index")
            not_enough_info_field = response_json.get("not_enough_info")
            confidence = response_json.get("confidence", 0.0)
            
            # Xác định xem có đủ thông tin không
            is_not_enough_info = (not_enough_info_field is not None) or (answer_index is None)
            
            return {
                "answer_index": answer_index if not is_not_enough_info else None,
                "not_enough_info": is_not_enough_info,
                "confidence": confidence
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response.content}")
            return {
                "answer_index": None,
                "not_enough_info": True,
                "confidence": 0.0
            }
