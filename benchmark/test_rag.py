import time
from agents.rag_agent import MedicalRAG
from utils.config import Config
import re 
import json 
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from utils.llm_config import get_fpt_llm
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger("httpx").disabled = True

llm = get_fpt_llm()

config = Config()
config.rag.llm = llm
config.rag.collection_name = "temp"
rag = MedicalRAG(config)

def parse_llm_response(response_text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)  # chỉ lấy phần { ... }
        
        response_json = json.loads(response_text)
        
        # Extract decision từ JSON đơn giản (không có wrapper)
        answer_index = response_json.get("answer_index")
        not_enough_info = response_json.get("not_enough_info")
        confidence = response_json.get("confidence", 0.0)

        is_not_enough_info = (not_enough_info is not None) or (answer_index is None)

        
        return {
            'answer_index': answer_index,
            'not_enough_info': is_not_enough_info,
            'confidence': confidence
        }
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            'answer_index': None,
            'not_enough_info': True,
            'confidence': 0.0
        }
def process_single_question(question_data):

    question_idx = question_data['question_idx']
    question = question_data['question']
    choices = question_data['choices']
    correct_answer = question_data['answer']
    
    try:
        result = rag.evaluate_mcq(question, choices)
        
        return {
            'question_idx': question_idx,
            'correct_answer': correct_answer,
            'answer_index': result['answer_index'],
            'not_enough_info': result['not_enough_info'],
            'confidence': result['confidence']
        }

    except Exception as e:
        print(f"Error processing question {question_idx}: {e}")
        return {
            'question_idx': question_idx,
            'correct_answer': correct_answer,
            'answer_index': None,
            'not_enough_info': True,
            'confidence': 0.0
        }

def benchmark_rag_agent():
    with open("data/benchmark/symptom_to_disease_mcq_one_hop.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    batch_size = 2
    correct_count = 0 
    total_count  = 0
    not_enough_info_count = 0  # Fixed variable name
    result_list = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        batch_start_time = time.time()
        
        print(f"\nProcessing batch {i//batch_size + 1}: Questions {i} to {min(i + batch_size - 1, len(data) - 1)}")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_question = {
                executor.submit(process_single_question, question_data): question_data 
                for question_data in batch
            }
            
            batch_results = []
            for future in as_completed(future_to_question):
                result = future.result()
                batch_results.append(result)
        
        batch_results.sort(key=lambda x: x['question_idx'])

        for result in batch_results: 
            question_idx = result['question_idx']
            correct_answer = result['correct_answer']
            answer_index = result['answer_index']
            not_enough_info = result['not_enough_info']
            confidence = result['confidence']
            
            print(f"Question {question_idx}: Correct={correct_answer}, Answer={answer_index}, NotEnoughInfo={not_enough_info}, Confidence={confidence:.2f}")
            print("--------------------------------")
            
            result_list.append(result)
            
            total_count += 1
            if not_enough_info:
                not_enough_info_count += 1  # Fixed variable name
            elif answer_index == correct_answer:
                correct_count += 1
                
        batch_time = time.time() - batch_start_time
        print(f"Batch completed in {batch_time:.2f}s")
    
    # Calculate final metrics
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    not_enough_info_rate = not_enough_info_count / total_count * 100 if total_count > 0 else 0
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Not enough info: {not_enough_info_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Not enough info rate: {not_enough_info_rate:.2f}%")
    
    # Add final metrics to result list
    result_list.append({
        'total_questions': total_count,
        'correct_count': correct_count,
        'not_enough_info_count': not_enough_info_count,
        'accuracy': accuracy,
        'not_enough_info_rate': not_enough_info_rate
    })
    
    # Save results to file
    with open(f"data/benchmark/rag_mcq_results_one_hop.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    benchmark_rag_agent()
