from utils.llm_config import get_llm
from utils.proxy_setting import * 
from utils.prompt import llm_base_mcq_evaluation_prompt
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

set_proxy()
llm = get_llm()

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
        # Tạo prompt sử dụng PromptTemplate
        prompt = llm_base_mcq_evaluation_prompt.format(
            question=question,
            choices=choices
        )
        
        from utils.llm_config import get_qwen_extra_body

        response = llm.bind(extra_body=get_qwen_extra_body()).invoke(prompt)
        
        response_text = response.content.strip()
        result = parse_llm_response(response_text)
        
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

def benchmark_llm_base_mcq_batch(batch_size=2): 
    with open('data/benchmark/symptom_to_disease_mcq_two_hop.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = 0
    not_enough_info_count = 0
    result_list = []
    
    # Xử lý theo batch
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_start_time = time.time()
        
        print(f"\nProcessing batch {i//batch_size + 1}: Questions {i} to {min(i + batch_size - 1, len(data) - 1)}")
        
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit tất cả tasks trong batch
            future_to_question = {
                executor.submit(process_single_question, question_data): question_data 
                for question_data in batch
            }
            
            # Collect results
            batch_results = []
            for future in as_completed(future_to_question):
                result = future.result()
                batch_results.append(result)
        
        # Sort results theo question_idx để giữ thứ tự
        batch_results.sort(key=lambda x: x['question_idx'])
        
        # Process results và tính accuracy
        for result in batch_results:
            question_idx = result['question_idx']
            correct_answer = result['correct_answer']
            answer_index = result['answer_index']
            not_enough_info = result['not_enough_info']
            confidence = result['confidence']
            
            print(f"Question: {question_idx}")
            print(f"Correct answer: {correct_answer}")
            print(f"Answer index: {answer_index}")
            print(f"Not enough info: {not_enough_info}")
            print(f"Confidence: {confidence}")
            print("--------------------------------")
            
            result_list.append(result)
            
            # Tính accuracy
            total_count += 1
            if not_enough_info:
                not_enough_info_count += 1
            elif answer_index == correct_answer:
                correct_count += 1
        
        batch_time = time.time() - batch_start_time
        print(f"  Batch completed in {batch_time:.2f}s")
            
    result_list.append({
        'accuracy': correct_count / total_count * 100,
        'not_enough_info': not_enough_info_count / total_count * 100
    })

    with open('data/benchmark/llm_base_mcq_two_hop_batch_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

def main():
    batch_size = 2
    benchmark_llm_base_mcq_batch(batch_size)

if __name__ == "__main__":
    main()
