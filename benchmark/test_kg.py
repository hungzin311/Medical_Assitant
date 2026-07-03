from agents.kg_agent import KGQueryEngine
import json

kg_agent = KGQueryEngine()

def benchmark_kg_agent_mcq():
    with open('data/benchmark/symptom_to_disease_mcq_two_hop.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = 0
    not_enough_info_count = 0
    
    result_list = []

    for record in data:
        question_idx = record['question_idx']
        question = record['question']
        choices = record['choices']
        correct_answer = record['answer']
    
        print(f"Question: {question_idx}")
        # Gọi evaluate_mcq
        result = kg_agent.evaluate_mcq(question, choices)
        result_list.append({
            'question_idx': question_idx,
            'correct_answer': correct_answer,
            'answer_index': result['answer_index'],
            'not_enough_info': result['not_enough_info'],
            'confidence': result['confidence']
        })

        print(f"Correct answer: {correct_answer}")
        print(f"Answer index: {result['answer_index']}")
        print(f"Not enough info: {result['not_enough_info']}")
        print(f"Confidence: {result['confidence']}")
        print("--------------------------------")
        # Tính accuracy
        total_count += 1
        if result['not_enough_info']:
            not_enough_info_count += 1
        elif result['answer_index'] == correct_answer:
            correct_count += 1
    
    result_list.append({
        'accuracy': correct_count / total_count * 100,
        'not_enough_info': not_enough_info_count / total_count * 100
    })

    with open('data/benchmark/symptom_to_disease_mcq_two_hop_result.json', 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    benchmark_kg_agent_mcq()