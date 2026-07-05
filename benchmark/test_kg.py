import json
import re

from agents.kg_agent import KGQueryEngine
from utils.llm_config import get_llm, get_qwen_extra_body
from utils.prompt import medical_mcq_evaluation_prompt


kg_agent = KGQueryEngine()
llm = get_llm(temperature=0.0)


def parse_mcq_response(response_text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)

        response_json = json.loads(response_text)
        answer_index = response_json.get("answer_index")
        not_enough_info = bool(response_json.get("not_enough_info", answer_index is None))

        return {
            "answer_index": None if not_enough_info else answer_index,
            "not_enough_info": not_enough_info,
            "confidence": response_json.get("confidence", 0.0),
        }
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response_text}")
        return {
            "answer_index": None,
            "not_enough_info": True,
            "confidence": 0.0,
        }


def evaluate_kg_mcq(question, choices):
    kg_context = kg_agent.retrieve_medical_context(question)

    if kg_context:
        filtered_context = kg_agent.filter_context_for_patient(
            kg_context,
            patient_info="",
            question=question,
        )
        kg_candidates_json = json.dumps(filtered_context, ensure_ascii=False)
    else:
        kg_candidates_json = json.dumps([], ensure_ascii=False)

    choices_formatted = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
    response = llm.bind(extra_body=get_qwen_extra_body()).invoke(
        medical_mcq_evaluation_prompt.format(
            kg_candidates=kg_candidates_json,
            question=question,
            choices=choices_formatted,
        )
    )

    return parse_mcq_response(response.content.strip())


def benchmark_kg_agent_mcq():
    with open("data/benchmark/symptom_to_disease_mcq_two_hop.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_count = 0
    total_count = 0
    not_enough_info_count = 0
    result_list = []

    for record in data:
        question_idx = record["question_idx"]
        question = record["question"]
        choices = record["choices"]
        correct_answer = record["answer"]

        print(f"Question: {question_idx}")
        result = evaluate_kg_mcq(question, choices)
        result_list.append(
            {
                "question_idx": question_idx,
                "correct_answer": correct_answer,
                "answer_index": result["answer_index"],
                "not_enough_info": result["not_enough_info"],
                "confidence": result["confidence"],
            }
        )

        print(f"Correct answer: {correct_answer}")
        print(f"Answer index: {result['answer_index']}")
        print(f"Not enough info: {result['not_enough_info']}")
        print(f"Confidence: {result['confidence']}")
        print("--------------------------------")

        total_count += 1
        if result["not_enough_info"]:
            not_enough_info_count += 1
        elif result["answer_index"] == correct_answer:
            correct_count += 1

    result_list.append(
        {
            "accuracy": correct_count / total_count * 100,
            "not_enough_info": not_enough_info_count / total_count * 100,
        }
    )

    with open("data/benchmark/symptom_to_disease_mcq_two_hop_result.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    benchmark_kg_agent_mcq()
