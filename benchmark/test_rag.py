import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.rag_agent import MedicalRAG
from utils.config import Config
from utils.llm_config import get_llm, get_qwen_extra_body
from utils.prompt import rag_agent_mcq_evaluation_prompt


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").disabled = True

llm = get_llm()

config = Config()
config.rag.llm = llm
config.rag.collection_name = "temp"
rag = MedicalRAG(config)


def parse_llm_response(response_text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)

        response_json = json.loads(response_text)
        answer_index = response_json.get("answer_index")
        not_enough_info = bool(response_json.get("not_enough_info", answer_index is None))
        confidence = response_json.get("confidence", 0.0)

        return {
            "answer_index": None if not_enough_info else answer_index,
            "not_enough_info": not_enough_info,
            "confidence": confidence,
        }
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            "answer_index": None,
            "not_enough_info": True,
            "confidence": 0.0,
        }


def generate_rag_mcq_response(question, choices, retrieved_docs):
    try:
        doc_texts = [
            f"[{doc.get('disease_name', '')}]({doc.get('description', '')})\n"
            f"{doc.get('cause', '')}\n{doc.get('symptom', '')}"
            for doc in retrieved_docs
        ]
        prompt = rag_agent_mcq_evaluation_prompt.format(
            question=question,
            choices=choices,
            context=doc_texts,
        )
        response = config.rag.llm.bind(extra_body=get_qwen_extra_body()).invoke(prompt)
        return parse_llm_response(response.content.strip())
    except Exception as e:
        logger.error(f"Error generating RAG benchmark response: {e}")
        return {
            "answer_index": None,
            "not_enough_info": True,
            "confidence": 0.0,
        }


def evaluate_rag_mcq(question, choices):
    try:
        expansion_result = rag.query_expander.expand_query(question, mode="rag", chat_history=None)
        expanded_query = expansion_result["expanded_query"]

        retrieved_documents = rag.vectorstore.similarity_search_with_score(
            query=expanded_query,
            k=config.rag.top_k,
        )
        reranked_documents = rag.reranker.rerank(expanded_query, retrieved_documents)
        return generate_rag_mcq_response(expanded_query, choices, reranked_documents)
    except Exception as e:
        logger.error(f"Error evaluating RAG MCQ: {e}")
        return {
            "answer_index": None,
            "not_enough_info": True,
            "confidence": 0.0,
        }


def process_single_question(question_data):
    question_idx = question_data["question_idx"]
    question = question_data["question"]
    choices = question_data["choices"]
    correct_answer = question_data["answer"]

    try:
        result = evaluate_rag_mcq(question, choices)

        return {
            "question_idx": question_idx,
            "correct_answer": correct_answer,
            "answer_index": result["answer_index"],
            "not_enough_info": result["not_enough_info"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        print(f"Error processing question {question_idx}: {e}")
        return {
            "question_idx": question_idx,
            "correct_answer": correct_answer,
            "answer_index": None,
            "not_enough_info": True,
            "confidence": 0.0,
        }


def benchmark_rag_agent():
    with open("data/benchmark/symptom_to_disease_mcq_one_hop.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    batch_size = 2
    correct_count = 0
    total_count = 0
    not_enough_info_count = 0
    result_list = []

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batch_start_time = time.time()

        print(f"\nProcessing batch {i // batch_size + 1}: Questions {i} to {min(i + batch_size - 1, len(data) - 1)}")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_question = {
                executor.submit(process_single_question, question_data): question_data
                for question_data in batch
            }

            batch_results = []
            for future in as_completed(future_to_question):
                result = future.result()
                batch_results.append(result)

        batch_results.sort(key=lambda x: x["question_idx"])

        for result in batch_results:
            question_idx = result["question_idx"]
            correct_answer = result["correct_answer"]
            answer_index = result["answer_index"]
            not_enough_info = result["not_enough_info"]
            confidence = result["confidence"]

            print(
                f"Question {question_idx}: Correct={correct_answer}, "
                f"Answer={answer_index}, NotEnoughInfo={not_enough_info}, Confidence={confidence:.2f}"
            )
            print("--------------------------------")

            result_list.append(result)

            total_count += 1
            if not_enough_info:
                not_enough_info_count += 1
            elif answer_index == correct_answer:
                correct_count += 1

        batch_time = time.time() - batch_start_time
        print(f"Batch completed in {batch_time:.2f}s")

    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    not_enough_info_rate = not_enough_info_count / total_count * 100 if total_count > 0 else 0

    print("\n=== FINAL RESULTS ===")
    print(f"Not enough info: {not_enough_info_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Not enough info rate: {not_enough_info_rate:.2f}%")

    result_list.append(
        {
            "total_questions": total_count,
            "correct_count": correct_count,
            "not_enough_info_count": not_enough_info_count,
            "accuracy": accuracy,
            "not_enough_info_rate": not_enough_info_rate,
        }
    )

    with open("data/benchmark/rag_mcq_results_one_hop.json", "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    benchmark_rag_agent()
