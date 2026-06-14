import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from agents.patient_memory_service.memory_service import get_patient_memory_service
from agents.patient_memory_service.schemas import (
    PatientConditionCreate,
    PatientConversationMemoryCreate,
    PatientMemorySearchRequest,
    MemoryMessage,
)


DATA_PATH = Path("data/benchmark/memory_eval_cases.json")
RESULT_PATH = Path("data/benchmark/memory_eval_results.json")
SUMMARY_PATH = Path("data/benchmark/memory_eval_summary.json")


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\sà-ỹđ]", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set:
    return {token for token in normalize(text).split() if len(token) > 2}


def overlap_score(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def is_match(expected_text: str, actual_text: str, threshold: float = 0.45) -> bool:
    expected_norm = normalize(expected_text)
    actual_norm = normalize(actual_text)
    return expected_norm in actual_norm or overlap_score(expected_norm, actual_norm) >= threshold


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python benchmark/generate_memory_eval_data.py` first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clear_patient(service, patient_id: str) -> None:
    try:
        service.delete_patient_memories(patient_id)
    except Exception:
        pass


def seed_expected_memories(service, case: Dict[str, Any]) -> None:
    for memory in case["expected_memories"]:
        service.add_condition(
            PatientConditionCreate(
                patient_id=case["patient_id"],
                condition_text=memory["text"],
                condition_type=memory["type"],
                metadata={
                    "source": memory["source"],
                    "expected_memory_id": memory["memory_id"],
                    "benchmark_case_id": case["case_id"],
                },
            )
        )


def write_conversation_memories(service, case: Dict[str, Any]) -> None:
    for session in case["sessions"]:
        messages = [
            MemoryMessage(role=message["role"], content=message["content"])
            for message in session["messages"]
            if message["role"] in {"user", "assistant", "system"}
        ]
        service.add_conversation(
            PatientConversationMemoryCreate(
                patient_id=case["patient_id"],
                run_id=session["session_id"],
                infer=True,
                messages=messages,
                metadata={
                    "source": "benchmark_conversation",
                    "benchmark_case_id": case["case_id"],
                },
            )
        )


def retrieve(service, case: Dict[str, Any], query: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
    started_at = time.time()
    response = service.search(
        PatientMemorySearchRequest(
            patient_id=case["patient_id"],
            query=query,
            top_k=top_k,
            threshold=0.1,
        )
    )
    latency = time.time() - started_at
    results = [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in response.get("results", [])
    ]
    return results, latency


def extraction_scores(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> Tuple[float, float]:
    actual_texts = [item.get("memory", "") for item in actual]
    matched_expected = sum(
        1 for expected_item in expected if any(is_match(expected_item["text"], actual_text) for actual_text in actual_texts)
    )
    matched_actual = sum(
        1 for actual_text in actual_texts if any(is_match(expected_item["text"], actual_text) for expected_item in expected)
    )
    precision = matched_actual / len(actual_texts) if actual_texts else 0.0
    recall = matched_expected / len(expected) if expected else 0.0
    return precision, recall


def retrieval_recall_at_k(expected: List[Dict[str, Any]], retrieved: List[Dict[str, Any]], k: int) -> float:
    top_texts = [item.get("memory", "") for item in retrieved[:k]]
    matched = sum(
        1 for expected_item in expected if any(is_match(expected_item["text"], text) for text in top_texts)
    )
    return matched / len(expected) if expected else 0.0


def heuristic_route(query: str, retrieved: List[Dict[str, Any]]) -> str:
    context = normalize(query + " " + " ".join(item.get("memory", "") for item in retrieved))
    if any(keyword in context for keyword in ["ảnh", "hình", "nội soi", "kết quả lần trước", "lần trước"]):
        return "CONVERSATION_AGENT"
    if any(keyword in context for keyword in ["lại", "tái diễn", "khó chịu", "nguy hiểm hơn", "đi viện"]):
        return "CONVERSATION_AGENT"
    return "PARALLEL_KG_RAG_AGENT"


def repeated_question_avoided(expected: List[Dict[str, Any]], retrieved: List[Dict[str, Any]]) -> bool:
    retrieved_text = " ".join(item.get("memory", "") for item in retrieved)
    return any(is_match(expected_item["text"], retrieved_text) for expected_item in expected)


def run_full_system_answer(
    query: str,
    case: Dict[str, Any],
    memory_enabled: bool,
    graph,
) -> str:
    from agents.agent_decision import process_query

    result = process_query(
        query,
        graph=graph,
        patient_id=case["patient_id"],
        session_id=f"{case['case_id']}_eval",
        memory_enabled=memory_enabled,
    )
    output = result.get("output")
    if hasattr(output, "content"):
        return output.content
    if isinstance(output, str):
        return output
    messages = result.get("messages") or []
    if messages and hasattr(messages[-1], "content"):
        return messages[-1].content
    return ""


def judge_answer_quality(answer: str, case: Dict[str, Any], query: str) -> Optional[float]:
    if not answer:
        return None
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is required for --llm-judge.")

    from utils.llm_config import get_gemini_llm

    reference_points = "\n".join(f"- {point}" for point in case["eval_queries"][0]["reference_answer_points"])
    prompt = f"""
You are judging a Vietnamese medical assistant answer.
Score from 1 to 5 using this rubric:
1 = unsafe or ignores the patient memory
3 = partially useful but misses important memory/safety context
5 = uses relevant memory correctly, avoids overdiagnosis, is safe and clear

Query:
{query}

Reference answer points:
{reference_points}

Answer:
{answer}

Return only JSON: {{"score": <number>, "reason": "<short reason>"}}
"""
    response = get_gemini_llm(temperature=0).invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return float(data.get("score"))
    except Exception:
        return None


def evaluate_mode(mode: str, cases: List[Dict[str, Any]], args, graph=None) -> Dict[str, Any]:
    service = get_patient_memory_service() if mode != "M0" else None
    records = []

    for case in cases:
        if service is not None:
            clear_patient(service, case["patient_id"])
        if mode == "M1" and service is not None:
            seed_expected_memories(service, case)
        elif mode == "M2" and service is not None:
            write_conversation_memories(service, case)

        query = case["eval_queries"][0]["query"]
        expected = case["expected_memories"]
        started_at = time.time()
        retrieved, retrieval_latency = ([], 0.0)
        if mode != "M0" and service is not None:
            retrieved, retrieval_latency = retrieve(service, case, query, top_k=5)

        precision, recall = extraction_scores(expected, retrieved)
        route = heuristic_route(query, retrieved)
        expected_agent = case["eval_queries"][0]["expected_agent"]
        answer = ""
        answer_quality = None

        if args.full_system:
            answer = run_full_system_answer(query, case, memory_enabled=(mode != "M0"), graph=graph)
            if args.llm_judge:
                answer_quality = judge_answer_quality(answer, case, query)

        latency = time.time() - started_at
        records.append(
            {
                "mode": mode,
                "case_id": case["case_id"],
                "category": case["category"],
                "patient_id": case["patient_id"],
                "query": query,
                "retrieved": retrieved,
                "memory_extraction_precision": precision,
                "memory_extraction_recall": recall,
                "memory_retrieval_recall_at_3": retrieval_recall_at_k(expected, retrieved, 3),
                "memory_retrieval_recall_at_5": retrieval_recall_at_k(expected, retrieved, 5),
                "expected_agent": expected_agent,
                "predicted_agent": route,
                "routing_correct": route == expected_agent,
                "repeated_question_avoided": repeated_question_avoided(expected, retrieved),
                "answer": answer,
                "answer_quality_score": answer_quality,
                "retrieval_latency": retrieval_latency,
                "latency": latency,
            }
        )

    return {
        "mode": mode,
        "records": records,
        "summary": summarize(records),
    }


def avg(values: List[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    return {
        "total_cases": total,
        "memory_extraction_precision": avg([record["memory_extraction_precision"] for record in records]),
        "memory_extraction_recall": avg([record["memory_extraction_recall"] for record in records]),
        "memory_retrieval_recall_at_3": avg([record["memory_retrieval_recall_at_3"] for record in records]),
        "memory_retrieval_recall_at_5": avg([record["memory_retrieval_recall_at_5"] for record in records]),
        "routing_accuracy": sum(1 for record in records if record["routing_correct"]) / total if total else 0.0,
        "repeated_question_avoidance_rate": sum(1 for record in records if record["repeated_question_avoided"]) / total if total else 0.0,
        "answer_quality_score": avg([record["answer_quality_score"] for record in records]),
        "avg_latency": avg([record["latency"] for record in records]),
        "avg_retrieval_latency": avg([record["retrieval_latency"] for record in records]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mem0 patient memory layer.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--modes", nargs="+", default=["M0", "M1", "M2"], choices=["M0", "M1", "M2"])
    parser.add_argument("--full-system", action="store_true", help="Call the full LangGraph agent for answer generation.")
    parser.add_argument("--llm-judge", action="store_true", help="Use Gemini as answer-quality judge. Requires --full-system.")
    args = parser.parse_args()

    if args.llm_judge and not args.full_system:
        raise ValueError("--llm-judge requires --full-system.")

    cases = load_cases(Path(args.data))
    graph = None
    if args.full_system:
        from agents.agent_decision import create_agent_graph
        from agents.patient_db_agent import PatientQueryEngine
        from utils.config import Config

        config = Config()
        graph = create_agent_graph(PatientQueryEngine(config))

    all_results = []
    summary = {}
    for mode in args.modes:
        mode_result = evaluate_mode(mode, cases, args, graph=graph)
        all_results.extend(mode_result["records"])
        summary[mode] = mode_result["summary"]

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=== MEMORY LAYER EVAL SUMMARY ===")
    for mode, metrics in summary.items():
        print(f"\n[{mode}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    print(f"\nWrote results to {RESULT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
