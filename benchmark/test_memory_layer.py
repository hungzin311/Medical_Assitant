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
    PatientMemoryListRequest,
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


def is_match(expected_text: str, actual_text: str, threshold: float = 0.5) -> bool:
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


def list_all_memories(service, case: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
    response = service.list_conditions(
        PatientMemoryListRequest(
            patient_id=case["patient_id"],
            top_k=top_k,
        )
    )
    return [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in response.get("results", [])
    ]


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


def retrieval_recall_at_k(
    expected: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    k: int,
    expected_ids: Optional[List[str]] = None,
) -> float:
    if not expected:
        return 0.0

    if expected_ids:
        retrieved_ids = set()
        for item in retrieved[:k]:
            metadata = item.get("metadata") or {}
            if metadata.get("expected_memory_id"):
                retrieved_ids.add(metadata["expected_memory_id"])
        matched = sum(1 for memory_id in expected_ids if memory_id in retrieved_ids)
        if matched:
            return matched / len(expected_ids)

    top_texts = [item.get("memory", "") for item in retrieved[:k]]
    matched = sum(
        1 for expected_item in expected if any(is_match(expected_item["text"], text) for text in top_texts)
    )
    return matched / len(expected)


def repeated_question_avoided(
    expected: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    expected_ids: Optional[List[str]] = None,
) -> bool:
    if expected_ids:
        retrieved_ids = {
            (item.get("metadata") or {}).get("expected_memory_id")
            for item in retrieved
        }
        if any(memory_id in retrieved_ids for memory_id in expected_ids):
            return True

    retrieved_text = " ".join(item.get("memory", "") for item in retrieved)
    return any(is_match(expected_item["text"], retrieved_text) for expected_item in expected)


def predict_route(
    query: str,
    case: Dict[str, Any],
    memory_enabled: bool,
) -> Tuple[str, float, str]:
    from agents.agent_decision import decide_agent_route

    started_at = time.time()
    route_result = decide_agent_route(
        query,
        patient_id=case["patient_id"],
        memory_enabled=memory_enabled,
    )
    latency = time.time() - started_at
    return route_result["agent"], latency, route_result.get("reasoning", "")


def run_full_system_answer(
    query: str,
    case: Dict[str, Any],
    memory_enabled: bool,
    graph,
) -> Tuple[str, Optional[str]]:
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
        answer = output.content
    elif isinstance(output, str):
        answer = output
    else:
        messages = result.get("messages") or []
        answer = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    return answer, result.get("routing_agent")


def judge_answer_quality(answer: str, eval_query: Dict[str, Any], query: str) -> Optional[float]:
    if not answer:
        return None

    from utils.llm_config import get_llm, get_qwen_extra_body

    reference_points = "\n".join(f"- {point}" for point in eval_query["reference_answer_points"])
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
    response = get_llm(temperature=0).bind(extra_body=get_qwen_extra_body()).invoke(prompt)
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
    memory_enabled = mode != "M0"

    for case in cases:
        if service is not None:
            clear_patient(service, case["patient_id"])
        if mode == "M1" and service is not None:
            seed_expected_memories(service, case)
        elif mode == "M2" and service is not None:
            write_conversation_memories(service, case)

        expected = case["expected_memories"]

        for eval_query in case["eval_queries"]:
            query = eval_query["query"]
            expected_agent = eval_query["expected_agent"]
            expected_ids = eval_query.get("expected_relevant_memory_ids", [])
            started_at = time.time()

            retrieved, retrieval_latency = ([], 0.0)
            stored_memories: List[Dict[str, Any]] = []
            if mode != "M0" and service is not None:
                retrieved, retrieval_latency = retrieve(service, case, query, top_k=5)
                if mode == "M2":
                    stored_memories = list_all_memories(service, case)

            extraction_source = stored_memories if mode == "M2" else retrieved
            precision, recall = extraction_scores(expected, extraction_source)

            predicted_agent, routing_latency, routing_reason = predict_route(
                query,
                case,
                memory_enabled=memory_enabled,
            )

            baseline_agent = None
            baseline_correct = None
            memory_routing_lift = False
            if eval_query.get("expected_agent_without_memory") is not None:
                if mode == "M0":
                    baseline_agent = predicted_agent
                    baseline_correct = predicted_agent == eval_query["expected_agent_without_memory"]
                else:
                    baseline_agent, _, _ = predict_route(
                        query,
                        case,
                        memory_enabled=False,
                    )
                    baseline_correct = baseline_agent == eval_query["expected_agent_without_memory"]
                    memory_routing_lift = (
                        predicted_agent == expected_agent
                        and baseline_agent != expected_agent
                    )

            answer = ""
            answer_quality = None
            executed_routing_agent = None
            if args.full_system:
                answer, executed_routing_agent = run_full_system_answer(
                    query,
                    case,
                    memory_enabled=memory_enabled,
                    graph=graph,
                )
                if args.llm_judge:
                    answer_quality = judge_answer_quality(answer, eval_query, query)

            latency = time.time() - started_at
            records.append(
                {
                    "mode": mode,
                    "case_id": case["case_id"],
                    "query_id": eval_query.get("query_id", "q1"),
                    "category": case["category"],
                    "difficulty": case.get("difficulty", "medium"),
                    "patient_id": case["patient_id"],
                    "query": query,
                    "retrieved": retrieved,
                    "stored_memories": stored_memories,
                    "memory_extraction_precision": precision,
                    "memory_extraction_recall": recall,
                    "memory_retrieval_recall_at_3": retrieval_recall_at_k(expected, retrieved, 3, expected_ids),
                    "memory_retrieval_recall_at_5": retrieval_recall_at_k(expected, retrieved, 5, expected_ids),
                    "expected_agent": expected_agent,
                    "predicted_agent": predicted_agent,
                    "routing_reason": routing_reason,
                    "routing_correct": predicted_agent == expected_agent,
                    "baseline_agent_without_memory": baseline_agent,
                    "baseline_routing_correct": baseline_correct,
                    "memory_routing_lift": memory_routing_lift,
                    "requires_memory": eval_query.get("requires_memory", False),
                    "repeated_question_avoided": repeated_question_avoided(expected, retrieved, expected_ids),
                    "answer": answer,
                    "executed_routing_agent": executed_routing_agent,
                    "answer_quality_score": answer_quality,
                    "retrieval_latency": retrieval_latency,
                    "routing_latency": routing_latency,
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
    memory_required = [record for record in records if record.get("requires_memory")]
    baseline_records = [record for record in records if record.get("baseline_agent_without_memory") is not None]
    hard_records = [record for record in records if record.get("difficulty") == "hard"]

    def rate(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        if not items:
            return None
        return sum(1 for item in items if item.get(key)) / len(items)

    return {
        "total_cases": total,
        "memory_extraction_precision": avg([record["memory_extraction_precision"] for record in records]),
        "memory_extraction_recall": avg([record["memory_extraction_recall"] for record in records]),
        "memory_retrieval_recall_at_3": avg([record["memory_retrieval_recall_at_3"] for record in records]),
        "memory_retrieval_recall_at_5": avg([record["memory_retrieval_recall_at_5"] for record in records]),
        "routing_accuracy": rate(records, "routing_correct"),
        "routing_accuracy_hard": rate(hard_records, "routing_correct"),
        "routing_accuracy_memory_required": rate(memory_required, "routing_correct"),
        "baseline_routing_accuracy": rate(baseline_records, "baseline_routing_correct"),
        "memory_routing_lift_rate": rate(records, "memory_routing_lift"),
        "repeated_question_avoidance_rate": rate(records, "repeated_question_avoided"),
        "answer_quality_score": avg([record["answer_quality_score"] for record in records]),
        "avg_latency": avg([record["latency"] for record in records]),
        "avg_retrieval_latency": avg([record["retrieval_latency"] for record in records]),
        "avg_routing_latency": avg([record["routing_latency"] for record in records]),
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

        graph = create_agent_graph()

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

# This function is only for evaluation process of memory layer to shorten the entire process (only get the chosen agent)
# def decide_agent_route(
#     query: Union[str, Dict],
#     *,
#     patient_id: str = "PAT_001",
#     conversation_history: Optional[List[BaseMessage]] = None,
#     memory_enabled: bool = True,
#     has_image: bool = False,
#     image_type: Optional[str] = None,
# ) -> Dict[str, Any]:
#     """Run the real Decision Agent routing logic for benchmarking or tooling."""
#     input_text = _input_to_text(query)
#     memory_result = retrieve_patient_memory_for_query(
#         patient_id,
#         input_text,
#         memory_enabled=memory_enabled,
#     )
#     decision_input = _build_decision_input(
#         input_text,
#         conversation_history=conversation_history,
#         patient_memory_context=memory_result["patient_memory_context"],
#         has_image=has_image,
#         image_type=image_type,
#     )
#     decision = get_decision_chain().invoke({"input": decision_input})
#     return {
#         "agent": decision["agent"],
#         "reasoning": decision.get("reasoning", ""),
#         "confidence": decision.get("confidence"),
#         "patient_memory_context": memory_result["patient_memory_context"],
#     }
