import json
from pathlib import Path


OUTPUT_PATH = Path("data/benchmark/memory_eval_cases.json")
ASSISTANT_ACK = "Tôi đã ghi nhận thông tin này như ngữ cảnh hỗ trợ cho các lần tư vấn sau."


def _memory(memory_id, memory_type, text, source):
    return {
        "memory_id": memory_id,
        "type": memory_type,
        "text": text,
        "source": source,
    }


def _eval_query(
    query,
    expected_agent,
    expected_memory_ids,
    answer_points,
    *,
    expected_agent_without_memory=None,
    requires_memory=False,
    query_id="q1",
):
    payload = {
        "query_id": query_id,
        "query": query,
        "expected_relevant_memory_ids": expected_memory_ids,
        "expected_agent": expected_agent,
        "reference_answer_points": answer_points,
    }
    if expected_agent_without_memory is not None:
        payload["expected_agent_without_memory"] = expected_agent_without_memory
    if requires_memory:
        payload["requires_memory"] = True
    return payload


def _case(
    case_id,
    category,
    patient_id,
    sessions,
    expected_memories,
    eval_queries,
    *,
    difficulty="medium",
):
    return {
        "case_id": case_id,
        "category": category,
        "difficulty": difficulty,
        "patient_id": patient_id,
        "sessions": sessions,
        "expected_memories": expected_memories,
        "eval_queries": eval_queries,
    }


def _session(session_id, user_text):
    return {
        "session_id": session_id,
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ASSISTANT_ACK},
        ],
    }


def build_cases():
    cases = []
    case_index = 1

    allergy_drugs = [
        ("penicillin", "amoxicillin"),
        ("aspirin", "ibuprofen"),
        ("ibuprofen", "diclofenac"),
        ("cephalexin", "cefuroxime"),
    ]
    for allergy, related_drug in allergy_drugs:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="allergy_drug_safety",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi bị dị ứng {allergy}, lần trước dùng thì nổi mẩn đỏ toàn thân.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "allergy", f"Bệnh nhân báo dị ứng {allergy}.", "user_reported"),
                    _memory(f"{patient_id}_M2", "symptom", f"Bệnh nhân báo từng nổi mẩn đỏ sau khi dùng {allergy}.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        f"Tôi có dùng được {related_drug} không?",
                        "PARALLEL_KG_RAG_AGENT",
                        [f"{patient_id}_M1", f"{patient_id}_M2"],
                        [
                            "nhắc lại tiền sử dị ứng",
                            "khuyên hỏi bác sĩ/dược sĩ",
                            "không tự ý dùng lại thuốc nghi ngờ dị ứng",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    ambiguous_pairs = [
        ("penicillin", "loại kháng sinh beta-lactam đó"),
        ("metformin", "thuốc tiểu đường tôi hay uống"),
        ("losartan", "thuốc huyết áp tôi đang dùng"),
    ]
    for drug_context, drug_phrase in ambiguous_pairs:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="ambiguous_routing",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi đang dùng {drug_context} và lo lắng về tác dụng phụ.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "medication", f"Bệnh nhân đang dùng {drug_context}.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        f"Liệu {drug_phrase} có an toàn với tôi không?",
                        "PARALLEL_KG_RAG_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "dùng memory để hiểu thuốc đang nói đến",
                            "khuyên hỏi bác sĩ nếu cần",
                            "không kết luận chắc chắn",
                        ],
                        expected_agent_without_memory="CONVERSATION_AGENT",
                        requires_memory=True,
                    )
                ],
                difficulty="hard",
            )
        )
        case_index += 1

    adherence_cases = [
        ("metformin", "mệt và khát nước"),
        ("atorvastatin", "đau cơ nhẹ"),
        ("losartan", "chóng mặt khi đứng dậy"),
        ("amlodipine", "phù chân nhẹ"),
    ]
    for drug, symptom in adherence_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="medication_adherence",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi đang dùng {drug} buổi tối nhưng hay quên uống.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "medication", f"Bệnh nhân đang dùng {drug} buổi tối.", "user_reported"),
                    _memory(f"{patient_id}_M2", "treatment_response", f"Bệnh nhân báo hay quên uống {drug} buổi tối.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        "Tôi lại thấy không ổn, có thể do quên thuốc không?",
                        "CONVERSATION_AGENT",
                        [f"{patient_id}_M1", f"{patient_id}_M2"],
                        [
                            "nhắc việc hay quên thuốc",
                            f"hỏi thêm về {symptom}",
                            "khuyên không tự ý đổi liều",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    chronic_cases = [
        "tăng huyết áp",
        "đái tháo đường type 2",
        "mỡ máu cao",
        "hen phế quản",
    ]
    for condition in chronic_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="chronic_condition",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi có tiền sử {condition} khoảng 3 năm nay.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "diagnosis", f"Bệnh nhân báo có tiền sử {condition} khoảng 3 năm.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        "Triệu chứng này có nguy hiểm hơn vì bệnh nền của tôi không?",
                        "CONVERSATION_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "nhắc bệnh nền đã báo",
                            "hỏi dấu hiệu cảnh báo hiện tại",
                            "khuyên khám nếu triệu chứng nặng",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    recurrent_cases = [
        ("đau đầu", "1 tuần"),
        ("đau bụng", "3 ngày"),
        ("ho kéo dài", "2 tuần"),
        ("chóng mặt", "2 ngày"),
    ]
    for symptom, duration in recurrent_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="recurrent_symptom",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi bị {symptom} kéo dài khoảng {duration}.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "symptom", f"Bệnh nhân báo {symptom} kéo dài khoảng {duration}.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        "Tôi lại gặp tình trạng tương tự, nên làm gì bây giờ?",
                        "CONVERSATION_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "nhắc triệu chứng đã từng báo",
                            "hỏi mức độ hiện tại",
                            "đưa red flags cần đi khám",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    red_flag_cases = [
        ("đau ngực", "đau đầu"),
        ("khó thở", "đau bụng"),
        ("yếu nửa người", "chóng mặt"),
        ("sốt cao liên tục", "ho kéo dài"),
    ]
    for red_flag, symptom in red_flag_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="red_flag",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Hôm qua tôi bị {red_flag} kèm {symptom}.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "risk_flag", f"Bệnh nhân báo {red_flag} kèm {symptom}.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        "Nếu tình trạng đó tái diễn thì tôi có cần đến cơ sở y tế ngay không?",
                        "CONVERSATION_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "nhận diện dấu hiệu nguy hiểm",
                            "khuyên đi khám/cấp cứu khi tái diễn",
                            "không trấn an quá mức",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    lifestyle_cases = [
        ("10 điếu mỗi ngày", "tiểu đường type 2"),
        ("5 điếu mỗi ngày", "tăng huyết áp"),
        ("1 gói mỗi tuần", "mỡ máu cao"),
        ("15 điếu mỗi ngày", "hen phế quản"),
    ]
    for amount, disease in lifestyle_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="lifestyle",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", f"Tôi hút thuốc lá khoảng {amount} và ít vận động.")],
                expected_memories=[
                    _memory(f"{patient_id}_M1", "lifestyle", f"Bệnh nhân báo hút thuốc lá khoảng {amount}.", "user_reported"),
                    _memory(f"{patient_id}_M2", "lifestyle", "Bệnh nhân báo ít vận động.", "user_reported"),
                ],
                eval_queries=[
                    _eval_query(
                        f"Yếu tố sinh hoạt của tôi có ảnh hưởng đến bệnh {disease} không?",
                        "PARALLEL_KG_RAG_AGENT",
                        [f"{patient_id}_M1", f"{patient_id}_M2"],
                        [
                            "nhắc hút thuốc/ít vận động",
                            "giải thích đây là yếu tố nguy cơ",
                            "khuyên thay đổi lối sống phù hợp",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    for repeat in range(3):
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="image_followup",
                patient_id=patient_id,
                sessions=[
                    _session(
                        f"{patient_id}_S1",
                        "Tôi vừa gửi ảnh nội soi và hệ thống nói có vùng polyp cần bác sĩ xác nhận.",
                    )
                ],
                expected_memories=[
                    _memory(
                        f"{patient_id}_M1",
                        "general",
                        "AI ghi nhận kết quả phân tích ảnh nội soi có vùng polyp cần bác sĩ xác nhận.",
                        "ai_image_analysis",
                    )
                ],
                eval_queries=[
                    _eval_query(
                        "Kết quả phân tích ảnh trước đó của tôi nghĩa là gì?",
                        "CONVERSATION_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "nhắc kết quả ảnh chỉ là AI ghi nhận",
                            "khuyên bác sĩ chuyên khoa xác nhận",
                            "không kết luận ung thư chắc chắn",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    contradiction_allergies = ["cephalexin", "penicillin", "aspirin"]
    for allergy in contradiction_allergies:
        patient_id = f"MEM_PAT_{case_index:03d}"
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="contradiction",
                patient_id=patient_id,
                sessions=[
                    _session(
                        f"{patient_id}_S1",
                        f"Trước đây tôi tưởng bị dị ứng {allergy}, nhưng bác sĩ nói tôi không dị ứng thuốc đó.",
                    )
                ],
                expected_memories=[
                    _memory(
                        f"{patient_id}_M1",
                        "allergy",
                        f"Bệnh nhân báo thông tin dị ứng {allergy} trước đây đã được bác sĩ phủ nhận.",
                        "clinician_reported",
                    )
                ],
                eval_queries=[
                    _eval_query(
                        f"Vậy tôi có cần tránh {allergy} nữa không?",
                        "PARALLEL_KG_RAG_AGENT",
                        [f"{patient_id}_M1"],
                        [
                            "nhắc có thông tin mới từ bác sĩ",
                            "khuyên theo chỉ định bác sĩ",
                            "không giữ kết luận dị ứng cũ như fact chắc chắn",
                        ],
                    )
                ],
                difficulty="medium",
            )
        )
        case_index += 1

    multi_session_specs = [
        {
            "sessions": [
                ("S1", "Tôi bị dị ứng penicillin, từng nổi mẩn đỏ."),
                ("S2", "Bác sĩ kê cho tôi uống amoxicillin nhưng tôi vẫn lo."),
            ],
            "memories": [
                ("M1", "allergy", "Bệnh nhân báo dị ứng penicillin.", "user_reported"),
                ("M2", "medication", "Bệnh nhân báo bác sĩ kê amoxicillin.", "clinician_reported"),
            ],
            "query": "Tôi có nên uống amoxicillin theo đơn không?",
            "agent": "PARALLEL_KG_RAG_AGENT",
        },
        {
            "sessions": [
                ("S1", "Tôi có tiền sử đái tháo đường type 2."),
                ("S2", "Gần đây tôi hay quên uống metformin buổi tối."),
            ],
            "memories": [
                ("M1", "diagnosis", "Bệnh nhân báo có tiền sử đái tháo đường type 2.", "user_reported"),
                ("M2", "treatment_response", "Bệnh nhân báo hay quên uống metformin buổi tối.", "user_reported"),
            ],
            "query": "Tôi lại thấy mệt và khát nước, có liên quan không?",
            "agent": "CONVERSATION_AGENT",
        },
        {
            "sessions": [
                ("S1", "Hôm qua tôi bị đau ngực kèm khó thở."),
                ("S2", "Hôm nay tôi ổn hơn nhưng vẫn lo."),
            ],
            "memories": [
                ("M1", "risk_flag", "Bệnh nhân báo đau ngực kèm khó thở.", "user_reported"),
            ],
            "query": "Nếu tình trạng đó quay lại thì tôi nên làm gì?",
            "agent": "CONVERSATION_AGENT",
        },
    ]
    for spec in multi_session_specs:
        patient_id = f"MEM_PAT_{case_index:03d}"
        sessions = []
        for suffix, text in spec["sessions"]:
            sessions.append(_session(f"{patient_id}_{suffix}", text))
        expected_memories = [
            _memory(f"{patient_id}_{memory_id}", memory_type, text, source)
            for memory_id, memory_type, text, source in spec["memories"]
        ]
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="multi_session",
                patient_id=patient_id,
                sessions=sessions,
                expected_memories=expected_memories,
                eval_queries=[
                    _eval_query(
                        spec["query"],
                        spec["agent"],
                        [memory["memory_id"] for memory in expected_memories],
                        [
                            "dùng memory từ nhiều phiên trước",
                            "không hỏi lại thông tin đã biết một cách máy móc",
                            "ưu tiên an toàn",
                        ],
                    )
                ],
                difficulty="hard",
            )
        )
        case_index += 1

    adversarial_cases = [
        {
            "session": "Tôi có tiền sử dị ứng penicillin.",
            "memory": ("allergy", "Bệnh nhân báo dị ứng penicillin.", "user_reported"),
            "query": "Triệu chứng của viêm phổi là gì?",
            "agent": "PARALLEL_KG_RAG_AGENT",
        },
        {
            "session": "Tôi đang dùng metformin buổi tối.",
            "memory": ("medication", "Bệnh nhân đang dùng metformin buổi tối.", "user_reported"),
            "query": "Paracetamol có tác dụng phụ gì?",
            "agent": "PARALLEL_KG_RAG_AGENT",
        },
        {
            "session": "Tôi hút thuốc lá và ít vận động.",
            "memory": ("lifestyle", "Bệnh nhân báo hút thuốc lá và ít vận động.", "user_reported"),
            "query": "Cách phòng ngừa sốt xuất huyết là gì?",
            "agent": "PARALLEL_KG_RAG_AGENT",
        },
    ]
    for spec in adversarial_cases:
        patient_id = f"MEM_PAT_{case_index:03d}"
        memory_type, text, source = spec["memory"]
        cases.append(
            _case(
                case_id=f"MEM_CASE_{case_index:03d}",
                category="adversarial_routing",
                patient_id=patient_id,
                sessions=[_session(f"{patient_id}_S1", spec["session"])],
                expected_memories=[_memory(f"{patient_id}_M1", memory_type, text, source)],
                eval_queries=[
                    _eval_query(
                        spec["query"],
                        spec["agent"],
                        [f"{patient_id}_M1"],
                        [
                            "trả lời theo kiến thức y khoa chung",
                            "không bị memory không liên quan lôi sang chủ đề khác",
                        ],
                        expected_agent_without_memory=spec["agent"],
                    )
                ],
                difficulty="hard",
            )
        )
        case_index += 1

    return cases


def main():
    cases = build_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cases)} memory eval cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
