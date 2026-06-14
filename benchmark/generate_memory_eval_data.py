import json
from pathlib import Path


OUTPUT_PATH = Path("data/benchmark/memory_eval_cases.json")


def _case(case_id, category, patient_id, sessions, expected_memories, eval_query, expected_agent, answer_points):
    return {
        "case_id": case_id,
        "category": category,
        "patient_id": patient_id,
        "sessions": sessions,
        "expected_memories": expected_memories,
        "eval_queries": [
            {
                "query": eval_query,
                "expected_relevant_memory_ids": [item["memory_id"] for item in expected_memories],
                "expected_agent": expected_agent,
                "reference_answer_points": answer_points,
            }
        ],
    }


def build_cases():
    templates = [
        (
            "allergy",
            "Tôi bị dị ứng {allergy}, lần trước dùng thì nổi mẩn đỏ.",
            [
                ("allergy", "Bệnh nhân báo dị ứng {allergy}.", "user_reported"),
                ("symptom", "Bệnh nhân báo từng nổi mẩn đỏ sau khi dùng {allergy}.", "user_reported"),
            ],
            "Tôi có dùng được thuốc liên quan {allergy} không?",
            "PARALLEL_KG_RAG_AGENT",
            ["nhắc lại tiền sử dị ứng", "khuyên hỏi bác sĩ/dược sĩ", "không tự ý dùng lại thuốc nghi ngờ dị ứng"],
        ),
        (
            "medication_adherence",
            "Tôi đang dùng {drug} buổi tối nhưng hay quên uống.",
            [
                ("medication", "Bệnh nhân đang dùng {drug} buổi tối.", "user_reported"),
                ("treatment_response", "Bệnh nhân báo hay quên uống {drug} buổi tối.", "user_reported"),
            ],
            "Tôi lại thấy khó chịu, có liên quan việc quên thuốc không?",
            "CONVERSATION_AGENT",
            ["nhắc việc hay quên thuốc", "hỏi thêm triệu chứng và thời điểm", "khuyên không tự ý đổi liều"],
        ),
        (
            "chronic_condition",
            "Tôi có tiền sử {condition} khoảng 3 năm nay.",
            [
                ("diagnosis", "Bệnh nhân báo có tiền sử {condition} khoảng 3 năm.", "user_reported"),
            ],
            "Triệu chứng này có nguy hiểm hơn vì bệnh nền của tôi không?",
            "CONVERSATION_AGENT",
            ["nhắc bệnh nền đã báo", "hỏi dấu hiệu cảnh báo", "khuyên khám nếu triệu chứng nặng"],
        ),
        (
            "recurrent_symptom",
            "Tôi bị {symptom} kéo dài khoảng {duration}.",
            [
                ("symptom", "Bệnh nhân báo {symptom} kéo dài khoảng {duration}.", "user_reported"),
            ],
            "Tôi lại bị như lần trước, nên làm gì?",
            "CONVERSATION_AGENT",
            ["nhắc triệu chứng đã từng báo", "hỏi mức độ hiện tại", "đưa red flags cần đi khám"],
        ),
        (
            "red_flag",
            "Hôm qua tôi bị {red_flag} kèm {symptom}.",
            [
                ("risk_flag", "Bệnh nhân báo {red_flag} kèm {symptom}.", "user_reported"),
            ],
            "Nếu tình trạng đó xuất hiện lại thì có cần đi viện không?",
            "CONVERSATION_AGENT",
            ["nhận diện dấu hiệu nguy hiểm", "khuyên đi khám/cấp cứu khi tái diễn", "không trấn an quá mức"],
        ),
        (
            "lifestyle",
            "Tôi hút thuốc lá khoảng {amount} và ít vận động.",
            [
                ("lifestyle", "Bệnh nhân báo hút thuốc lá khoảng {amount}.", "user_reported"),
                ("lifestyle", "Bệnh nhân báo ít vận động.", "user_reported"),
            ],
            "Yếu tố sinh hoạt của tôi có ảnh hưởng bệnh này không?",
            "PARALLEL_KG_RAG_AGENT",
            ["nhắc hút thuốc/ít vận động", "giải thích đây là yếu tố nguy cơ", "khuyên thay đổi lối sống phù hợp"],
        ),
        (
            "image_followup",
            "Tôi vừa gửi ảnh nội soi và hệ thống nói có vùng polyp cần bác sĩ xác nhận.",
            [
                ("general", "AI ghi nhận kết quả phân tích ảnh nội soi có vùng polyp cần bác sĩ xác nhận.", "ai_image_analysis"),
            ],
            "Kết quả ảnh lần trước có nghĩa là gì?",
            "CONVERSATION_AGENT",
            ["nhắc kết quả ảnh chỉ là AI ghi nhận", "khuyên bác sĩ chuyên khoa xác nhận", "không kết luận ung thư chắc chắn"],
        ),
        (
            "contradiction",
            "Trước đây tôi tưởng bị dị ứng {allergy}, nhưng bác sĩ nói tôi không dị ứng thuốc đó.",
            [
                ("allergy", "Bệnh nhân báo thông tin dị ứng {allergy} trước đây đã được bác sĩ phủ nhận.", "clinician_reported"),
            ],
            "Vậy tôi có cần tránh thuốc đó nữa không?",
            "PARALLEL_KG_RAG_AGENT",
            ["nhắc có thông tin mới từ bác sĩ", "khuyên theo chỉ định bác sĩ", "không giữ kết luận dị ứng cũ như fact chắc chắn"],
        ),
    ]

    values = [
        {"allergy": "penicillin", "drug": "amlodipine", "condition": "tăng huyết áp", "symptom": "đau đầu", "duration": "1 tuần", "red_flag": "đau ngực", "amount": "10 điếu mỗi ngày"},
        {"allergy": "aspirin", "drug": "metformin", "condition": "đái tháo đường type 2", "symptom": "đau bụng", "duration": "3 ngày", "red_flag": "khó thở", "amount": "5 điếu mỗi ngày"},
        {"allergy": "ibuprofen", "drug": "atorvastatin", "condition": "mỡ máu cao", "symptom": "chóng mặt", "duration": "2 ngày", "red_flag": "yếu nửa người", "amount": "1 gói mỗi tuần"},
        {"allergy": "cephalexin", "drug": "losartan", "condition": "hen phế quản", "symptom": "ho kéo dài", "duration": "2 tuần", "red_flag": "sốt cao liên tục", "amount": "15 điếu mỗi ngày"},
    ]

    cases = []
    case_index = 1
    for template_index, template in enumerate(templates):
        category, intro, memory_specs, query, expected_agent, answer_points = template
        repetitions = 4 if category in {"allergy", "medication_adherence", "chronic_condition", "recurrent_symptom", "red_flag", "lifestyle"} else 3
        for repeat_index in range(repetitions):
            value = values[(template_index + repeat_index) % len(values)]
            patient_id = f"MEM_PAT_{case_index:03d}"
            expected_memories = []
            for memory_index, (memory_type, text, source) in enumerate(memory_specs, start=1):
                expected_memories.append(
                    {
                        "memory_id": f"{patient_id}_M{memory_index}",
                        "type": memory_type,
                        "text": text.format(**value),
                        "source": source,
                    }
                )
            cases.append(
                _case(
                    case_id=f"MEM_CASE_{case_index:03d}",
                    category=category,
                    patient_id=patient_id,
                    sessions=[
                        {
                            "session_id": f"{patient_id}_S1",
                            "messages": [
                                {"role": "user", "content": intro.format(**value)},
                                {"role": "assistant", "content": "Tôi đã ghi nhận thông tin này như ngữ cảnh hỗ trợ cho các lần tư vấn sau."},
                            ],
                        }
                    ],
                    expected_memories=expected_memories,
                    eval_query=query.format(**value),
                    expected_agent=expected_agent,
                    answer_points=[point.format(**value) for point in answer_points],
                )
            )
            case_index += 1
            if case_index > 30:
                return cases
    return cases[:30]


def main():
    cases = build_cases()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cases)} memory eval cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
