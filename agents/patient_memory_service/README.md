# Patient Memory Service

Service này gắn Mem0 vào project mà không sửa các file cũ. Nó dùng Mem0 để lưu và truy vấn các tình trạng dài hạn của bệnh nhân, ví dụ triệu chứng đã báo, dị ứng, thuốc đang dùng, phản ứng điều trị, red flag, hoặc preference liên quan chăm sóc.

## Cách Chạy Riêng

```bash
pip install "mem0ai[nlp]"
pip install posthog
uvicorn agents.services.patient_memory_service.api:app --host 127.0.0.1 --port 3010
```

Nếu không muốn dùng repo local `./mem0`, có thể cài package:

```bash
pip install "mem0ai[nlp]"
```

Service sẽ dùng các biến môi trường hiện có của project:

- `QDRANT_URL`, `QDRANT_API_KEY`
- `GOOGLE_API_KEY`
- `FPT_BASE_URL`, `FPT_API_KEY`, `FPT_EMBEDDING_MODEL`

Collection mặc định là `medical_patient_memories`, tách riêng khỏi RAG, patient DB và KG.

## Endpoint Chính

Base URL khi chạy riêng:

```text
http://127.0.0.1:3010/api/patient-memory
```

### 1. Health Check

```bash
curl http://127.0.0.1:3010/api/patient-memory/health
```

### 2. Lưu Một Tình Trạng Patient

Endpoint này lưu fact rõ ràng, không dùng LLM để suy diễn thêm.

```bash
curl -X POST http://127.0.0.1:3010/api/patient-memory/conditions \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_001",
    "condition_text": "Bệnh nhân báo dị ứng penicillin, từng nổi mẩn đỏ sau khi dùng.",
    "condition_type": "allergy",
    "status": "active",
    "source": "user_reported",
    "run_id": "session_001",
    "confidence": 0.9,
    "tags": ["drug_allergy", "safety"]
  }'
```

### 3. Trích Xuất Memory Từ Hội Thoại

Endpoint này dùng Mem0 `infer=True`, phù hợp khi muốn lưu các fact condition từ transcript.

```bash
curl -X POST http://127.0.0.1:3010/api/patient-memory/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_001",
    "run_id": "session_001",
    "infer": true,
    "messages": [
      {"role": "user", "content": "Tôi đau đầu 1 tuần nay và hay quên uống thuốc huyết áp buổi tối."},
      {"role": "assistant", "content": "Tôi đã ghi nhận tình trạng đau đầu kéo dài và vấn đề tuân thủ thuốc."}
    ],
    "metadata": {
      "source": "chat"
    }
  }'
```

### 4. Query Memory Theo Patient

```bash
curl -X POST http://127.0.0.1:3010/api/patient-memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_001",
    "query": "Bệnh nhân có dị ứng thuốc gì hoặc vấn đề tuân thủ thuốc không?",
    "top_k": 5,
    "threshold": 0.1
  }'
```

### 5. List Memory Của Patient

```bash
curl "http://127.0.0.1:3010/api/patient-memory/patients/PAT_001/conditions?top_k=20"
```

### 6. Delete

Xóa một memory:

```bash
curl -X DELETE http://127.0.0.1:3010/api/patient-memory/memories/<memory_id>
```

Xóa toàn bộ memory của patient:

```bash
curl -X DELETE "http://127.0.0.1:3010/api/patient-memory/patients/PAT_001/conditions?confirm=true"
```

## Cách Tích Hợp Vào App Hiện Tại Sau Này

Vì yêu cầu hiện tại là không sửa file cũ, service chưa được include vào `app.py`. Khi bạn muốn gắn vào app chính, thêm thủ công vào `app.py`:

```python
from agents.services.patient_memory_service.api import router as patient_memory_router

app.include_router(patient_memory_router)
```

Hoặc gọi wrapper trực tiếp trong agent:

```python
from agents.services.patient_memory_service import get_patient_memory_service
from agents.services.patient_memory_service.schemas import PatientMemorySearchRequest

memory_service = get_patient_memory_service()
memory_context = memory_service.search(
    PatientMemorySearchRequest(
        patient_id="PAT_001",
        query="dị ứng và thuốc đang dùng",
        top_k=5,
    )
)
```

## Biến Môi Trường Tuỳ Chọn

```env
PATIENT_MEMORY_COLLECTION=medical_patient_memories
PATIENT_MEMORY_HISTORY_DB_PATH=data/mem0/patient_memory_history.db
PATIENT_MEMORY_LOCAL_QDRANT_PATH=data/mem0/qdrant
PATIENT_MEMORY_EMBEDDING_DIMS=1024
PATIENT_MEMORY_AGENT_ID=medical_assistant_patient_memory
PATIENT_MEMORY_SERVICE_HOST=127.0.0.1
PATIENT_MEMORY_SERVICE_PORT=3010
PATIENT_MEMORY_USE_PROJECT_PROXY=false
```

## Lưu Ý An Toàn

- Không lưu raw image/base64 vào Mem0.
- Không lưu secrets, số điện thoại, địa chỉ, API key.
- Với chẩn đoán chưa chắc chắn, lưu dạng "bệnh nhân báo..." hoặc "AI nghi ngờ..." thay vì fact khẳng định.
- Patient memory chỉ là ngữ cảnh hỗ trợ, không thay thế RAG, KG hoặc patient profile/form store.
