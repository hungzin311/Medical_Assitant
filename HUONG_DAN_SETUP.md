# Hướng dẫn thiết lập và chạy Medical Assistant

Tài liệu này mô tả cách cài đặt môi trường, cấu hình API/dịch vụ ngoài, và chạy ứng dụng **Multi-Agent Medical Chatbot** (FastAPI + LangGraph + RAG/Qdrant + Neo4j + Gemini + embedding FPT).

---

## 1. Yêu cầu hệ thống

| Thành phần | Gợi ý |
|-------------|--------|
| **Hệ điều hành** | Windows 10/11, Linux, hoặc macOS |
| **Python** | 3.10 hoặc 3.11 (khuyến nghị; tránh phiên bản quá cũ vì LangChain/LangGraph) |
| **RAM** | Tối thiểu 8 GB; khuyến nghị 16 GB+ (PyTorch, sentence-transformers, Docling) |
| **GPU** | Không bắt buộc; có CUDA giúp inference nhanh hơn cho một số model |
| **Ổ đĩa** | Vài GB trống cho môi trường ảo, package, và model (nếu bạn thêm file `.pth`) |

---

## 2. Clone repository

```powershell
cd D:\
git clone <URL-repo-của-bạn> Medical_Assitant
cd Medical_Assitant
```

---

## 3. Môi trường ảo Python (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 4. Cài đặt dependencies

Từ thư mục gốc project (nơi có `requirements.txt`):

```powershell
pip install -r requirements.txt
```

**Lưu ý:**

- File này kéo theo **PyTorch**, **transformers**, **opencv**, v.v. — lần cài đầu có thể mất nhiều thời gian và băng thông.
- Nếu bạn làm việc với pipeline NLP/KG bổ sung (spaCy, py2neo, v.v.), có thể cài thêm:

```powershell
pip install -r requirements_nlp.txt
```

---

## 5. Biến môi trường (`.env`)

Tạo file **`.env`** ở thư mục gốc project (cùng cấp với `app.py`). Các module dùng `python-dotenv` để đọc biến.

### 5.1. Bắt buộc cho chạy chính (theo code hiện tại)

| Biến | Mô tả |
|------|--------|
| `GOOGLE_API_KEY` | API key Google AI (Gemini) — dùng cho hội thoại, vision, một số agent |
| `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY_3` | Tuỳ chọn — dùng cho luồng KG / tách quota (xem `utils/llm_config.py`) |
| `QDRANT_URL` | URL cluster **Qdrant Cloud** (ví dụ `https://xxxx.cloud.qdrant.io`) |
| `QDRANT_API_KEY` | API key Qdrant Cloud |
| `FPT_BASE_URL` | Base URL API embedding/OpenAI-compatible của FPT |
| `FPT_API_KEY` | API key tương ứng |
| `FPT_EMBEDDING_MODEL` | Tên model embedding (project còn dùng alias `Vietnamese_Embedding` trong một số chỗ) |
| `NEO4J_URI` | URI Neo4j, ví dụ `neo4j://127.0.0.1:7687` |
| `NEO4J_USER` | User Neo4j (thường `neo4j`) |
| `NEO4J_PASSWORD` | Mật khẩu Neo4j |

### 5.2. Tuỳ chọn nhưng nên có đủ tính năng

| Biến | Mô tả |
|------|--------|
| `TAVILY_API_KEY` | Web search qua Tavily (`langchain_community`) |
| `TOGETHER_API_KEY` | Nếu bạn dùng Together embeddings ở đâu đó trong pipeline |
| `HUGGINGFACE_TOKEN` | Tải model private từ Hugging Face (nếu cần) |
| `FPT_LLM_COMPLETION` | Base URL cho `get_llm` (MedGemma qua FPT) — chỉ khi bạn gọi luồng đó |

### 5.3. Ví dụ khung `.env` (thay giá trị thật)

```env
# Google Gemini
GOOGLE_API_KEY=
GOOGLE_API_KEY_2=
GOOGLE_API_KEY_3=

# Qdrant Cloud
QDRANT_URL=
QDRANT_API_KEY=

# FPT (embedding / OpenAI-compatible)
FPT_BASE_URL=
FPT_API_KEY=
FPT_EMBEDDING_MODEL=

# Neo4j
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# Web search
TAVILY_API_KEY=

# Tuỳ chọn
TOGETHER_API_KEY=
HUGGINGFACE_TOKEN=
FPT_LLM_COMPLETION=
```

---

## 6. Qdrant Cloud và collection

Ứng dụng chính kết nối **Qdrant qua URL + API key** (không chạy Qdrant embedded cho RAG chính trong `QdrantClientManager`).

Trong `utils/config.py`:

- **RAG:** collection mặc định `medical_assistance_rag_vietnamese` (vector `dense`, chiều **1024**, Cosine).
- **Patient DB (vector):** collection `medical_records`.

Bạn cần:

1. Tạo cluster trên [Qdrant Cloud](https://cloud.qdrant.io/).
2. Tạo các collection đúng tên và cấu hình vector (hoặc chạy script ingest — một số script có thể tạo collection khi ingest).

Nếu collection RAG chưa tồn tại, agent RAG có thể báo lỗi dạng collection không tồn tại; khi đó hãy ingest dữ liệu (mục 9) hoặc tạo collection thủ công trên dashboard.

---

## 7. Neo4j (Knowledge Graph)

1. Cài [Neo4j Desktop](https://neo4j.com/download/) hoặc Neo4j Server / Docker.
2. Tạo database, đặt mật khẩu, gán vào `NEO4J_*` trong `.env`.
3. **Nạp dữ liệu graph:** dùng các script trong `ingest_data/` (ví dụ `create_kg.py`).  
   **Lưu ý bảo mật:** trong `ingest_data/create_kg.py` có đoạn `URI` / `AUTH` hardcode phục vụ môi trường dev; trên máy bạn nên **đồng bộ với Neo4j thật** hoặc chuyển hẳn sang đọc từ biến môi trường để tránh lệch cấu hình.

Ứng dụng runtime (`KGManager`) dùng `get_graph_db()` → `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

---

## 8. Proxy HTTP/HTTPS (quan trọng nếu chạy ngoài mạng nội bộ)

Trong `utils/proxy_setting.py`, hàm `set_proxy()` đang gán proxy cố định (`http://10.61.11.42:3128`).  
`app.py` và `agents/agent_decision.py` (và KG) gọi `set_proxy()` khi khởi động.

- Nếu bạn **không** dùng proxy đó: sửa `proxy_setting.py` để trỏ proxy đúng của bạn, hoặc để trống / gọi `unset_proxy()`, hoặc bọc điều kiện theo biến môi trường — nếu không, request ra internet (Gemini, Qdrant, v.v.) có thể thất bại.

---

## 9. Dữ liệu RAG / ingest (tuỳ nhu cầu)

Thư mục `ingest_data/` chứa các script ví dụ:

| Script | Gợi ý mục đích |
|--------|----------------|
| `ingest_rag_data.py` | Ingest file/thư mục vào vector store RAG (cần `QDRANT_URL`, `QDRANT_API_KEY`) |
| `ingest_rag_with_kg_docs.py` | Ingest kết hợp tài liệu liên quan KG |
| `ingest_jsonl_to_qdrant.py` | Đẩy payload JSONL lên Qdrant |
| `create_kg.py` | Xây/tải dữ liệu vào Neo4j |

Chạy ví dụ (đọc `--help` hoặc mở file để xem tham số):

```powershell
python ingest_data/ingest_rag_data.py --file "duong/den/file.pdf"
```

Một số script có đường dẫn mặc định trỏ ổ `D:/...` — cần chỉnh lại cho máy bạn.

---

## 10. Model phân tích ảnh (file trọng số)

Trong `utils/config.py`, `MedicalCVConfig` trỏ tới:

- **Skin lesion:** `./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar`
- **Polyp segmentation:** `./agents/image_analysis_agent/polyp_seg_tool/models/deeplabv3_resnet50.pth`

Repository có thể **không** kèm sẵn các file nặng này. Bạn cần đặt đúng đường dẫn hoặc cập nhật config nếu file nằm chỗ khác; nếu thiếu file, các endpoint phân tích ảnh liên quan có thể lỗi khi gọi model.

---

## 11. Chạy ứng dụng chính

Luôn chạy từ **thư mục gốc** project (để mount `templates/`, `uploads/`, `data/` đúng).

```powershell
cd D:\Medical_Assitant
.\.venv\Scripts\Activate.ps1
python app.py
```

Mặc định trong `utils/config.py`, `APIConfig` dùng `host=127.0.0.1`, `port=3000`.

- **Dashboard:** [http://127.0.0.1:3000/](http://127.0.0.1:3000/)
- **Giao diện chat:** [http://127.0.0.1:3000/chat](http://127.0.0.1:3000/chat)

Có thể chạy tương đương bằng Uvicorn:

```powershell
uvicorn app:app --host 127.0.0.1 --port 3000 --reload
```

Thư mục `uploads/`, `logs/` sẽ được tạo tự động nếu chưa có (`app.py`).

---

## 12. Dịch vụ Patient Memory (tuỳ chọn)

Service Mem0 nằm trong `agents/services/patient_memory_service/`. **Mặc định chưa được mount vào `app.py`**; chạy riêng khi cần.

1. Cài Mem0:

```powershell
pip install "mem0ai[nlp]"
pip install posthog
```

2. Chạy API:

```powershell
uvicorn agents.services.patient_memory_service.api:app --host 127.0.0.1 --port 3010
```

Chi tiết endpoint và biến môi trường bổ sung: xem `agents/services/patient_memory_service/README.md`.

---

## 13. Kiểm tra nhanh sau khi cài

1. `.env` đã đủ `GOOGLE_API_KEY`, `QDRANT_*`, `FPT_*`, `NEO4J_*`.
2. Neo4j đang chạy và URI đúng.
3. Qdrant Cloud bật và collection tồn tại (sau ingest hoặc tạo tay).
4. Proxy không chặn traffic (hoặc đã sửa `proxy_setting.py`).
5. `python app.py` không báo lỗi import; mở trình duyệt tới cổng 3000.

---

## 14. Benchmark / test (tuỳ chọn)

Thư mục `benchmark/` có các script kiểm thử RAG, KG, LLM. Chạy trực tiếp bằng Python sau khi đã cấu hình `.env` và dịch vụ tương ứng.

---

## 15. Tài liệu thêm trong repo

- `README.md` — giới thiệu ngắn và quick start.
- `Report.pdf` — mô tả kiến trúc/hệ thống (nếu có trong repo).
- `agents/services/patient_memory_service/README.md` — memory service.

---

## 16. Gỡ lỗi thường gặp

| Hiện tượng | Hướng xử lý |
|-------------|-------------|
| Lỗi kết nối SSL / timeout ra ngoài | Kiểm tra proxy trong `utils/proxy_setting.py` |
| Qdrant: URL/API key invalid | Kiểm tra `QDRANT_URL` (có `https://`) và API key trên Cloud |
| Neo4j authentication failed | Đúng `NEO4J_PASSWORD`, service Neo4j đã start |
| Collection does not exist | Ingest dữ liệu hoặc tạo collection đúng tên trên Qdrant |
| Thiếu module sau cài | `pip install -r requirements.txt` trong đúng venv |
| Lỗi khi phân tích ảnh | Kiểm tra file model `.pth` / `.pth.tar` theo `MedicalCVConfig` |

---

*Tài liệu được tạo dựa trên cấu trúc và cấu hình trong repository tại thời điểm viết. Nếu team thay đổi cổng, collection, hoặc luồng khởi động, hãy cập nhật song song file này.*

