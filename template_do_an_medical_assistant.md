# 3. Mục tiêu của đồ án tốt nghiệp

## 3.1. Kiến thức sinh viên thu thập được

- Kiến thức về hệ thống AI đa tác nhân (Multi-Agent) và cơ chế điều phối tác vụ bằng workflow dạng đồ thị.
- Kiến thức về xử lý ngôn ngữ tự nhiên trong miền y tế, bao gồm truy xuất tri thức bằng RAG, Knowledge Graph và cơ chế hỏi đáp đa bước.
- Kiến thức về thị giác máy tính và Vision-Language cho bài toán phân tích ảnh y tế, đặc biệt là ảnh nội soi và ảnh tổn thương da.
- Kiến thức về xây dựng mô hình học sâu cho bài toán phân đoạn ảnh y tế, cụ thể cho bài toán polyp segmentation.
- Kiến thức về fine-tune mô hình cho bài toán Medical VQA, kết hợp giữa ảnh và câu hỏi ngôn ngữ tự nhiên để sinh câu trả lời phù hợp.
- Kiến thức về quản lý bộ nhớ dài hạn cho hội thoại y tế, sử dụng Mem0 để lưu và tái sử dụng ngữ cảnh bệnh nhân qua nhiều lần tương tác.

## 3.2. Công nghệ sinh viên thu thập được

- Các thư viện học sâu và xử lý ảnh: PyTorch, Torchvision, OpenCV, Pillow, Segmentation Models PyTorch.
- Các thư viện xử lý ngôn ngữ và mô hình ngôn ngữ lớn: Transformers, Sentence Transformers, LangChain, LangGraph.
- Các công nghệ truy xuất và quản lý tri thức: Qdrant Vector Database, Neo4j Knowledge Graph, mô hình embedding.
- Kỹ thuật tinh chỉnh mô hình bằng SFT và LoRA cho các mô hình VLM trên bộ dữ liệu Kvasir-VQA-x1.
- Công nghệ quản lý long-term memory: Mem0 để lưu context của người bệnh qua nhiều phiên hỏi đáp.
- Các công nghệ triển khai hệ thống: FastAPI, Jinja2 Templates, Uvicorn.

## 3.3. Kỹ năng sinh viên phát triển được

- Kỹ năng phân tích bài toán và thiết kế kiến trúc cho một hệ thống trợ lý y tế đa mô-đun.
- Kỹ năng xây dựng pipeline kết hợp giữa Text Module, Vision Module và cơ chế bộ nhớ dài hạn.
- Kỹ năng xây dựng, huấn luyện và đánh giá mô hình DeepLabV3+ cho bài toán phân đoạn polyp trên ảnh nội soi.
- Kỹ năng huấn luyện và fine-tune mô hình ngôn ngữ - thị giác bằng phương pháp SFT với LoRA trên dữ liệu chuyên biệt.
- Kỹ năng xây dựng hệ thống hỏi đáp có ngữ cảnh, giúp mô hình ghi nhớ thông tin bệnh nhân giữa các lần trao đổi.
- Kỹ năng phát triển API backend, tích hợp mô hình AI vào giao diện web demo và đánh giá chất lượng hệ thống.

## 3.4. Sản phẩm kỳ vọng

- Một hệ thống **Multi-Agent Medical Assistant** có khả năng tiếp nhận câu hỏi y tế bằng tiếng Việt, xử lý câu hỏi văn bản hoặc câu hỏi kèm ảnh y tế, và đưa ra phản hồi có cấu trúc.
- Một **Text Module** hỗ trợ truy xuất song song từ RAG và Knowledge Graph, kết hợp cơ chế multi-hop QA để hỏi thêm thông tin khi dữ kiện chưa đủ.
- Một **Vision Module** hỗ trợ phân tích ảnh y tế, đặc biệt trên ảnh nội soi và ảnh tổn thương da, đồng thời kết nối với bài toán VQA.
- Một mô-đun **polyp segmentation** sử dụng các kiến trúc Deep Learning để hỗ trợ phân đoạn vùng polyp trên ảnh nội soi.
-  mô hình Vision Language Model  được fine-tune bằng **SFT với LoRA** trên bộ dữ liệu **Kvasir-VQA-x1** để cải thiện khả năng trả lời câu hỏi liên quan đến ảnh nội soi.
- Một thành phần **Mem0 long-term memory** có khả năng lưu context người bệnh đã trả lời trước đó và tái sử dụng trong các phiên hội thoại tiếp theo.
- Một giao diện web demo và các API backend phục vụ cho thử nghiệm, trình diễn và đánh giá hệ thống.

## 3.5. Vấn đề thực tiễn đã giải quyết

- Hỗ trợ người dùng tiếp cận thông tin y tế ban đầu theo cách có hệ thống hơn, thay vì chỉ trả lời một lượt thiếu ngữ cảnh.
- Hỗ trợ kết hợp nhiều nguồn tri thức như RAG, Knowledge Graph và web search để tăng độ bao phủ và độ tin cậy của phản hồi.
- Hỗ trợ phân đoạn polyp trên ảnh nội soi bằng mô hình học sâu, từ đó làm rõ hơn vùng tổn thương cần quan tâm.
- Hỗ trợ bài toán hỏi đáp trên ảnh y tế thông qua mô hình MedGemma 4B đã được fine-tune cho dữ liệu chuyên biệt.
- Hỗ trợ ghi nhớ thông tin bệnh nhân dài hạn bằng Mem0, giúp hệ thống không cần hỏi lại các thông tin đã có và tăng mức độ cá nhân hóa của phản hồi.
- Góp phần xây dựng một nền tảng AI y tế hỗ trợ ra quyết định ban đầu; tuy nhiên hệ thống vẫn chỉ mang tính tham khảo và không thay thế bác sĩ.

# 4. Các nội dung sẽ thực hiện và kế hoạch triển khai

**Lưu ý:** Tiến độ dưới đây được trình bày theo mẫu kế hoạch triển khai theo tuần, có thể điều chỉnh tùy theo tình hình thực tế của đề tài.

## Nội dung 1: Tìm hiểu tổng quan về bài toán và công nghệ liên quan, từ tuần 1 đến tuần 3

**Chi tiết:**

- Tìm hiểu các công trình nghiên cứu liên quan đến Medical VQA, multi-agent, RAG, Knowledge Graph và long-term memory trong hội thoại.
- Tìm hiểu các phương pháp xây dựng hệ thống hỏi đáp y tế có kết hợp văn bản và hình ảnh.
- Tìm hiểu bộ dữ liệu Kvasir-VQA-x1 và các đặc điểm của dữ liệu ảnh nội soi, câu hỏi và câu trả lời.
- Tìm hiểu bài toán phân đoạn polyp trên ảnh nội soi và các kiến trúc học sâu phù hợp như DeepLabV3+, PraNet.
- Tìm hiểu các công nghệ chính sẽ sử dụng như PyTorch, Transformers, LangGraph, FastAPI, Mem0, Qdrant và Neo4j.
- Tìm hiểu phương pháp fine-tune mô hình VLM bằng SFT và LoRA.

## Nội dung 2: Tìm hiểu tổng quan về công nghệ liên quan, từ tuần 3 đến tuần 5

**Chi tiết:**

- Tìm hiểu sâu hơn cơ chế hoạt động của Vision Transformers, Transformers và các kỹ thuật huấn luyện, fine-tune mô hình.
- Tìm hiểu các thư viện xử lý ảnh như OpenCV và các thư viện phục vụ huấn luyện như HuggingFace, Accelerate, Weights \& Biases.
- Tìm hiểu cách triển khai Mem0 để lưu trữ và truy xuất context người bệnh trong các cuộc hội thoại nhiều phiên.
- Tìm hiểu các thư viện và công cụ phục vụ xây dựng API và giao diện như FastAPI, Jinja2 hoặc Gradio.
- Tìm hiểu các thư viện tính toán khoa học và trực quan hóa dữ liệu như NumPy, Pandas, SciPy, Matplotlib.

## Nội dung 3: Phân tích, thiết kế, từ tuần 5 đến tuần 10

**Chi tiết:**

- Phân tích yêu cầu chức năng của hệ thống gồm hỏi đáp y tế, truy xuất tri thức, phân tích ảnh y tế và Medical VQA.
- Thiết kế mô-đun segmentation cho ảnh nội soi, lựa chọn DeepLabV3+ làm kiến trúc nền cho bài toán polyp segmentation.
- Thiết kế kiến trúc tổng thể của hệ thống theo mô hình multi-agent, xác định vai trò của Decision Agent, Text Module, Vision Module và Memory Module.
- Thiết kế luồng tích hợp giữa RAG, Knowledge Graph và mô hình VLM được fine-tune cho bài toán VQA.
- Thiết kế cơ chế long-term memory bằng Mem0 để lưu context bệnh nhân qua nhiều lần trao đổi.
- Thiết kế giao diện demo và các API chính để phục vụ quá trình kiểm thử và trình diễn hệ thống.

## Nội dung 4: Xây dựng chương trình, từ tuần 8 đến tuần 15

**Chi tiết:**

- Xây dựng Decision Agent để định tuyến truy vấn theo loại đầu vào và ngữ cảnh hội thoại.
- Xây dựng Text Module gồm KG Agent, RAG Agent, Conversational Agent và Web Search Agent cho bài toán hỏi đáp y tế.
- Xây dựng Vision Module cho phân tích ảnh nội soi, ảnh tổn thương da và các ảnh y tế tổng quát.
- Xây dựng và thử nghiệm mô hình DeepLabV3+, PraNet cho bài toán polyp segmentation, sau đó tích hợp đầu ra phân đoạn vào Vision Module.
- Fine-tune mô hình MedGemma 4B, Qwen_3.5 4B bằng SFT với LoRA trên bộ dữ liệu Kvasir-VQA-x1 để cải thiện chất lượng trả lời cho câu hỏi liên quan đến ảnh nội soi.
- Tích hợp Mem0 vào hệ thống để lưu trữ, cập nhật và tái sử dụng thông tin người bệnh qua nhiều phiên hội thoại.
- Cài đặt giao diện demo để người dùng nhập câu hỏi, tải ảnh y tế và nhận phản hồi từ hệ thống.

## Nội dung 5: Thử nghiệm và đánh giá, từ tuần 14 đến tuần 17

**Chi tiết:**

- Thực hiện đánh giá Vision Module trên các bài toán phân tích ảnh y tế phù hợp, bao gồm phân đoạn và mô tả ảnh.
- Đánh giá mô hình DeepLabV3+, PraNet trên bài toán polyp segmentation bằng các chỉ số phù hợp như Dice Score, IoU hoặc F1-score.
- Đánh giá chất lượng mô hình MedGemma 4B, Qwen_3.5 4B sau fine-tune trên Kvasir-VQA-x1 thông qua các chỉ số như Accuracy, F1-score, BLEU hoặc các thước đo phù hợp cho VQA.
- Đánh giá hiệu quả của Mem0 trong việc duy trì ngữ cảnh dài hạn và giảm số lần hệ thống phải hỏi lại thông tin đã có.
- Phân tích ưu điểm, hạn chế và các trường hợp hệ thống còn trả lời sai hoặc thiếu ngữ cảnh để đề xuất hướng cải tiến.
- Hoàn thiện báo cáo, tổng hợp kết quả thực nghiệm và chuẩn bị tài liệu phục vụ bảo vệ đồ án.
