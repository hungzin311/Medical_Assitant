import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .retriever import MedlinePlusRetriever
from utils.streaming import invoke_with_streaming


class MedlinePlusAgent:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config.medlineplus
        self.retriever = MedlinePlusRetriever(self.config)
        self.llm = self.config.llm

    def process_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aprocess_query(query=query, chat_history=chat_history, top_k=top_k))
        raise RuntimeError("process_query() was called inside an event loop; use await aprocess_query() instead")

    async def aprocess_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            retrieval_result = await self.retriever.aretrieve(query=query, top_k=top_k)
            documents = retrieval_result["documents"]

            if not documents:
                return {
                    "response": "Tôi không tìm thấy thông tin phù hợp trong MedlinePlus KB.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieval": retrieval_result,
                    "processing_time": time.time() - start_time,
                }

            prompt = self._build_prompt(
                query=query,
                documents=documents,
                linked_entities=retrieval_result.get("linked_entities", []),
                expanded_relations=retrieval_result.get("expanded_relations", []),
                chat_history=chat_history,
            )
            response = invoke_with_streaming(self.llm, prompt)
            response_text = getattr(response, "content", str(response))

            return {
                "response": response_text,
                "sources": self._extract_sources(documents),
                "confidence": self._estimate_confidence(documents),
                "retrieval": retrieval_result,
                "processing_time": time.time() - start_time,
            }
        except Exception as exc:
            self.logger.error("Error processing MedlinePlus query: %s", exc)
            return {
                "response": f"Tôi gặp lỗi khi truy xuất MedlinePlus KB: {exc}",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
            }

    def _build_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        linked_entities: List[Dict[str, Any]],
        expanded_relations: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        history_text = ""
        for message in chat_history or []:
            role = message.get("role", "user")
            content = message.get("content", "")
            history_text += f"{role}: {content}\n"

        context_blocks = []
        for index, doc in enumerate(documents, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Source: MedlinePlus",
                        f"Title: {doc.get('title', '')}",
                        f"Doc type: {doc.get('doc_type', '')}",
                        f"Section type: {doc.get('section_type', '')}",
                        f"URL: {doc.get('source_path', '')}",
                        f"Content: {doc.get('content', '')}",
                    ]
                )
            )

        relation_text = json.dumps(expanded_relations[:10], ensure_ascii=False, indent=2)
        linked_text = json.dumps(linked_entities, ensure_ascii=False, indent=2)

        return f"""Bạn là trợ lý y tế. Trả lời bằng tiếng Việt, chỉ dựa trên ngữ cảnh MedlinePlus được cung cấp.
Nếu ngữ cảnh không đủ, nói rõ là không đủ thông tin. Không chẩn đoán chắc chắn và khuyến nghị người dùng trao đổi với bác sĩ khi phù hợp.

Câu hỏi:
{query}

Lịch sử hội thoại:
{history_text}

Thực thể nhận diện được:
{linked_text}

Quan hệ liên quan:
{relation_text}

Ngữ cảnh MedlinePlus:
{chr(10).join(context_blocks)}

Yêu cầu trả lời:
- Trả lời ngắn gọn, thực tế.
- Nêu rõ ý nghĩa nếu câu hỏi hỏi về kết quả xét nghiệm.
- Không bịa thông tin ngoài ngữ cảnh.
- Cuối câu trả lời ghi nguồn theo dạng [1], [2] nếu sử dụng thông tin từ đoạn đó.
"""

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen = set()
        for doc in documents:
            key = (doc.get("title"), doc.get("source_path"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "title": doc.get("title", "MedlinePlus"),
                    "source": doc.get("source", "MedlinePlus"),
                    "source_path": doc.get("source_path", ""),
                }
            )
        return sources

    def _estimate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0
        top_score = max(float(doc.get("score") or 0.0) for doc in documents)
        return max(0.0, min(1.0, top_score))
