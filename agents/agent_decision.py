import logging
import json
import time
from typing import Any, Dict, List, Optional, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import END
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.medlineplus_agent import MedlinePlusAgent
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from langgraph.checkpoint.memory import MemorySaver
from agents.kg_agent import KGQueryEngine
from agents.patient_db_agent import PatientQueryEngine
from utils.llm_config import *
from utils.proxy_setting import *
from utils.prompt import decision_agent_prompt, conversation_agent_prompt, medical_multi_source_cot_prompt
from utils.config import Config
from utils.streaming import emit_stream_event, invoke_with_streaming
import concurrent.futures

#Set proxy  
set_proxy()
load_dotenv()
config = Config()
memory = MemorySaver()
image_analyzer = ImageAnalysisAgent(config=config)
logger = logging.getLogger(__name__)

def get_agent_status_message(agent_name: str) -> str:
    messages = {
        "CONVERSATION_AGENT": "Processing your conversation request...",
        "PARALLEL_KG_RAG_AGENT": "Searching KG, RAG, and MedlinePlus in parallel...",
        "KG_AGENT": "Searching the medical knowledge graph...",
        "RAG_AGENT": "Searching the medical knowledge database...",
        "MEDLINEPLUS_AGENT": "Searching MedlinePlus knowledge base...",
        "WEB_SEARCH_PROCESSOR_AGENT": "Searching the web for latest medical information...",
        "POLYP_SEGMENTATION_AGENT": "Analyzing polyp image...",
        "POLYP_VQA_AGENT": "Answering your polyp image question...",
        "GENERAL_MEDICAL_IMAGE_AGENT": "Analyzing medical image...",
    }
    return messages.get(agent_name, "Processing your medical query...")

class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    patient_id: Optional[str]  # Patient ID for KG and patient database queries
    session_id: Optional[str]  # Session ID for LangGraph checkpoint and Mem0 run scope
    patient_memory_context: Optional[str]  # Retrieved long-term memory context
    patient_memory_items: List[Dict[str, Any]]  # Raw normalized Mem0 search results
    memory_enabled: bool  # Whether long-term memory is enabled for this run
    routing_agent: Optional[str]  # Agent selected at routing time (preserved for evaluation)
    polyp_segmentation_path: Optional[str]  # Segmentation overlay path for polyp VQA


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def _input_to_text(current_input: Union[str, Dict, None], include_image_hint: bool = False) -> str:
    if isinstance(current_input, dict):
        text = current_input.get("text", "") or ""
        if include_image_hint and current_input.get("image"):
            text = f"{text}, user uploaded an image for diagnosis.".strip(", ")
        return text
    return current_input or ""


def _shorten_text(text: str, max_chars: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


_decision_chain = None


def get_decision_chain():
    """Return the shared Decision Agent chain used by LangGraph routing."""
    global _decision_chain
    if _decision_chain is None:
        decision_model = config.agent_decision.llm
        json_parser = JsonOutputParser(pydantic_object=AgentDecision)
        decision_prompt = ChatPromptTemplate.from_messages([
            ("human", f"System: {decision_agent_prompt}\n\nUser: {{input}}"),
        ])
        _decision_chain = decision_prompt | decision_model | json_parser
    return _decision_chain


def retrieve_patient_memory_for_query(
    patient_id: str,
    query: str,
    *,
    memory_enabled: bool = True,
    top_k: int = 5,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Retrieve durable patient memory for routing or answer generation."""
    if not memory_enabled or not query.strip():
        return {
            "patient_memory_context": "",
            "patient_memory_items": [],
        }

    try:
        from agents.patient_memory_service.memory_service import get_patient_memory_service
        from agents.patient_memory_service.schemas import PatientMemorySearchRequest

        service = get_patient_memory_service()
        search_result = service.search(
            PatientMemorySearchRequest(
                patient_id=patient_id,
                query=query,
                top_k=top_k,
                threshold=threshold,
            )
        )
        memory_items = [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in search_result.get("results", [])
        ]
        return {
            "patient_memory_context": _format_patient_memory_context(memory_items),
            "patient_memory_items": memory_items,
        }
    except Exception as exc:
        logger.warning("Patient memory retrieval skipped: %s", exc)
        return {
            "patient_memory_context": "",
            "patient_memory_items": [],
        }


def _build_decision_input(
    query: str,
    *,
    conversation_history: Optional[List[BaseMessage]] = None,
    patient_memory_context: str = "",
    has_image: bool = False,
    image_type: Optional[str] = None,
) -> str:
    recent_context = ""
    for message in (conversation_history or [])[-6:]:
        if isinstance(message, HumanMessage):
            recent_context += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            recent_context += f"Assistant: {message.content}\n"

    return f"""
        User query: {query}

        Recent conversation context:
        {recent_context or 'None'}

        Relevant long-term patient memory:
        {patient_memory_context if patient_memory_context else 'None'}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """


def decide_agent_route(
    query: Union[str, Dict],
    *,
    patient_id: str = "PAT_001",
    conversation_history: Optional[List[BaseMessage]] = None,
    memory_enabled: bool = True,
    has_image: bool = False,
    image_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the real Decision Agent routing logic for benchmarking or tooling."""
    input_text = _input_to_text(query)
    memory_result = retrieve_patient_memory_for_query(
        patient_id,
        input_text,
        memory_enabled=memory_enabled,
    )
    decision_input = _build_decision_input(
        input_text,
        conversation_history=conversation_history,
        patient_memory_context=memory_result["patient_memory_context"],
        has_image=has_image,
        image_type=image_type,
    )
    decision = get_decision_chain().invoke({"input": decision_input})
    return {
        "agent": decision["agent"],
        "reasoning": decision.get("reasoning", ""),
        "confidence": decision.get("confidence"),
        "patient_memory_context": memory_result["patient_memory_context"],
        "patient_memory_items": memory_result["patient_memory_items"],
    }


def _format_patient_memory_context(memory_items: List[Dict[str, Any]]) -> str:
    if not memory_items:
        return ""

    lines = []
    for index, item in enumerate(memory_items[:5], start=1):
        memory_text = item.get("memory") or item.get("text") or ""
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or metadata.get("memory_kind") or "memory"
        score = item.get("score")

        if source in {"ai_image_analysis", "assistant", "ai"}:
            prefix = "AI ghi nhận chưa xác nhận"
        else:
            prefix = "Bệnh nhân báo"

        score_text = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"{index}. {prefix}: {_shorten_text(memory_text, 300)} (source={source}{score_text})")

    return "\n".join(lines)


def _memory_context_message(patient_memory_context: Optional[str]) -> List[Dict[str, str]]:
    if not patient_memory_context:
        return []
    return [
        {
            "role": "system",
            "content": (
                "LONG-TERM PATIENT MEMORY (supporting context only, not a confirmed diagnosis):\n"
                f"{patient_memory_context}"
            ),
        }
    ]


def create_agent_graph(patient_query_engine: PatientQueryEngine):
    """Create and configure the LangGraph for agent orchestration."""
    decision_chain = get_decision_chain()

    kg_agent = KGQueryEngine(patient_query_engine)
    rag_agent = MedicalRAG(config)
    medlineplus_agent = MedlinePlusAgent(config)

    def is_polyp_segmentation_request(input_text: str) -> bool:
        input_text = (input_text or "").lower()
        segmentation_keywords = [
            "phân vùng",
            "phan vung",
            "segment",
            "segmentation",
            "mask",
            "overlay",
            "vùng polyp",
            "vung polyp",
        ]
        return any(keyword in input_text for keyword in segmentation_keywords)
    
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = _input_to_text(current_input)
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = image_analyzer.analyze_image(image_path, input_text)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Set to False to ensure normal routing
        }

    def retrieve_patient_memory(state: AgentState) -> AgentState:
        """Retrieve durable patient memory before routing and answer generation."""
        patient_id = state.get("patient_id") or "PAT_001"
        query = _input_to_text(state.get("current_input"), include_image_hint=True)
        memory_result = retrieve_patient_memory_for_query(
            patient_id,
            query,
            memory_enabled=state.get("memory_enabled", True),
        )
        if memory_result["patient_memory_context"]:
            emit_stream_event("status", {"message": "Retrieved relevant patient memory..."})
        return {
            **state,
            **memory_result,
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        patient_memory_context = state.get("patient_memory_context") or ""
        has_image = state["has_image"]
        image_type = state["image_type"]

        # Check for NON-MEDICAL images and reject them
        if image_type == "NON-MEDICAL":
            updated_state = {
                **state,
                "output": AIMessage(content="Tôi rất xin lỗi, nhưng hình ảnh bạn tải lên không phải là hình ảnh y tế. Tôi chỉ có thể phân tích các hình ảnh y tế như X-quang, CT, MRI, ảnh da, nội soi, v.v. Vui lòng tải lên hình ảnh y tế để tôi có thể hỗ trợ bạn."),
            }
            # print("Updated state: ", updated_state['output'])
            return {**updated_state,
                    "agent_name": "NON_MEDICAL_FILTER",
                    "next": "apply_guardrails"}
        
        # Prepare input for decision model
        input_text = _input_to_text(current_input)
        patient_memory_context = state.get("patient_memory_context") or ""

        if has_image and image_type == "POLYP SEGMENTATION":
            selected_agent = (
                "POLYP_SEGMENTATION_AGENT"
                if not input_text.strip() or is_polyp_segmentation_request(input_text)
                else "POLYP_VQA_AGENT"
            )
            emit_stream_event("agent", {
                "agent": selected_agent,
                "message": get_agent_status_message(selected_agent),
                "transition": True,
            })
            updated_state = {
                **state,
                "agent_name": selected_agent,
                "routing_agent": selected_agent,
            }
            return {"agent_state": updated_state, "next": selected_agent}

        decision_input = _build_decision_input(
            input_text,
            conversation_history=messages,
            patient_memory_context=patient_memory_context,
            has_image=has_image,
            image_type=image_type,
        )

        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        emit_stream_event("agent", {
            "agent": decision["agent"],
            "message": get_agent_status_message(decision["agent"]),
            "transition": True,
        })
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
            "routing_agent": decision["agent"],
        }

        # if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
        #     return {"agent_state": updated_state, "next": "needs_validation"}
        return {"agent_state": updated_state, "next": decision["agent"]}

    def run_conversation_agent(state: AgentState) -> AgentState:

        print(f"Selected agent: CONVERSATION_AGENT")
        emit_stream_event("agent", {
            "agent": "CONVERSATION_AGENT",
            "message": "Processing your conversation request...",
            "transition": False,
        })

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = _input_to_text(current_input)
        
        follow_up_keywords = [
            # Từ khóa về chẩn đoán trước đó
            "chẩn đoán trước", "chẩn đoán lúc nãy", "kết quả vừa rồi", "kết quả trước",
            "chẩn đoán ban đầu", "chẩn đoán lần trước", "phân tích trước đó",
            
            # Từ khóa về hình ảnh
            "hình ảnh này", "ảnh này", "bức ảnh", "hình vừa gửi", "ảnh lúc nãy",
            "hình ảnh trước", "ảnh đó", "hình đó", "X-quang này", "CT này",
            
            # Từ khóa về kết quả
            "kết quả này", "kết quả đó", "những gì bạn tìm thấy", "phát hiện của bạn",
            "những gì phát hiện", "kết quả phân tích",
            
            # Yêu cầu giải thích thêm
            "nói thêm về", "giải thích thêm", "chi tiết hơn", "thông tin thêm",
            "mô tả rõ hơn", "phân tích kỹ hơn", "làm rõ", "cụ thể hơn",
            
            # Câu hỏi theo sau
            "về cái này", "về điều đó", "về vấn đề này", "liên quan đến",
            "dựa trên", "căn cứ vào", "theo kết quả"
        ]
        
        is_follow_up = any(keyword in input_text.lower() for keyword in follow_up_keywords)
        
        # If it seems like a follow-up question and we have previous agent names in the state
        if is_follow_up and len(messages) > 2:
            # Look for the most recent image ID in the conversation history
            image_id = None
            for i in range(len(messages)-1, -1, -1):
                if isinstance(messages[i], AIMessage): 
                    if "image_id" in getattr(messages[i], "metadata", {}): 
                        image_id = getattr(messages[i], "metadata", {}).get("image_id", None)
                        print(f"Found image_id: {image_id}")
                        break
            
            # If we found an image ID, try to generate a follow-up response
            if image_id:
                try:
                    print(f"Found image_id: {image_id}, generating follow-up response")
                    follow_up_response = image_analyzer.generate_followup_response(image_id, input_text)
                    return {
                        **state,
                        "output": AIMessage(content=follow_up_response),
                        "agent_name": "CONVERSATION_AGENT"
                    }
                except Exception as e:
                    print(f"Error generating follow-up response: {e}")

        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:  # currently considering complete history - limit control from utils.config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
                
        # Combine everything for the decision input
        conversation_prompt = f"""Câu hỏi người dùng: {input_text}

        Ngữ cảnh cuộc trò chuyện gần đây: {recent_context}
        {conversation_agent_prompt}
        """

        if patient_memory_context:
            conversation_prompt += (
                "\nLong-term patient memory (supporting context only, not a confirmed diagnosis):\n"
                f"{patient_memory_context}\n"
            )

        response = invoke_with_streaming(config.conversation.llm, conversation_prompt)

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_kg_rag_parallel(state: AgentState) -> AgentState:
        print(f"Selected agent: PARALLEL_KG_RAG_AGENT")
        emit_stream_event("agent", {
            "agent": "PARALLEL_KG_RAG_AGENT",
            "message": "Searching KG, RAG, and MedlinePlus in parallel...",
            "transition": False,
        })
        
        messages = state["messages"]
        query = _input_to_text(state["current_input"])
        kg_context_limit = config.rag.context_limit 
        rag_context_limit = config.rag.context_limit
        patient_id = state.get('patient_id', 'PAT_001')
        patient_memory_messages = _memory_context_message(state.get("patient_memory_context"))

        print(f"Patient ID: {patient_id}")

        def build_chat_history(context_limit: int) -> List[Dict[str, str]]:
            chat_history = list(patient_memory_messages)
            for msg in messages[-context_limit:]:
                if isinstance(msg, HumanMessage): 
                    chat_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    chat_history.append({"role": "assistant", "content": msg.content})
            return chat_history

        def run_kg_retrieval(): 
            try:
                patient_profile = patient_query_engine.get_patient_profile(patient_id)
                expanded_result = kg_agent.response_generator.query_expander.expand_query(
                    query,
                    patient_info=patient_profile,
                    mode="kg",
                    chat_history=build_chat_history(kg_context_limit),
                )
                expanded_query = expanded_result["expanded_query"]
                refined_question = expanded_query["refined_question"]
                patient_context = expanded_query["patient_context"]
                kg_context = kg_agent.response_generator.cypher_query_llm.retrieve_context_from_kg(refined_question)
                filtered_context = []
                if kg_context:
                    filtered_context = kg_agent.response_generator.context_filter.filter_context(
                        kg_context,
                        patient_context,
                        refined_question,
                    )

                result = {
                    "agent_name": "KG_AGENT",
                    "query": refined_question,
                    "patient_context": patient_context,
                    "documents": filtered_context,
                    "confidence": 0.7 if filtered_context else 0.0,
                    "sources": [],
                }
                with open('kg_response.json', 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4, default=str)
                return result
            except Exception as e:
                logger.exception("Error in KG agent")
                return {
                    "agent_name": "KG_AGENT",
                    "query": query,
                    "patient_context": "",
                    "documents": [],
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(e),
                }
        
        def run_rag_retrieval(): 
            try:
                expansion_result = rag_agent.query_expander.expand_query(
                    query,
                    mode="rag",
                    chat_history=build_chat_history(rag_context_limit),
                )
                expanded_query = expansion_result["expanded_query"]
                documents = rag_agent.vector_store.retrieve_relevant_chunks(
                    query=expanded_query,
                    vectorstore=rag_agent.vectorstore,
                )
                confidence = max([float(doc.get("score") or 0.0) for doc in documents], default=0.0)
                result = {
                    "agent_name": "RAG_AGENT",
                    "query": expanded_query,
                    "documents": documents,
                    "confidence": confidence,
                    "sources": rag_agent.response_generator._extract_sources(documents),
                }
                with open('rag_response.json', 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4, default=str)
                return result
            except Exception as e:
                logger.exception("Error in RAG agent")
                return {
                    "agent_name": "RAG_AGENT",
                    "query": query,
                    "documents": [],
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(e),
                }

        def run_medlineplus_retrieval():
            try:
                expansion_result = rag_agent.query_expander.expand_query(
                    query,
                    mode="medlineplus",
                    chat_history=build_chat_history(config.medlineplus.context_limit),
                )
                medlineplus_query = expansion_result["expanded_query"]
                retrieval_result = medlineplus_agent.retriever.retrieve(query=medlineplus_query)
                documents = retrieval_result.get("documents", [])
                confidence = medlineplus_agent._estimate_confidence(documents)
                result = {
                    "agent_name": "MEDLINEPLUS_AGENT",
                    "original_query": query,
                    "query": medlineplus_query,
                    "documents": documents,
                    "linked_entities": retrieval_result.get("linked_entities", []),
                    "expanded_relations": retrieval_result.get("expanded_relations", []),
                    "confidence": confidence,
                    "sources": medlineplus_agent._extract_sources(documents),
                }
                with open("medlineplus_response.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4, default=str)
                return result
            except Exception as e:
                logger.exception("Error in MedlinePlus agent")
                return {
                    "agent_name": "MEDLINEPLUS_AGENT",
                    "query": query,
                    "documents": [],
                    "linked_entities": [],
                    "expanded_relations": [],
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(e),
                }

        def compact_text(text: str, max_chars: int = 1400) -> str:
            text = (text or "").strip()
            if len(text) <= max_chars:
                return text
            return text[: max_chars - 3].rstrip() + "..."

        def format_kg_context(result: Dict[str, Any]) -> str:
            blocks = []
            for index, item in enumerate(result.get("documents", [])[:5], start=1):
                blocks.append(f"[KG-{index}] {compact_text(json.dumps(item, ensure_ascii=False, default=str), 1800)}")
            return "\n\n".join(blocks) if blocks else "Không có context KG phù hợp."

        def format_documents(result: Dict[str, Any], prefix: str, limit: int = 6) -> str:
            blocks = []
            for index, doc in enumerate(result.get("documents", [])[:limit], start=1):
                blocks.append(
                    "\n".join([
                        f"[{prefix}-{index}]",
                        f"Title: {doc.get('title') or doc.get('source') or doc.get('disease_name') or ''}",
                        f"Score: {doc.get('score', '')}",
                        f"URL: {doc.get('source_path', '')}",
                        f"Content: {compact_text(doc.get('content') or json.dumps(doc, ensure_ascii=False, default=str), 1800)}",
                    ])
                )
            return "\n\n".join(blocks) if blocks else f"Không có context {prefix} phù hợp."

        def format_sources(*results: Dict[str, Any]) -> List[Dict[str, str]]:
            seen = set()
            sources = []
            for result in results:
                for source in result.get("sources", []):
                    title = source.get("title") or source.get("source") or result.get("agent_name", "source")
                    path = source.get("path") or source.get("source_path") or ""
                    key = (title, path)
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append({"title": title, "path": path})
            return sources
            
        with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as executor: 
            kg_future = executor.submit(run_kg_retrieval)
            rag_future = executor.submit(run_rag_retrieval)
            medlineplus_future = executor.submit(run_medlineplus_retrieval)

            kg_result = kg_future.result()
            rag_result = rag_future.result()
            medlineplus_result = medlineplus_future.result()
            
        has_kg_context = bool(kg_result.get("documents"))
        has_rag_context = bool(rag_result.get("documents")) and rag_result.get("confidence", 0.0) >= config.rag.min_retrieval_confidence
        has_medlineplus_context = bool(medlineplus_result.get("documents")) and medlineplus_result.get("confidence", 0.0) >= config.medlineplus.min_retrieval_confidence

        print(f"KG context count: {len(kg_result.get('documents', []))}")
        print(f"RAG context count: {len(rag_result.get('documents', []))}")
        print(f"MedlinePlus context count: {len(medlineplus_result.get('documents', []))}")
        print(f"RAG confidence: {rag_result.get('confidence', 0.0)}")
        print(f"MedlinePlus confidence: {medlineplus_result.get('confidence', 0.0)}")
        
        if not (has_kg_context or has_rag_context or has_medlineplus_context):
            print("KG, RAG, and MedlinePlus all have insufficient info -> Routing to Web Search")
            emit_stream_event("agent", {
                "agent": "WEB_SEARCH_PROCESSOR_AGENT",
                "message": "KG/RAG did not have enough information. Switching to web search...",
                "transition": True,
            })
            return {
                **state,
                "output": AIMessage(content=""),
                "needs_human_validation": False,
                "agent_name": "PARALLEL_KG_RAG_AGENT",
                "next": "WEB_SEARCH_PROCESSOR_AGENT",
                "kg_result": kg_result,
                "rag_result": rag_result,
                "medlineplus_result": medlineplus_result
            }

        sources = format_sources(rag_result, medlineplus_result)
        source_text = "\n".join(
            f"- [{source['title']}]({source['path']})" if source.get("path") else f"- {source['title']}"
            for source in sources
        )
        history_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in build_chat_history(config.rag.context_limit)
        )

        emit_stream_event("agent", {
            "agent": "KG_RAG_PARALLEL",
            "message": "Generating final answer from KG, RAG, and MedlinePlus context...",
            "transition": True,
        })

        synthesis_prompt = medical_multi_source_cot_prompt.format(
            patient_context=kg_result.get("patient_context") or state.get("patient_memory_context") or "Không có.",
            user_query=query,
            history=history_text or "Không có.",
            kg_context=format_kg_context(kg_result),
            rag_context=format_documents(rag_result, "RAG"),
            medlineplus_query=medlineplus_result.get("query") or query,
            medlineplus_context=format_documents(medlineplus_result, "MED"),
            sources=source_text or "Không có nguồn URL.",
        )
        ## TO DO: change llm model
        last_model_response = get_llm()
        response = invoke_with_streaming(last_model_response, synthesis_prompt)
        response_text = getattr(response, "content", str(response))
        try:
            json_text = response_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json", 1)[1].split("```", 1)[0].strip()
            parsed_response = json.loads(json_text)
            response_text = parsed_response.get("step3_action", {}).get("content", response_text)
        except Exception as exc:
            logger.warning("Could not parse multi-source COT response as JSON: %s", exc)

        return {
            **state,
            "output": AIMessage(content=response_text),
            "needs_human_validation": False,
            "agent_name": "KG_RAG_PARALLEL",
            "next": "check_validation",
            "retrieval_confidence": max(
                float(kg_result.get("confidence", 0.0)),
                float(rag_result.get("confidence", 0.0)),
                float(medlineplus_result.get("confidence", 0.0)),
            ),
            "kg_result": kg_result,
            "rag_result": rag_result,
            "medlineplus_result": medlineplus_result
        }
        
    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        emit_stream_event("agent", {
            "agent": "WEB_SEARCH_PROCESSOR_AGENT",
            "message": "Searching the web for latest medical information...",
            "transition": True,
        })
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        if state.get("patient_memory_context"):
            recent_context += (
                "Long-term patient memory (supporting context only, not a confirmed diagnosis):\n"
                f"{state.get('patient_memory_context')}\n"
            )
        for msg in messages[-web_search_context_limit:]: # limit controlled from utils.config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Ensure response is an AIMessage
        if isinstance(processed_response, str):
            output_message = AIMessage(content=processed_response)
        else:
            output_message = processed_response

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            "output": output_message,
            "agent_name": involved_agents
        }

    # Add the new general medical image agent function
    def run_general_medical_image_agent(state: AgentState) -> AgentState:
        print(f"Selected agent: GENERAL_MEDICAL_IMAGE_AGENT")

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        
        # Get user query if available
        user_query = ""
        if isinstance(current_input, dict) and "text" in current_input:
            user_query = current_input.get("text", "")
        if state.get("patient_memory_context"):
            user_query = (
                f"{user_query}\n\nLong-term patient memory (supporting context only, not a confirmed diagnosis):\n"
                f"{state.get('patient_memory_context')}"
            ).strip()
        
        # Process the image with the general medical image agent
        diagnosis_result = image_analyzer.diagnose_general_medical_image(image_path, user_query)
        
        if diagnosis_result["success"]:
            response = AIMessage(content=diagnosis_result["diagnosis"])                
        else:
            response = AIMessage(content="Tôi đã gặp lỗi khi phân tích hình ảnh y tế này. Vui lòng thử lại hoặc tham khảo ý kiến bác sĩ chuyên khoa.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "GENERAL_MEDICAL_IMAGE_AGENT"
        }
        
    def run_polyp_segmentation_agent(state: AgentState) -> AgentState:

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        messages = state["messages"]
        
        # Get user query if available
        user_query = ""
        if isinstance(current_input, dict) and "text" in current_input:
            user_query = current_input.get("text", "")
        if state.get("patient_memory_context"):
            user_query = (
                f"{user_query}\n\nLong-term patient memory (supporting context only, not a confirmed diagnosis):\n"
                f"{state.get('patient_memory_context')}"
            ).strip()

        print(f"Selected agent: POLYP_SEGMENTATION_AGENT")

        # Segment the polyp
        try:
            segmentation_path = image_analyzer.segment_polyp(image_path, config.medical_cv.polyp_seg_output_path)
            segmentation_success = True
        except Exception as e:
            print(f"Error in polyp segmentation: {e}")
            segmentation_success = False
            segmentation_path = None

        if segmentation_success:
            # Create a diagnosis result for the polyp segmentation
            diagnosis_result = {
                "diagnosis": "Hình ảnh nội soi đại tràng đã được phân vùng để phân tích polyp. Kết quả phân vùng:\n\n" +
                           "🔴 **Vùng màu đỏ**: Polyp tân sinh (neoplastic) - có khả năng tiến triển thành ung thư, cần theo dõi chặt chẽ và có thể cần can thiệp\n" +
                           "🟢 **Vùng màu xanh**: Polyp không tân sinh (non-neoplastic) - thường lành tính, ít nguy cơ ung thư hóa\n\n" +
                           "Ảnh kết quả đã được overlay mask lên hình gốc để dễ quan sát. Việc phân loại này giúp bác sĩ đưa ra quyết định điều trị phù hợp.",
                "success": True,
                "image_path": image_path,
                "analysis_type": "polyp_segmentation"
            }
            
            # Summarize the diagnosis with the summarizer agent
            summarized_result = image_analyzer.summarize_diagnosis(
                diagnosis_result=diagnosis_result,
                chat_history=messages[-10:] if len(messages) > 0 else None,
                user_query=user_query
            )
            
            # Use the summarized content as the response
            if summarized_result["success"]:
                response = AIMessage(content=f"Dưới đây là kết quả phân vùng polyp từ hình ảnh nội soi đại tràng:\n\n{summarized_result['summary']}", metadata={"image_id":summarized_result['image_id']})
            else:
                response = AIMessage(content=f"Dưới đây là kết quả phân vùng polyp từ hình ảnh nội soi đại tràng:\n\n{diagnosis_result['diagnosis']}")
        else:
            response = AIMessage(content="Không thể thực hiện phân vùng polyp trên hình ảnh này. Hình ảnh có thể không đủ rõ nét hoặc không phải là hình ảnh nội soi đại tràng phù hợp.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "POLYP_SEGMENTATION_AGENT",
            "polyp_segmentation_path": segmentation_path
        }

    def run_polyp_vqa_agent(state: AgentState) -> AgentState:
        print(f"Selected agent: POLYP_VQA_AGENT")
        emit_stream_event("agent", {
            "agent": "POLYP_VQA_AGENT",
            "message": "Answering your polyp image question...",
            "transition": False,
        })

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        user_query = current_input.get("text", "") if isinstance(current_input, dict) else ""
        if state.get("patient_memory_context"):
            user_query = (
                f"{user_query}\n\nLong-term patient memory (supporting context only, not a confirmed diagnosis):\n"
                f"{state.get('patient_memory_context')}"
            ).strip()

        try:
            segmentation_path = image_analyzer.segment_polyp(image_path)
            vqa_result = image_analyzer.answer_polyp_vqa(
                image_path=image_path,
                segmentation_image_path=segmentation_path,
                user_query=user_query,
            )
        except Exception as e:
            print(f"Error in polyp VQA agent: {e}")
            segmentation_path = None
            vqa_result = {
                "success": False,
                "answer": "Tôi đã gặp lỗi khi phân tích VQA polyp. Vui lòng thử lại hoặc tham khảo bác sĩ chuyên khoa.",
                "error": str(e),
            }

        if vqa_result["success"]:
            response = AIMessage(content=vqa_result["answer"])
        else:
            response = AIMessage(content=vqa_result["answer"])

        return {
            **state,
            "output": response,
            "needs_human_validation": True,
            "agent_name": "POLYP_VQA_AGENT",
            "polyp_segmentation_path": segmentation_path
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"""{state['output'].content}"""

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        print("Choosing Apply Output Guardrails")
        
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
      
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Create appropriate thank you message based on response
                if validation_input.lower().startswith('yes'):
                    thank_you_message = AIMessage(content="Cảm ơn bạn đã xác nhận! Tôi rất vui khi thông tin đã hữu ích cho bạn. Nếu bạn có thêm câu hỏi nào khác về sức khỏe, tôi luôn sẵn sàng hỗ trợ.")
                else:  # starts with 'no'
                    thank_you_message = AIMessage(content="Cảm ơn bạn đã đưa ra nhận xét! Ý kiến của bạn rất quan trọng giúp chúng tôi cải thiện chất lượng dịch vụ. Tôi khuyến khích bạn tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác nhất.")
                
                return {
                    **state,
                    "output": thank_you_message,
                    "messages": thank_you_message
                }
                
        # Apply output sanitization
        sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    def write_patient_memory(state: AgentState) -> AgentState:
        """Persist selected long-term patient facts after response generation."""
        if not state.get("memory_enabled", True):
            return state

        patient_id = state.get("patient_id") or "PAT_001"
        session_id = state.get("session_id")
        current_input = state.get("current_input")
        input_text = _input_to_text(current_input)
        output = state.get("output")
        output_text = output.content if hasattr(output, "content") else str(output or "")

        if not input_text.strip() and not output_text.strip():
            return state
        if state.get("agent_name") == "NON_MEDICAL_FILTER":
            return state

        try:
            from agents.patient_memory_service.memory_service import get_patient_memory_service
            from agents.patient_memory_service.schemas import (
                MemoryMessage,
                PatientConditionCreate,
                PatientConversationMemoryCreate,
            )

            service = get_patient_memory_service()
            is_image_request = isinstance(current_input, dict) and bool(current_input.get("image"))
            is_validation = input_text.lower().startswith("validation result:")

            if is_image_request and output_text.strip():
                service.add_condition(
                    PatientConditionCreate(
                        patient_id=patient_id,
                        condition_text=(
                            "AI ghi nhận kết quả phân tích ảnh y tế sau đây, cần bác sĩ xác nhận: "
                            f"{_shorten_text(output_text, 900)}"
                        ),
                        condition_type="general",
                        run_id=session_id,
                        metadata={
                            "source": "ai_image_analysis",
                            "validated": False,
                            "stored_by": "langgraph_write_patient_memory",
                        },
                    )
                )
                return state

            metadata = {
                "source": "human_validation" if is_validation else "chat",
                "stored_by": "langgraph_write_patient_memory",
            }
            messages_to_store = []
            if input_text.strip():
                messages_to_store.append(MemoryMessage(role="user", content=_shorten_text(input_text, 1500)))
            if output_text.strip() and not is_validation:
                messages_to_store.append(MemoryMessage(role="assistant", content=_shorten_text(output_text, 1500)))

            if messages_to_store:
                service.add_conversation(
                    PatientConversationMemoryCreate(
                        patient_id=patient_id,
                        run_id=session_id,
                        infer=True,
                        messages=messages_to_store,
                        metadata=metadata,
                    )
                )
        except Exception as exc:
            logger.warning("Patient memory write skipped: %s", exc)

        return state

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("retrieve_patient_memory", retrieve_patient_memory)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node('PARALLEL_KG_RAG_AGENT', run_kg_rag_parallel)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("POLYP_SEGMENTATION_AGENT", run_polyp_segmentation_agent)
    workflow.add_node("POLYP_VQA_AGENT", run_polyp_vqa_agent)
    workflow.add_node("GENERAL_MEDICAL_IMAGE_AGENT", run_general_medical_image_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    workflow.add_node("write_patient_memory", write_patient_memory)
    
    workflow.set_entry_point("analyze_input")
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "retrieve_patient_memory"
        }
    )
    workflow.add_edge("retrieve_patient_memory", "route_to_agent")
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "PARALLEL_KG_RAG_AGENT": "PARALLEL_KG_RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "POLYP_SEGMENTATION_AGENT": "POLYP_SEGMENTATION_AGENT",
            "POLYP_VQA_AGENT": "POLYP_VQA_AGENT",
            "GENERAL_MEDICAL_IMAGE_AGENT": "GENERAL_MEDICAL_IMAGE_AGENT",
            "apply_guardrails": "apply_guardrails"  
            # "needs_validation": "RAG_AGENT"
        }
    )

    workflow.add_conditional_edges(
        "PARALLEL_KG_RAG_AGENT",
        lambda x: x['next'],
        {
            "check_validation": "check_validation",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT"
        }
    )
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_edge("POLYP_SEGMENTATION_AGENT", "check_validation")
    workflow.add_edge("POLYP_VQA_AGENT", "check_validation")
    workflow.add_edge("GENERAL_MEDICAL_IMAGE_AGENT", "check_validation")
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", "write_patient_memory")
    workflow.add_edge("write_patient_memory", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  
        }
    )
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "patient_id": None,
        "session_id": None,
        "patient_memory_context": "",
        "patient_memory_items": [],
        "memory_enabled": True,
        "routing_agent": None,
        "polyp_segmentation_path": None
    }


def process_query(
    query: Union[str, Dict],
    conversation_history: List[BaseMessage] = None,
    graph: StateGraph = None,
    patient_id: str = "PAT_001",
    session_id: Optional[str] = None,
    memory_enabled: bool = True,
) -> Dict:
    state = init_agent_state()
    
    if conversation_history:
        state["messages"] = conversation_history
    
    state["current_input"] = query
    state["patient_id"] = patient_id or "PAT_001"
    state["session_id"] = session_id
    state["memory_enabled"] = memory_enabled

    if isinstance(query, dict):
        query = _input_to_text(query, include_image_hint=True)
    
    if not conversation_history:
        state["messages"] = [HumanMessage(content=query)]

    thread_id = f"{state['patient_id']}:{session_id}" if session_id else state["patient_id"]
    dynamic_thread_config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(state, dynamic_thread_config)

    if len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    return result
