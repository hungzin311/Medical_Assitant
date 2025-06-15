"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Specify a thread
thread_config = {"configurable": {"thread_id": "1"}}


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gemini-2.0-flash"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gemini-2.0-flash"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """Bạn là một hệ thống phân loại y tế thông minh có nhiệm vụ chuyển các câu hỏi của người dùng đến 
    tác nhân chuyên biệt phù hợp. Công việc của bạn là phân tích yêu cầu của người dùng và xác định tác nhân 
    nào phù hợp nhất để xử lý dựa trên nội dung truy vấn, sự hiện diện của hình ảnh và ngữ cảnh cuộc trò chuyện.

    Các tác nhân có sẵn:
    1. CONVERSATION_AGENT - Cho trò chuyện chung, lời chào và câu hỏi không liên quan đến y tế. 
    2. RAG_AGENT - Cho các câu hỏi kiến thức y tế cụ thể có thể được trả lời từ tài liệu y khoa đã được thiết lập. Kiến thức y tế hiện đã được thu thập bao gồm: giải phẫu và chức năng các bộ phận cơ thể, các loại bệnh và phương pháp điều trị, thông tin về thuốc, quy trình và hướng dẫn y học lâm sàng.
    3. WEB_SEARCH_PROCESSOR_AGENT - Cho câu hỏi về phát triển y tế gần đây, dịch bệnh hiện tại hoặc thông tin y tế nhạy cảm theo thời gian.
    4. SKIN_LESION_AGENT - Để phân tích hình ảnh mà người dùng cần phân vùng tổn thương da.
    5. GENERAL_MEDICAL_IMAGE_AGENT - Để phân tích và chẩn đoán các loại hình ảnh y tế khác không thuộc các danh mục trên.

    Đưa ra quyết định của bạn dựa trên những hướng dẫn sau:
    - Nếu người dùng không tải lên bất kỳ hình ảnh nào, luôn chuyển đến tác nhân hội thoại.
    - Nếu người dùng tải lên hình ảnh y tế, quyết định tác nhân thị giác y tế nào phù hợp dựa trên loại hình ảnh và truy vấn của người dùng. Nếu hình ảnh được tải lên mà không có truy vấn, luôn chuyển đến tác nhân thị giác y tế chính xác dựa trên loại hình ảnh.
    - Nếu hình ảnh được phân loại là "OTHER" hoặc "GENERAL MEDICAL IMAGE", sử dụng tác nhân GENERAL_MEDICAL_IMAGE_AGENT.
    - Nếu người dùng hỏi về phát triển y tế gần đây hoặc tình hình sức khỏe hiện tại, sử dụng tác nhân xử lý tìm kiếm web.
    - Nếu người dùng hỏi các câu hỏi kiến thức y tế cụ thể, sử dụng tác nhân RAG.
    - Đối với cuộc trò chuyện chung, lời chào hoặc câu hỏi **không liên quan** đến y tế, sử dụng tác nhân hội thoại. Nhưng nếu hình ảnh được tải lên, luôn ưu tiên các tác nhân thị giác y tế trước.

    Bạn phải cung cấp câu trả lời của mình ở định dạng JSON với cấu trúc sau:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Lý luận từng bước của bạn để chọn tác nhân này",
    "confidence": 0.95  // Giá trị từ 0.0 đến 1.0 cho biết mức độ tin cậy của bạn trong quyết định này
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


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
    insufficient_info: bool  # Flag indicating RAG response has insufficient information


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""
    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            input_text = current_input.get("text", "")
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path, input_text)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Set to False to ensure normal routing
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check if this is a follow-up question about a previous diagnosis
        # This is a simple check - in a real system, you might use more sophisticated methods
        follow_up_keywords = ["previous diagnosis", "earlier diagnosis", "the image", "that image", 
                             "the diagnosis", "that diagnosis", "what you found", "tell me more about"]
        
        is_follow_up = any(keyword in input_text.lower() for keyword in follow_up_keywords)
        
        # If it seems like a follow-up question and we have previous agent names in the state
        if is_follow_up and len(messages) > 2:
            # Look for the most recent image ID in the conversation history
            image_id = None
            for i in range(len(messages)-1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    content = messages[i].content
                    # Look for the image_id pattern in the content
                    if "[image_id:" in content.lower():
                        start_idx = content.lower().find("[image_id:") + 10
                        end_idx = content.lower().find("]", start_idx)
                        image_id = content[start_idx:end_idx].strip()
                        break
                    # Alternative format
                    elif "image_id:" in content.lower():
                        start_idx = content.lower().find("image_id:") + 9
                        end_idx = content.find("\n", start_idx) if "\n" in content[start_idx:] else len(content)
                        image_id = content[start_idx:end_idx].strip()
                        break
            
            # If we found an image ID, try to generate a follow-up response
            if image_id:
                try:
                    print(f"Found image_id: {image_id}, generating follow-up response")
                    follow_up_response = AgentConfig.image_analyzer.generate_followup_response(image_id, input_text)
                    return {
                        **state,
                        "output": AIMessage(content=follow_up_response),
                        "agent_name": "CONVERSATION_AGENT"
                    }
                except Exception as e:
                    print(f"Error generating follow-up response: {e}")
                    # Fall back to regular conversation if there's an error
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        conversation_prompt = f"""Câu hỏi người dùng: {input_text}

        Ngữ cảnh cuộc trò chuyện gần đây: {recent_context}

        Bạn là một Trợ lý Hội thoại Y tế được hỗ trợ bởi AI. Mục tiêu của bạn là tạo điều kiện cho các cuộc trò chuyện suôn sẻ và nhiều thông tin với người dùng, xử lý cả các truy vấn thông thường và liên quan đến y tế. Bạn phải trả lời một cách tự nhiên đồng thời đảm bảo độ chính xác và rõ ràng về mặt y tế.

        ### Vai trò & Khả năng
        - Tham gia vào **cuộc trò chuyện chung** trong khi duy trì tính chuyên nghiệp.
        - Trả lời **câu hỏi y tế** sử dụng kiến thức đã được xác minh.
        - Chuyển hướng **truy vấn phức tạp** đến RAG (tạo sinh tăng cường truy xuất) hoặc tìm kiếm web nếu cần.
        - Xử lý **câu hỏi theo dõi** trong khi theo dõi ngữ cảnh cuộc trò chuyện.
        - Chuyển hướng **hình ảnh y tế** đến tác nhân phân tích AI thích hợp.

        ### Hướng dẫn trả lời:
        1. **Cuộc trò chuyện chung:**
        - Nếu người dùng tham gia trò chuyện thông thường (ví dụ: lời chào, trò chuyện nhỏ), hãy trả lời một cách thân thiện, hấp dẫn.
        - Giữ câu trả lời **ngắn gọn và hấp dẫn**, trừ khi cần câu trả lời chi tiết.

        2. **Câu hỏi y tế:**
        - Nếu bạn có **độ tin cậy cao** trong việc trả lời, hãy cung cấp câu trả lời chính xác về mặt y tế.
        - Đảm bảo câu trả lời **rõ ràng, ngắn gọn và dựa trên sự thật**.

        3. **Theo dõi & Làm rõ:**
        - Duy trì lịch sử cuộc trò chuyện để có câu trả lời tốt hơn.
        - Nếu truy vấn không rõ ràng, hãy đặt **câu hỏi theo dõi** trước khi trả lời.

        4. **Xử lý phân tích hình ảnh y tế:**
        - **Không** cố gắng tự phân tích hình ảnh.
        - Nếu người dùng nói về việc phân tích hoặc xử lý hoặc phát hiện hoặc phân đoạn hoặc phân loại bất kỳ bệnh nào từ bất kỳ hình ảnh nào, hãy yêu cầu người dùng tải lên hình ảnh để trong lượt tiếp theo nó được chuyển đến các tác nhân thị giác y tế thích hợp.
        - Nếu hình ảnh đã được tải lên, nó sẽ được chuyển đến các tác nhân thị giác máy tính y tế. Đọc lịch sử để biết về kết quả chẩn đoán và tiếp tục cuộc trò chuyện nếu người dùng hỏi bất cứ điều gì liên quan đến chẩn đoán.
        - Sau khi xử lý, **giúp người dùng hiểu kết quả**.

        5. **Không chắc chắn & Cân nhắc đạo đức:**
        - Nếu không chắc chắn, **không bao giờ giả định** các sự kiện y tế.
        - Khuyến nghị tham khảo ý kiến **chuyên gia y tế có giấy phép** đối với các vấn đề y tế nghiêm trọng.
        - Tránh đưa ra **chẩn đoán y tế** hoặc **đơn thuốc**—chỉ nên đề cập đến kiến thức chung.

        ### Định dạng trả lời:
        - Duy trì **giọng điệu trò chuyện nhưng chuyên nghiệp**.
        - Sử dụng **điểm đánh dấu hoặc danh sách đánh số** để làm rõ khi cần thiết.
        - Nếu lấy thông tin từ nguồn bên ngoài (RAG/Tìm kiếm Web), hãy đề cập **thông tin đến từ đâu** (ví dụ: "Theo Mayo Clinic...").
        - Nếu người dùng yêu cầu chẩn đoán, hãy nhắc họ **tìm kiếm tư vấn y tế**.

        ### Ví dụ về truy vấn & câu trả lời của người dùng:

        **Người dùng:** "Này, ngày của bạn thế nào?"
        **Bạn:** "Tôi đang ở đây và sẵn sàng giúp đỡ! Tôi có thể hỗ trợ bạn như thế nào hôm nay?"

        **Người dùng:** "Tôi bị đau đầu và sốt. Tôi nên làm gì?"
        **Bạn:** "Tôi không phải là bác sĩ, nhưng đau đầu và sốt có thể có nhiều nguyên nhân, từ nhiễm trùng đến mất nước. Nếu các triệu chứng của bạn kéo dài, bạn nên gặp chuyên gia y tế."

        Câu trả lời của LLM hội thoại:"""

        response = config.conversation.llm.invoke(conversation_prompt)

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        try:
            response = rag_agent.process_query(query, chat_history=recent_context)
            
            # Ensure response has the required keys
            if "response" not in response:
                print("Error: RAG response is missing 'response' key")
                response["response"] = "I apologize, but I encountered an error while processing your query. Please try again."
                
            retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

            print(f"Retrieval Confidence: {retrieval_confidence}")
            print(f"Sources: {len(response.get('sources', []))}")

            # Check if response indicates insufficient information
            insufficient_info = False
            response_text = response["response"]
            
            # Ensure response_text is a string
            if not isinstance(response_text, str):
                print(f"Warning: Response text is not a string, converting from {type(response_text)}")
                response_text = str(response_text)
                
            if (
                "Tôi không có đủ thông tin" in response_text or 
                "không đủ thông tin" in response_text.lower() or
                "thông tin không đầy đủ" in response_text.lower() or
                "không thể trả lời" in response_text.lower() or
                "không trả lời được" in response_text.lower()   
                ):
                
                print("RAG response indicates insufficient information")
                insufficient_info = True

            print(f"Insufficient info flag set to: {insufficient_info}")

            # Store RAG output ONLY if confidence is high
            if retrieval_confidence >= config.rag.min_retrieval_confidence:
                response_output = AIMessage(content=response_text)
            else:
                response_output = AIMessage(content="")
            
            return {
                **state,
                "output": response_output,
                "needs_human_validation": False,  # Assuming no validation needed for RAG responses
                "retrieval_confidence": retrieval_confidence,
                "agent_name": "RAG_AGENT",
                "insufficient_info": insufficient_info
            }
            
        except Exception as e:
            import traceback
            print(f"Error in RAG agent: {e}")
            print(traceback.format_exc())
            
            return {
                **state,
                "output": AIMessage(content="I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."),
                "needs_human_validation": True,
                "retrieval_confidence": 0.0,
                "agent_name": "RAG_AGENT",
                "insufficient_info": True
            }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Add the new general medical image agent function
    def run_general_medical_image_agent(state: AgentState) -> AgentState:
        """Handle general medical image analysis."""

        print(f"Selected agent: GENERAL_MEDICAL_IMAGE_AGENT")

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        messages = state["messages"]
        
        # Get user query if available
        user_query = ""
        if isinstance(current_input, dict) and "text" in current_input:
            user_query = current_input.get("text", "")
        
        # Process the image with the general medical image agent
        diagnosis_result = AgentConfig.image_analyzer.diagnose_general_medical_image(image_path, user_query)
        
        # Summarize the diagnosis with the summarizer agent
        if diagnosis_result["success"]:
            # Pass the diagnosis result, chat history, and user query to the summarizer
            summarized_result = AgentConfig.image_analyzer.summarize_diagnosis(
                diagnosis_result=diagnosis_result,
                chat_history=messages[-10:] if len(messages) > 0 else None,  # Pass last 10 messages as context
                user_query=user_query
            )
            
            # Use the summarized content as the response
            if summarized_result["success"]:
                response = AIMessage(content=summarized_result["summary"])
            else:
                response = AIMessage(content=diagnosis_result["diagnosis"])
        else:
            response = AIMessage(content="I encountered an error while analyzing this medical image. Please try again or consult a healthcare professional.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "GENERAL_MEDICAL_IMAGE_AGENT"
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        messages = state["messages"]
        
        # Get user query if available
        user_query = ""
        if isinstance(current_input, dict) and "text" in current_input:
            user_query = current_input.get("text", "")

        print(f"Selected agent: SKIN_LESION_AGENT")

        # Segment the skin lesion
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        if predicted_mask:
            # Create a basic diagnosis result for the skin lesion
            diagnosis_result = {
                "diagnosis": "The image shows a skin lesion that has been segmented for analysis. The segmentation highlights the boundaries of the lesion, which is an important step in determining whether it may be benign or malignant. Further analysis by a dermatologist is recommended for a definitive diagnosis.",
                "success": True,
                "image_path": image_path
            }
            
            # Summarize the diagnosis with the summarizer agent
            summarized_result = AgentConfig.image_analyzer.summarize_diagnosis(
                diagnosis_result=diagnosis_result,
                chat_history=messages[-10:] if len(messages) > 0 else None,
                user_query=user_query
            )
            
            # Use the summarized content as the response
            if summarized_result["success"]:
                response = AIMessage(content=f"Dưới đây là ảnh phân vùng vết thương trên da dựa trên ảnh đã được cung cấp:\n\n{summarized_result['summary']}")
            else:
                response = AIMessage(content="Dưới đây là ảnh phân vùng vết thương trên da dựa trên ảnh đã được cung cấp:")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
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
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- Nếu bạn là chuyên gia, hãy đánh giá lại kết quả. Chọn **Yes** or **No**. Nếu No, cung cấp nhận xét.\n- Nếu bạn là bệnh nhân: Chỉ cần chọn Yes để xác nhận."

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
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("GENERAL_MEDICAL_IMAGE_AGENT", run_general_medical_image_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "GENERAL_MEDICAL_IMAGE_AGENT": "GENERAL_MEDICAL_IMAGE_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    # workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")    
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    workflow.add_edge("GENERAL_MEDICAL_IMAGE_AGENT", "check_validation")
    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
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
        "insufficient_info": False
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> Dict:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()
    
    # Initialize state
    state = init_agent_state()
    
    # Add conversation history if provided
    if conversation_history:
        state["messages"] = conversation_history
    
    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    # Add query to messages if no conversation history was provided
    if not conversation_history:
        state["messages"] = [HumanMessage(content=query)]

    # Run the graph
    result = graph.invoke(state, thread_config)

    # Keep history to reasonable size
    if len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    
    return result