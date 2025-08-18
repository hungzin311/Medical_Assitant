from typing import Dict, List, Optional, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import END
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from langgraph.checkpoint.memory import MemorySaver
from proxy_setting import *
from agents.prompt import *
from config import Config

#Set proxy  
set_proxy()

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
    DECISION_MODEL = "gemini-2.5-flash"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gemini-2.5-flash"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
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
    ("human", f"System: {decision_agent_prompt}\n\nUser: {{input}}")])

    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
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
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)
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
        
        # Check for NON-MEDICAL images and reject them
        if image_type == "NON-MEDICAL":
            updated_state = {
                **state,
                "output": AIMessage(content="Tôi rất xin lỗi, nhưng hình ảnh bạn tải lên không phải là hình ảnh y tế. Tôi chỉ có thể phân tích các hình ảnh y tế như X-quang, CT, MRI, ảnh da, nội soi, v.v. Vui lòng tải lên hình ảnh y tế để tôi có thể hỗ trợ bạn."),
                "agent_name": "NON_MEDICAL_FILTER"
            }
            return {"agent_state": updated_state, "next": "apply_guardrails"}


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
                        image_id = getattr(messages[i], "meatadata", {}).get("image_id", None)
                        print(f"Found image_id: {image_id}")
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
        {conversation_agent_prompt}
        """

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
            
            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            return {
                **state,
                "output": AIMessage(content="Tôi xin lỗi, nhưng tôi đã gặp lỗi khi xử lý truy vấn của bạn. Vui lòng thử lại hoặc diễn đạt lại câu hỏi của bạn." + safety_disclaimer),
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
                response = AIMessage(content=summarized_result["summary"], metadata={"image_id":summarized_result['image_id']})
            else:
                response = AIMessage(content=diagnosis_result["diagnosis"])
        else:
            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            response = AIMessage(content="Tôi đã gặp lỗi khi phân tích hình ảnh y tế này. Vui lòng thử lại hoặc tham khảo ý kiến bác sĩ chuyên khoa." + safety_disclaimer)

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
                "diagnosis": "Hình ảnh cho thấy một tổn thương da đã được phân vùng để phân tích. Việc phân vùng làm nổi bật ranh giới của tổn thương, đây là một bước quan trọng trong việc xác định liệu nó có thể là lành tính hay ác tính. Cần tham khảo ý kiến bác sĩ da liễu để có chẩn đoán chính xác.",
                "success": True,
                "image_path": image_path,
                "analysis_type": "skin_lesion_segmentation"
            }
            
            # Summarize the diagnosis with the summarizer agent
            summarized_result = AgentConfig.image_analyzer.summarize_diagnosis(
                diagnosis_result=diagnosis_result,
                chat_history=messages[-10:] if len(messages) > 0 else None,
                user_query=user_query
            )
            
            # Use the summarized content as the response
            if summarized_result["success"]:
                response = AIMessage(content=f"Dưới đây là kết quả phân vùng tổn thương da dựa trên ảnh đã được cung cấp:\n\n{summarized_result['summary']}", metadata={"image_id":summarized_result['image_id']})
            else:
                response = AIMessage(content="Dưới đây là kết quả phân vùng tổn thương da dựa trên ảnh đã được cung cấp:")
        else:
            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            response = AIMessage(content="Hình ảnh được tải lên không đủ rõ nét để có thể chẩn đoán hoặc hình ảnh này không phải là hình ảnh y tế." + safety_disclaimer)

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }

    def run_polyp_segmentation_agent(state: AgentState) -> AgentState:
        """Handle polyp segmentation image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)
        messages = state["messages"]
        
        # Get user query if available
        user_query = ""
        if isinstance(current_input, dict) and "text" in current_input:
            user_query = current_input.get("text", "")

        print(f"Selected agent: POLYP_SEGMENTATION_AGENT")

        # Segment the polyp
        try:
            AgentConfig.image_analyzer.segment_polyp(image_path)
            segmentation_success = True
        except Exception as e:
            print(f"Error in polyp segmentation: {e}")
            segmentation_success = False

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
            summarized_result = AgentConfig.image_analyzer.summarize_diagnosis(
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
            safety_disclaimer = "\n\n⚠️ **Lưu ý quan trọng:** Thông tin trên chỉ mang tính chất tham khảo và được tạo ra bởi AI. Đây không phải là chẩn đoán y tế chính thức. Bạn nên đi khám bác sĩ chuyên khoa sớm nhất có thể để được thăm khám và điều trị phù hợp."
            response = AIMessage(content="Không thể thực hiện phân vùng polyp trên hình ảnh này. Hình ảnh có thể không đủ rõ nét hoặc không phải là hình ảnh nội soi đại tràng phù hợp." + safety_disclaimer)

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "POLYP_SEGMENTATION_AGENT"
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

    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("POLYP_SEGMENTATION_AGENT", run_polyp_segmentation_agent)
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
            "POLYP_SEGMENTATION_AGENT": "POLYP_SEGMENTATION_AGENT",
            "GENERAL_MEDICAL_IMAGE_AGENT": "GENERAL_MEDICAL_IMAGE_AGENT",
            "apply_guardrails": "apply_guardrails",  # For NON-MEDICAL images
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    workflow.add_edge("POLYP_SEGMENTATION_AGENT", "check_validation")
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


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, graph: StateGraph = None) -> Dict:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages
        
    Returns:
        Response from the appropriate agent
    """
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