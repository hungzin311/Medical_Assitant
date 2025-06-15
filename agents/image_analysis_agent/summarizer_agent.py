import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage

class MedicalImageSummarizer:
    """
    Summarizer agent that processes image analysis results, maintains context memory,
    and provides follow-up recommendations for medical images.
    """
    
    def __init__(self, llm):
        """
        Initialize the medical image summarizer.
        
        Args:
            llm: The LLM model for summarization and follow-up recommendations
        """
        self.logger = logging.getLogger(__name__)
        self.llm = llm
        self.memory = {}  # Dictionary to store diagnosis info by image_id
        self.logger.info("Medical Image Summarizer initialized")
    
    def store_diagnosis(self, image_id: str, diagnosis_data: Dict[str, Any]) -> None:
        """
        Store diagnosis information in memory for future reference.
        
        Args:
            image_id: Unique identifier for the image
            diagnosis_data: Diagnosis information to store
        """
        self.memory[image_id] = diagnosis_data
        self.logger.info(f"Stored diagnosis for image {image_id}")
    
    def get_stored_diagnosis(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored diagnosis information.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Stored diagnosis data or None if not found
        """
        return self.memory.get(image_id)
    
    def summarize_diagnosis(self, diagnosis_result: Dict[str, Any], 
                           chat_history: List = None, 
                           user_query: str = None) -> Dict[str, Any]:
        """
        Summarize the diagnosis results and provide follow-up recommendations.
        
        Args:
            diagnosis_result: The raw diagnosis result from the image analysis
            chat_history: Optional chat history for context
            user_query: Optional user query for context
            
        Returns:
            Dictionary containing summarized diagnosis and follow-up recommendations
        """
        self.logger.info("Summarizing diagnosis results")
        
        # Extract diagnosis from the result
        diagnosis = diagnosis_result.get("diagnosis", "")
        image_path = diagnosis_result.get("image_path", "")
        
        # Generate a unique image ID (using the image path for simplicity)
        image_id = str(hash(image_path))
        
        # Store the diagnosis for future reference
        self.store_diagnosis(image_id, diagnosis_result)
        
        # Format chat history if provided
        history_context = ""
        if chat_history:
            for message in chat_history[-5:]:  # Use the last 5 messages for context
                if isinstance(message, HumanMessage):
                    history_context += f"User: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_context += f"Assistant: {message.content}\n"
        
        # Prepare the prompt for the summarizer
        summarizer_prompt = f"""
        Bạn là một trợ lý y tế tóm tắt kết quả phân tích hình ảnh và đưa ra các khuyến nghị hữu ích tiếp theo.
        THÔNG TIN CHẨN ĐOÁN:
        {diagnosis}
        {f'CÂU HỎI CỦA NGƯỜI DÙNG: {user_query}' if user_query else ''}
        {f'BỐI CẢNH CUỘC TRÒ CHUYỆN GẦN ĐÂY: {history_context}' if history_context else ''}
        Vui lòng cung cấp:

        Tóm tắt rõ ràng, súc tích về chẩn đoán bằng ngôn ngữ đơn giản
        Các điểm chính mà bệnh nhân cần biết
        Các bước tiếp theo được khuyến nghị hoặc xét nghiệm bổ sung nếu có
        Các câu hỏi tiềm ẩn mà bệnh nhân có thể có và câu trả lời cho chúng
        Kết thúc bằng một câu hỏi về việc họ có cần thêm thông tin, khuyến nghị về thuốc, hoặc hỗ trợ khác không

        Hãy nhớ duy trì giọng điệu thông cảm, chuyên nghiệp và nhấn mạnh rằng đây là phân tích hỗ trợ bởi AI, không thay thế cho lời khuyên y tế chuyên nghiệp.
        """
        
        try:
            # Invoke LLM for summarization
            response = self.llm.invoke(summarizer_prompt)
            
            # Add image_id to the response content for tracking
            summary_with_id = f"{response.content}\n\n[Image_ID: {image_id}]"
            
            # Store the summarized result along with the original diagnosis
            summarized_result = {
                "image_id": image_id,
                "original_diagnosis": diagnosis,
                "summary": summary_with_id,
                "success": True
            }
            
            return summarized_result
        except Exception as e:
            self.logger.error(f"Error summarizing diagnosis: {e}")
            return {
                "image_id": image_id,
                "original_diagnosis": diagnosis,
                "summary": f"I encountered an error while summarizing this diagnosis. Please review the original diagnosis or consult a healthcare professional.\n\n[Image_ID: {image_id}]",
                "success": False,
                "error": str(e)
            }
    
    def generate_followup_response(self, image_id: str, follow_up_query: str) -> str:
        """
        Generate a response to a follow-up query based on stored diagnosis.
        
        Args:
            image_id: Unique identifier for the previously diagnosed image
            follow_up_query: User's follow-up question
            
        Returns:
            Response to the follow-up query
        """
        # Retrieve the stored diagnosis
        stored_diagnosis = self.get_stored_diagnosis(image_id)
        
        if not stored_diagnosis:
            return "I'm sorry, I don't have information about that previous diagnosis. Could you please provide more details or upload the image again?"
        
        # Prepare the prompt for follow-up
        followup_prompt = f"""
        You are a medical assistant responding to a follow-up question about a previous diagnosis.
        
        ORIGINAL DIAGNOSIS:
        {stored_diagnosis.get('diagnosis', '')}
        
        USER'S FOLLOW-UP QUESTION:
        {follow_up_query}
        
        Please provide:
        1. A clear, helpful response to the user's follow-up question
        2. Any additional relevant information that might be helpful
        3. If the question requires information beyond what's available in the diagnosis, suggest appropriate next steps
        4. End with a question asking if they need more information or have other questions
        
        Remember to maintain a compassionate, professional tone and emphasize that this is AI-assisted analysis, not a replacement for professional medical advice.
        """
        
        try:
            # Invoke LLM for follow-up response
            response = self.llm.invoke(followup_prompt)
            # Include the image_id in the response for continued tracking
            return f"{response.content}\n\n[Image_ID: {image_id}]"
        except Exception as e:
            self.logger.error(f"Error generating follow-up response: {e}")
            return f"I'm sorry, I encountered an error while processing your follow-up question. Please try rephrasing your question or consult a healthcare professional for accurate information.\n\n[Image_ID: {image_id}]" 