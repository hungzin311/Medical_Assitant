import sys
from pathlib import Path
from config import Config  
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typing import List, Dict, Any, Optional
import logging
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


class DiseaseEvaluation:
    def __init__(self, patient_vector_store):
        self.logger = logging.getLogger(__name__)
        self.client = patient_vector_store.client 
        self.collection_name = patient_vector_store.collection_name
        self.embedding_model = patient_vector_store.embedding_model
        config = Config()
        self.llm = config.patient_db.llm
        
    def retrieve_patient_records(
        self, 
        patient_id: str,
        query: Optional[str] =None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Patient record retrieval.
        Args:
            patient_id: patient ID for same-patient history
            query: Optional query for similarity search
            limit: Maximum number of results
            
        Returns:
            List of retrieved patient records with scores
        """
        try:
            #Retrieved records list
            patient_records = []
            if query: 
                query_vector = self.embedding_model.embed_query(query)
                patient_history = self.client.query_points(
                    collection_name = self.collection_name, 
                    query = query_vector, 
                    query_filter = Filter( 
                        must = [
                            FieldCondition(key='patient_id', match = MatchValue(value = patient_id))
                        ]
                    ), 
                    limit = limit, 
                    using = 'dense', 
                    with_payload = True
                )
            else:
                patient_history = self.client.query_points( 
                    collection_name=self.collection_name, 
                    query_filter = Filter(
                        must = [
                            FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
                        ]
                    ), 
                    limit = limit, 
                    with_payload= True,
                    using="dense"
                )
            #Format results
            for result in patient_history.points: 
                record = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                patient_records.append(record)
                    
            self.logger.info(f"Retrieved {len(patient_records)} patient records")
            return patient_records
            
        except Exception as e:
            self.logger.error(f"Error retrieving patient records: {e}")
        return []

    def evaluate_based_record_llm(
        self, 
        patient_record: List[Dict[str, Any]], 
        patient_form: List[Dict[str, Any]],
        evaluation_type: str = "disease_progression"
    ) -> Dict[str, Any]:
        
        # Simple version. Does not contain RAG. Like a demo, about 70% of the full version.
        self.logger.info(f"Starting LLM-based {evaluation_type} evaluation")
        try:
            # Prepare patient data for analysis
            current_patient = self._format_patient_for_analysis(patient_record)
            
            # Generate evaluation prompt based on type
            evaluation_prompt = self._create_evaluation_prompt(
                current_patient, patient_form, evaluation_type
            )
            # Get LLM evaluation
            response = self.llm.invoke(evaluation_prompt)
            
            # Parse and structure the response
            evaluation_result = self._parse_evaluation_response(response.content, evaluation_type)
            
            self.logger.info(f"Successfully completed {evaluation_type} evaluation")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error in LLM-based evaluation: {e}")
            return {
                "error": str(e),
                "evaluation_type": evaluation_type,
                "status": "failed"
            }
    
    def _format_patient_for_analysis(self, patient_records: List[Dict[str, Any]]) -> str:

        first_payload = patient_records[0].get('payload', {})
        basic_info = f"""
        THÔNG TIN CƠ BẢN BỆNH NHÂN:
        - ID: {patient_records[0].get('id', 'N/A')}
        - Tuổi: {first_payload.get('age', 'N/A')}
        - Giới tính: {first_payload.get('sex', 'N/A')}
        - BMI: {first_payload.get('bmi', 'N/A')} (Phân loại: {first_payload.get('bmi_category', 'N/A')})
        """

        # Lịch sử khám và điều trị
        history_info = []
        for i, record in enumerate(patient_records, start=1):
            payload = record.get('payload', {})
            visit_info = f"""
            LẦN KHÁM {i}:
            - Triệu chứng chính: {payload.get('chief_complaint', 'N/A')}
            - Chẩn đoán: {payload.get('diagnosis', 'N/A')}
            - Bệnh kèm theo: {', '.join(payload.get('comorbidities', []))}
            - Điểm nghiêm trọng: {payload.get('severity_score', 'N/A')}/10
            - Phân loại nghiêm trọng: {payload.get('severity_category', 'N/A')}
            - Các điều trị đã thử: {', '.join(payload.get('treatments_tried', []))}
            - Kết quả điều trị chính: {payload.get('primary_outcome', 'N/A')}
            - Chống chỉ định: {', '.join(payload.get('contraindications', []))}
            - Tóm tắt tình trạng: {payload.get('summary_text', 'N/A')}
            """
            history_info.append(visit_info.strip())

        return basic_info.strip() + "\n\n" + "\n\n".join(history_info)
    
    def _create_evaluation_prompt(self, patient_info: str, patient_form: List[Dict[str, Any]], evaluation_type: str) -> str:
        """Create evaluation prompt based on type."""
        
        base_prompt = f"""
        Bạn là một chuyên gia y tế có kinh nghiệm cao trong việc phân tích và đánh giá tình trạng bệnh nhân. 
        Nhiệm vụ của bạn là đánh giá bệnh nhân hiện tại dựa trên các thông tin mà bệnh nhân vừa điền và dữ liệu của các lần khám trước.
        -Thông tin bệnh nhân vừa cung cấp:
        {patient_info}
        -Thông tin về các lần khám trước của bệnh nhân:
        {patient_form}        
        """
        
        if evaluation_type == "disease_progression":
            specific_prompt = """
            NHIỆM VỤ: ĐÁNH GIÁ TIẾN TRIỂN BỆNH

            Hãy phân tích và đánh giá:
            1. **Dự báo tiến triển bệnh** (tốt lên/xấu đi/ổn định) trong 3-6 tháng tới
            2. **Các yếu tố nguy cơ** có thể ảnh hưởng đến tiến triển
            3. **Dấu hiệu cảnh báo** cần theo dõi
            4. **Khuyến nghị theo dõi** (tần suất khám, xét nghiệm cần làm)
            5. **So sánh với các trường hợp tương tự** và rút ra bài học

            Trả lời theo định dạng JSON:
            {
                "disease_progression_forecast": "tốt lên/xấu đi/ổn định",
                "progression_probability": 0.85,
                "risk_factors": ["yếu tố 1", "yếu tố 2"],
                "warning_signs": ["dấu hiệu 1", "dấu hiệu 2"],
                "monitoring_recommendations": {
                    "follow_up_frequency": "mỗi 3 tháng",
                    "tests_needed": ["xét nghiệm 1", "xét nghiệm 2"],
                    "symptoms_to_watch": ["triệu chứng 1", "triệu chứng 2"]
                },
                "comparison_insights": "So sánh với các trường hợp tương tự...",
                "confidence_score": 0.90,
                "reasoning": "Lý do chi tiết cho đánh giá..."
            }
            """
                    
        elif evaluation_type == "risk_assessment":
            specific_prompt = """
            NHIỆM VỤ: ĐÁNH GIÁ RỦI RO SỨC KHỎE

            Hãy phân tích và đánh giá:
            1. **Mức độ rủi ro tổng thể** (thấp/trung bình/cao/rất cao)
            2. **Rủi ro biến chứng cụ thể** và xác suất xảy ra
            3. **Yếu tố bảo vệ** có thể giảm rủi ro
            4. **Biện pháp phòng ngừa** cần thiết
            5. **Ưu tiên can thiệp** dựa trên mức độ cấp thiết

            Trả lời theo định dạng JSON:
            {
                "overall_risk_level": "thấp/trung bình/cao/rất cao",
                "risk_score": 0.75,
                "specific_risks": [
                    {
                        "complication": "biến chứng 1",
                        "probability": 0.30,
                        "severity": "nghiêm trọng",
                        "timeframe": "6 tháng"
                    }
                ],
                "protective_factors": ["yếu tố bảo vệ 1", "yếu tố bảo vệ 2"],
                "prevention_measures": ["biện pháp 1", "biện pháp 2"],
                "intervention_priorities": [
                    {
                        "action": "hành động 1",
                        "urgency": "cao",
                        "expected_benefit": "giảm 30% rủi ro"
                    }
                ],
                "confidence_score": 0.85,
                "reasoning": "Lý do chi tiết cho đánh giá rủi ro..."
            }
            """
        else:
            specific_prompt = """
            NHIỆM VỤ: ĐÁNH GIÁ TỔNG QUAN

            Hãy đưa ra đánh giá tổng quan về tình trạng bệnh nhân và khuyến nghị chung.
            """
        
        return base_prompt + specific_prompt
    
    def _parse_evaluation_response(self, response_content: str, evaluation_type: str) -> Dict[str, Any]:
        """Parse LLM response and structure the result."""
        try:
            # Try to parse as JSON first
            # Extract JSON from response if it's wrapped in other text
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_content[start_idx:end_idx]
                parsed_result = json.loads(json_str)
                parsed_result['evaluation_type'] = evaluation_type
                parsed_result['status'] = 'success'
                return parsed_result
            else:
                # If JSON parsing fails, return structured text response
                return {
                    'evaluation_type': evaluation_type,
                    'status': 'success',
                    'analysis': response_content,
                    'confidence_score': 0.70  # Default confidence for text response
                }
                
        except json.JSONDecodeError:
            # Fallback to text analysis
            return {
                'evaluation_type': evaluation_type,
                'status': 'success',
                'analysis': response_content,
                'confidence_score': 0.70,
                'note': 'Response parsed as text due to JSON parsing error'
            }
