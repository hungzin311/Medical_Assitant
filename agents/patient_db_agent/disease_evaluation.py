import sys
import logging
from pathlib import Path
from config import Config  
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import Config
import json
from agents.patient_db_agent.patient_vectorstore import PatientVectorStore
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

class DiseaseEvaluation:
    """
    Handles treatment-related operations for patient records including:
    - Patient record retrieval for treatment planning
    - Finding similar treatment cases for optimization
    """
    
    def __init__(self):
        """
        Initialize with reference to the patient vector store.
        
        Args:
            patient_vector_store: Instance of PatientVectorStore for data access
        """
        self.logger = logging.getLogger(__name__)
        self.client = PatientVectorStore.client 
        self.collection_name = PatientVectorStore.collection_name
        self.embeddings = PatientVectorStore.embedding_model
        # Initialize LLM for disease evaluation
        config = Config()
        self.llm = config.patient_db.llm
        
    def retrieve_patient_records(
        self, 
        query: str, 
        patient_id: Optional[str] = None,
        demographic_filters: Optional[Dict[str, Any]] = None,
        clinical_filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage patient record retrieval (Problem 1).
        
        Args:
            query: Query text
            patient_id: Optional patient ID for same-patient history
            demographic_filters: Age, sex, BMI filters
            clinical_filters: Comorbidities, severity, diagnosis filters
            limit: Maximum number of results
            
        Returns:
            List of retrieved patient records with scores
        """
        must_conditions = []
        should_conditions = []
        
        # Stage 1: Same patient history (highest priority)
        if patient_id:
            should_conditions.append(
                FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
            )
        
        # Stage 2: Demographic filters
        if demographic_filters:
            if "age_range" in demographic_filters:
                min_age, max_age = demographic_filters["age_range"]
                must_conditions.append(
                    FieldCondition(key="age", range=Range(gte=min_age, lte=max_age))
                )
            if "sex" in demographic_filters:
                should_conditions.append(
                    FieldCondition(key="sex", match=MatchValue(value=demographic_filters["sex"]))
                )
            if "bmi_category" in demographic_filters:
                should_conditions.append(
                    FieldCondition(key="bmi_category", match=MatchValue(value=demographic_filters["bmi_category"]))
                )
        
        # Stage 3: Clinical filters
        if clinical_filters:
            if "comorbidities" in clinical_filters:
                for comorbidity in clinical_filters["comorbidities"]:
                    should_conditions.append(
                        FieldCondition(key="comorbidities", match=MatchValue(value=comorbidity))
                    )
            if "severity_category" in clinical_filters:
                must_conditions.append(
                    FieldCondition(key="severity_category", match=MatchValue(value=clinical_filters["severity_category"]))
                )
        
        # Build filter
        query_filter = None
        if must_conditions or should_conditions:
            query_filter = Filter(
                must=must_conditions if must_conditions else None,
                should=should_conditions if should_conditions else None
            )
        
        query = self.embeddings.embed_query(query)
        # Execute search
        try:
            # Use similarity_search_with_score for vector-based search
            results_with_scores = self.client.query_points(
                query = query,
                collection_name=self.collection_name,
                query_filter=query_filter, 
                limit=limit,
                with_payload = True,
                using="dense"
            )
            
            # The response is a QueryResponse object with a .points attribute
            results = []
            for point in results_with_scores.points:
                result_obj = type('SearchResult', (), {
                    'id': point.id,
                    'score': point.score,
                    'payload': point.payload
                })()
                results.append(result_obj)

            #Retrieved records list
            patient_records = []
            reference_records = []

            # Second search for same patient records only if patient_id is provided
            if patient_id:
                result2 = self.client.query_points( 
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
                result2 = result2.points
                #Format results
                for result in result2: 
                    record = {
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload
                    }
                    patient_records.append(record)
                        
            # Format results
            for result in results:
                record = {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                if record['id'] != patient_id:
                    reference_records.append(record)
            
            self.logger.info(f"Retrieved {len(reference_records)} patient records")
            return patient_records, reference_records
            
        except Exception as e:
            self.logger.error(f"Error retrieving patient records: {e}")
            return [], []

    def evaluate_based_record_llm(
        self, 
        patient_record: List[Dict[str, Any]], 
        reference_records: List[Dict[str, Any]], 
        evaluation_type: str = "disease_progression"
    ) -> Dict[str, Any]:
        
        # Simple version. Does not contain RAG. Like a demo, about 70% of the full version.

        """
        Evaluate patient disease progression and risk using LLM analysis of similar records.
        
        Args:
            patient_record: Current patient record to evaluate
            reference_records: Similar patient records for comparison
            evaluation_type: Type of evaluation ("disease_progression", "risk_assessment", "treatment_outcome")
            
        Returns:
            Dictionary containing LLM evaluation results
        """
        self.logger.info(f"Starting LLM-based {evaluation_type} evaluation")
        
        try:
            # Prepare patient data for analysis
            current_patient = self._format_patient_for_analysis(patient_record)
            reference_cases = self._format_reference_cases(reference_records[:5])  # Top 5 similar cases
            
            # Generate evaluation prompt based on type
            evaluation_prompt = self._create_evaluation_prompt(
                current_patient, reference_cases, evaluation_type
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
        """Format patient records for LLM analysis."""

        # Lấy thông tin cơ bản từ bản ghi đầu tiên
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

    
    def _format_reference_cases(self, reference_records: List[Dict[str, Any]]) -> str:
        """Format reference cases for LLM analysis."""
        reference_text = "CÁC TRƯỜNG HỢP TƯƠNG TỰ ĐỂ THAM KHẢO:\n\n"
        
        for i, record in enumerate(reference_records, 1):
            payload = record.get('payload', {})
            similarity_score = record.get('score', 0)
            
            reference_text += f"""
            TRƯỜNG HỢP {i} (Độ tương đồng: {similarity_score:.3f}):
            - Tuổi: {payload.get('age', 'N/A')}, Giới tính: {payload.get('sex', 'N/A')}
            - Chẩn đoán: {payload.get('diagnosis', 'N/A')}
            - Triệu chứng: {payload.get('chief_complaint', 'N/A')}
            - Bệnh kèm theo: {', '.join(payload.get('comorbidities', []))}
            - Điểm nghiêm trọng: {payload.get('severity_score', 'N/A')}/10
            - Điều trị đã dùng: {', '.join(payload.get('treatments_tried', []))}
            - Kết quả điều trị: {payload.get('primary_outcome', 'N/A')}
            - Tóm tắt: {payload.get('summary_text', 'N/A')[:200]}...
            """
        return reference_text
    
    def _create_evaluation_prompt(self, patient_info: str, reference_cases: str, evaluation_type: str) -> str:
        """Create evaluation prompt based on type."""
        
        base_prompt = f"""
        Bạn là một chuyên gia y tế có kinh nghiệm cao trong việc phân tích và đánh giá tình trạng bệnh nhân. 
        Nhiệm vụ của bạn là đánh giá bệnh nhân hiện tại dựa trên các trường hợp tương tự đã có.

        {patient_info}

        {reference_cases}
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
            import json
            # Extract JSON from response if it's wrapped in other text
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_content[start_idx:end_idx]
                parsed_result = json.loads(json_str)
                parsed_result['evaluation_type'] = evaluation_type
                parsed_result['status'] = 'success'
                parsed_result['raw_response'] = response_content
                return parsed_result
            else:
                # If JSON parsing fails, return structured text response
                return {
                    'evaluation_type': evaluation_type,
                    'status': 'success',
                    'analysis': response_content,
                    'raw_response': response_content,
                    'confidence_score': 0.70  # Default confidence for text response
                }
                
        except json.JSONDecodeError:
            # Fallback to text analysis
            return {
                'evaluation_type': evaluation_type,
                'status': 'success',
                'analysis': response_content,
                'raw_response': response_content,
                'confidence_score': 0.70,
                'note': 'Response parsed as text due to JSON parsing error'
            }
