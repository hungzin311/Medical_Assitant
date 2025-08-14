import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


class DiseaseEvaluator:
    """
    Handles disease evaluation and population health monitoring including:
    - Early warning system for population patterns
    - Risk assessment and monitoring
    - Population health analytics
    """
    
    def __init__(self, patient_vector_store):
        """
        Initialize with reference to the patient vector store.
        
        Args:
            patient_vector_store: Instance of PatientVectorStore for data access
        """
        self.logger = logging.getLogger(__name__)
        self.client = patient_vector_store.client
        self.collection_name = patient_vector_store.collection_name
        self.embeddings = patient_vector_store.embedding_model
        
    def monitor_population_patterns(
        self, 
        time_window: str = "7d",
        geographic_region: Optional[str] = None,
        facility_id: Optional[str] = None,
        risk_flags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Monitor population patterns for early warning (Problem 3).
        
        Args:
            time_window: Time window (e.g., "7d", "30d")
            geographic_region: Optional geographic filter
            facility_id: Optional facility filter
            risk_flags: List of risk flags to monitor
            limit: Maximum number of results
            
        Returns:
            List of flagged cases
        """
        must_conditions = []
        should_conditions = []
        
        # Time filter - for simplicity, using is_recent flag
        # Only apply time filter for short time windows
        if time_window in ["7d", "30d"]:
            must_conditions.append(
                FieldCondition(key="is_recent", match=MatchValue(value=True))
            )
        # For longer time windows or general monitoring, don't filter by time
        
        # Geographic filter
        if geographic_region:
            must_conditions.append(
                FieldCondition(key="geographic_region", match=MatchValue(value=geographic_region))
            )
        
        # Facility filter
        if facility_id:
            must_conditions.append(
                FieldCondition(key="facility_id", match=MatchValue(value=facility_id))
            )
        
        # Risk flags
        risk_flag_mapping = {
            "outbreak": "outbreak_risk",
            "unusual": "unusual_presentation",
            "resistance": "treatment_resistance",
            "quality": "quality_alert",
            "high_risk": "high_risk_patient"
        }
        
        # Simplify the risk flag logic - just look for any records without complex filtering
        # The risk scoring will be done in post-processing
        if not risk_flags:
            # If no specific flags requested, get all recent records
            pass  # We already have time and location filters
        
        query_filter = Filter(
            must=must_conditions if must_conditions else None,
            should=should_conditions if should_conditions else None
        )
        
        try:
            # Get all records first, then filter in Python
            all_results = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=Filter(must=must_conditions) if must_conditions else None,
                limit=limit,
                with_payload=True
            )
            
            flagged_cases = []
            # Filter and score results
            for result in all_results.points:
                payload = result.payload
                risk_score = self._calculate_risk_score(payload)
                
                # Apply risk flag filtering in Python if specified
                include_case = True
                if risk_flags:
                    include_case = False
                    for flag in risk_flags:
                        if flag in risk_flag_mapping:
                            flag_field = risk_flag_mapping[flag]
                            if payload.get(flag_field, False):
                                include_case = True
                                break
                else:
                    # If no specific flags, include cases with any risk flags or high risk scores
                    has_risk_flag = any(payload.get(flag_field, False) for flag_field in risk_flag_mapping.values())
                    if has_risk_flag or risk_score > 0.5:
                        include_case = True
                    else:
                        include_case = False
                
                if include_case:
                    case = {
                        "id": result.id,
                        "payload": payload,
                        "risk_score": risk_score
                    }
                    flagged_cases.append(case)
            
            # Sort by risk score
            flagged_cases.sort(key=lambda x: x["risk_score"], reverse=True)
            
            self.logger.info(f"Found {len(flagged_cases)} flagged cases for monitoring")
            return flagged_cases
            
        except Exception as e:
            self.logger.error(f"Error monitoring population patterns: {e}")
            return []

    def _calculate_risk_score(self, payload: Dict[str, Any]) -> float:
        """Calculate composite risk score for population monitoring."""
        risk_score = 0.0
        
        # Risk flags
        if payload.get("outbreak_risk", False):
            risk_score += 0.3
        if payload.get("unusual_presentation", False):
            risk_score += 0.25
        if payload.get("treatment_resistance", False):
            risk_score += 0.2
        if payload.get("quality_alert", False):
            risk_score += 0.15
        if payload.get("high_risk_patient", False):
            risk_score += 0.1
        
        # Severity contribution
        severity_score = payload.get("severity_score", 0.0)
        risk_score += severity_score * 0.2
        
        # Comorbidity contribution
        comorbidity_count = payload.get("comorbidity_count", 0)
        risk_score += min(comorbidity_count * 0.05, 0.2)
        
        return min(risk_score, 1.0)

    def evaluate_outbreak_risk(
        self,
        geographic_region: Optional[str] = None,
        facility_id: Optional[str] = None,
        time_window: str = "7d"
    ) -> Dict[str, Any]:
        """
        Evaluate outbreak risk for a specific region or facility.
        
        Args:
            geographic_region: Geographic region to analyze
            facility_id: Facility ID to analyze
            time_window: Time window for analysis
            
        Returns:
            Outbreak risk assessment
        """
        flagged_cases = self.monitor_population_patterns(
            time_window=time_window,
            geographic_region=geographic_region,
            facility_id=facility_id,
            risk_flags=["outbreak", "unusual"],
            limit=200
        )
        
        # Calculate outbreak metrics
        total_cases = len(flagged_cases)
        high_risk_cases = len([case for case in flagged_cases if case["risk_score"] > 0.7])
        outbreak_cases = len([case for case in flagged_cases 
                            if case["payload"].get("outbreak_risk", False)])
        unusual_cases = len([case for case in flagged_cases 
                           if case["payload"].get("unusual_presentation", False)])
        
        # Calculate risk level
        if total_cases == 0:
            risk_level = "low"
        elif high_risk_cases / total_cases > 0.3:
            risk_level = "high"
        elif high_risk_cases / total_cases > 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "total_cases": total_cases,
            "high_risk_cases": high_risk_cases,
            "outbreak_cases": outbreak_cases,
            "unusual_cases": unusual_cases,
            "risk_ratio": high_risk_cases / total_cases if total_cases > 0 else 0,
            "geographic_region": geographic_region,
            "facility_id": facility_id,
            "time_window": time_window
        }

    def get_population_health_summary(
        self,
        geographic_region: Optional[str] = None,
        time_window: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get population health summary for a region.
        
        Args:
            geographic_region: Geographic region to analyze
            time_window: Time window for analysis
            
        Returns:
            Population health summary
        """
        # Simplified query for population health summary
        must_conditions = []
        
        # Time filter - for simplicity, using is_recent flag
        # Only apply time filter for short time windows
        if time_window in ["7d", "30d"]:
            must_conditions.append(
                FieldCondition(key="is_recent", match=MatchValue(value=True))
            )
        # For longer time windows, don't filter by time
        
        # Geographic filter
        if geographic_region:
            must_conditions.append(
                FieldCondition(key="geographic_region", match=MatchValue(value=geographic_region))
            )
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        try:
            # Get all cases without complex filtering
            results = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=query_filter,
                limit=1000,
                with_payload=True
            )
            
            all_cases = []
            for result in results.points:
                case = {
                    "id": result.id,
                    "payload": result.payload,
                    "risk_score": self._calculate_risk_score(result.payload)
                }
                all_cases.append(case)
        except Exception as e:
            self.logger.error(f"Error getting population health data: {e}")
            all_cases = []
        
        if not all_cases:
            return {
                "total_patients": 0,
                "average_risk_score": 0,
                "high_risk_percentage": 0,
                "common_conditions": [],
                "geographic_region": geographic_region
            }
        
        # Calculate metrics
        total_patients = len(all_cases)
        average_risk_score = sum(case["risk_score"] for case in all_cases) / total_patients
        high_risk_count = len([case for case in all_cases if case["risk_score"] > 0.5])
        high_risk_percentage = (high_risk_count / total_patients) * 100
        
        # Extract common conditions
        condition_counts = {}
        for case in all_cases:
            comorbidities = case["payload"].get("comorbidities", [])
            for condition in comorbidities:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # Get top 5 most common conditions
        common_conditions = sorted(condition_counts.items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_patients": total_patients,
            "average_risk_score": round(average_risk_score, 3),
            "high_risk_percentage": round(high_risk_percentage, 2),
            "common_conditions": [{"condition": cond, "count": count} 
                                for cond, count in common_conditions],
            "geographic_region": geographic_region,
            "time_window": time_window
        }