"""
Patient Query Engine

This module provides high-level query interfaces for the three core problems:
1. Patient record retrieval and similarity search
2. Treatment optimization with outcome prediction
3. Early warning system and population health monitoring
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from agents.patient_db_agent.disease_evaluation import DiseaseEvaluation

from .patient_vectorstore import PatientVectorStore
from .find_treatment import TreatmentFinder


@dataclass
class RetrievalResult:
    """Result from patient record retrieval."""
    records: List[Dict[str, Any]]
    query_time: float
    total_found: int
    filters_applied: Dict[str, Any]


@dataclass
class TreatmentRecommendation:
    """Treatment recommendation result."""
    treatment: str
    expected_outcome: float
    confidence: float
    rationale: str
    supporting_cases: List[Dict[str, Any]]


@dataclass
class PopulationAlert:
    """Population health monitoring alert."""
    alert_type: str
    risk_level: str
    affected_cases: List[Dict[str, Any]]
    geographic_region: Optional[str]
    facility_id: Optional[str]
    description: str


class PatientQueryEngine:
    """
    High-level query interface for patient database operations.
    """
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.patient_store = PatientVectorStore(config)
        self.treatment_finder = TreatmentFinder(self.patient_store)
        self.disease_evaluator = DiseaseEvaluation()
        self.config = config
    
    # Problem 3: Early Warning System Methods
    
    def monitor_outbreak_risk(
        self,
        time_window: str = "7d",
        geographic_region: Optional[str] = None,
        facility_id: Optional[str] = None,
        threshold: int = 5
    ) -> List[PopulationAlert]:
        """
        Monitor for potential disease outbreaks or unusual patterns.
        
        Args:
            time_window: Time window for monitoring ("7d", "30d")
            geographic_region: Optional geographic filter
            facility_id: Optional facility filter
            threshold: Minimum number of cases to trigger alert
            
        Returns:
            List of population health alerts
        """
        alerts = []
        
        try:
            # Monitor various risk patterns
            risk_patterns = [
                ("outbreak_risk", "outbreak"),
                ("unusual_presentation", "unusual_presentation"),
                ("treatment_resistance", "treatment_resistance"),
                ("quality_alert", "quality_issues")
            ]
            
            for risk_flag, alert_type in risk_patterns:
                flagged_cases = self.disease_evaluator.monitor_population_patterns(
                    time_window=time_window,
                    geographic_region=geographic_region,
                    facility_id=facility_id,
                    risk_flags=[risk_flag.replace("_risk", "").replace("_alert", "")],
                    limit=100
                )
                
                if len(flagged_cases) >= threshold:
                    # Determine risk level
                    avg_risk_score = np.mean([case["risk_score"] for case in flagged_cases])
                    
                    if avg_risk_score >= 0.7:
                        risk_level = "HIGH"
                    elif avg_risk_score >= 0.5:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "LOW"
                    
                    # Generate description
                    description = self._generate_alert_description(
                        alert_type, len(flagged_cases), geographic_region, facility_id
                    )
                    
                    alert = PopulationAlert(
                        alert_type=alert_type,
                        risk_level=risk_level,
                        affected_cases=flagged_cases[:20],  # Limit to first 20 cases
                        geographic_region=geographic_region,
                        facility_id=facility_id,
                        description=description
                    )
                    alerts.append(alert)
            
            self.logger.info(f"Generated {len(alerts)} population health alerts")
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error monitoring outbreak risk: {e}")
            return []
    
    def _analyze_common_conditions(self, population_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze most common conditions in population data."""
        condition_counts = {}
        
        for case in population_data:
            comorbidities = case.get("payload", {}).get("comorbidities", [])
            for condition in comorbidities:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # Return top 10 conditions
        return dict(sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _calculate_trend_changes(
        self, 
        trends: Dict[str, Any], 
        time_periods: List[str]
    ) -> Dict[str, Any]:
        """Calculate changes in trends between time periods."""
        if len(time_periods) < 2:
            return {}
        
        # Compare most recent period to previous
        current_period = time_periods[0]  # Assuming first is most recent
        previous_period = time_periods[1]
        
        current_data = trends[current_period]
        previous_data = trends[previous_period]
        
        trend_changes = {}
        
        # Calculate percentage changes
        for metric in ["total_cases", "high_risk_cases", "average_risk_score"]:
            current_val = current_data.get(metric, 0)
            previous_val = previous_data.get(metric, 0)
            
            if previous_val > 0:
                change_pct = ((current_val - previous_val) / previous_val) * 100
                trend_changes[f"{metric}_change_pct"] = round(change_pct, 1)
            else:
                trend_changes[f"{metric}_change_pct"] = 0.0
        
        return trend_changes
