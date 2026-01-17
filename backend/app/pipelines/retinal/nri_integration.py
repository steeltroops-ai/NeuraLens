"""
NRI (Neurological Risk Index) Integration Service for Retinal Analysis

Integrates retinal analysis results into the NRI composite score:
- Format retinal results for NRI API (Requirement 9.1)
- Send retinal data with 30% base weight (Requirement 9.2)
- Include confidence scores (Requirement 9.3)
- Handle NRI failure fallback (Requirement 9.4)
- Update dashboard with contribution (Requirement 9.5)
- Dynamic weight adjustment (Requirement 9.6)
- Cascade recalculation on deletion (Requirement 9.7)
- Track contribution percentage (Requirement 9.8)

Author: NeuraLens Team
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx

from .schemas import RetinalAnalysisResponse, RiskAssessment

logger = logging.getLogger(__name__)


class NRIStatus(str, Enum):
    """NRI integration status codes"""
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    PARTIAL = "partial"
    STANDALONE = "standalone"


@dataclass
class NRIContribution:
    """Data structure for NRI contribution from retinal analysis"""
    base_weight: float = 0.30  # 30% per Requirement 9.2
    actual_weight: float = 0.30
    contribution_score: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_weight": self.base_weight,
            "actual_weight": self.actual_weight,
            "contribution_score": self.contribution_score,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class NRIPayload:
    """
    Payload format for NRI API integration.
    
    Requirement 9.1: Format retinal results for NRI API
    """
    assessment_id: str
    patient_id: str
    source: str = "retinal"
    
    # Risk data
    risk_score: float = 0.0
    risk_category: str = "minimal"
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    
    # Biomarker summary
    vessel_density: float = 0.0
    tortuosity_index: float = 0.0
    cup_to_disc_ratio: float = 0.0
    amyloid_presence: float = 0.0
    
    # Weight and contribution
    weight: float = 0.30
    adjusted_contribution: float = 0.0
    
    # Metadata
    model_version: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        return {
            "assessment_id": self.assessment_id,
            "patient_id": self.patient_id,
            "source": self.source,
            "risk_data": {
                "score": self.risk_score,
                "category": self.risk_category,
                "confidence_interval": {
                    "lower": self.confidence_lower,
                    "upper": self.confidence_upper
                }
            },
            "biomarkers": {
                "vessel_density": self.vessel_density,
                "tortuosity_index": self.tortuosity_index,
                "cup_to_disc_ratio": self.cup_to_disc_ratio,
                "amyloid_presence": self.amyloid_presence
            },
            "contribution": {
                "weight": self.weight,
                "adjusted_score": self.adjusted_contribution
            },
            "metadata": {
                "model_version": self.model_version,
                "timestamp": self.timestamp.isoformat()
            }
        }


@dataclass
class NRIResponse:
    """Response from NRI API"""
    success: bool
    status: NRIStatus
    nri_score: Optional[float] = None
    nri_category: Optional[str] = None
    retinal_contribution: Optional[float] = None
    message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NRIResponse":
        return cls(
            success=data.get("success", False),
            status=NRIStatus(data.get("status", "failed")),
            nri_score=data.get("nri_score"),
            nri_category=data.get("nri_category"),
            retinal_contribution=data.get("retinal_contribution"),
            message=data.get("message")
        )


class NRIIntegrationService:
    """
    Service for integrating retinal analysis with NRI (Neurological Risk Index).
    
    Handles:
    - Formatting retinal results for NRI API
    - Sending contributions with proper weights
    - Fallback to standalone mode on failure
    - Dynamic weight adjustment
    - Cascade recalculation
    
    Requirements: 9.1-9.8
    """
    
    # Weight constants per requirements
    BASE_WEIGHT = 0.30  # 30% base weight per Requirement 9.2
    MIN_WEIGHT = 0.15   # Minimum weight for dynamic adjustment
    MAX_WEIGHT = 0.40   # Maximum weight for dynamic adjustment
    
    def __init__(
        self,
        nri_api_url: Optional[str] = None,
        timeout_seconds: float = 10.0
    ):
        self.nri_api_url = nri_api_url or "http://localhost:8000/api/v1/nri"
        self.timeout = timeout_seconds
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    def format_for_nri(
        self,
        assessment: RetinalAnalysisResponse,
        adjusted_weight: Optional[float] = None
    ) -> NRIPayload:
        """
        Format retinal analysis results for NRI API.
        
        Requirement 9.1: Format retinal results for NRI API
        Requirement 9.3: Include confidence scores
        
        Args:
            assessment: Complete retinal analysis response
            adjusted_weight: Optional adjusted weight (default: 30%)
            
        Returns:
            NRIPayload ready for API submission
        """
        weight = adjusted_weight or self.BASE_WEIGHT
        
        # Calculate adjusted contribution
        adjusted_contribution = assessment.risk_assessment.risk_score * weight
        
        return NRIPayload(
            assessment_id=assessment.assessment_id,
            patient_id=assessment.patient_id,
            risk_score=assessment.risk_assessment.risk_score,
            risk_category=assessment.risk_assessment.risk_category,
            confidence_lower=assessment.risk_assessment.confidence_interval[0],
            confidence_upper=assessment.risk_assessment.confidence_interval[1],
            vessel_density=assessment.biomarkers.vessels.density_percentage,
            tortuosity_index=assessment.biomarkers.vessels.tortuosity_index,
            cup_to_disc_ratio=assessment.biomarkers.optic_disc.cup_to_disc_ratio,
            amyloid_presence=assessment.biomarkers.amyloid_beta.presence_score,
            weight=weight,
            adjusted_contribution=adjusted_contribution,
            model_version=assessment.model_version
        )
    
    async def send_contribution(
        self,
        assessment: RetinalAnalysisResponse,
        adjusted_weight: Optional[float] = None
    ) -> Tuple[NRIResponse, NRIContribution]:
        """
        Send retinal analysis contribution to NRI service.
        
        Requirement 9.2: Send retinal data with 30% base weight
        
        Args:
            assessment: Complete retinal analysis response
            adjusted_weight: Optional adjusted weight
            
        Returns:
            Tuple of (NRIResponse, NRIContribution)
        """
        payload = self.format_for_nri(assessment, adjusted_weight)
        contribution = NRIContribution(
            base_weight=self.BASE_WEIGHT,
            actual_weight=payload.weight,
            contribution_score=payload.adjusted_contribution,
            confidence=self._calculate_confidence(assessment)
        )
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.nri_api_url}/contribute",
                json=payload.to_dict()
            )
            
            if response.status_code == 200:
                data = response.json()
                nri_response = NRIResponse.from_dict(data)
                logger.info(
                    f"NRI contribution sent successfully: "
                    f"assessment={assessment.assessment_id}, "
                    f"contribution={contribution.contribution_score:.2f}"
                )
                return nri_response, contribution
            else:
                logger.warning(
                    f"NRI API returned non-200 status: {response.status_code}"
                )
                return self._create_fallback_response(
                    f"API returned status {response.status_code}"
                ), contribution
                
        except httpx.TimeoutException:
            logger.error("NRI API request timed out")
            return self._create_fallback_response("Request timed out"), contribution
        except httpx.RequestError as e:
            logger.error(f"NRI API request failed: {str(e)}")
            return self._create_fallback_response(str(e)), contribution
        except Exception as e:
            logger.exception(f"Unexpected error in NRI integration: {str(e)}")
            return self._create_fallback_response(str(e)), contribution
    
    def _create_fallback_response(self, message: str) -> NRIResponse:
        """
        Create fallback response for NRI failure.
        
        Requirement 9.4: Display retinal results independently on NRI failure
        """
        return NRIResponse(
            success=False,
            status=NRIStatus.STANDALONE,
            message=f"NRI integration failed, retinal results available standalone. Error: {message}"
        )
    
    def _calculate_confidence(self, assessment: RetinalAnalysisResponse) -> float:
        """Calculate overall confidence from biomarker confidences"""
        confidences = [
            assessment.biomarkers.vessels.confidence,
            assessment.biomarkers.optic_disc.confidence,
            assessment.biomarkers.macula.confidence,
            assessment.biomarkers.amyloid_beta.confidence
        ]
        return sum(confidences) / len(confidences)
    
    def calculate_dynamic_weight(
        self,
        assessment: RetinalAnalysisResponse,
        other_modalities: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate dynamically adjusted weight based on confidence and other modalities.
        
        Requirement 9.6: Handle dynamic weight adjustment
        
        Args:
            assessment: Retinal analysis response
            other_modalities: Dictionary of other modality weights (e.g., {"speech": 0.25, "gait": 0.20})
            
        Returns:
            Adjusted weight value
        """
        confidence = self._calculate_confidence(assessment)
        
        # Base adjustment based on confidence
        # High confidence (>0.9) -> increase weight
        # Low confidence (<0.7) -> decrease weight
        if confidence >= 0.9:
            confidence_factor = 1.1
        elif confidence >= 0.8:
            confidence_factor = 1.0
        elif confidence >= 0.7:
            confidence_factor = 0.9
        else:
            confidence_factor = 0.8
        
        adjusted_weight = self.BASE_WEIGHT * confidence_factor
        
        # Adjust based on other modalities if provided
        if other_modalities:
            total_other_weight = sum(other_modalities.values())
            available_weight = 1.0 - total_other_weight
            
            # Ensure we don't exceed available weight
            adjusted_weight = min(adjusted_weight, available_weight * 0.5)
        
        # Clamp to valid range
        return max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, adjusted_weight))
    
    async def trigger_recalculation(
        self,
        patient_id: str,
        deleted_assessment_id: str
    ) -> NRIResponse:
        """
        Trigger NRI recalculation after assessment deletion.
        
        Requirement 9.7: Trigger NRI recalculation on assessment deletion
        
        Args:
            patient_id: Patient identifier
            deleted_assessment_id: ID of deleted assessment
            
        Returns:
            NRIResponse with recalculation result
        """
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.nri_api_url}/recalculate",
                json={
                    "patient_id": patient_id,
                    "deleted_assessment_id": deleted_assessment_id,
                    "source": "retinal"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"NRI recalculation triggered for patient {patient_id}"
                )
                return NRIResponse.from_dict(data)
            else:
                return self._create_fallback_response(
                    f"Recalculation API returned status {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"NRI recalculation failed: {str(e)}")
            return self._create_fallback_response(str(e))
    
    def get_dashboard_data(
        self,
        assessment: RetinalAnalysisResponse,
        nri_response: Optional[NRIResponse],
        contribution: NRIContribution
    ) -> Dict[str, Any]:
        """
        Get data formatted for dashboard display.
        
        Requirement 9.5: Update NRI display with retinal contribution
        Requirement 9.8: Show contribution percentage
        
        Args:
            assessment: Retinal analysis response
            nri_response: Optional NRI API response
            contribution: NRI contribution data
            
        Returns:
            Dictionary with dashboard display data
        """
        dashboard_data = {
            "retinal_analysis": {
                "assessment_id": assessment.assessment_id,
                "risk_score": assessment.risk_assessment.risk_score,
                "risk_category": assessment.risk_assessment.risk_category,
                "quality_score": assessment.quality_score,
                "created_at": assessment.created_at.isoformat()
            },
            "nri_contribution": {
                "weight_percentage": contribution.actual_weight * 100,
                "contribution_score": contribution.contribution_score,
                "confidence": contribution.confidence * 100,
                "status": NRIStatus.STANDALONE.value
            },
            "biomarker_summary": {
                "vessel_density": {
                    "value": assessment.biomarkers.vessels.density_percentage,
                    "status": self._get_biomarker_status(
                        assessment.biomarkers.vessels.density_percentage, 4.0, 7.0
                    )
                },
                "tortuosity": {
                    "value": assessment.biomarkers.vessels.tortuosity_index,
                    "status": self._get_biomarker_status(
                        assessment.biomarkers.vessels.tortuosity_index, 0.8, 1.3
                    )
                },
                "cup_to_disc": {
                    "value": assessment.biomarkers.optic_disc.cup_to_disc_ratio,
                    "status": self._get_biomarker_status(
                        assessment.biomarkers.optic_disc.cup_to_disc_ratio, 0.3, 0.5
                    )
                },
                "amyloid_beta": {
                    "value": assessment.biomarkers.amyloid_beta.presence_score,
                    "status": self._get_biomarker_status(
                        assessment.biomarkers.amyloid_beta.presence_score, 0.0, 0.2
                    )
                }
            }
        }
        
        # Add NRI data if available
        if nri_response and nri_response.success:
            dashboard_data["nri_contribution"]["status"] = nri_response.status.value
            if nri_response.nri_score is not None:
                dashboard_data["nri_total"] = {
                    "score": nri_response.nri_score,
                    "category": nri_response.nri_category,
                    "retinal_contribution_percentage": (
                        (contribution.contribution_score / nri_response.nri_score) * 100
                        if nri_response.nri_score > 0 else 0
                    )
                }
        
        return dashboard_data
    
    def _get_biomarker_status(
        self, value: float, min_val: float, max_val: float
    ) -> str:
        """Get status string for biomarker value"""
        if min_val <= value <= max_val:
            return "normal"
        elif value < min_val:
            return "low"
        else:
            return "high"
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
nri_integration_service = NRIIntegrationService()
