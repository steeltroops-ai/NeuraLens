"""
Dermatology Pipeline Core Service

Main analysis service orchestrating all pipeline components.
"""

import logging
import time
import uuid
import base64
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

from ..config import MANDATORY_DISCLAIMER, CRITICAL_DISCLAIMER
from ..schemas import (
    DermatologyRequest,
    DermatologySuccessResponse,
    DermatologyFailureResponse,
    ValidationResult,
    PreprocessingResult,
    SegmentationResult,
    ABCDEFeatures,
    MelanomaResult,
    MalignancyResult,
    SubtypeResult,
    RiskTierResult,
    StageInfo
)
from ..input import DermatologyInputValidator
from ..preprocessing import DermatologyPreprocessor
from ..segmentation import DermatologySegmenter
from ..analysis import ABCDEExtractor, DermatologyClassifier
from ..clinical import ClinicalScorer
from ..errors import get_error_response

logger = logging.getLogger(__name__)


def _convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class DermatologyService:
    """
    Main dermatology analysis service.
    """
    
    def __init__(self):
        self.validator = DermatologyInputValidator()
        self.preprocessor = DermatologyPreprocessor()
        self.segmenter = DermatologySegmenter()
        self.abcde_extractor = ABCDEExtractor()
        self.classifier = DermatologyClassifier()
        self.scorer = ClinicalScorer()
    
    async def analyze(
        self,
        image_data: bytes,
        content_type: str = None,
        request: DermatologyRequest = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete dermatology analysis pipeline.
        
        Returns:
            - success: bool
            - response: dict (success or failure response)
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        stages = []
        
        request = request or DermatologyRequest()
        
        try:
            # === STAGE 1: Validation ===
            stage_start = time.time()
            validation, image, image_hash = self.validator.validate(
                image_data, content_type
            )
            stage_duration = int((time.time() - stage_start) * 1000)
            
            stages.append(StageInfo(
                name="validation",
                status="success" if validation.passed else "warning",
                duration_ms=stage_duration,
                confidence=validation.quality_score if validation.quality else 0,
                warnings=validation.warnings
            ))
            
            # Only block on critical errors (file cannot be read)
            if image is None:
                return self._format_failure(
                    validation.errors[0] if validation.errors else get_error_response("E_VAL_001"),
                    request_id,
                    "validation",
                    stages,
                    start_time
                )
            
            # === STAGE 2: Preprocessing ===
            stage_start = time.time()
            preprocessing = self.preprocessor.preprocess(image)
            stage_duration = int((time.time() - stage_start) * 1000)
            
            stages.append(StageInfo(
                name="preprocessing",
                status="success",
                duration_ms=stage_duration,
                confidence=preprocessing.overall_confidence,
                warnings=preprocessing.warnings
            ))
            
            # === STAGE 3: Segmentation ===
            stage_start = time.time()
            segmentation = self.segmenter.segment(preprocessing.image)
            stage_duration = int((time.time() - stage_start) * 1000)
            
            seg_status = "success" if segmentation.detected else "failure"
            stages.append(StageInfo(
                name="segmentation",
                status=seg_status,
                duration_ms=stage_duration,
                confidence=segmentation.confidence,
                warnings=segmentation.warnings
            ))
            
            if not segmentation.detected or segmentation.geometry is None:
                return self._format_failure(
                    get_error_response("E_SEG_001"),
                    request_id,
                    "segmentation",
                    stages,
                    start_time
                )
            
            # === STAGE 4: ABCDE Feature Extraction ===
            stage_start = time.time()
            abcde = self.abcde_extractor.extract(
                preprocessing.image,
                segmentation.mask,
                segmentation.geometry
            )
            stage_duration = int((time.time() - stage_start) * 1000)
            
            stages.append(StageInfo(
                name="feature_extraction",
                status="success",
                duration_ms=stage_duration
            ))
            
            # === STAGE 5: Classification ===
            stage_start = time.time()
            classification = self.classifier.classify(
                preprocessing.image,
                segmentation.mask,
                abcde
            )
            melanoma = classification['melanoma']
            malignancy = classification['malignancy']
            subtype = classification['subtype']
            stage_duration = int((time.time() - stage_start) * 1000)
            
            stages.append(StageInfo(
                name="classification",
                status="success",
                duration_ms=stage_duration,
                confidence=melanoma.probability
            ))
            
            # === STAGE 6: Risk Scoring ===
            stage_start = time.time()
            risk = self.scorer.score(melanoma, malignancy, subtype, abcde)
            stage_duration = int((time.time() - stage_start) * 1000)
            
            stages.append(StageInfo(
                name="scoring",
                status="success",
                duration_ms=stage_duration
            ))
            
            # === STAGE 7: Generate Visualizations ===
            visualizations = None
            if request.include_visualizations:
                stage_start = time.time()
                visualizations = self._generate_visualizations(
                    preprocessing.image,
                    segmentation.mask,
                    segmentation.geometry
                )
                stage_duration = int((time.time() - stage_start) * 1000)
                
                stages.append(StageInfo(
                    name="visualization",
                    status="success",
                    duration_ms=stage_duration
                ))
            
            # === STAGE 8: Generate Explanation ===
            explanation = None
            if request.generate_explanation:
                stage_start = time.time()
                explanation = self._generate_explanation(
                    melanoma, malignancy, subtype, abcde, risk
                )
                stage_duration = int((time.time() - stage_start) * 1000)
                
                stages.append(StageInfo(
                    name="explanation",
                    status="success",
                    duration_ms=stage_duration
                ))
            
            # === Format Success Response ===
            total_time = int((time.time() - start_time) * 1000)
            
            response = self._format_success(
                request_id=request_id,
                image_hash=image_hash,
                processing_time_ms=total_time,
                stages=stages,
                segmentation=segmentation,
                abcde=abcde,
                melanoma=melanoma,
                malignancy=malignancy,
                subtype=subtype,
                risk=risk,
                visualizations=visualizations,
                explanation=explanation,
                validation=validation
            )
            
            return True, response
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            total_time = int((time.time() - start_time) * 1000)
            
            return self._format_failure(
                get_error_response("E_SYS_001"),
                request_id,
                "unknown",
                stages,
                start_time
            )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"derm_{timestamp}_{unique}"
    
    def _generate_visualizations(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        geometry
    ) -> Dict[str, str]:
        """Generate visualization images."""
        visualizations = {}
        
        try:
            # Segmentation overlay
            overlay = image.copy()
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            
            # Mark center
            if geometry and geometry.center:
                cv2.circle(overlay, geometry.center, 5, (255, 0, 0), -1)
            
            # Convert to base64
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', overlay_rgb)
            visualizations['segmentation_overlay_base64'] = base64.b64encode(
                buffer
            ).decode('utf-8')
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        return visualizations
    
    def _generate_explanation(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures,
        risk: RiskTierResult
    ) -> Dict[str, Any]:
        """Generate AI explanation."""
        # Summary
        summary = self._generate_summary(melanoma, risk)
        
        # Detailed explanation
        detailed = self._generate_detailed(melanoma, malignancy, subtype, abcde, risk)
        
        # Recommendations
        recommendations = self._generate_recommendations(risk)
        
        # Disclaimers
        disclaimers = [MANDATORY_DISCLAIMER.strip()]
        if risk.tier <= 2:
            disclaimers.insert(0, CRITICAL_DISCLAIMER.strip())
        
        return {
            "summary": summary,
            "detailed": detailed,
            "recommendations": recommendations,
            "disclaimers": disclaimers
        }
    
    def _generate_summary(
        self,
        melanoma: MelanomaResult,
        risk: RiskTierResult
    ) -> str:
        """Generate brief summary."""
        if risk.tier == 1:
            return (
                "This lesion shows features that warrant urgent medical attention. "
                "Please seek dermatology evaluation within 24-48 hours."
            )
        elif risk.tier == 2:
            return (
                "This lesion has concerning features that should be evaluated by a dermatologist "
                "within the next 1-2 weeks."
            )
        elif risk.tier == 3:
            return (
                "This lesion shows some features that warrant professional evaluation. "
                "We recommend scheduling a dermatology appointment within 1-3 months."
            )
        elif risk.tier == 4:
            return (
                "This lesion appears low risk but should be monitored. "
                "Consider follow-up during your next routine skin check."
            )
        else:
            return (
                "This lesion appears benign based on our analysis. "
                "Continue routine skin health monitoring."
            )
    
    def _generate_detailed(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        abcde: ABCDEFeatures,
        risk: RiskTierResult
    ) -> str:
        """Generate detailed explanation."""
        sections = []
        
        # Overview
        sections.append("## Analysis Summary\n")
        sections.append(
            "Your skin lesion has been analyzed using our AI dermatology screening system. "
            "Here's what we found:\n"
        )
        
        # Key findings
        sections.append("### Key Findings\n")
        
        # ABCDE breakdown
        if abcde.asymmetry.is_concerning:
            sections.append(
                f"**Asymmetry**: The lesion shows {abcde.asymmetry.classification.replace('_', ' ')}. "
                "Asymmetry can be an indicator of abnormal cell growth.\n"
            )
        
        if abcde.border.is_concerning:
            sections.append(
                f"**Border**: The border appears {abcde.border.classification.replace('_', ' ')}. "
                "Irregular borders can indicate concerning lesions.\n"
            )
        
        if abcde.color.is_concerning:
            color_info = f"We detected {abcde.color.num_colors} different color tones"
            if abcde.color.has_blue_white_veil:
                color_info += " including a blue-white veil pattern"
            sections.append(f"**Color**: {color_info}. Multiple colors warrant attention.\n")
        
        if abcde.diameter.is_concerning:
            sections.append(
                f"**Diameter**: At {abcde.diameter.max_dimension_mm:.1f}mm, this lesion exceeds "
                "the typical 6mm guideline used in screening.\n"
            )
        
        if abcde.criteria_met == 0:
            sections.append(
                "No significant concerning features were detected in the ABCDE analysis.\n"
            )
        
        # Risk level
        sections.append(f"\n### Risk Level: {risk.tier_name}\n")
        sections.append(f"{risk.reasoning}\n")
        
        return "\n".join(sections)
    
    def _generate_recommendations(self, risk: RiskTierResult) -> list:
        """Generate recommendations based on risk."""
        recommendations = []
        
        if risk.tier == 1:
            recommendations = [
                "Seek immediate dermatology evaluation within 24-48 hours",
                "Do not delay medical attention",
                "Bring these results to your appointment",
                "Note any recent changes in the lesion"
            ]
        elif risk.tier == 2:
            recommendations = [
                "Schedule a dermatology appointment within 1-2 weeks",
                "Monitor for any rapid changes",
                "Take photos to track any changes",
                "Bring these results to your appointment"
            ]
        elif risk.tier == 3:
            recommendations = [
                "Schedule a dermatology consultation within 1-3 months",
                "Take monthly photos to track any changes",
                "Note any new symptoms (itching, bleeding, crusting)",
                "Bring these results to your appointment"
            ]
        elif risk.tier == 4:
            recommendations = [
                "Consider evaluation during your next routine skin check",
                "Monitor for any changes in size, shape, or color",
                "Practice sun protection",
                "Perform regular self-examinations"
            ]
        else:
            recommendations = [
                "Continue routine skin health monitoring",
                "Practice sun protection",
                "Perform regular self-examinations",
                "See a dermatologist annually for full skin check"
            ]
        
        return recommendations
    
    def _format_success(
        self,
        request_id: str,
        image_hash: str,
        processing_time_ms: int,
        stages: list,
        segmentation: SegmentationResult,
        abcde: ABCDEFeatures,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        risk: RiskTierResult,
        visualizations: dict,
        explanation: dict,
        validation: ValidationResult
    ) -> Dict[str, Any]:
        """Format success response."""
        # Format stages
        stages_list = [
            {
                "name": s.name,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "confidence": s.confidence,
                "warnings": s.warnings
            }
            for s in stages
        ]
        
        # Format geometry
        geometry_dict = None
        if segmentation.geometry:
            g = segmentation.geometry
            geometry_dict = {
                "area_mm2": g.area_mm2,
                "diameter_mm": g.diameter_mm,
                "major_axis_mm": g.major_axis_mm,
                "minor_axis_mm": g.minor_axis_mm,
                "circularity": g.circularity,
                "asymmetry_index": g.asymmetry_index
            }
        
        # Format ABCDE
        abcde_dict = {
            "asymmetry": {
                "score": abcde.asymmetry.combined_score,
                "is_concerning": abcde.asymmetry.is_concerning,
                "classification": abcde.asymmetry.classification
            },
            "border": {
                "score": abcde.border.irregularity_score,
                "is_concerning": abcde.border.is_concerning,
                "classification": abcde.border.classification
            },
            "color": {
                "score": abcde.color.color_score,
                "is_concerning": abcde.color.is_concerning,
                "num_colors": abcde.color.num_colors,
                "has_blue_white_veil": abcde.color.has_blue_white_veil
            },
            "diameter": {
                "score": abcde.diameter.risk_contribution,
                "is_concerning": abcde.diameter.is_concerning,
                "value_mm": abcde.diameter.max_dimension_mm
            },
            "evolution": {
                "score": abcde.evolution.evolution_score,
                "is_concerning": abcde.evolution.is_concerning,
                "classification": abcde.evolution.classification
            }
        }
        
        # Format escalations
        escalations_list = [
            {
                "rule": e.rule_name,
                "action": e.action,
                "reason": e.reason,
                "priority": e.priority
            }
            for e in risk.escalations
        ]
        
        # Collect warnings
        all_warnings = []
        for s in stages:
            all_warnings.extend(s.warnings)
        
        # Convert all numpy types to native Python types for JSON serialization
        return _convert_numpy({
            "success": True,
            
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image_hash": image_hash,
            "processing_time_ms": processing_time_ms,
            
            "stages_completed": stages_list,
            
            "lesion_detected": segmentation.detected,
            "lesion_confidence": segmentation.confidence,
            "geometry": geometry_dict,
            
            "risk_tier": risk.tier,
            "risk_tier_name": risk.tier_name,
            "risk_score": risk.risk_score,
            "urgency": risk.urgency,
            "action": risk.action,
            "escalations": escalations_list,
            
            "melanoma_probability": melanoma.probability,
            "melanoma_classification": melanoma.classification.value,
            "malignancy_classification": malignancy.classification,
            "benign_probability": malignancy.benign_probability,
            "malignant_probability": malignancy.malignant_probability,
            "primary_subtype": subtype.primary_subtype,
            "subtype_probability": subtype.primary_probability,
            
            "abcde_score": abcde.total_score,
            "abcde_criteria_met": abcde.criteria_met,
            "abcde_details": abcde_dict,
            
            "visualizations": visualizations,
            "explanation": explanation,
            
            "image_quality": validation.quality_score if validation.quality else 0,
            "analysis_confidence": melanoma.probability,
            "warnings": all_warnings
        })
    
    def _format_failure(
        self,
        error: Dict[str, Any],
        request_id: str,
        stage: str,
        stages: list,
        start_time: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Format failure response."""
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        stages_completed = [
            s.name for s in stages if s.status == "success"
        ]
        
        response = {
            "success": False,
            
            "error_code": error.get("code", "E_SYS_001"),
            "error_category": error.get("category", "system"),
            "error_title": error.get("title", "Processing Error"),
            "error_message": error.get("message", "An error occurred"),
            "error_action": error.get("action", "Please try again"),
            
            "tips": error.get("tips", []),
            "recoverable": error.get("recoverable", False),
            "retry_recommended": error.get("recoverable", False),
            
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "processing_time_ms": processing_time_ms,
            
            "stages_completed": stages_completed
        }
        
        return False, response
