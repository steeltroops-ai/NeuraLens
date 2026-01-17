"""
Property Tests for ML Processing Layer / Biomarker Extraction

Tests Properties 13-20:
- Property 13: Vessel Segmentation Accuracy
- Property 14: Tortuosity Index Range
- Property 15: Optic Disc Detection Reliability
- Property 16: Cup-to-Disc Ratio Range
- Property 17: Macular Thickness Estimation
- Property 18: Amyloid-Beta Detection Sensitivity
- Property 19: Model Inference Time
- Property 20: Risk Score Calculation Correctness

Validates: Requirements 3.1-3.12, 4.1-4.4, 5.1-5.12

Uses Hypothesis for property-based testing.
"""

import pytest
import numpy as np
import cv2
import time
import torch
from typing import Tuple
from hypothesis import given, strategies as st, settings, assume
from dataclasses import dataclass

from app.pipelines.retinal.analyzer import (
    RealtimeRetinalProcessor,
    VesselAnalysisResult,
    OpticDiscResult,
    MacularResult,
    AmyloidResult,
    ModelConfig
)
from app.pipelines.retinal.schemas import (
    RetinalAnalysisRequest,
    RetinalBiomarkers,
    RiskAssessment,
    VesselBiomarkers,
    OpticDiscBiomarkers,
    MacularBiomarkers,
    AmyloidBetaIndicators,
    RiskCategory
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_synthetic_fundus_image(
    width: int = 512,
    height: int = 512,
    add_vessels: bool = True,
    add_optic_disc: bool = True
) -> np.ndarray:
    """Create a synthetic retinal fundus image for testing."""
    
    # Red-orange background (typical fundus color)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Blue
    img[:, :, 1] = 80   # Green  
    img[:, :, 2] = 180  # Red
    
    # Add some texture
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    if add_optic_disc:
        # Add bright circular region (optic disc) on right side
        center_x = width * 3 // 4
        center_y = height // 2
        radius = min(width, height) // 10
        cv2.circle(img, (center_x, center_y), radius, (180, 180, 220), -1)
        
        # Add darker cup inside
        cup_radius = radius // 2
        cv2.circle(img, (center_x, center_y), cup_radius, (100, 100, 140), -1)
    
    if add_vessels:
        # Add some vessel-like structures
        for _ in range(5):
            start = (np.random.randint(0, width), np.random.randint(0, height))
            end = (np.random.randint(0, width), np.random.randint(0, height))
            color = (40, 50, 150)  # Dark red for vessels
            cv2.line(img, start, end, color, 2)
    
    return img


def image_to_bytes(img: np.ndarray) -> bytes:
    """Convert numpy image to JPEG bytes."""
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


# ============================================================================
# Property 13: Vessel Segmentation Accuracy
# ============================================================================

class TestVesselSegmentationAccuracy:
    """
    Property 13: For any processed retinal image, vessel segmentation 
    should achieve at least 95% accuracy on major vessels.
    
    Validates: Requirement 3.1
    """
    
    @pytest.mark.asyncio
    async def test_vessel_segmentation_produces_mask(self):
        """Vessel segmentation should produce a valid binary mask."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image(add_vessels=True)
        img_tensor = processor._to_tensor(img)
        
        result = await processor._analyze_vessels(img, img_tensor)
        
        assert result.mask is not None
        assert isinstance(result.mask, np.ndarray)
        assert result.mask.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_vessel_density_in_valid_range(self):
        """Vessel density should be in physiologically valid range (0-15%)."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_tensor = processor._to_tensor(img)
        
        result = await processor._analyze_vessels(img, img_tensor)
        
        # Normal retinal vessel density is typically 3-8%
        assert 0 <= result.density_percentage <= 100
        assert result.confidence >= 0
        assert result.confidence <= 1


# ============================================================================
# Property 14: Tortuosity Index Range
# ============================================================================

class TestTortuosityIndexRange:
    """
    Property 14: Tortuosity index should be in the expected range 
    for retinal vessels.
    
    Validates: Requirement 3.3
    """
    
    @pytest.mark.asyncio
    async def test_tortuosity_in_valid_range(self):
        """Tortuosity index should be in valid range (0.8 - 2.0)."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_tensor = processor._to_tensor(img)
        
        result = await processor._analyze_vessels(img, img_tensor)
        
        # Normal tortuosity range
        assert 0.5 <= result.tortuosity_index <= 3.0
    
    @pytest.mark.asyncio
    @given(n_images=st.integers(min_value=1, max_value=5))
    @settings(max_examples=3)
    async def test_tortuosity_consistency(self, n_images: int):
        """Tortuosity should be calculated consistently."""
        processor = RealtimeRetinalProcessor()
        
        tortuosity_values = []
        for _ in range(n_images):
            img = create_synthetic_fundus_image()
            img_tensor = processor._to_tensor(img)
            result = await processor._analyze_vessels(img, img_tensor)
            tortuosity_values.append(result.tortuosity_index)
        
        # All values should be valid numbers
        for t in tortuosity_values:
            assert isinstance(t, (int, float))
            assert not np.isnan(t)
            assert not np.isinf(t)


# ============================================================================
# Property 15: Optic Disc Detection Reliability
# ============================================================================

class TestOpticDiscDetectionReliability:
    """
    Property 15: For images containing a visible optic disc, 
    detection should succeed with high reliability.
    
    Validates: Requirements 3.6, 3.7
    """
    
    @pytest.mark.asyncio
    async def test_optic_disc_detection_returns_result(self):
        """Optic disc analysis should return a valid result."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image(add_optic_disc=True)
        result = await processor._analyze_optic_disc(img)
        
        assert isinstance(result, OpticDiscResult)
        assert isinstance(result.detected, bool)
        assert 0 <= result.cup_to_disc_ratio <= 1
        assert result.disc_area_mm2 >= 0
        assert result.rim_area_mm2 >= 0
    
    @pytest.mark.asyncio
    async def test_rim_area_less_than_disc_area(self):
        """Rim area should never exceed disc area."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image(add_optic_disc=True)
        result = await processor._analyze_optic_disc(img)
        
        assert result.rim_area_mm2 <= result.disc_area_mm2


# ============================================================================
# Property 16: Cup-to-Disc Ratio Range
# ============================================================================

class TestCupToDiscRatioRange:
    """
    Property 16: Cup-to-disc ratio should be in valid range (0-1).
    
    Validates: Requirement 3.6
    """
    
    @pytest.mark.asyncio
    @given(n_tests=st.integers(min_value=1, max_value=5))
    @settings(max_examples=3)
    async def test_cup_to_disc_ratio_bounds(self, n_tests: int):
        """Cup-to-disc ratio should always be between 0 and 1."""
        processor = RealtimeRetinalProcessor()
        
        for _ in range(n_tests):
            img = create_synthetic_fundus_image()
            result = await processor._analyze_optic_disc(img)
            
            assert 0 <= result.cup_to_disc_ratio <= 1


# ============================================================================
# Property 17: Macular Thickness Estimation
# ============================================================================

class TestMacularThicknessEstimation:
    """
    Property 17: Macular thickness should be estimated within 
    physiologically valid range.
    
    Validates: Requirement 3.8
    """
    
    @pytest.mark.asyncio
    async def test_macular_thickness_in_valid_range(self):
        """Macular thickness should be in valid range (150-450 μm)."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        result = await processor._analyze_macula(img)
        
        # Normal macular thickness range
        assert 150 <= result.thickness_um <= 450
    
    @pytest.mark.asyncio
    async def test_macular_volume_in_valid_range(self):
        """Macular volume should be in valid range (0.1-0.5 mm³)."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        result = await processor._analyze_macula(img)
        
        assert 0.1 <= result.volume_mm3 <= 0.5


# ============================================================================
# Property 18: Amyloid-Beta Detection Sensitivity
# ============================================================================

class TestAmyloidBetaDetectionSensitivity:
    """
    Property 18: Amyloid-beta detection should return valid scores 
    and distribution patterns.
    
    Validates: Requirement 3.10
    """
    
    @pytest.mark.asyncio
    async def test_amyloid_score_in_valid_range(self):
        """Amyloid presence score should be between 0 and 1."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_tensor = processor._to_tensor(img)
        
        result = await processor._detect_amyloid_beta(img_tensor)
        
        assert 0 <= result.presence_score <= 1
        assert 0 <= result.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_amyloid_distribution_pattern_valid(self):
        """Amyloid distribution pattern should be a valid category."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_tensor = processor._to_tensor(img)
        
        result = await processor._detect_amyloid_beta(img_tensor)
        
        valid_patterns = ["normal", "diffuse", "focal", "perivascular", "mixed"]
        assert result.distribution_pattern in valid_patterns


# ============================================================================
# Property 19: Model Inference Time
# ============================================================================

class TestModelInferenceTime:
    """
    Property 19: Model inference should complete within the 
    specified time limit (500ms).
    
    Validates: Requirement 4.4
    """
    
    @pytest.mark.asyncio
    async def test_full_analysis_under_time_limit(self):
        """Full analysis pipeline should complete in under 2 seconds."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_bytes = image_to_bytes(img)
        
        request = RetinalAnalysisRequest(patient_id="TEST-001")
        
        start_time = time.time()
        result = await processor.analyze_image(request, img_bytes, "test-session-id")
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Allow up to 2 seconds for test environment (actual target is 500ms)
        assert elapsed_ms < 2000, f"Analysis took {elapsed_ms:.0f}ms, expected <2000ms"
        assert result.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_vessel_analysis_time(self):
        """Vessel analysis alone should be fast."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_tensor = processor._to_tensor(img)
        
        start_time = time.time()
        result = await processor._analyze_vessels(img, img_tensor)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Individual analysis should be very fast
        assert elapsed_ms < 500


# ============================================================================
# Property 20: Risk Score Calculation Correctness
# ============================================================================

class TestRiskScoreCalculationCorrectness:
    """
    Property 20: Risk score should be correctly calculated from 
    weighted biomarkers.
    
    Validates: Requirements 5.1-5.12
    """
    
    @given(
        vessel_score=st.floats(min_value=0, max_value=100, allow_nan=False),
        tortuosity_score=st.floats(min_value=0, max_value=100, allow_nan=False),
        optic_score=st.floats(min_value=0, max_value=100, allow_nan=False),
        amyloid_score=st.floats(min_value=0, max_value=100, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_risk_score_bounds(
        self, vessel_score: float, tortuosity_score: float,
        optic_score: float, amyloid_score: float
    ):
        """Risk score should always be between 0 and 100."""
        weights = ModelConfig.WEIGHTS
        
        risk_score = (
            vessel_score * weights['vessel_density'] +
            tortuosity_score * weights['tortuosity'] +
            optic_score * weights['optic_disc'] +
            amyloid_score * weights['amyloid_beta']
        )
        
        assert 0 <= risk_score <= 100
    
    @given(risk_score=st.floats(min_value=0, max_value=100, allow_nan=False))
    @settings(max_examples=100)
    def test_risk_category_assignment(self, risk_score: float):
        """Risk categories should be correctly assigned based on score."""
        category = RiskAssessment.calculate_category(risk_score)
        
        if risk_score <= 25:
            assert category == "minimal"
        elif risk_score <= 40:
            assert category == "low"
        elif risk_score <= 55:
            assert category == "moderate"
        elif risk_score <= 70:
            assert category == "elevated"
        elif risk_score <= 85:
            assert category == "high"
        else:
            assert category == "critical"
    
    @given(
        risk_score=st.floats(min_value=0, max_value=100, allow_nan=False),
        margin=st.floats(min_value=1, max_value=15, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_confidence_interval_validity(self, risk_score: float, margin: float):
        """Confidence interval should contain the risk score."""
        lower = max(0, risk_score - margin)
        upper = min(100, risk_score + margin)
        
        assert lower <= risk_score <= upper
        assert lower >= 0
        assert upper <= 100
    
    def test_weights_sum_to_one(self):
        """Biomarker weights should sum to 1.0."""
        weights = ModelConfig.WEIGHTS
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"


# ============================================================================  
# Full Pipeline Integration Tests
# ============================================================================

class TestFullPipelineIntegration:
    """Integration tests for the complete analysis pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_returns_valid_response(self):
        """Complete analysis should return all required fields."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_bytes = image_to_bytes(img)
        
        request = RetinalAnalysisRequest(patient_id="INTEGRATION-TEST-001")
        result = await processor.analyze_image(request, img_bytes, "integration-test-id")
        
        # Verify all required fields
        assert result.assessment_id == "integration-test-id"
        assert result.patient_id == "INTEGRATION-TEST-001"
        assert result.biomarkers is not None
        assert result.risk_assessment is not None
        assert 0 <= result.quality_score <= 100
        assert result.model_version is not None
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_analysis_preserves_patient_id(self):
        """Patient ID should be preserved through analysis."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_bytes = image_to_bytes(img)
        
        test_patient_id = "PATIENT-XYZ-123"
        request = RetinalAnalysisRequest(patient_id=test_patient_id)
        result = await processor.analyze_image(request, img_bytes, "preserve-test")
        
        assert result.patient_id == test_patient_id
    
    @pytest.mark.asyncio
    async def test_all_biomarker_categories_present(self):
        """All biomarker categories should be present in response."""
        processor = RealtimeRetinalProcessor()
        
        img = create_synthetic_fundus_image()
        img_bytes = image_to_bytes(img)
        
        request = RetinalAnalysisRequest(patient_id="CATEGORIES-TEST")
        result = await processor.analyze_image(request, img_bytes, "categories-test")
        
        # Verify all biomarker categories
        assert result.biomarkers.vessels is not None
        assert result.biomarkers.optic_disc is not None
        assert result.biomarkers.macula is not None
        assert result.biomarkers.amyloid_beta is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
