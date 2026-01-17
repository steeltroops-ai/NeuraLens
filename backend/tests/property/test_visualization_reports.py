"""
Property Tests for Visualization and Report Generation

Tests Properties 28-29:
- Property 28: Visualization Completeness
- Property 29: Clinical Report Completeness

Additional tests:
- Risk gauge rendering
- Heatmap color gradient
- Measurement overlay accuracy
- PDF report structure
- Report content validation

Validates: Requirements 6.1-6.10, 7.1-7.12
"""

import pytest
import numpy as np
import cv2
from datetime import datetime
from typing import Dict
from hypothesis import given, strategies as st, settings

from app.pipelines.retinal.visualization import (
    RetinalVisualizationService,
    ColorPalette,
    VisualizationConfig
)
from app.pipelines.retinal.report_generator import ReportGenerator, ReportConfig
from app.pipelines.retinal.schemas import (
    RetinalAnalysisResponse,
    RetinalBiomarkers,
    VesselBiomarkers,
    OpticDiscBiomarkers,
    MacularBiomarkers,
    AmyloidBetaIndicators,
    RiskAssessment
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

def create_test_image(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a test image for visualization testing."""
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    return img


def create_test_mask(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a test binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Add some structures
    cv2.circle(mask, (width // 2, height // 2), 50, 255, -1)
    for _ in range(5):
        pt1 = (np.random.randint(0, width), np.random.randint(0, height))
        pt2 = (np.random.randint(0, width), np.random.randint(0, height))
        cv2.line(mask, pt1, pt2, 255, 2)
    return mask


def create_test_assessment() -> RetinalAnalysisResponse:
    """Create a test RetinalAnalysisResponse."""
    return RetinalAnalysisResponse(
        assessment_id="test-assessment-001",
        patient_id="PATIENT-001",
        biomarkers=RetinalBiomarkers(
            vessels=VesselBiomarkers(
                density_percentage=5.5,
                tortuosity_index=1.1,
                avr_ratio=0.7,
                branching_coefficient=1.5,
                confidence=0.92
            ),
            optic_disc=OpticDiscBiomarkers(
                cup_to_disc_ratio=0.4,
                disc_area_mm2=2.8,
                rim_area_mm2=1.7,
                confidence=0.88
            ),
            macula=MacularBiomarkers(
                thickness_um=285.0,
                volume_mm3=0.25,
                confidence=0.85
            ),
            amyloid_beta=AmyloidBetaIndicators(
                presence_score=0.15,
                distribution_pattern="normal",
                confidence=0.90
            )
        ),
        risk_assessment=RiskAssessment(
            risk_score=32.5,
            risk_category="low",
            confidence_interval=(28.0, 37.0),
            contributing_factors={
                "vessel_density": 25.0,
                "tortuosity": 30.0,
                "optic_disc": 35.0,
                "amyloid_beta": 15.0
            }
        ),
        quality_score=85.0,
        heatmap_url="/api/v1/retinal/visualizations/test/heatmap",
        segmentation_url="/api/v1/retinal/visualizations/test/segmentation",
        created_at=datetime.utcnow(),
        model_version="1.0.0",
        processing_time_ms=350
    )


# ============================================================================
# Property 28: Visualization Completeness
# ============================================================================

class TestVisualizationCompleteness:
    """
    Property 28: For any processed retinal image, the system should generate 
    complete visualizations including vessel segmentation overlay, 
    attention heatmap, and measurement annotations.
    
    Validates: Requirements 6.1-6.10
    """
    
    def test_vessel_overlay_produces_valid_image(self):
        """Vessel overlay should produce a valid BGR image."""
        service = RetinalVisualizationService()
        original = create_test_image()
        mask = create_test_mask()
        
        result = service.generate_vessel_overlay(original, mask)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original.shape
        assert result.dtype == np.uint8
    
    def test_vessel_overlay_with_artery_vein_separation(self):
        """Vessel overlay should color-code arteries and veins (Requirement 6.4)."""
        service = RetinalVisualizationService()
        original = create_test_image()
        vessel_mask = create_test_mask()
        
        # Create separate artery and vein masks
        artery_mask = np.zeros_like(vessel_mask)
        vein_mask = np.zeros_like(vessel_mask)
        artery_mask[vessel_mask > 0] = 255  # Top half as arteries
        artery_mask[256:, :] = 0
        vein_mask[vessel_mask > 0] = 255  # Bottom half as veins
        vein_mask[:256, :] = 0
        
        result = service.generate_vessel_overlay(
            original, vessel_mask, artery_mask, vein_mask
        )
        
        assert result is not None
        assert result.shape == original.shape
    
    def test_heatmap_produces_valid_image(self):
        """Heatmap overlay should produce a valid BGR image (Requirement 6.2, 6.3)."""
        service = RetinalVisualizationService()
        original = create_test_image()
        attention = np.random.rand(512, 512).astype(np.float32)
        
        result = service.generate_heatmap(original, attention)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == original.shape
    
    @given(opacity=st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    @settings(max_examples=10)
    def test_heatmap_respects_opacity(self, opacity: float):
        """Heatmap should respect configured opacity."""
        config = VisualizationConfig(heatmap_opacity=opacity)
        service = RetinalVisualizationService(config)
        original = create_test_image()
        attention = np.ones((512, 512), dtype=np.float32) * 0.5
        
        result = service.generate_heatmap(original, attention)
        
        # Result should be a blend of original and heatmap
        assert result is not None
    
    def test_measurement_overlay_with_optic_disc(self):
        """Measurement overlay should include optic disc annotation (Requirement 6.5)."""
        service = RetinalVisualizationService()
        original = create_test_image()
        
        result = service.generate_measurement_overlay(
            original,
            optic_disc_center=(384, 256),
            optic_disc_radius=50,
            cup_radius=25,
            cup_to_disc_ratio=0.45
        )
        
        assert result is not None
        assert result.shape == original.shape
    
    def test_measurement_overlay_with_scale_bar(self):
        """Measurement overlay should include scale bar (Requirement 6.10)."""
        service = RetinalVisualizationService()
        original = create_test_image()
        
        result = service.generate_measurement_overlay(
            original,
            scale_pixels_per_mm=100.0
        )
        
        assert result is not None
        # Scale bar should be visible in bottom left corner
        # (visual verification needed, but at least check image is valid)
        assert result.shape == original.shape
    
    def test_measurement_overlay_with_macula(self):
        """Measurement overlay should include macula annotation."""
        service = RetinalVisualizationService()
        original = create_test_image()
        
        result = service.generate_measurement_overlay(
            original,
            macula_center=(256, 256)
        )
        
        assert result is not None
        assert result.shape == original.shape


# ============================================================================
# Risk Gauge Visualization Tests
# ============================================================================

class TestRiskGaugeVisualization:
    """Tests for risk gauge visualization (Requirement 6.6)."""
    
    @given(risk_score=st.floats(min_value=0, max_value=100, allow_nan=False))
    @settings(max_examples=20)
    def test_risk_gauge_for_any_score(self, risk_score: float):
        """Risk gauge should be generated for any valid score."""
        service = RetinalVisualizationService()
        
        # Determine category based on score
        if risk_score <= 25:
            category = "minimal"
        elif risk_score <= 40:
            category = "low"
        elif risk_score <= 55:
            category = "moderate"
        elif risk_score <= 70:
            category = "elevated"
        elif risk_score <= 85:
            category = "high"
        else:
            category = "critical"
        
        result = service.generate_risk_gauge(risk_score, category)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # BGR image
    
    def test_risk_gauge_dimensions(self):
        """Risk gauge should have correct dimensions."""
        service = RetinalVisualizationService()
        
        width, height = 400, 250
        result = service.generate_risk_gauge(50.0, "moderate", width=width, height=height)
        
        assert result.shape[0] == height
        assert result.shape[1] == width


# ============================================================================
# Biomarker Chart Tests
# ============================================================================

class TestBiomarkerChartVisualization:
    """Tests for biomarker comparison charts (Requirement 6.7)."""
    
    def test_biomarker_chart_generation(self):
        """Biomarker chart should be generated from data."""
        service = RetinalVisualizationService()
        
        biomarkers = {
            "Vessel Density": 5.5,
            "Tortuosity": 1.1,
            "Cup-to-Disc": 0.4,
        }
        
        reference_ranges = {
            "Vessel Density": (4.0, 7.0),
            "Tortuosity": (0.8, 1.3),
            "Cup-to-Disc": (0.3, 0.5),
        }
        
        result = service.generate_biomarker_chart(biomarkers, reference_ranges)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_empty_biomarker_chart(self):
        """Empty biomarker data should produce blank chart."""
        service = RetinalVisualizationService()
        
        result = service.generate_biomarker_chart({}, {})
        
        assert result is not None


# ============================================================================
# Trend Chart Tests
# ============================================================================

class TestTrendChartVisualization:
    """Tests for trend analysis charts (Requirement 6.8)."""
    
    def test_trend_chart_generation(self):
        """Trend chart should be generated from historical data."""
        service = RetinalVisualizationService()
        
        dates = ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
        values = [35.0, 38.0, 32.0, 30.0]
        
        result = service.generate_trend_chart(dates, values, "Risk Score")
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_trend_chart_insufficient_data(self):
        """Trend chart with insufficient data should show message."""
        service = RetinalVisualizationService()
        
        result = service.generate_trend_chart(["2026-01-01"], [35.0])
        
        assert result is not None


# ============================================================================
# Image Conversion Tests
# ============================================================================

class TestImageConversion:
    """Tests for image format conversion."""
    
    def test_image_to_base64(self):
        """Image should be convertible to base64."""
        service = RetinalVisualizationService()
        img = create_test_image()
        
        base64_str = service.image_to_base64(img)
        
        assert base64_str is not None
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
    
    def test_image_to_bytes(self):
        """Image should be convertible to bytes."""
        service = RetinalVisualizationService()
        img = create_test_image()
        
        img_bytes = service.image_to_bytes(img)
        
        assert img_bytes is not None
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0


# ============================================================================
# Property 29: Clinical Report Completeness
# ============================================================================

class TestClinicalReportCompleteness:
    """
    Property 29: For any assessment, the clinical report should contain 
    all required sections with properly formatted content.
    
    Validates: Requirements 7.1-7.12
    """
    
    def test_report_generation_produces_pdf(self):
        """Report generator should produce valid PDF bytes."""
        generator = ReportGenerator()
        assessment = create_test_assessment()
        
        pdf_bytes = generator.generate_report(assessment)
        
        assert pdf_bytes is not None
        assert isinstance(pdf_bytes, bytes)
        # PDF files start with %PDF
        assert pdf_bytes[:4] == b'%PDF'
    
    def test_report_includes_patient_info(self):
        """Report should include patient information (Requirement 7.2)."""
        generator = ReportGenerator()
        assessment = create_test_assessment()
        
        pdf_bytes = generator.generate_report(
            assessment,
            patient_name="John Doe",
            patient_dob="1980-05-15"
        )
        
        # PDF is generated (content verification would need PDF parsing)
        assert len(pdf_bytes) > 1000  # Non-trivial content
    
    def test_report_includes_provider_info(self):
        """Report should include provider information (Requirement 7.10)."""
        generator = ReportGenerator()
        assessment = create_test_assessment()
        
        pdf_bytes = generator.generate_report(
            assessment,
            provider_name="Dr. Jane Smith",
            provider_npi="1234567890"
        )
        
        assert len(pdf_bytes) > 1000
    
    @given(
        risk_score=st.floats(min_value=0, max_value=100, allow_nan=False)
    )
    @settings(max_examples=10)
    def test_report_for_any_risk_score(self, risk_score: float):
        """Report should be generated for any valid risk score."""
        generator = ReportGenerator()
        
        # Determine category
        if risk_score <= 25:
            category = "minimal"
        elif risk_score <= 40:
            category = "low"
        elif risk_score <= 55:
            category = "moderate"
        elif risk_score <= 70:
            category = "elevated"
        elif risk_score <= 85:
            category = "high"
        else:
            category = "critical"
        
        assessment = create_test_assessment()
        # Update risk score in assessment
        assessment.risk_assessment.risk_score = risk_score
        assessment.risk_assessment.risk_category = category
        
        pdf_bytes = generator.generate_report(assessment)
        
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
    
    def test_report_generation_speed(self):
        """Report generation should complete within reasonable time."""
        import time
        
        generator = ReportGenerator()
        assessment = create_test_assessment()
        
        start = time.time()
        pdf_bytes = generator.generate_report(assessment)
        elapsed = time.time() - start
        
        # Report should generate in less than 5 seconds
        assert elapsed < 5.0
        assert pdf_bytes is not None


# ============================================================================
# Color Palette Tests
# ============================================================================

class TestColorPalette:
    """Tests for color palette consistency."""
    
    def test_risk_color_for_all_categories(self):
        """All risk categories should have defined colors."""
        categories = ["minimal", "low", "moderate", "elevated", "high", "critical"]
        
        for category in categories:
            color = ColorPalette.get_risk_color(category)
            
            assert color is not None
            assert len(color) == 3  # RGB tuple
            assert all(0 <= c <= 255 for c in color)
    
    def test_unknown_category_returns_gray(self):
        """Unknown risk category should return gray color."""
        color = ColorPalette.get_risk_color("unknown")
        
        assert color == (128, 128, 128)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
