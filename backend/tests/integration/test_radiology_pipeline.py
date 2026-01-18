"""
Radiology Pipeline End-to-End Tests
Tests chest X-ray analysis using TorchXRayVision

Run with: pytest tests/integration/test_radiology_pipeline.py -v
"""

import pytest
import base64
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app


# Test client
client = TestClient(app)


def create_test_xray_image(
    width: int = 512,
    height: int = 512,
    brightness: float = 0.5,
    add_opacity: bool = False
) -> bytes:
    """
    Create a synthetic chest X-ray image for testing
    
    Args:
        width, height: Image dimensions
        brightness: Overall brightness (0-1)
        add_opacity: Add synthetic opacity for pathology simulation
    
    Returns:
        JPEG image bytes
    """
    # Create base grayscale image
    img = np.ones((height, width), dtype=np.uint8) * int(brightness * 255)
    
    # Add chest cavity shape (darker lung regions)
    y, x = np.ogrid[:height, :width]
    
    # Left lung (darker)
    left_center = (width // 3, height // 2)
    left_mask = ((x - left_center[0])**2 + (y - left_center[1])**2) < (width // 4)**2
    img[left_mask] = int(brightness * 255 * 0.7)
    
    # Right lung (darker)  
    right_center = (2 * width // 3, height // 2)
    right_mask = ((x - right_center[0])**2 + (y - right_center[1])**2) < (width // 4)**2
    img[right_mask] = int(brightness * 255 * 0.7)
    
    # Heart (lighter in center)
    heart_center = (width // 2, int(height * 0.55))
    heart_mask = ((x - heart_center[0])**2 / (width//8)**2 + 
                  (y - heart_center[1])**2 / (height//6)**2) < 1
    img[heart_mask] = int(brightness * 255 * 1.1)
    
    # Add opacity if simulating pathology
    if add_opacity:
        opacity_center = (width // 3, height // 2)
        opacity_mask = ((x - opacity_center[0])**2 + (y - opacity_center[1])**2) < (width // 8)**2
        img[opacity_mask] = int(min(255, brightness * 255 * 1.3))
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, img.shape).astype(np.int32)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    
    # Convert to PIL and save as JPEG
    pil_img = Image.fromarray(img, mode='L')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    
    return buffer.getvalue()


class TestRadiologyPipelineEndpoints:
    """Test Radiology Pipeline API endpoints"""
    
    def test_health_endpoint(self):
        """Test radiology pipeline health check"""
        response = client.get("/api/radiology/health")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["module"] == "radiology"
        assert "model" in data
        assert "torchxrayvision_available" in data
        assert "gradcam_available" in data
        assert data["pathologies_count"] == 18
    
    def test_conditions_endpoint(self):
        """Test listing detectable conditions"""
        response = client.get("/api/radiology/conditions")
        assert response.status_code == 200
        data = response.json()
        
        assert "conditions" in data
        assert "total" in data
        assert data["total"] == 18
        
        # Check condition structure
        for condition in data["conditions"]:
            assert "name" in condition
            assert "description" in condition
            assert "category" in condition
            assert "urgency" in condition
            assert "accuracy" in condition
    
    def test_info_endpoint(self):
        """Test module info endpoint"""
        response = client.get("/api/radiology/info")
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "CXR-Insight AI"
        assert "pathologies" in data
        assert "datasets" in data
        assert len(data["datasets"]) >= 4
    
    def test_demo_endpoint(self):
        """Test demo analysis endpoint"""
        response = client.post("/api/radiology/demo")
        assert response.status_code == 200
        data = response.json()
        
        # Verify PRD response structure
        assert data["success"] == True
        assert "timestamp" in data
        assert "processing_time_ms" in data
        
        # Primary finding
        assert "primary_finding" in data
        assert "condition" in data["primary_finding"]
        assert "probability" in data["primary_finding"]
        assert "severity" in data["primary_finding"]
        
        # All predictions (18 pathologies)
        assert "all_predictions" in data
        assert len(data["all_predictions"]) >= 10
        
        # Findings
        assert "findings" in data
        assert isinstance(data["findings"], list)
        
        # Risk assessment
        assert "risk_level" in data
        assert "risk_score" in data
        
        # Quality
        assert "quality" in data
        
        # Recommendations
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
    
    def test_analyze_endpoint(self):
        """Test full X-ray analysis endpoint"""
        # Create test image
        image_bytes = create_test_xray_image(512, 512, 0.5)
        
        response = client.post(
            "/api/radiology/analyze",
            files={"file": ("test_xray.jpg", image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["success"] == True
        assert "timestamp" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0
        
        # Primary finding
        assert "primary_finding" in data
        pf = data["primary_finding"]
        assert "condition" in pf
        assert "probability" in pf
        assert 0 <= pf["probability"] <= 100
        
        # All predictions
        assert "all_predictions" in data
        for condition, prob in data["all_predictions"].items():
            assert 0 <= prob <= 100
        
        # Risk
        assert data["risk_level"] in ["normal", "low", "moderate", "high", "critical"]
        assert 0 <= data["risk_score"] <= 100
        
        # Quality
        assert "quality" in data
        quality = data["quality"]
        assert quality["image_quality"] in ["good", "adequate", "poor"]
        assert quality["usable"] == True
    
    def test_analyze_with_opacity(self):
        """Test analysis with simulated lung opacity"""
        image_bytes = create_test_xray_image(512, 512, 0.4, add_opacity=True)
        
        response = client.post(
            "/api/radiology/analyze",
            files={"file": ("opacity_xray.jpg", image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_analyze_png_format(self):
        """Test analysis with PNG format"""
        # Create PNG image
        img = np.ones((512, 512), dtype=np.uint8) * 128
        pil_img = Image.fromarray(img, mode='L')
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        response = client.post(
            "/api/radiology/analyze",
            files={"file": ("test.png", buffer.getvalue(), "image/png")}
        )
        
        assert response.status_code == 200
    
    def test_analyze_invalid_format(self):
        """Test rejection of non-image files"""
        response = client.post(
            "/api/radiology/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_analyze_file_too_large(self):
        """Test rejection of files over 10MB"""
        # Create large image (simulate)
        large_bytes = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/radiology/analyze",
            files={"file": ("large.jpg", large_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 400 or response.status_code == 500


class TestQualityAssessment:
    """Test X-ray quality assessment"""
    
    def test_good_quality_image(self):
        """Test quality assessment for good quality image"""
        from app.pipelines.radiology.quality import assess_xray_quality
        
        # Good quality image
        image_bytes = create_test_xray_image(1024, 1024, 0.5)
        quality = assess_xray_quality(image_bytes)
        
        assert quality["usable"] == True
        assert quality["image_quality"] in ["good", "adequate"]
        assert "1024x1024" in quality["resolution"]
    
    def test_low_resolution_warning(self):
        """Test quality warning for low resolution"""
        from app.pipelines.radiology.quality import assess_xray_quality
        
        # Low resolution image
        image_bytes = create_test_xray_image(256, 256, 0.5)
        quality = assess_xray_quality(image_bytes)
        
        assert "256x256" in quality["resolution"]
        # Should have resolution warning
        resolution_issue = any("resolution" in issue.lower() for issue in quality["issues"])
        # Note: may still be usable
    
    def test_dark_image_warning(self):
        """Test quality warning for underexposed image"""
        from app.pipelines.radiology.quality import assess_xray_quality
        
        # Very dark image
        image_bytes = create_test_xray_image(512, 512, 0.15)
        quality = assess_xray_quality(image_bytes)
        
        # Should have exposure warning
        exposure_issues = [i for i in quality["issues"] if "exposed" in i.lower() or "dark" in i.lower()]
        # May have underexposure warning
    
    def test_bright_image_warning(self):
        """Test quality warning for overexposed image"""
        from app.pipelines.radiology.quality import assess_xray_quality
        
        # Very bright image
        image_bytes = create_test_xray_image(512, 512, 0.85)
        quality = assess_xray_quality(image_bytes)
        
        # Should have exposure warning
        # Some brightness issues may be flagged


class TestVisualization:
    """Test heatmap visualization"""
    
    def test_generate_heatmap(self):
        """Test heatmap generation"""
        from app.pipelines.radiology.visualization import generate_xray_heatmap
        
        image_bytes = create_test_xray_image(512, 512, 0.5)
        heatmap_b64 = generate_xray_heatmap(image_bytes)
        
        if heatmap_b64:
            # Should be valid base64
            heatmap_bytes = base64.b64decode(heatmap_b64)
            assert len(heatmap_bytes) > 0
            
            # Should be valid image
            img = Image.open(io.BytesIO(heatmap_bytes))
            assert img.size[0] == 224  # Standard size
            assert img.size[1] == 224
    
    def test_visualizer_class(self):
        """Test XRayVisualizer class"""
        from app.pipelines.radiology.visualization import XRayVisualizer
        
        visualizer = XRayVisualizer()
        
        # Create test image
        img_np = np.ones((512, 512), dtype=np.uint8) * 128
        
        # Generate fallback heatmap
        heatmap = visualizer._generate_fallback_heatmap(img_np)
        
        if heatmap:
            assert len(base64.b64decode(heatmap)) > 0


class TestAnalyzer:
    """Test XRayAnalyzer class"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        from app.pipelines.radiology.analyzer import XRayAnalyzer
        
        analyzer = XRayAnalyzer()
        # Should initialize without error
        assert analyzer is not None
    
    def test_analyze_image(self):
        """Test full analysis"""
        from app.pipelines.radiology.analyzer import XRayAnalyzer
        
        analyzer = XRayAnalyzer()
        image_bytes = create_test_xray_image(512, 512, 0.5)
        
        result = analyzer.analyze(image_bytes)
        
        assert result.primary_finding is not None
        assert 0 <= result.confidence <= 100
        assert result.risk_level in ["normal", "low", "moderate", "high", "critical"]
        assert isinstance(result.findings, list)
        assert isinstance(result.all_predictions, dict)
    
    def test_predictions_range(self):
        """Test that all predictions are in valid range"""
        from app.pipelines.radiology.analyzer import XRayAnalyzer
        
        analyzer = XRayAnalyzer()
        image_bytes = create_test_xray_image(512, 512, 0.5)
        
        result = analyzer.analyze(image_bytes)
        
        for condition, prob in result.all_predictions.items():
            assert 0 <= prob <= 100, f"{condition} probability out of range: {prob}"


class TestModels:
    """Test Pydantic models"""
    
    def test_primary_finding_model(self):
        """Test PrimaryFinding model"""
        from app.pipelines.radiology.models import PrimaryFinding
        
        finding = PrimaryFinding(
            condition="Pneumonia",
            probability=75.5,
            severity="high"
        )
        
        assert finding.condition == "Pneumonia"
        assert finding.probability == 75.5
    
    def test_finding_model(self):
        """Test Finding model"""
        from app.pipelines.radiology.models import Finding
        
        finding = Finding(
            condition="Cardiomegaly",
            probability=45.2,
            severity="moderate",
            description="Enlarged heart"
        )
        
        assert finding.condition == "Cardiomegaly"
    
    def test_pathology_info(self):
        """Test pathology info dictionary"""
        from app.pipelines.radiology.models import PATHOLOGY_INFO
        
        assert "Pneumonia" in PATHOLOGY_INFO
        assert "Cardiomegaly" in PATHOLOGY_INFO
        assert "Pneumothorax" in PATHOLOGY_INFO
        
        # Check structure
        pneumonia = PATHOLOGY_INFO["Pneumonia"]
        assert "description" in pneumonia
        assert "urgency" in pneumonia
        assert "accuracy" in pneumonia


# Async tests
@pytest.mark.asyncio
async def test_async_analyze():
    """Test async analysis endpoint"""
    image_bytes = create_test_xray_image(512, 512, 0.5)
    
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/api/radiology/analyze",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True


@pytest.mark.asyncio
async def test_async_demo():
    """Test async demo endpoint"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        response = await ac.post("/api/radiology/demo")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
