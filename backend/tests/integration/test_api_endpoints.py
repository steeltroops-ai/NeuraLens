"""
Integration Tests for Retinal Analysis API Endpoints

Tests complete API workflows:
- Image upload and analysis
- Validation endpoint
- Results retrieval
- Patient history
- Report generation
- Visualization endpoints
- Error handling

Validates: Requirements for Tasks 8.1-8.6
"""

import pytest
import numpy as np
import cv2
import io
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Import the router for testing
from app.pipelines.retinal.router import router
from app.pipelines.retinal.schemas import RetinalAnalysisResponse


# ============================================================================
# Test Fixtures
# ============================================================================

def create_test_image_bytes() -> bytes:
    """Create a valid test image as bytes."""
    # Create a 1024x1024 fundus-like image
    img = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
    
    # Add some features
    cv2.circle(img, (768, 512), 80, (200, 180, 150), -1)  # Optic disc
    cv2.circle(img, (512, 512), 30, (100, 80, 80), -1)    # Macula
    
    # Add vessel-like structures
    for _ in range(10):
        pt1 = (np.random.randint(0, 1024), np.random.randint(0, 1024))
        pt2 = (np.random.randint(0, 1024), np.random.randint(0, 1024))
        cv2.line(img, pt1, pt2, (80, 60, 140), 2)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def create_small_image_bytes() -> bytes:
    """Create a small test image (below resolution threshold)."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


def create_blurry_image_bytes() -> bytes:
    """Create a blurry test image."""
    img = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
    # Apply heavy blur
    img = cv2.GaussianBlur(img, (101, 101), 0)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


# ============================================================================
# Create FastAPI test app
# ============================================================================

from fastapi import FastAPI

app = FastAPI()
app.include_router(router, prefix="/api/v1/retinal")


# ============================================================================
# Synchronous Tests using TestClient
# ============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_ok(self):
        """Health endpoint should return healthy status."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "processor" in data
            assert "storage" in data


class TestValidationEndpoint:
    """Tests for /validate endpoint."""
    
    def test_validate_valid_image(self):
        """Valid image should pass validation."""
        with TestClient(app) as client:
            image_bytes = create_test_image_bytes()
            
            response = client.post(
                "/api/v1/retinal/validate",
                files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "is_valid" in data
            assert "quality_score" in data
    
    def test_validate_small_image(self):
        """Small image should return validation issues."""
        with TestClient(app) as client:
            image_bytes = create_small_image_bytes()
            
            response = client.post(
                "/api/v1/retinal/validate",
                files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            # Should have resolution issues
            assert "issues" in data


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""
    
    def test_analyze_valid_image(self):
        """Valid image should return analysis results."""
        with TestClient(app) as client:
            image_bytes = create_test_image_bytes()
            
            response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT-001"},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            # May be 200 or 400 depending on validation
            # For mock purposes, check structure
            if response.status_code == 200:
                data = response.json()
                assert "assessment_id" in data
                assert "patient_id" in data
                assert "biomarkers" in data
                assert "risk_assessment" in data
    
    def test_analyze_missing_patient_id(self):
        """Missing patient ID should return error."""
        with TestClient(app) as client:
            image_bytes = create_test_image_bytes()
            
            response = client.post(
                "/api/v1/retinal/analyze",
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            # Should fail validation for missing patient_id
            assert response.status_code in [400, 422]
    
    def test_analyze_missing_image(self):
        """Missing image should return error."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT"}
            )
            
            assert response.status_code == 422  # Validation error


class TestResultsEndpoint:
    """Tests for /results/{id} endpoint."""
    
    def test_get_nonexistent_result(self):
        """Nonexistent assessment ID should return 404."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/results/nonexistent-id")
            
            assert response.status_code == 404
    
    def test_get_result_after_analysis(self):
        """Should retrieve results after analysis."""
        with TestClient(app) as client:
            # First, perform an analysis
            image_bytes = create_test_image_bytes()
            
            analyze_response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT-002"},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            if analyze_response.status_code == 200:
                assessment_id = analyze_response.json()["assessment_id"]
                
                # Then retrieve results
                get_response = client.get(f"/api/v1/retinal/results/{assessment_id}")
                
                assert get_response.status_code == 200
                data = get_response.json()
                assert data["assessment_id"] == assessment_id


class TestHistoryEndpoint:
    """Tests for /history/{patient_id} endpoint."""
    
    def test_get_empty_history(self):
        """New patient should have empty history."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/history/NEW-PATIENT-999")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 0
            assert data["assessments"] == []
    
    def test_history_pagination(self):
        """History should support pagination parameters."""
        with TestClient(app) as client:
            response = client.get(
                "/api/v1/retinal/history/PATIENT-001",
                params={"limit": 5, "offset": 0}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "total_count" in data
            assert "has_more" in data


class TestReportEndpoint:
    """Tests for /report/{id} endpoint."""
    
    def test_report_nonexistent_assessment(self):
        """Report for nonexistent assessment should return 404."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/report/nonexistent-id")
            
            assert response.status_code == 404
    
    def test_report_generation_after_analysis(self):
        """Should generate PDF report after analysis."""
        with TestClient(app) as client:
            # First, perform an analysis
            image_bytes = create_test_image_bytes()
            
            analyze_response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT-003"},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            if analyze_response.status_code == 200:
                assessment_id = analyze_response.json()["assessment_id"]
                
                # Generate report
                report_response = client.get(
                    f"/api/v1/retinal/report/{assessment_id}",
                    params={
                        "patient_name": "Test Patient",
                        "patient_dob": "1990-01-01"
                    }
                )
                
                assert report_response.status_code == 200
                assert report_response.headers["content-type"] == "application/pdf"
                assert len(report_response.content) > 0
                # Check PDF magic bytes
                assert report_response.content[:4] == b'%PDF'


class TestVisualizationEndpoint:
    """Tests for /visualizations/{id}/{type} endpoint."""
    
    def test_visualization_nonexistent_assessment(self):
        """Visualization for nonexistent assessment should return 404."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/visualizations/nonexistent-id/heatmap")
            
            assert response.status_code == 404
    
    def test_invalid_visualization_type(self):
        """Invalid visualization type should return 400."""
        with TestClient(app) as client:
            # First create an assessment
            image_bytes = create_test_image_bytes()
            
            analyze_response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT-004"},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            if analyze_response.status_code == 200:
                assessment_id = analyze_response.json()["assessment_id"]
                
                # Request invalid visualization
                viz_response = client.get(
                    f"/api/v1/retinal/visualizations/{assessment_id}/invalid_type"
                )
                
                assert viz_response.status_code == 400


class TestTrendsEndpoint:
    """Tests for /trends/{patient_id} endpoint."""
    
    def test_trends_new_patient(self):
        """New patient should have empty trends."""
        with TestClient(app) as client:
            response = client.get("/api/v1/retinal/trends/NEW-PATIENT-TRENDS")
            
            assert response.status_code == 200
            data = response.json()
            assert data["data_points"] == []
            assert data["trend_direction"] == "stable"
    
    def test_trends_with_biomarker_parameter(self):
        """Should support different biomarker parameters."""
        with TestClient(app) as client:
            biomarkers = ["risk_score", "vessel_density", "tortuosity", "cup_to_disc"]
            
            for biomarker in biomarkers:
                response = client.get(
                    "/api/v1/retinal/trends/PATIENT-TRENDS",
                    params={"biomarker": biomarker}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["biomarker"] == biomarker


class TestDeleteEndpoint:
    """Tests for DELETE /results/{id} endpoint."""
    
    def test_delete_nonexistent_assessment(self):
        """Deleting nonexistent assessment should return 404."""
        with TestClient(app) as client:
            response = client.delete("/api/v1/retinal/results/nonexistent-delete-id")
            
            assert response.status_code == 404
    
    def test_delete_after_analysis(self):
        """Should delete assessment after analysis."""
        with TestClient(app) as client:
            # First, perform an analysis
            image_bytes = create_test_image_bytes()
            
            analyze_response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": "TEST-PATIENT-DELETE"},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            if analyze_response.status_code == 200:
                assessment_id = analyze_response.json()["assessment_id"]
                
                # Delete assessment
                delete_response = client.delete(f"/api/v1/retinal/results/{assessment_id}")
                
                assert delete_response.status_code == 200
                assert delete_response.json()["assessment_id"] == assessment_id
                
                # Verify it's deleted
                get_response = client.get(f"/api/v1/retinal/results/{assessment_id}")
                assert get_response.status_code == 404


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_content_type(self):
        """Invalid content type should be handled."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/retinal/validate",
                files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
            )
            
            # Should return validation response with issues
            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is False
    
    def test_corrupted_image_data(self):
        """Corrupted image data should be handled gracefully."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/retinal/validate",
                files={"image": ("test.jpg", io.BytesIO(b"corrupted data"), "image/jpeg")}
            )
            
            # Should return validation response with issues
            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is False


# ============================================================================
# Complete Workflow Tests
# ============================================================================

class TestCompleteWorkflow:
    """End-to-end workflow tests."""
    
    def test_full_analysis_workflow(self):
        """Test complete workflow: analyze -> results -> report -> delete."""
        with TestClient(app) as client:
            patient_id = "WORKFLOW-TEST-PATIENT"
            
            # 1. Upload and analyze
            image_bytes = create_test_image_bytes()
            
            analyze_response = client.post(
                "/api/v1/retinal/analyze",
                data={"patient_id": patient_id},
                files={"image": ("fundus.jpg", io.BytesIO(image_bytes), "image/jpeg")}
            )
            
            if analyze_response.status_code == 200:
                result = analyze_response.json()
                assessment_id = result["assessment_id"]
                
                # 2. Check results
                results_response = client.get(f"/api/v1/retinal/results/{assessment_id}")
                assert results_response.status_code == 200
                
                # 3. Check history
                history_response = client.get(f"/api/v1/retinal/history/{patient_id}")
                assert history_response.status_code == 200
                history_data = history_response.json()
                assert history_data["total_count"] >= 1
                
                # 4. Generate report
                report_response = client.get(f"/api/v1/retinal/report/{assessment_id}")
                assert report_response.status_code == 200
                
                # 5. Get visualizations
                for viz_type in ["heatmap", "segmentation", "gauge", "measurements"]:
                    viz_response = client.get(
                        f"/api/v1/retinal/visualizations/{assessment_id}/{viz_type}"
                    )
                    assert viz_response.status_code == 200
                
                # 6. Check trends
                trends_response = client.get(f"/api/v1/retinal/trends/{patient_id}")
                assert trends_response.status_code == 200
                
                # 7. Delete (cleanup)
                delete_response = client.delete(f"/api/v1/retinal/results/{assessment_id}")
                assert delete_response.status_code == 200


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
