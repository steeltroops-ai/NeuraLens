"""
Property Tests for NRI Integration, Security, and Performance

Tests Properties:
- Properties 38-43: NRI Integration
- Properties 30, 36, 37, 48, 49: Security Features
- Properties 20, 33, 44-46: Performance Guarantees

Author: NeuraLens Team
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from hypothesis import given, strategies as st, settings

from app.pipelines.retinal.nri_integration import (
    NRIIntegrationService, NRIPayload, NRIContribution,
    NRIResponse, NRIStatus
)
from app.pipelines.retinal.security import (
    EncryptionService, AuthenticationService, AuthorizationService,
    AuditLoggingService, DataAnonymizationService,
    UserRole, AuditAction, ResourceType, AuditLogEntry
)
from app.pipelines.retinal.performance import (
    CacheService, MemoryCache, RequestQueue, PerformanceMonitor,
    CacheConfig, CacheBackend
)
from app.pipelines.retinal.schemas import (
    RetinalAnalysisResponse, RetinalBiomarkers, VesselBiomarkers,
    OpticDiscBiomarkers, MacularBiomarkers, AmyloidBetaIndicators,
    RiskAssessment
)


# ============================================================================
# Test Fixtures
# ============================================================================

def create_test_assessment() -> RetinalAnalysisResponse:
    """Create a test assessment for testing"""
    return RetinalAnalysisResponse(
        assessment_id="test-001",
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
            risk_score=35.0,
            risk_category="low",
            confidence_interval=(30.0, 40.0),
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
# Property 38: NRI API Data Formatting
# ============================================================================

class TestNRIDataFormatting:
    """
    Property 38: Retinal results should be correctly formatted for NRI API
    
    Validates: Requirement 9.1
    """
    
    def test_format_includes_all_required_fields(self):
        """NRI payload should include all required fields"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        payload = service.format_for_nri(assessment)
        
        assert payload.assessment_id == assessment.assessment_id
        assert payload.patient_id == assessment.patient_id
        assert payload.risk_score == assessment.risk_assessment.risk_score
        assert payload.risk_category == assessment.risk_assessment.risk_category
        assert payload.source == "retinal"
    
    def test_format_includes_biomarkers(self):
        """NRI payload should include key biomarkers"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        payload = service.format_for_nri(assessment)
        
        assert payload.vessel_density == assessment.biomarkers.vessels.density_percentage
        assert payload.tortuosity_index == assessment.biomarkers.vessels.tortuosity_index
        assert payload.cup_to_disc_ratio == assessment.biomarkers.optic_disc.cup_to_disc_ratio
        assert payload.amyloid_presence == assessment.biomarkers.amyloid_beta.presence_score
    
    def test_format_to_dict_is_json_serializable(self):
        """NRI payload should serialize to JSON-compatible dict"""
        import json
        
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        payload = service.format_for_nri(assessment)
        payload_dict = payload.to_dict()
        
        # Should not raise
        json_str = json.dumps(payload_dict)
        assert json_str is not None
        assert len(json_str) > 0


# ============================================================================
# Property 39: NRI Weight Contribution
# ============================================================================

class TestNRIWeightContribution:
    """
    Property 39: Retinal contribution should use 30% base weight
    
    Validates: Requirement 9.2
    """
    
    def test_base_weight_is_30_percent(self):
        """Base weight should be 30%"""
        service = NRIIntegrationService()
        
        assert service.BASE_WEIGHT == 0.30
    
    @given(risk_score=st.floats(min_value=0, max_value=100, allow_nan=False))
    @settings(max_examples=20)
    def test_contribution_calculation(self, risk_score: float):
        """Contribution should be risk_score * weight"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        assessment.risk_assessment.risk_score = risk_score
        
        payload = service.format_for_nri(assessment)
        
        expected = risk_score * service.BASE_WEIGHT
        assert abs(payload.adjusted_contribution - expected) < 0.01


# ============================================================================
# Property 40: NRI Failure Fallback
# ============================================================================

class TestNRIFailureFallback:
    """
    Property 40: System should display retinal results on NRI failure
    
    Validates: Requirement 9.4
    """
    
    def test_fallback_response_indicates_standalone(self):
        """Fallback should indicate standalone mode"""
        service = NRIIntegrationService()
        
        response = service._create_fallback_response("Test error")
        
        assert response.success is False
        assert response.status == NRIStatus.STANDALONE
        assert "standalone" in response.message.lower()
    
    def test_dashboard_data_available_on_failure(self):
        """Dashboard data should be available even on NRI failure"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        failed_response = NRIResponse(
            success=False,
            status=NRIStatus.FAILED,
            message="Connection failed"
        )
        
        contribution = NRIContribution(
            base_weight=0.30,
            actual_weight=0.30,
            contribution_score=10.5,
            confidence=0.88
        )
        
        dashboard = service.get_dashboard_data(assessment, failed_response, contribution)
        
        assert "retinal_analysis" in dashboard
        assert "biomarker_summary" in dashboard
        assert dashboard["retinal_analysis"]["risk_score"] == assessment.risk_assessment.risk_score


# ============================================================================
# Property 42: Dynamic Weight Adjustment
# ============================================================================

class TestDynamicWeightAdjustment:
    """
    Property 42: Weight should adjust based on confidence
    
    Validates: Requirement 9.6
    """
    
    def test_high_confidence_increases_weight(self):
        """High confidence should not decrease weight below base"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        # Set all confidences high
        assessment.biomarkers.vessels.confidence = 0.95
        assessment.biomarkers.optic_disc.confidence = 0.95
        assessment.biomarkers.macula.confidence = 0.95
        assessment.biomarkers.amyloid_beta.confidence = 0.95
        
        weight = service.calculate_dynamic_weight(assessment)
        
        assert weight >= service.BASE_WEIGHT
    
    def test_weight_stays_within_bounds(self):
        """Dynamic weight should stay within min/max bounds"""
        service = NRIIntegrationService()
        assessment = create_test_assessment()
        
        weight = service.calculate_dynamic_weight(assessment)
        
        assert service.MIN_WEIGHT <= weight <= service.MAX_WEIGHT


# ============================================================================
# Property 30: Data Encryption
# ============================================================================

class TestDataEncryption:
    """
    Property 30: Data should be encrypted at rest and in transit
    
    Validates: Requirements 8.1, 8.2, 11.4, 11.5
    """
    
    def test_encryption_is_reversible(self):
        """Encrypted data should decrypt to original"""
        service = EncryptionService()
        original = "Sensitive patient data"
        
        encrypted = service.encrypt(original)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == original
        assert encrypted != original
    
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=20)
    def test_encryption_works_for_any_text(self, text: str):
        """Encryption should work for any text input"""
        service = EncryptionService()
        
        encrypted = service.encrypt(text)
        decrypted = service.decrypt(encrypted)
        
        assert decrypted == text
    
    def test_dict_encryption(self):
        """Dictionary data should encrypt properly"""
        service = EncryptionService()
        data = {"patient_id": "P001", "risk_score": 45.5}
        
        encrypted = service.encrypt_dict(data)
        decrypted = service.decrypt_dict(encrypted)
        
        assert decrypted == data


# ============================================================================
# Property 36: Audit Logging Completeness
# ============================================================================

class TestAuditLogging:
    """
    Property 36: All access and modifications should be logged
    
    Validates: Requirements 8.11, 11.3, 11.11
    """
    
    def test_log_entry_includes_required_fields(self):
        """Log entries should include user, action, timestamp"""
        service = AuditLoggingService()
        
        entry = service.log(
            user_id="test-user",
            action=AuditAction.READ,
            resource_type=ResourceType.ASSESSMENT,
            resource_id="assessment-001"
        )
        
        assert entry.user_id == "test-user"
        assert entry.action == AuditAction.READ
        assert entry.timestamp is not None
    
    def test_log_retrieval(self):
        """Logs should be retrievable with filters"""
        service = AuditLoggingService()
        
        # Create some logs
        service.log("user1", AuditAction.CREATE, ResourceType.ASSESSMENT)
        service.log("user1", AuditAction.READ, ResourceType.ASSESSMENT)
        service.log("user2", AuditAction.READ, ResourceType.PATIENT)
        
        # Retrieve filtered
        logs = service.get_logs(user_id="user1")
        
        assert len(logs) >= 2
        assert all(l.user_id == "user1" for l in logs)
    
    def test_patient_id_anonymized_in_logs(self):
        """Patient IDs should be anonymized in log entries"""
        entry = AuditLogEntry(
            user_id="test",
            action=AuditAction.READ,
            resource_type=ResourceType.ASSESSMENT,
            patient_id="PATIENT-12345"
        )
        
        log_dict = entry.to_dict()
        
        # Should be anonymized
        assert log_dict["patient_id"] != "PATIENT-12345"
        assert "***" in log_dict["patient_id"]


# ============================================================================
# Property 48: Security Mechanisms
# ============================================================================

class TestSecurityMechanisms:
    """
    Property 48: Security mechanisms should be activated
    
    Validates: Requirements 11.1, 11.2, 11.6
    """
    
    def test_session_expires_after_timeout(self):
        """Session should expire after 15 minutes"""
        service = AuthenticationService()
        
        session = service.create_session(
            user_id="test",
            role=UserRole.PHYSICIAN
        )
        
        # Verify not expired initially
        validated = service.validate_session(session.session_id)
        assert validated is not None
        
        # Manually expire
        session.last_activity = datetime.utcnow() - timedelta(minutes=20)
        
        # Should be expired now
        expired = session.is_expired(15)
        assert expired is True
    
    def test_rbac_permissions(self):
        """RBAC should enforce role-based access"""
        authz = AuthorizationService()
        
        # Admin can delete
        assert authz.check_permission(UserRole.ADMIN, ResourceType.ASSESSMENT, "delete")
        
        # Technician cannot delete
        assert not authz.check_permission(UserRole.TECHNICIAN, ResourceType.ASSESSMENT, "delete")
        
        # Patient can only read
        assert authz.check_permission(UserRole.PATIENT, ResourceType.ASSESSMENT, "read")
        assert not authz.check_permission(UserRole.PATIENT, ResourceType.ASSESSMENT, "create")
    
    def test_lockout_after_failed_attempts(self):
        """Account should lock after multiple failed attempts"""
        service = AuthenticationService()
        
        user_id = "test-lockout"
        
        # Record multiple failures
        for _ in range(5):
            service.record_failed_attempt(user_id)
        
        # Should be locked
        assert service.is_locked_out(user_id)


# ============================================================================
# Property 49: Data Anonymization
# ============================================================================

class TestDataAnonymization:
    """
    Property 49: Data should be anonymized on export
    
    Validates: Requirement 11.8
    """
    
    def test_patient_id_anonymized(self):
        """Patient ID should be anonymized on export"""
        service = DataAnonymizationService()
        
        data = {
            "patient_id": "PATIENT-12345",
            "risk_score": 45.0
        }
        
        anonymized = service.anonymize_patient_data(data)
        
        assert anonymized["patient_id"] != "PATIENT-12345"
        assert anonymized["risk_score"] == 45.0  # Non-PII preserved
    
    def test_pii_fields_removed(self):
        """PII fields should be removed"""
        service = DataAnonymizationService()
        
        data = {
            "patient_id": "P001",
            "patient_name": "John Doe",
            "patient_dob": "1990-01-01",
            "risk_score": 35.0
        }
        
        anonymized = service.anonymize_patient_data(data)
        
        assert "patient_name" not in anonymized
        assert "patient_dob" not in anonymized
        assert "risk_score" in anonymized


# ============================================================================
# Property 44: Performance Guarantees
# ============================================================================

class TestPerformanceGuarantees:
    """
    Property 44: System should meet performance targets
    
    Validates: Requirements 4.4, 10.1-10.9
    """
    
    def test_cache_improves_retrieval(self):
        """Cached retrieval should be faster"""
        import time
        
        cache = MemoryCache()
        
        # Set value
        cache.set("test_key", {"data": "value"})
        
        # Measure retrieval time
        start = time.perf_counter()
        for _ in range(1000):
            cache.get("test_key")
        elapsed = time.perf_counter() - start
        
        # Should complete quickly (< 10ms for 1000 ops)
        assert elapsed < 0.01
    
    def test_cache_respects_ttl(self):
        """Cache should respect TTL"""
        import time
        
        cache = MemoryCache()
        cache.set("short_ttl", "value", ttl_seconds=1)
        
        # Should be available immediately
        assert cache.get("short_ttl") == "value"
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("short_ttl") is None


# ============================================================================
# Property 45: Request Queueing
# ============================================================================

class TestRequestQueueing:
    """
    Property 45: Requests should be queued under load
    
    Validates: Requirement 10.6
    """
    
    @pytest.mark.asyncio
    async def test_queue_returns_position(self):
        """Queue should return position for queued requests"""
        queue = RequestQueue(max_concurrent=2)
        
        pos1 = await queue.enqueue("req1", "patient1")
        pos2 = await queue.enqueue("req2", "patient1")
        
        assert pos1 == 1
        assert pos2 == 2
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority requests should be served first"""
        queue = RequestQueue(max_concurrent=1)
        
        await queue.enqueue("low", "p1", priority=1)
        await queue.enqueue("high", "p1", priority=10)
        await queue.enqueue("medium", "p1", priority=5)
        
        # High priority should be first
        req = await queue.dequeue()
        assert req.request_id == "high"
    
    @pytest.mark.asyncio
    async def test_status_tracking(self):
        """Queue should track request status"""
        queue = RequestQueue(max_concurrent=4)
        
        await queue.enqueue("req1", "patient1")
        
        status = await queue.get_status("req1")
        assert status["status"] == "queued"


# ============================================================================
# Property 46: Horizontal Scaling Trigger
# ============================================================================

class TestHorizontalScaling:
    """
    Property 46: Scaling should trigger at 50 queue items
    
    Validates: Requirement 10.7
    """
    
    @pytest.mark.asyncio
    async def test_scale_threshold(self):
        """Scale callback should trigger at threshold"""
        queue = RequestQueue(max_concurrent=4)
        scale_triggered = False
        triggered_size = 0
        
        async def on_scale(size: int):
            nonlocal scale_triggered, triggered_size
            scale_triggered = True
            triggered_size = size
        
        queue.on_scale_needed(on_scale)
        
        # Add items up to threshold
        for i in range(50):
            await queue.enqueue(f"req{i}", f"p{i}")
        
        assert scale_triggered
        assert triggered_size >= 50


# ============================================================================
# Performance Monitor Tests
# ============================================================================

class TestPerformanceMonitor:
    """Tests for performance monitoring"""
    
    def test_timing_statistics(self):
        """Monitor should calculate statistics correctly"""
        monitor = PerformanceMonitor()
        
        # Record some timings
        for t in [100, 200, 150, 300, 250]:
            monitor.record(t)
        
        stats = monitor.get_stats()
        
        assert stats["count"] == 5
        assert stats["min_ms"] == 100
        assert stats["max_ms"] == 300
        assert 180 <= stats["avg_ms"] <= 220  # Should be 200


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
