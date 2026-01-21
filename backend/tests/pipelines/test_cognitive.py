"""
Cognitive Pipeline Tests - Backend
Production-grade test suite for the cognitive assessment pipeline.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List

# Import pipeline components
from app.pipelines.cognitive.schemas import (
    CognitiveSessionInput, TaskSession, TaskEvent,
    CognitiveResponse, PipelineStage, RiskLevel, TaskCompletionStatus
)
from app.pipelines.cognitive.core.service import CognitiveService
from app.pipelines.cognitive.input.validator import CognitiveValidator
from app.pipelines.cognitive.features.extractor import FeatureExtractor
from app.pipelines.cognitive.clinical.risk_scorer import RiskScorer
from app.pipelines.cognitive.errors.codes import ErrorCode, PipelineError


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_reaction_time_task() -> TaskSession:
    """Create a valid reaction time task with realistic events."""
    now = datetime.now()
    events = []
    base_time = 0.0
    
    # Simulate 5 trials
    for i in range(5):
        # Stimulus shown
        events.append(TaskEvent(
            timestamp=base_time,
            event_type="stimulus_shown",
            payload={"trial": i}
        ))
        # Response after 250-400ms (realistic)
        rt = 250 + (i * 30)
        base_time += rt
        events.append(TaskEvent(
            timestamp=base_time,
            event_type="response_received",
            payload={"trial": i, "rt": rt}
        ))
        base_time += 1000  # 1s between trials
    
    return TaskSession(
        task_id="reaction_time_v1",
        start_time=now,
        end_time=now + timedelta(seconds=10),
        events=events,
        metadata={"test": True}
    )


@pytest.fixture
def valid_nback_task() -> TaskSession:
    """Create a valid N-Back task with trial results."""
    now = datetime.now()
    events = [TaskEvent(timestamp=0, event_type="test_start", payload={"n": 2})]
    base_time = 100.0
    
    # Simulate 20 trials with mixed results
    results = ["hit"] * 8 + ["miss"] * 2 + ["false_alarm"] * 3 + ["correct_rejection"] * 7
    for i, result in enumerate(results):
        events.append(TaskEvent(
            timestamp=base_time,
            event_type="trial_result",
            payload={"result": result, "trial": i}
        ))
        base_time += 2500  # 2.5s per trial
    
    return TaskSession(
        task_id="n_back_2",
        start_time=now,
        end_time=now + timedelta(seconds=60),
        events=events,
        metadata={"n": 2}
    )


@pytest.fixture
def valid_session(valid_reaction_time_task, valid_nback_task) -> CognitiveSessionInput:
    """Create a valid session with multiple tasks."""
    return CognitiveSessionInput(
        session_id="sess_test_12345",
        tasks=[valid_reaction_time_task, valid_nback_task],
        user_metadata={"test": True}
    )


@pytest.fixture
def service() -> CognitiveService:
    return CognitiveService()


@pytest.fixture
def validator() -> CognitiveValidator:
    return CognitiveValidator()


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Tests for input validation layer."""
    
    def test_valid_session_passes(self, validator, valid_session):
        """Valid session should pass validation."""
        errors = validator.validate(valid_session)
        assert len(errors) == 0
    
    def test_empty_tasks_fails(self, validator):
        """Session with no tasks should fail."""
        session = CognitiveSessionInput(
            session_id="sess_empty",
            tasks=[],
            user_metadata={}
        )
        errors = validator.validate(session)
        assert len(errors) > 0
        assert any("Empty" in e or "empty" in e.lower() for e in errors)
    
    def test_invalid_session_id_fails(self, validator, valid_reaction_time_task):
        """Session ID not starting with 'sess_' should fail."""
        # Note: This is enforced by Pydantic, but we test the validator
        try:
            session = CognitiveSessionInput(
                session_id="invalid_123",
                tasks=[valid_reaction_time_task],
                user_metadata={}
            )
            # If Pydantic allows it, validator should catch it
            errors = validator.validate(session)
            assert len(errors) > 0
        except Exception:
            pass  # Expected - Pydantic validation error
    
    def test_non_monotonic_timestamps_fails(self, validator):
        """Events with non-monotonic timestamps should fail."""
        now = datetime.now()
        task = TaskSession(
            task_id="bad_task_1",
            start_time=now,
            end_time=now + timedelta(seconds=10),
            events=[
                TaskEvent(timestamp=100, event_type="stimulus_shown", payload={}),
                TaskEvent(timestamp=50, event_type="response_received", payload={}),  # Out of order!
            ],
            metadata={}
        )
        session = CognitiveSessionInput(
            session_id="sess_bad_timestamps",
            tasks=[task],
            user_metadata={}
        )
        errors = validator.validate(session)
        assert len(errors) > 0


# =============================================================================
# FEATURE EXTRACTION TESTS
# =============================================================================

class TestFeatureExtraction:
    """Tests for feature extraction layer."""
    
    def test_reaction_time_extraction(self, extractor, valid_reaction_time_task):
        """Should extract valid metrics from reaction time task."""
        metrics = extractor.extract([valid_reaction_time_task])
        
        assert len(metrics) == 1
        assert metrics[0].task_id == "reaction_time_v1"
        assert metrics[0].completion_status == TaskCompletionStatus.COMPLETE
        assert metrics[0].validity_flag == True
        assert "mean_rt" in metrics[0].parameters
        assert metrics[0].parameters["mean_rt"] > 0
    
    def test_nback_extraction(self, extractor, valid_nback_task):
        """Should extract valid metrics from N-Back task."""
        metrics = extractor.extract([valid_nback_task])
        
        assert len(metrics) == 1
        assert "n_back" in metrics[0].task_id
        assert metrics[0].completion_status == TaskCompletionStatus.COMPLETE
        assert "accuracy" in metrics[0].parameters
        assert 0 <= metrics[0].parameters["accuracy"] <= 1
    
    def test_unknown_task_handled(self, extractor):
        """Unknown task type should return invalid metrics gracefully."""
        now = datetime.now()
        unknown_task = TaskSession(
            task_id="unknown_task_xyz",
            start_time=now,
            end_time=now + timedelta(seconds=10),
            events=[TaskEvent(timestamp=0, event_type="test_start", payload={})],
            metadata={}
        )
        metrics = extractor.extract([unknown_task])
        
        assert len(metrics) == 1
        assert metrics[0].validity_flag == False
        assert metrics[0].completion_status == TaskCompletionStatus.UNKNOWN


# =============================================================================
# SCORING TESTS
# =============================================================================

class TestScoring:
    """Tests for clinical scoring layer."""
    
    def test_low_risk_scoring(self, scorer):
        """High performance should result in low risk."""
        from app.pipelines.cognitive.schemas import CognitiveFeatures, TaskMetrics
        
        features = CognitiveFeatures(
            domain_scores={"memory": 0.9, "processing_speed": 0.85},
            raw_metrics=[],
            valid_task_count=2,
            total_task_count=2
        )
        
        risk, explainability = scorer.score_with_explanation(features)
        
        assert risk.risk_level == RiskLevel.LOW
        assert risk.overall_risk_score < 0.3
        assert risk.confidence_score > 0.7
    
    def test_high_risk_scoring(self, scorer):
        """Low performance should result in high risk."""
        from app.pipelines.cognitive.schemas import CognitiveFeatures
        
        features = CognitiveFeatures(
            domain_scores={"memory": 0.2, "processing_speed": 0.25},
            raw_metrics=[],
            valid_task_count=2,
            total_task_count=2
        )
        
        risk, explainability = scorer.score_with_explanation(features)
        
        assert risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert risk.overall_risk_score > 0.6
    
    def test_recommendations_generated(self, scorer):
        """Should generate appropriate recommendations."""
        from app.pipelines.cognitive.schemas import CognitiveFeatures
        
        features = CognitiveFeatures(
            domain_scores={"memory": 0.4},
            raw_metrics=[],
            valid_task_count=1,
            total_task_count=1
        )
        
        risk, _ = scorer.score_with_explanation(features)
        recommendations = scorer.generate_recommendations(risk)
        
        assert len(recommendations) > 0
        assert all(r.category in ["clinical", "lifestyle", "routine", "specific"] for r in recommendations)


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, service, valid_session):
        """Full pipeline should complete successfully with valid input."""
        response = await service.process_session(valid_session)
        
        assert response.status == "success"
        assert response.risk_assessment is not None
        assert response.features is not None
        assert len(response.stages) == 4
        assert response.error_code is None
    
    @pytest.mark.asyncio
    async def test_pipeline_with_single_task(self, service, valid_reaction_time_task):
        """Pipeline should work with just one task."""
        session = CognitiveSessionInput(
            session_id="sess_single_task",
            tasks=[valid_reaction_time_task],
            user_metadata={}
        )
        
        response = await service.process_session(session)
        
        assert response.status == "success"
        assert response.risk_assessment is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_error_propagation(self, service):
        """Pipeline should return structured error on invalid input."""
        session = CognitiveSessionInput(
            session_id="sess_empty_test",
            tasks=[],
            user_metadata={}
        )
        
        # Empty tasks should fail validation
        with pytest.raises(Exception):
            await service.process_session(session)


# =============================================================================
# API CONTRACT TESTS
# =============================================================================

class TestAPIContract:
    """Tests to verify API contract compliance."""
    
    @pytest.mark.asyncio
    async def test_response_schema_compliance(self, service, valid_session):
        """Response should match CognitiveResponse schema."""
        response = await service.process_session(valid_session)
        
        # Verify all required fields are present
        assert hasattr(response, 'session_id')
        assert hasattr(response, 'timestamp')
        assert hasattr(response, 'processing_time_ms')
        assert hasattr(response, 'status')
        assert hasattr(response, 'stages')
        assert hasattr(response, 'risk_assessment')
        assert hasattr(response, 'features')
        assert hasattr(response, 'recommendations')
        assert hasattr(response, 'pipeline_version')
    
    @pytest.mark.asyncio
    async def test_stage_progress_tracking(self, service, valid_session):
        """All pipeline stages should be tracked."""
        response = await service.process_session(valid_session)
        
        stages = response.stages
        assert len(stages) >= 3
        
        # Verify stage structure
        for stage in stages:
            assert hasattr(stage, 'stage')
            assert hasattr(stage, 'stage_index')
            assert hasattr(stage, 'message')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
