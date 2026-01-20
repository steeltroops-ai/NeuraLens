# 07 - Pipeline Orchestration and State Tracking

## Purpose
Design the internal pipeline behavior including stage execution, state management, error recovery, and completion tracking.

---

## 1. Pipeline State Machine

### 1.1 State Definitions

```python
from enum import Enum, auto

class PipelineState(Enum):
    """Pipeline execution states."""
    
    # Initial states
    RECEIVED = auto()           # Request received, not yet processed
    VALIDATING = auto()         # Input validation in progress
    
    # Processing states
    PREPROCESSING = auto()      # Image preprocessing
    SEGMENTING = auto()         # Lesion segmentation
    EXTRACTING_FEATURES = auto()  # ABCDE feature extraction
    CLASSIFYING = auto()        # Model inference
    SCORING = auto()            # Risk scoring
    GENERATING_EXPLANATION = auto()  # AI explanation
    FORMATTING = auto()         # Output formatting
    
    # Terminal states
    COMPLETED = auto()          # Success
    FAILED = auto()             # Unrecoverable failure
    REJECTED = auto()           # Input rejected (validation failure)
    TIMEOUT = auto()            # Processing timeout
    
    # Special states
    AWAITING_RETRY = auto()     # Retryable failure, waiting for retry
    PARTIAL_SUCCESS = auto()    # Some stages succeeded, some failed
```

### 1.2 State Transitions

```
                    +----------+
                    | RECEIVED |
                    +----------+
                         |
                         v
                    +-----------+
                    | VALIDATING|
                    +-----------+
                    /     |      \
                   /      |       \
                  v       v        v
          +--------+ +----------+ +----------+
          |REJECTED| |PREPROCESS| |  FAILED  |
          +--------+ +----------+ +----------+
                         |
                         v
                    +-----------+
                    | SEGMENTING|
                    +-----------+
                    /           \
                   v             v
          +----------+     +-----------+
          | EXTRACT  |     |AWAIT_RETRY|
          | FEATURES |     +-----------+
          +----------+
                |
                v
          +-----------+
          |CLASSIFYING|
          +-----------+
                |
                v
          +----------+
          | SCORING  |
          +----------+
                |
                v
          +----------+     +----------+
          | EXPLAIN  |---->| FORMAT   |
          +----------+     +----------+
                                |
                                v
                          +-----------+
                          | COMPLETED |
                          +-----------+
```

### 1.3 State Machine Implementation

```python
class PipelineStateMachine:
    """
    Manages pipeline state transitions and history.
    """
    
    VALID_TRANSITIONS = {
        PipelineState.RECEIVED: [PipelineState.VALIDATING],
        PipelineState.VALIDATING: [
            PipelineState.PREPROCESSING, 
            PipelineState.REJECTED, 
            PipelineState.FAILED
        ],
        PipelineState.PREPROCESSING: [
            PipelineState.SEGMENTING, 
            PipelineState.FAILED,
            PipelineState.AWAITING_RETRY
        ],
        PipelineState.SEGMENTING: [
            PipelineState.EXTRACTING_FEATURES, 
            PipelineState.FAILED,
            PipelineState.AWAITING_RETRY
        ],
        PipelineState.EXTRACTING_FEATURES: [
            PipelineState.CLASSIFYING, 
            PipelineState.PARTIAL_SUCCESS,
            PipelineState.FAILED
        ],
        PipelineState.CLASSIFYING: [
            PipelineState.SCORING, 
            PipelineState.FAILED
        ],
        PipelineState.SCORING: [
            PipelineState.GENERATING_EXPLANATION, 
            PipelineState.FORMATTING,  # Skip explanation if disabled
            PipelineState.FAILED
        ],
        PipelineState.GENERATING_EXPLANATION: [
            PipelineState.FORMATTING, 
            PipelineState.PARTIAL_SUCCESS  # Explanation optional
        ],
        PipelineState.FORMATTING: [
            PipelineState.COMPLETED, 
            PipelineState.FAILED
        ],
        PipelineState.AWAITING_RETRY: [
            PipelineState.PREPROCESSING,
            PipelineState.SEGMENTING,
            PipelineState.FAILED
        ]
    }
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.current_state = PipelineState.RECEIVED
        self.history: List[StateTransition] = []
        self.start_time = datetime.utcnow()
    
    def transition(
        self, 
        new_state: PipelineState, 
        context: dict = None
    ) -> bool:
        """Attempt to transition to a new state."""
        if new_state not in self.VALID_TRANSITIONS.get(self.current_state, []):
            logger.error(
                f"Invalid transition: {self.current_state.name} -> {new_state.name}"
            )
            return False
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=datetime.utcnow(),
            context=context or {}
        )
        self.history.append(transition)
        
        # Update current state
        old_state = self.current_state
        self.current_state = new_state
        
        logger.info(
            f"Pipeline {self.request_id}: {old_state.name} -> {new_state.name}"
        )
        
        return True
    
    def is_terminal(self) -> bool:
        """Check if pipeline is in a terminal state."""
        return self.current_state in [
            PipelineState.COMPLETED,
            PipelineState.FAILED,
            PipelineState.REJECTED,
            PipelineState.TIMEOUT
        ]
    
    def get_elapsed_time(self) -> timedelta:
        """Get elapsed time since pipeline started."""
        return datetime.utcnow() - self.start_time
    
    def get_stage_timing(self) -> dict:
        """Get timing for each stage."""
        timings = {}
        
        for i, transition in enumerate(self.history[:-1]):
            stage_name = transition.from_state.name
            next_transition = self.history[i + 1]
            duration = (next_transition.timestamp - transition.timestamp).total_seconds()
            timings[stage_name] = duration
        
        return timings
```

---

## 2. Stage Execution Tracking

### 2.1 Stage Results

```python
@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    
    stage: str
    status: str                      # "success", "warning", "failure"
    start_time: datetime
    end_time: datetime
    duration_ms: int
    
    # Output data
    output: Any                      # Stage-specific output
    
    # Quality metrics
    confidence: Optional[float]
    quality_score: Optional[float]
    
    # Issues
    warnings: List[str]
    errors: List[str]
    
    # Retry info
    retry_count: int
    is_retryable: bool


class StageTracker:
    """Tracks stage execution and results."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, StageResult] = {}
        self.current_stage: Optional[str] = None
    
    def start_stage(self, stage_name: str) -> None:
        """Mark stage as started."""
        self.current_stage = stage_name
        self.stages[stage_name] = StageResult(
            stage=stage_name,
            status="running",
            start_time=datetime.utcnow(),
            end_time=None,
            duration_ms=0,
            output=None,
            confidence=None,
            quality_score=None,
            warnings=[],
            errors=[],
            retry_count=0,
            is_retryable=True
        )
        
        logger.info(f"[{self.request_id}] Stage started: {stage_name}")
    
    def complete_stage(
        self, 
        stage_name: str, 
        output: Any,
        confidence: float = None,
        quality_score: float = None,
        warnings: List[str] = None
    ) -> None:
        """Mark stage as completed."""
        if stage_name not in self.stages:
            raise ValueError(f"Stage not started: {stage_name}")
        
        stage = self.stages[stage_name]
        stage.end_time = datetime.utcnow()
        stage.duration_ms = int(
            (stage.end_time - stage.start_time).total_seconds() * 1000
        )
        stage.status = "success"
        stage.output = output
        stage.confidence = confidence
        stage.quality_score = quality_score
        stage.warnings = warnings or []
        
        logger.info(
            f"[{self.request_id}] Stage completed: {stage_name} "
            f"({stage.duration_ms}ms)"
        )
    
    def fail_stage(
        self, 
        stage_name: str, 
        error: str,
        is_retryable: bool = True
    ) -> None:
        """Mark stage as failed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageResult(
                stage=stage_name,
                status="failure",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_ms=0,
                output=None,
                confidence=None,
                quality_score=None,
                warnings=[],
                errors=[error],
                retry_count=0,
                is_retryable=is_retryable
            )
        else:
            stage = self.stages[stage_name]
            stage.end_time = datetime.utcnow()
            stage.duration_ms = int(
                (stage.end_time - stage.start_time).total_seconds() * 1000
            )
            stage.status = "failure"
            stage.errors.append(error)
            stage.is_retryable = is_retryable
        
        logger.error(f"[{self.request_id}] Stage failed: {stage_name} - {error}")
    
    def get_summary(self) -> dict:
        """Get summary of all stages."""
        return {
            "stages": [
                {
                    "name": s.stage,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "confidence": s.confidence,
                    "warnings": len(s.warnings),
                    "errors": len(s.errors)
                }
                for s in self.stages.values()
            ],
            "total_duration_ms": sum(s.duration_ms for s in self.stages.values()),
            "success_count": sum(1 for s in self.stages.values() if s.status == "success"),
            "failure_count": sum(1 for s in self.stages.values() if s.status == "failure")
        }
```

---

## 3. Pipeline Orchestrator

### 3.1 Main Orchestrator

```python
class DermatologyPipelineOrchestrator:
    """
    Orchestrates the complete dermatology analysis pipeline.
    """
    
    TIMEOUT_SECONDS = 30
    MAX_RETRIES = 2
    
    def __init__(self):
        self.validator = InputValidator()
        self.preprocessor = ImagePreprocessor()
        self.segmenter = LesionSegmenter()
        self.feature_extractor = FeatureExtractor()
        self.classifier = DermatologyClassifier()
        self.scorer = ClinicalScorer()
        self.explainer = ExplanationGenerator()
        self.formatter = OutputFormatter()
    
    async def analyze(
        self, 
        request: DermatologyRequest
    ) -> DermatologyResponse:
        """Run complete analysis pipeline."""
        # Initialize tracking
        request_id = self._generate_request_id()
        state_machine = PipelineStateMachine(request_id)
        tracker = StageTracker(request_id)
        
        try:
            # Stage 1: Validation
            state_machine.transition(PipelineState.VALIDATING)
            tracker.start_stage("validation")
            
            validation_result = await self._validate(request)
            
            if not validation_result.passed:
                state_machine.transition(PipelineState.REJECTED)
                return self._format_rejection(validation_result, tracker)
            
            tracker.complete_stage(
                "validation", 
                validation_result,
                quality_score=validation_result.quality_score
            )
            
            # Stage 2: Preprocessing
            state_machine.transition(PipelineState.PREPROCESSING)
            tracker.start_stage("preprocessing")
            
            preprocessed = await self._preprocess(
                request.image, 
                validation_result
            )
            
            tracker.complete_stage(
                "preprocessing",
                preprocessed,
                confidence=preprocessed.confidence,
                warnings=preprocessed.warnings
            )
            
            # Stage 3: Segmentation
            state_machine.transition(PipelineState.SEGMENTING)
            tracker.start_stage("segmentation")
            
            segmentation = await self._segment(preprocessed.image)
            
            if not segmentation.validation.valid:
                if tracker.stages["segmentation"].retry_count < self.MAX_RETRIES:
                    state_machine.transition(PipelineState.AWAITING_RETRY)
                    return await self._retry_segmentation(
                        preprocessed, state_machine, tracker
                    )
                else:
                    state_machine.transition(PipelineState.FAILED)
                    return self._format_failure(
                        "Lesion segmentation failed",
                        tracker
                    )
            
            tracker.complete_stage(
                "segmentation",
                segmentation,
                confidence=segmentation.mean_confidence
            )
            
            # Stage 4: Feature Extraction
            state_machine.transition(PipelineState.EXTRACTING_FEATURES)
            tracker.start_stage("feature_extraction")
            
            features = await self._extract_features(
                preprocessed.image,
                segmentation.mask
            )
            
            tracker.complete_stage("feature_extraction", features)
            
            # Stage 5: Classification
            state_machine.transition(PipelineState.CLASSIFYING)
            tracker.start_stage("classification")
            
            classification = await self._classify(
                preprocessed.image,
                segmentation.mask,
                features
            )
            
            tracker.complete_stage(
                "classification",
                classification,
                confidence=classification.confidence
            )
            
            # Stage 6: Scoring
            state_machine.transition(PipelineState.SCORING)
            tracker.start_stage("scoring")
            
            scoring = await self._score(
                classification,
                features,
                request.prior_analysis
            )
            
            tracker.complete_stage("scoring", scoring)
            
            # Stage 7: Explanation (optional)
            if request.generate_explanation:
                state_machine.transition(PipelineState.GENERATING_EXPLANATION)
                tracker.start_stage("explanation")
                
                try:
                    explanation = await self._explain(
                        classification,
                        features,
                        scoring
                    )
                    tracker.complete_stage("explanation", explanation)
                except Exception as e:
                    tracker.fail_stage("explanation", str(e))
                    explanation = None
            else:
                explanation = None
            
            # Stage 8: Formatting
            state_machine.transition(PipelineState.FORMATTING)
            tracker.start_stage("formatting")
            
            response = self._format_response(
                validation_result,
                preprocessed,
                segmentation,
                features,
                classification,
                scoring,
                explanation,
                tracker
            )
            
            tracker.complete_stage("formatting", response)
            state_machine.transition(PipelineState.COMPLETED)
            
            return response
            
        except asyncio.TimeoutError:
            state_machine.transition(PipelineState.TIMEOUT)
            return self._format_timeout(tracker)
            
        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            state_machine.transition(PipelineState.FAILED)
            return self._format_failure(str(e), tracker)
```

---

## 4. Retry Logic

### 4.1 Retryable Conditions

```python
class RetryPolicy:
    """
    Defines retry policies for pipeline stages.
    """
    
    RETRY_CONFIGS = {
        "preprocessing": {
            "max_retries": 2,
            "retry_delay_ms": 100,
            "conditions": ["color_correction_failed", "artifact_removal_failed"],
            "fallback": "skip_failed_step"
        },
        "segmentation": {
            "max_retries": 3,
            "retry_delay_ms": 200,
            "conditions": ["low_confidence", "fragmented_mask", "no_lesion_detected"],
            "fallback": "use_bounding_box"
        },
        "classification": {
            "max_retries": 2,
            "retry_delay_ms": 100,
            "conditions": ["model_timeout", "cuda_error"],
            "fallback": "use_single_model"
        },
        "explanation": {
            "max_retries": 1,
            "retry_delay_ms": 500,
            "conditions": ["llm_timeout", "api_error"],
            "fallback": "use_template_explanation"
        }
    }
    
    def should_retry(
        self, 
        stage: str, 
        error: Exception, 
        retry_count: int
    ) -> RetryDecision:
        """Determine if stage should be retried."""
        config = self.RETRY_CONFIGS.get(stage)
        
        if config is None:
            return RetryDecision(should_retry=False)
        
        if retry_count >= config["max_retries"]:
            return RetryDecision(
                should_retry=False,
                fallback=config["fallback"]
            )
        
        error_type = self._classify_error(error)
        
        if error_type in config["conditions"]:
            return RetryDecision(
                should_retry=True,
                delay_ms=config["retry_delay_ms"],
                retry_number=retry_count + 1
            )
        
        return RetryDecision(
            should_retry=False,
            fallback=config["fallback"]
        )
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for retry logic."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return "model_timeout"
        elif "cuda" in error_str or "gpu" in error_str:
            return "cuda_error"
        elif "confidence" in error_str:
            return "low_confidence"
        elif "segmentation" in error_str:
            return "fragmented_mask"
        else:
            return "unknown"
```

### 4.2 Fallback Strategies

```python
class FallbackExecutor:
    """
    Executes fallback strategies when stages fail.
    """
    
    def execute_fallback(
        self, 
        stage: str, 
        fallback_type: str,
        context: dict
    ) -> FallbackResult:
        """Execute fallback strategy."""
        if fallback_type == "skip_failed_step":
            return self._skip_step(stage, context)
        elif fallback_type == "use_bounding_box":
            return self._use_bounding_box(context)
        elif fallback_type == "use_single_model":
            return self._use_single_model(context)
        elif fallback_type == "use_template_explanation":
            return self._use_template_explanation(context)
        else:
            raise ValueError(f"Unknown fallback: {fallback_type}")
    
    def _use_bounding_box(self, context: dict) -> FallbackResult:
        """Use detection bounding box instead of segmentation mask."""
        image = context["image"]
        detection = context.get("detection")
        
        if detection is None:
            # Run quick detection
            detection = self._quick_detect(image)
        
        # Create rectangular mask from bounding box
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        x, y, w, h = detection.bounding_box
        mask[y:y+h, x:x+w] = 255
        
        return FallbackResult(
            success=True,
            output={"mask": mask, "source": "bounding_box"},
            quality_degradation=0.2,  # 20% quality loss
            warning="Using bounding box instead of precise segmentation"
        )
    
    def _use_single_model(self, context: dict) -> FallbackResult:
        """Use single model instead of ensemble."""
        image = context["image"]
        mask = context["mask"]
        
        # Use primary model only
        primary_model = context["models"][0]
        
        with torch.no_grad():
            prediction = primary_model(image, mask)
        
        return FallbackResult(
            success=True,
            output=prediction,
            quality_degradation=0.1,
            warning="Using single model due to ensemble failure"
        )
    
    def _use_template_explanation(self, context: dict) -> FallbackResult:
        """Use template-based explanation."""
        results = context["results"]
        
        template = self._select_template(results.risk_tier)
        explanation = template.format(
            risk_tier=results.risk_tier,
            melanoma_prob=results.melanoma.probability * 100,
            abcde_count=results.abcde.criteria_met
        )
        
        return FallbackResult(
            success=True,
            output={"explanation": explanation, "source": "template"},
            quality_degradation=0.3,
            warning="Using template explanation (AI explanation unavailable)"
        )
```

---

## 5. Hard-Stop Conditions

### 5.1 Unsafe Input Detection

```python
class HardStopChecker:
    """
    Checks for conditions that require immediate pipeline termination.
    """
    
    HARD_STOP_CONDITIONS = [
        {
            "name": "no_image_data",
            "check": lambda r: r.image is None or r.image.size == 0,
            "message": "No valid image data received"
        },
        {
            "name": "non_skin_image",
            "check": lambda r: r.validation and r.validation.skin_ratio < 0.05,
            "message": "Image does not appear to contain skin"
        },
        {
            "name": "corrupted_file",
            "check": lambda r: r.validation and r.validation.is_corrupted,
            "message": "Image file is corrupted"
        },
        {
            "name": "unsafe_content",
            "check": lambda r: r.validation and r.validation.contains_unsafe_content,
            "message": "Image contains inappropriate content"
        },
        {
            "name": "extreme_quality_failure",
            "check": lambda r: r.validation and r.validation.quality_score < 0.1,
            "message": "Image quality too poor for any analysis"
        },
        {
            "name": "processing_timeout",
            "check": lambda r: r.elapsed_time.total_seconds() > 60,
            "message": "Processing timeout exceeded"
        }
    ]
    
    def check(self, request: Any) -> Optional[HardStop]:
        """Check if hard stop condition is met."""
        for condition in self.HARD_STOP_CONDITIONS:
            try:
                if condition["check"](request):
                    return HardStop(
                        condition=condition["name"],
                        message=condition["message"],
                        recoverable=False
                    )
            except Exception:
                pass
        
        return None
    
    def check_during_processing(
        self, 
        stage: str, 
        result: Any
    ) -> Optional[HardStop]:
        """Check for hard stops during processing."""
        # Segmentation produces empty mask
        if stage == "segmentation" and result.mask is not None:
            if np.sum(result.mask > 0) == 0:
                return HardStop(
                    condition="empty_segmentation",
                    message="No lesion could be detected in the image",
                    recoverable=True
                )
        
        # Classification confidence extremely low
        if stage == "classification":
            if result.confidence < 0.2:
                return HardStop(
                    condition="extremely_low_confidence",
                    message="Unable to make reliable classification",
                    recoverable=False
                )
        
        return None
```

---

## 6. Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REQUEST RECEIVED                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VALIDATION                                                          │
│  ├── File validation (type, size, integrity)                        │
│  ├── Image quality (resolution, focus, exposure)                    │
│  └── Content validation (skin detection, lesion presence)           │
└─────────────────────────────────────────────────────────────────────┘
                │                               │
        [PASSED]│                               │[FAILED]
                ▼                               ▼
┌────────────────────────────┐    ┌────────────────────────────┐
│  PREPROCESSING             │    │  REJECTION RESPONSE        │
│  ├── Color constancy       │    │  └── Error details         │
│  ├── Illumination          │    │  └── Retake guidance       │
│  ├── Hair removal          │    └────────────────────────────┘
│  ├── Contrast enhance      │
│  └── Resize/align          │
└────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SEGMENTATION                                                        │
│  ├── Detection (YOLO)                                               │
│  ├── Semantic segmentation (U-Net)                                  │
│  ├── Boundary refinement (CRF)                                      │
│  └── Geometry extraction                                            │
└─────────────────────────────────────────────────────────────────────┘
                │                               │
        [SUCCESS]│                              │[RETRY?]
                │                               │
                │            ┌──────────────────┘
                │            │
                ▼            ▼
┌─────────────────────┐  ┌─────────────────────┐
│ FEATURE EXTRACTION  │  │ RETRY WITH FALLBACK │
│ ├── A: Asymmetry    │  │ └──> Bounding box   │
│ ├── B: Border       │  └─────────────────────┘
│ ├── C: Color        │           │
│ ├── D: Diameter     │           │
│ └── E: Evolution    │◄──────────┘
└─────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION (Parallel)                                           │
│  ├── Melanoma classifier (EfficientNet)                             │
│  ├── Malignancy classifier                                          │
│  ├── Subtype classifier                                             │
│  └── Ensemble aggregation                                           │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SCORING                                                             │
│  ├── ABCDE score computation                                        │
│  ├── Risk tier assignment                                           │
│  ├── Longitudinal comparison (if prior available)                   │
│  └── Escalation check                                               │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXPLANATION GENERATION (Optional)                                   │
│  ├── Grad-CAM heatmap                                               │
│  ├── Feature attribution                                            │
│  └── Natural language summary                                       │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT FORMATTING                                                   │
│  ├── Structure response                                             │
│  ├── Generate overlays                                              │
│  └── Add disclaimers                                                │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RESPONSE SENT                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. State Transition Events

```python
@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: PipelineState
    to_state: PipelineState
    timestamp: datetime
    context: dict
    
    # Stage results if applicable
    stage_result: Optional[StageResult] = None
    
    # Metrics
    memory_usage_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class StateEventEmitter:
    """Emits events on state transitions for monitoring."""
    
    def __init__(self):
        self.listeners: List[Callable] = []
    
    def on_transition(self, callback: Callable) -> None:
        """Register transition callback."""
        self.listeners.append(callback)
    
    def emit(self, transition: StateTransition) -> None:
        """Emit transition event."""
        for listener in self.listeners:
            try:
                listener(transition)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    # Usage: logging, metrics, alerting
    def log_transition(self, t: StateTransition):
        logger.info(f"State: {t.from_state.name} -> {t.to_state.name}")
    
    def record_metric(self, t: StateTransition):
        metrics.state_transitions.labels(
            from_state=t.from_state.name,
            to_state=t.to_state.name
        ).inc()
```
