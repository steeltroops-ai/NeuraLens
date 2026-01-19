# Retinal Pipeline - Orchestration & State Tracking

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 7 - Orchestration |

---

## 1. Pipeline State Machine

### 1.1 State Diagram (Text)
```
                    +-------------+
                    |   PENDING   |
                    +------+------+
                           |
                           v
                +------------------+
                | INPUT_VALIDATION |-----> [FAILED]
                +--------+---------+
                         |
                         v
              +---------------------+
              | IMAGE_PREPROCESSING |-----> [FAILED]
              +---------+-----------+
                        |
                        v
              +-------------------+
              | QUALITY_ASSESSMENT|-----> [FAILED] (if quality < 0.3)
              +---------+---------+
                        |
                        v
          +------------------------+
          | ANATOMICAL_DETECTION   |-----> [WARN] (anatomy issues)
          +-----------+------------+
                      |
                      v
           +---------------------+
           | BIOMARKER_EXTRACTION|-----> [WARN] (low confidence)
           +----------+----------+
                      |
                      v
              +-------------+
              | DR_GRADING  |
              +------+------+
                     |
                     v
           +-------------------+
           | RISK_CALCULATION  |
           +---------+---------+
                     |
                     v
           +-------------------+
           | HEATMAP_GENERATION|-----> [WARN] (generation failed)
           +---------+---------+
                     |
                     v
          +---------------------+
          | CLINICAL_ASSESSMENT |
          +----------+----------+
                     |
                     v
           +------------------+
           | OUTPUT_FORMATTING|
           +---------+--------+
                     |
                     v
               +-----------+
               | COMPLETED |
               +-----------+
```

---

## 2. Stage Definitions

### 2.1 Pipeline Stages Enum
```python
class PipelineStage(str, Enum):
    PENDING = "pending"
    INPUT_VALIDATION = "input_validation"
    IMAGE_PREPROCESSING = "image_preprocessing"
    QUALITY_ASSESSMENT = "quality_assessment"
    ANATOMICAL_DETECTION = "anatomical_detection"
    BIOMARKER_EXTRACTION = "biomarker_extraction"
    DR_GRADING = "dr_grading"
    RISK_CALCULATION = "risk_calculation"
    HEATMAP_GENERATION = "heatmap_generation"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    OUTPUT_FORMATTING = "output_formatting"
    COMPLETED = "completed"
    FAILED = "failed"
```

### 2.2 Stage Configuration
```python
STAGE_CONFIG = {
    "input_validation": {
        "timeout_ms": 5000,
        "retryable": False,
        "required": True,
        "fail_action": "hard_stop",
    },
    "image_preprocessing": {
        "timeout_ms": 10000,
        "retryable": True,
        "max_retries": 2,
        "required": True,
        "fail_action": "hard_stop",
    },
    "quality_assessment": {
        "timeout_ms": 5000,
        "retryable": False,
        "required": True,
        "fail_action": "gate",  # Proceed with warning if quality low
    },
    "anatomical_detection": {
        "timeout_ms": 15000,
        "retryable": True,
        "max_retries": 1,
        "required": False,  # Can proceed with fallback
        "fail_action": "fallback",
    },
    "biomarker_extraction": {
        "timeout_ms": 20000,
        "retryable": True,
        "max_retries": 1,
        "required": True,
        "fail_action": "partial_result",
    },
    "dr_grading": {
        "timeout_ms": 10000,
        "retryable": True,
        "max_retries": 2,
        "required": True,
        "fail_action": "hard_stop",
    },
    "risk_calculation": {
        "timeout_ms": 2000,
        "retryable": False,
        "required": True,
        "fail_action": "hard_stop",
    },
    "heatmap_generation": {
        "timeout_ms": 10000,
        "retryable": True,
        "max_retries": 1,
        "required": False,
        "fail_action": "skip",
    },
    "clinical_assessment": {
        "timeout_ms": 5000,
        "retryable": False,
        "required": True,
        "fail_action": "hard_stop",
    },
    "output_formatting": {
        "timeout_ms": 2000,
        "retryable": False,
        "required": True,
        "fail_action": "hard_stop",
    },
}
```

---

## 3. State Tracking

### 3.1 Pipeline State Object
```python
@dataclass
class PipelineState:
    session_id: str
    current_stage: PipelineStage = PipelineStage.PENDING
    stages_completed: List[str] = field(default_factory=list)
    stages_timing_ms: Dict[str, float] = field(default_factory=dict)
    errors: List[PipelineError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    
    # Intermediate results
    image_bytes: Optional[bytes] = None
    preprocessed_image: Optional[np.ndarray] = None
    quality_score: Optional[float] = None
    anatomy: Optional[AnatomicalMap] = None
    biomarkers: Optional[CompleteBiomarkers] = None
    dr_result: Optional[DRResult] = None
    
    def transition_to(self, stage: PipelineStage):
        self.current_stage = stage
        logger.info(f"[{self.session_id}] Transitioning to {stage}")
    
    def complete_stage(self, stage: str, duration_ms: float):
        self.stages_completed.append(stage)
        self.stages_timing_ms[stage] = duration_ms
        logger.info(f"[{self.session_id}] Completed {stage} in {duration_ms:.1f}ms")
    
    def add_error(self, error: PipelineError):
        self.errors.append(error)
        logger.error(f"[{self.session_id}] Error in {error.stage}: {error.message}")
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        logger.warning(f"[{self.session_id}] Warning: {warning}")
```

---

## 4. Retry Logic

### 4.1 Retry Implementation
```python
async def execute_with_retry(
    stage: str,
    operation: Callable,
    state: PipelineState,
    *args, **kwargs
) -> Any:
    config = STAGE_CONFIG[stage]
    max_retries = config.get("max_retries", 0)
    
    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=config["timeout_ms"] / 1000
            )
            duration_ms = (time.time() - start_time) * 1000
            state.complete_stage(stage, duration_ms)
            return result
            
        except asyncio.TimeoutError:
            error = PipelineError(
                stage=stage,
                error_type="timeout",
                message=f"Stage timeout after {config['timeout_ms']}ms",
                details={"attempt": attempt + 1, "max_attempts": max_retries + 1}
            )
            
            if attempt < max_retries:
                state.add_warning(f"Retry {attempt + 1}/{max_retries} for {stage}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
            else:
                state.add_error(error)
                raise PipelineException(error)
                
        except Exception as e:
            error = PipelineError(
                stage=stage,
                error_type="execution_error",
                message=str(e),
                details={"attempt": attempt + 1, "exception_type": type(e).__name__}
            )
            
            if attempt < max_retries and config.get("retryable", False):
                state.add_warning(f"Retry {attempt + 1}/{max_retries} for {stage} due to: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
            else:
                state.add_error(error)
                raise PipelineException(error)
```

---

## 5. Hard-Stop Conditions

### 5.1 Conditions That Halt Pipeline
```python
HARD_STOP_CONDITIONS = {
    "input_validation": [
        "Invalid file format",
        "Corrupted file",
        "Resolution below minimum",
        "Not a fundus image",
    ],
    "quality_assessment": [
        "Quality score < 0.2 (unusable)",
        "Completely dark/overexposed image",
    ],
    "dr_grading": [
        "Model inference failure after retries",
        "Invalid probability distribution",
    ],
    "risk_calculation": [
        "Missing required biomarkers",
        "Mathematical error in scoring",
    ],
}
```

### 5.2 Hard-Stop Handler
```python
def handle_hard_stop(state: PipelineState, error: PipelineError):
    state.current_stage = PipelineStage.FAILED
    state.completed_at = datetime.utcnow().isoformat()
    
    return {
        "success": False,
        "session_id": state.session_id,
        "error": {
            "stage": error.stage,
            "error_type": error.error_type,
            "message": error.message,
            "details": error.details,
        },
        "stages_completed": state.stages_completed,
        "stages_timing_ms": state.stages_timing_ms,
        "total_time_ms": calculate_total_time(state),
        "resubmission_recommended": error.stage == "input_validation",
    }
```

---

## 6. Execution Flow Implementation

### 6.1 Main Orchestrator
```python
class PipelineOrchestrator:
    def __init__(self):
        self.input_layer = InputLayer()
        self.preprocessing_layer = PreprocessingLayer()
        self.analysis_layer = AnalysisLayer()
        self.grading_layer = GradingLayer()
        self.risk_layer = RiskLayer()
        self.clinical_layer = ClinicalLayer()
        self.visualization_layer = VisualizationLayer()
        self.output_layer = OutputLayer()
    
    async def execute(self, image: UploadFile, session_id: str) -> dict:
        state = PipelineState(session_id=session_id)
        
        try:
            # Layer 1: Input Validation
            state.transition_to(PipelineStage.INPUT_VALIDATION)
            image_bytes = await self.input_layer.process(image, state)
            
            # Layer 2: Preprocessing
            state.transition_to(PipelineStage.IMAGE_PREPROCESSING)
            preprocessed = await self.preprocessing_layer.process(image_bytes, state)
            
            # Layer 3: Quality Gate
            state.transition_to(PipelineStage.QUALITY_ASSESSMENT)
            if state.quality_score < 0.3:
                raise PipelineException(PipelineError(
                    stage="quality_assessment",
                    error_type="quality_gate_failed",
                    message="Image quality too low for reliable analysis"
                ))
            
            # Layer 4: Biomarker Extraction
            state.transition_to(PipelineStage.BIOMARKER_EXTRACTION)
            biomarkers = await self.analysis_layer.process(
                state.preprocessed_image, state.quality_score, None, state
            )
            
            # Layer 5: DR Grading
            state.transition_to(PipelineStage.DR_GRADING)
            dr_result = await self.grading_layer.process(biomarkers, state)
            
            # Layer 6: Risk Calculation
            state.transition_to(PipelineStage.RISK_CALCULATION)
            risk = await self.risk_layer.process(biomarkers, dr_result, state)
            
            # Layer 7: Visualization (non-critical)
            state.transition_to(PipelineStage.HEATMAP_GENERATION)
            try:
                heatmap = await self.visualization_layer.process(image_bytes, state)
            except Exception as e:
                state.add_warning(f"Heatmap generation failed: {e}")
                heatmap = None
            
            # Layer 8: Clinical Assessment
            state.transition_to(PipelineStage.CLINICAL_ASSESSMENT)
            clinical = await self.clinical_layer.process(
                biomarkers, dr_result, state.dme_result, risk, state
            )
            
            # Layer 9: Output Formatting
            state.transition_to(PipelineStage.OUTPUT_FORMATTING)
            response = await self.output_layer.format(state, heatmap)
            
            state.current_stage = PipelineStage.COMPLETED
            state.completed_at = datetime.utcnow().isoformat()
            
            return response
            
        except PipelineException as e:
            return handle_hard_stop(state, e.error)
```

---

## 7. Logging & Telemetry

### 7.1 Stage Logging Format
```python
LOG_FORMAT = {
    "timestamp": "ISO8601",
    "session_id": "UUID",
    "stage": "PipelineStage",
    "event": "start|complete|error|warning",
    "duration_ms": "float (for complete events)",
    "details": "dict (optional additional context)",
}

# Example log entries:
# {"timestamp":"2026-01-19T08:00:00Z","session_id":"abc123","stage":"dr_grading","event":"start"}
# {"timestamp":"2026-01-19T08:00:01Z","session_id":"abc123","stage":"dr_grading","event":"complete","duration_ms":1250.5}
```
