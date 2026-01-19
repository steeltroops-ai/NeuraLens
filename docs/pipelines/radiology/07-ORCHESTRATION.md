# 07 - Pipeline Orchestration and State Tracking

## Document Info
| Field | Value |
|-------|-------|
| Stage | Pipeline Orchestration |
| Owner | ML Systems Architect |
| Reviewer | All Team Members |

---

## 1. Overview

### 1.1 Purpose
Manage pipeline execution flow:
- Modality detection and routing
- Parallel and sequential stage execution
- State tracking and checkpointing
- Retry logic and failure recovery

---

## 2. Pipeline State Machine

### 2.1 State Definitions

```
                    +----------+
                    |  START   |
                    +----------+
                         |
                         v
                    +----------+
                    | RECEIPT  |<----+
                    +----------+     |
                         |           | (retry)
                         v           |
                    +----------+     |
                    |VALIDATION|-----+
                    +----------+
                         |
            +------------+------------+
            |                         |
            v                         v
    +---------------+         +---------------+
    |PREPROCESSING  |         |PREPROCESSING  |
    |(Image Branch) |         |(Volume Branch)|
    +---------------+         +---------------+
            |                         |
            +------------+------------+
                         |
                         v
                    +----------+
                    | DETECTION|
                    +----------+
                         |
                         v
                    +----------+
                    | ANALYSIS |
                    +----------+
                         |
                         v
                    +----------+
                    |AGGREGATION|
                    +----------+
                         |
                         v
                    +----------+
                    | SCORING  |
                    +----------+
                         |
                         v
                    +----------+
                    |FORMATTING|
                    +----------+
                         |
            +------------+------------+
            |                         |
            v                         v
       +--------+               +--------+
       |COMPLETE|               | FAILED |
       +--------+               +--------+
```

### 2.2 State Enum
```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

class PipelineState(Enum):
    START = auto()
    RECEIPT = auto()
    VALIDATION = auto()
    PREPROCESSING = auto()
    DETECTION = auto()
    ANALYSIS = auto()
    AGGREGATION = auto()
    SCORING = auto()
    FORMATTING = auto()
    COMPLETE = auto()
    FAILED = auto()

@dataclass
class StateTransition:
    from_state: PipelineState
    to_state: PipelineState
    timestamp: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None
```

---

## 3. Modality Detection and Routing

### 3.1 Modality Detection
```python
class ModalityRouter:
    """Detect modality and route to appropriate pipeline."""
    
    MODALITY_MAP = {
        "CR": "chest_xray",
        "DX": "chest_xray",
        "CT": "ct",
        "MR": "mri",
        "PT": "pet",
        "NM": "nuclear"
    }
    
    def detect_modality(self, input_data: dict) -> dict:
        """
        Detect modality from input.
        
        Returns:
            dict with modality, body_region, is_volumetric
        """
        # Check DICOM modality tag
        if "dicom_metadata" in input_data:
            dicom_modality = input_data["dicom_metadata"].get("modality")
            if dicom_modality:
                modality = self.MODALITY_MAP.get(dicom_modality, "unknown")
                body_region = input_data["dicom_metadata"].get("body_part")
                return {
                    "modality": modality,
                    "body_region": body_region,
                    "is_volumetric": input_data.get("slice_count", 1) > 1,
                    "detection_method": "dicom_tag"
                }
        
        # Check file type hints
        if input_data.get("file_extension") in [".dcm", ".dicom"]:
            return self._infer_from_image(input_data)
        
        # Check user-provided hint
        if "modality_hint" in input_data:
            return {
                "modality": input_data["modality_hint"],
                "body_region": input_data.get("body_region_hint"),
                "is_volumetric": input_data.get("slice_count", 1) > 1,
                "detection_method": "user_hint"
            }
        
        # Default: infer from image characteristics
        return self._infer_from_image(input_data)
    
    def get_pipeline_config(self, modality: str, body_region: str = None) -> dict:
        """Get pipeline configuration for modality."""
        
        configs = {
            "chest_xray": {
                "model": "torchxrayvision",
                "preprocessing": ["normalize", "resize_224"],
                "analysis_modules": ["pathology_18"],
                "parallel_stages": False
            },
            "ct": {
                "model": "3d_unet",
                "preprocessing": ["windowing", "normalize", "resample"],
                "analysis_modules": ["lesion_detection", "organ_segmentation"],
                "parallel_stages": True
            },
            "mri": {
                "model": "nnunet",
                "preprocessing": ["bias_correction", "normalize", "resample"],
                "analysis_modules": ["structure_segmentation", "lesion_detection"],
                "parallel_stages": True
            }
        }
        
        return configs.get(modality, configs["chest_xray"])
```

---

## 4. Parallelizable Stages

### 4.1 Stage Dependencies

| Stage | Dependencies | Can Parallelize |
|-------|--------------|-----------------|
| Receipt | None | N/A |
| Validation | Receipt | No |
| Preprocessing | Validation | **Yes** (if multi-input) |
| Detection | Preprocessing | **Yes** (per structure) |
| Analysis | Detection | **Yes** (per condition) |
| Aggregation | Analysis | No |
| Scoring | Aggregation | No |
| Formatting | Scoring | No |

### 4.2 Parallel Execution
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelExecutor:
    """Execute parallelizable stages concurrently."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def execute_parallel(
        self, 
        tasks: list, 
        timeout_seconds: float = 30.0
    ) -> list:
        """
        Execute tasks in parallel with timeout.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            timeout_seconds: Maximum time for all tasks
        
        Returns:
            List of results
        """
        loop = asyncio.get_event_loop()
        
        futures = []
        for func, args, kwargs in tasks:
            future = loop.run_in_executor(
                self.executor, 
                lambda f=func, a=args, k=kwargs: f(*a, **k)
            )
            futures.append(future)
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout_seconds
            )
            return results
        except asyncio.TimeoutError:
            # Cancel pending futures
            for f in futures:
                f.cancel()
            raise TimeoutError(f"Parallel execution timed out after {timeout_seconds}s")
    
    async def execute_detection_parallel(
        self, 
        image: np.ndarray,
        detectors: dict
    ) -> dict:
        """Run multiple detection modules in parallel."""
        
        tasks = []
        for name, detector in detectors.items():
            tasks.append((detector.detect, (image,), {}))
        
        results = await self.execute_parallel(tasks)
        
        return {
            name: result 
            for name, result in zip(detectors.keys(), results)
            if not isinstance(result, Exception)
        }
```

---

## 5. Pipeline State Tracking

### 5.1 State Tracker
```python
class PipelineStateTracker:
    """Track pipeline execution state."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.current_state = PipelineState.START
        self.transitions: List[StateTransition] = []
        self.stage_outputs: dict = {}
        self.start_time = datetime.utcnow()
    
    def transition(
        self, 
        to_state: PipelineState,
        success: bool = True,
        error: str = None,
        output: dict = None
    ):
        """Record state transition."""
        
        now = datetime.utcnow()
        
        # Calculate duration
        if self.transitions:
            last_time = self.transitions[-1].timestamp
        else:
            last_time = self.start_time
        duration_ms = (now - last_time).total_seconds() * 1000
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=now,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
        self.transitions.append(transition)
        
        # Update state
        self.current_state = to_state
        
        # Store output
        if output:
            self.stage_outputs[to_state.name] = output
    
    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "request_id": self.request_id,
            "current_state": self.current_state.name,
            "is_complete": self.current_state in [PipelineState.COMPLETE, PipelineState.FAILED],
            "is_failed": self.current_state == PipelineState.FAILED,
            "stages_completed": [t.to_state.name for t in self.transitions if t.success],
            "total_duration_ms": sum(t.duration_ms for t in self.transitions),
            "last_error": next(
                (t.error for t in reversed(self.transitions) if t.error), 
                None
            )
        }
    
    def get_stage_timings(self) -> dict:
        """Get timing breakdown by stage."""
        return {
            t.to_state.name: t.duration_ms 
            for t in self.transitions
        }
```

---

## 6. Retry Logic

### 6.1 Retry Configuration
```python
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay_seconds": 1.0,
    "exponential_backoff": True,
    "backoff_multiplier": 2.0,
    
    # Retryable errors
    "retryable_errors": [
        "E_INF_001",  # Model timeout
        "E_INF_002",  # Model load failed
        "E_SYS_001",  # Internal server error
        "E_SYS_003",  # Rate limit
    ],
    
    # Non-retryable errors
    "fatal_errors": [
        "E_VAL_001",  # No valid input
        "E_VAL_002",  # Invalid format
        "E_GEN_007",  # Non-medical image
    ]
}
```

### 6.2 Retry Executor
```python
import time
from typing import Callable

class RetryExecutor:
    """Execute with retry logic."""
    
    def __init__(self, config: dict = None):
        self.config = config or RETRY_CONFIG
    
    def execute_with_retry(
        self, 
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        stage_name: str = "unknown"
    ) -> dict:
        """Execute function with retry on recoverable errors."""
        
        kwargs = kwargs or {}
        max_retries = self.config["max_retries"]
        delay = self.config["retry_delay_seconds"]
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1
                }
            
            except Exception as e:
                last_error = e
                error_code = getattr(e, 'code', 'UNKNOWN')
                
                # Check if error is fatal
                if error_code in self.config["fatal_errors"]:
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": error_code,
                        "retryable": False,
                        "attempts": attempt + 1
                    }
                
                # Check if retryable
                if error_code not in self.config["retryable_errors"]:
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": error_code,
                        "retryable": False,
                        "attempts": attempt + 1
                    }
                
                # Wait before retry
                if attempt < max_retries:
                    if self.config["exponential_backoff"]:
                        sleep_time = delay * (self.config["backoff_multiplier"] ** attempt)
                    else:
                        sleep_time = delay
                    time.sleep(sleep_time)
        
        return {
            "success": False,
            "error": str(last_error),
            "retryable": True,
            "attempts": max_retries + 1,
            "exhausted_retries": True
        }
```

---

## 7. Hard-Stop Conditions

### 7.1 Conditions That Halt Pipeline

| Condition | Error Code | Stage | Recovery |
|-----------|------------|-------|----------|
| No valid input | E_VAL_001 | Validation | None |
| Invalid file format | E_VAL_002 | Validation | None |
| Non-medical image | E_GEN_007 | Validation | None |
| Critical decode failure | E_PREP_001 | Preprocessing | None |
| Out of memory | E_INF_003 | Any | Reduce input |
| Model not loaded | E_INF_002 | Inference | System restart |

### 7.2 Hard-Stop Handler
```python
class HardStopHandler:
    """Handle unrecoverable pipeline failures."""
    
    HARD_STOP_CODES = {
        "E_VAL_001", "E_VAL_002", "E_GEN_007",
        "E_PREP_001", "E_DCM_001"
    }
    
    def should_hard_stop(self, error_code: str) -> bool:
        """Determine if error should halt pipeline."""
        return error_code in self.HARD_STOP_CODES
    
    def create_failure_response(
        self,
        error: Exception,
        state_tracker: PipelineStateTracker
    ) -> dict:
        """Create failure response for hard stop."""
        
        return {
            "success": False,
            "request_id": state_tracker.request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_time_ms": sum(
                t.duration_ms for t in state_tracker.transitions
            ),
            
            "error": {
                "code": getattr(error, 'code', 'E_SYS_001'),
                "message": str(error),
                "stage": state_tracker.current_state.name,
                "recoverable": False
            },
            
            "stages_completed": [
                {"stage": t.to_state.name, "time_ms": t.duration_ms}
                for t in state_tracker.transitions if t.success
            ],
            
            "stages_failed": [
                {"stage": t.to_state.name, "error": t.error}
                for t in state_tracker.transitions if not t.success
            ]
        }
```

---

## 8. Pipeline Orchestrator

### 8.1 Main Orchestrator
```python
class RadiologyPipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        self.router = ModalityRouter()
        self.parallel_executor = ParallelExecutor()
        self.retry_executor = RetryExecutor()
        self.hard_stop_handler = HardStopHandler()
    
    async def execute(self, request: dict) -> dict:
        """Execute full pipeline."""
        
        request_id = request.get("request_id", str(uuid.uuid4()))
        tracker = PipelineStateTracker(request_id)
        
        try:
            # Stage 0: Receipt
            tracker.transition(PipelineState.RECEIPT)
            receipt = self._process_receipt(request)
            
            # Stage 1: Validation
            tracker.transition(PipelineState.VALIDATION)
            validation = self._run_validation(request)
            if not validation["is_valid"]:
                raise ValidationError(validation["errors"])
            
            # Detect modality and get config
            modality_info = self.router.detect_modality(request)
            config = self.router.get_pipeline_config(modality_info["modality"])
            
            # Stage 2: Preprocessing
            tracker.transition(PipelineState.PREPROCESSING)
            preprocessed = self._run_preprocessing(request, config)
            
            # Stage 3: Detection
            tracker.transition(PipelineState.DETECTION)
            if config.get("parallel_stages"):
                detection = await self._run_detection_parallel(preprocessed, config)
            else:
                detection = self._run_detection(preprocessed, config)
            
            # Stage 4: Analysis
            tracker.transition(PipelineState.ANALYSIS)
            analysis = self._run_analysis(preprocessed, detection, config)
            
            # Stage 5: Aggregation
            tracker.transition(PipelineState.AGGREGATION)
            aggregated = self._run_aggregation(analysis)
            
            # Stage 6: Scoring
            tracker.transition(PipelineState.SCORING)
            scored = self._run_scoring(aggregated)
            
            # Stage 7: Formatting
            tracker.transition(PipelineState.FORMATTING)
            formatted = self._format_response(scored, tracker)
            
            # Complete
            tracker.transition(PipelineState.COMPLETE)
            
            return formatted
            
        except Exception as e:
            tracker.transition(
                PipelineState.FAILED,
                success=False,
                error=str(e)
            )
            
            return self.hard_stop_handler.create_failure_response(e, tracker)
```

---

## 9. Execution Flow Diagram (Text)

```
REQUEST RECEIVED
       |
       v
[1. RECEIPT] -----> Log request, generate ID
       |
       v
[2. VALIDATION] --> Check formats, validate inputs
       |            |
       |            +--> [FAIL] Return validation error
       v
[3. ROUTE] -------> Detect modality, select pipeline
       |
       +------------+------------+
       |            |            |
       v            v            v
   [X-RAY]       [CT]        [MRI]
   Pipeline     Pipeline     Pipeline
       |            |            |
       +------------+------------+
                    |
                    v
[4. PREPROCESS] --> Normalize, enhance, mask
       |
       v
[5. DETECT] ------> Anatomical structures
       |            (parallel if multi-structure)
       v
[6. ANALYZE] -----> Pathology detection
       |            (parallel if multi-condition)
       v
[7. AGGREGATE] ---> Combine slice/region predictions
       |
       v
[8. SCORE] -------> Risk calculation, recommendations
       |
       v
[9. FORMAT] ------> JSON response, heatmaps
       |
       v
[10. COMPLETE] ---> Return response
```

---

## 10. Stage Confirmation

```json
{
  "stage_complete": "ORCHESTRATION",
  "pipeline_status": {
    "request_id": "req_abc123",
    "modality": "chest_xray",
    "total_stages": 8,
    "completed_stages": 8,
    "failed_stages": 0,
    "total_duration_ms": 2450
  },
  "stage_timings": {
    "RECEIPT": 5,
    "VALIDATION": 120,
    "PREPROCESSING": 450,
    "DETECTION": 380,
    "ANALYSIS": 980,
    "AGGREGATION": 45,
    "SCORING": 70,
    "FORMATTING": 400
  }
}
```
