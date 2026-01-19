# 07 - Pipeline Orchestration and State Tracking

## Document Info
| Field | Value |
|-------|-------|
| Stage | Orchestration |
| Owner | ML Systems Architect |
| Reviewer | All Team Members |

---

## 1. Pipeline State Machine

### 1.1 State Diagram
```
[INIT] --> [RECEIPT] --> [VALIDATION] --> [PREPROCESSING] -->
    |          |              |                 |
    |          v              v                 v
    |      (log receipt)  (validate)        (preprocess)
    |          |              |                 |
    |          +--[ERROR]-----+--------+--------+
    |                                  |
    +----------------------------------+
    
[PREPROCESSING] --> [DETECTION] --> [ANALYSIS] --> [FUSION] -->
       |                |               |              |
       v                v               v              v
   (clean data)    (find structures)  (compute)    (combine)
       |                |               |              |
       +---[ERROR]------+-------+-------+------+-------+
                                               |
[FUSION] --> [SCORING] --> [FORMATTING] --> [COMPLETE]
    |            |              |               |
    v            v              v               v
 (fuse)      (risk score)   (JSON out)      (return)
    |            |              |
    +--[ERROR]---+------+-------+
```

### 1.2 State Definitions
| State ID | State Name | Entry Condition | Exit Condition |
|----------|------------|-----------------|----------------|
| 0 | INIT | Request received | Logging complete |
| 1 | RECEIPT | Init complete | Acknowledgment sent |
| 2 | VALIDATION | Receipt confirmed | All inputs valid |
| 3 | PREPROCESSING | Validation passed | Quality gates passed |
| 4 | DETECTION | Preprocessing done | Structures found |
| 5 | ANALYSIS | Detection complete | Metrics computed |
| 6 | FUSION | Analysis done | Features combined |
| 7 | SCORING | Fusion complete | Risk calculated |
| 8 | FORMATTING | Scoring done | Response built |
| 9 | COMPLETE | Formatting done | Response sent |
| -1 | ERROR | Any stage failure | Error response sent |

---

## 2. Execution Flow

### 2.1 Modality-Specific Branches
```python
class PipelineOrchestrator:
    """Orchestrate cardiology pipeline execution."""
    
    def execute(self, request: CardiologyRequest) -> CardiologyResponse:
        state = PipelineState()
        
        try:
            # Stage 0: Receipt
            state.transition("RECEIPT")
            self._log_receipt(request)
            
            # Stage 1: Validation
            state.transition("VALIDATION")
            validation = self._validate(request)
            if not validation.is_valid:
                raise ValidationError(validation.errors)
            
            # Stage 2: Preprocessing (parallel branches)
            state.transition("PREPROCESSING")
            preprocessed = {}
            
            if request.has_echo:
                preprocessed['echo'] = self._preprocess_echo(request.echo_data)
            
            if request.has_ecg:
                preprocessed['ecg'] = self._preprocess_ecg(request.ecg_data)
            
            if request.has_metadata:
                preprocessed['metadata'] = self._normalize_metadata(request.metadata)
            
            # Stage 3: Detection (echo only)
            if 'echo' in preprocessed:
                state.transition("DETECTION")
                detection = self._detect_structures(preprocessed['echo'])
            else:
                detection = None
            
            # Stage 4: Analysis (parallel)
            state.transition("ANALYSIS")
            analysis = {}
            
            if detection:
                analysis['echo'] = self._analyze_echo(detection)
            
            if 'ecg' in preprocessed:
                analysis['ecg'] = self._analyze_ecg(preprocessed['ecg'])
            
            # Stage 5: Fusion
            state.transition("FUSION")
            fused = self._fuse_modalities(analysis, preprocessed.get('metadata'))
            
            # Stage 6: Scoring
            state.transition("SCORING")
            risk = self._compute_risk(fused)
            
            # Stage 7: Formatting
            state.transition("FORMATTING")
            response = self._format_response(analysis, risk, state)
            
            # Stage 8: Complete
            state.transition("COMPLETE")
            return response
            
        except PipelineError as e:
            state.transition("ERROR")
            return self._format_error(e, state)
```

### 2.2 Parallel Execution
```
Request
   |
   +-- Echo Branch (if present) --+
   |                              |
   +-- ECG Branch (if present) ---+--> Barrier --> Fusion --> Output
   |                              |
   +-- Metadata Branch -----------+
```

---

## 3. State Checkpoints

### 3.1 Checkpoint Data
```python
@dataclass
class StateCheckpoint:
    """Checkpoint for pipeline state recovery."""
    
    state_id: int
    state_name: str
    timestamp: datetime
    
    # Input hashes for validation
    input_hashes: Dict[str, str]
    
    # Intermediate results
    validation_result: Optional[ValidationReport]
    preprocessed_echo: Optional[bytes]  # Serialized
    preprocessed_ecg: Optional[bytes]
    detection_result: Optional[bytes]
    analysis_result: Optional[bytes]
    
    # Metrics
    processing_times: Dict[str, int]
    memory_usage_mb: float
    
    def save(self, checkpoint_dir: str):
        """Save checkpoint to disk."""
        path = f"{checkpoint_dir}/checkpoint_{self.state_id}_{self.timestamp.isoformat()}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'StateCheckpoint':
        """Load checkpoint from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
```

### 3.2 Checkpoint Strategy
| Stage | Checkpoint? | Reason |
|-------|-------------|--------|
| RECEIPT | No | Lightweight |
| VALIDATION | Yes | Input verified |
| PREPROCESSING | Yes | Expensive computation |
| DETECTION | Yes | Model inference done |
| ANALYSIS | Yes | Metrics computed |
| FUSION | No | Quick operation |
| SCORING | No | Quick operation |
| FORMATTING | No | Quick operation |

---

## 4. Retry Logic

### 4.1 Recoverable Failures
| Failure Type | Retry Strategy | Max Attempts |
|--------------|---------------|--------------|
| Timeout | Exponential backoff | 3 |
| Memory error | Reduce batch size, retry | 2 |
| Model load failure | Reload model, retry | 2 |
| Transient network | Fixed delay, retry | 3 |

### 4.2 Non-Recoverable Failures
| Failure Type | Action |
|--------------|--------|
| Validation failure | Return error immediately |
| Corrupted input | Return error with details |
| All retries exhausted | Return error with diagnostics |
| Critical model failure | Alert, return error |

### 4.3 Retry Implementation
```python
def with_retry(
    func: Callable,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    recoverable_errors: Tuple = (TimeoutError, MemoryError)
) -> Any:
    """Execute function with retry logic."""
    
    last_error = None
    delay = 1.0
    
    for attempt in range(max_attempts):
        try:
            return func()
        except recoverable_errors as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise RetryExhaustedError(f"Failed after {max_attempts} attempts: {e}")
    
    raise last_error
```

---

## 5. Hard-Stop Conditions

### 5.1 Immediate Termination
| Condition | Reason | Response Code |
|-----------|--------|---------------|
| No valid modality | Cannot analyze nothing | E_HARD_001 |
| Critical memory exceeded | System stability | E_HARD_002 |
| Model file corrupted | Cannot proceed | E_HARD_003 |
| Input size limit exceeded | Security | E_HARD_004 |

### 5.2 Graceful Degradation
| Condition | Degraded Behavior |
|-----------|------------------|
| Echo model fails, ECG available | ECG-only analysis |
| Low confidence detection | Skip functional analysis |
| Fusion model fails | Return individual modality results |

---

## 6. Execution Flow Diagram (Text)

```
START
  |
  v
[Log Request Receipt]
  |
  v
[Detect Modalities]
  |
  +-- No valid modality? --> [ERROR: No input]
  |
  v
[Validate Each Modality]
  |
  +-- Validation fails? --> [ERROR: Validation]
  |
  v
[Parallel Preprocessing]
  |-- Echo: Frame select, filter, normalize
  |-- ECG: Filter, normalize, segment
  |-- Metadata: Normalize
  |
  +-- Quality gate fails? --> [ERROR: Quality]
  |
  v
[Echo Detection] (if echo present)
  |
  +-- Detection fails? --> [WARN, continue ECG-only]
  |
  v
[Parallel Analysis]
  |-- Echo: EF, wall motion, chambers
  |-- ECG: HR, HRV, rhythm, arrhythmia
  |
  +-- Analysis fails? --> [ERROR: Analysis]
  |
  v
[Multimodal Fusion]
  |
  v
[Risk Scoring]
  |
  v
[Format Response]
  |
  v
[Return Success Response]
  |
END
```

---

## 7. Stage Output

```json
{
  "stage_complete": "ORCHESTRATION",
  "pipeline_execution": {
    "states_traversed": ["RECEIPT", "VALIDATION", "PREPROCESSING", "DETECTION", "ANALYSIS", "FUSION", "SCORING", "FORMATTING", "COMPLETE"],
    "branches_executed": ["echo", "ecg", "metadata"],
    "checkpoints_saved": 4,
    "retries_performed": 0,
    "total_time_ms": 2450
  },
  "stage_timings": {
    "RECEIPT": 5,
    "VALIDATION": 120,
    "PREPROCESSING": 800,
    "DETECTION": 600,
    "ANALYSIS": 450,
    "FUSION": 50,
    "SCORING": 25,
    "FORMATTING": 100
  }
}
```
