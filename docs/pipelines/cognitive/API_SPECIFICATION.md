# Cognitive Pipeline API Specification v2.0.0

## Overview

The Cognitive Assessment Pipeline provides browser-based cognitive testing with clinical-grade analysis. This document defines the complete API contract for frontend-backend integration.

---

## Endpoints

### POST `/api/cognitive/analyze`

Process a completed cognitive assessment session.

#### Request

```json
{
  "session_id": "sess_1705881234567",
  "patient_id": "optional_patient_id",
  "tasks": [
    {
      "task_id": "reaction_time_v1",
      "start_time": "2026-01-22T00:00:00Z",
      "end_time": "2026-01-22T00:02:00Z",
      "events": [
        {
          "timestamp": 0,
          "event_type": "test_start",
          "payload": {}
        },
        {
          "timestamp": 2500,
          "event_type": "stimulus_shown",
          "payload": { "trial": 0 }
        },
        {
          "timestamp": 2750,
          "event_type": "response_received",
          "payload": { "trial": 0, "rt": 250 }
        }
      ],
      "metadata": { "browser": "Chrome" }
    }
  ],
  "user_metadata": {
    "userAgent": "Mozilla/5.0..."
  }
}
```

#### Response (Success)

```json
{
  "session_id": "sess_1705881234567",
  "pipeline_version": "2.0.0",
  "timestamp": "2026-01-22T00:02:30Z",
  "processing_time_ms": 145.3,
  "status": "success",
  "stages": [
    {
      "stage": "validating",
      "stage_index": 0,
      "total_stages": 4,
      "message": "Validation passed",
      "duration_ms": 12.5
    },
    {
      "stage": "extracting",
      "stage_index": 1,
      "total_stages": 4,
      "message": "Extracted 2 domain scores",
      "duration_ms": 45.2
    },
    {
      "stage": "scoring",
      "stage_index": 2,
      "total_stages": 4,
      "message": "Risk level: low",
      "duration_ms": 33.1
    },
    {
      "stage": "complete",
      "stage_index": 3,
      "total_stages": 4,
      "message": "Generating response",
      "duration_ms": 5.0
    }
  ],
  "risk_assessment": {
    "overall_risk_score": 0.23,
    "risk_level": "low",
    "confidence_score": 0.87,
    "confidence_interval": [0.18, 0.28],
    "domain_risks": {
      "memory": {
        "score": 0.15,
        "risk_level": "low",
        "percentile": 85,
        "confidence": 0.88,
        "contributing_factors": []
      },
      "processing_speed": {
        "score": 0.31,
        "risk_level": "low",
        "percentile": 69,
        "confidence": 0.85,
        "contributing_factors": []
      }
    }
  },
  "features": {
    "domain_scores": {
      "memory": 0.85,
      "processing_speed": 0.69
    },
    "raw_metrics": [
      {
        "task_id": "reaction_time_v1",
        "completion_status": "complete",
        "performance_score": 75.5,
        "parameters": {
          "mean_rt": 285.3,
          "std_rt": 45.2,
          "valid_trials": 5,
          "error_rate": 0.1
        },
        "validity_flag": true,
        "quality_warnings": []
      }
    ],
    "fatigue_index": 0.12,
    "consistency_score": 0.91,
    "valid_task_count": 2,
    "total_task_count": 2
  },
  "recommendations": [
    {
      "category": "routine",
      "description": "Cognitive function appears within normal range. Re-test in 6-12 months.",
      "priority": "low"
    }
  ],
  "explainability": {
    "summary": "Cognitive screening indicates normal function across tested domains. Assessment reliability: high.",
    "key_factors": ["All domains within normal range"],
    "domain_contributions": {
      "memory": 0.0375,
      "processing_speed": 0.062
    },
    "methodology_note": "Risk calculated using weighted domain aggregation with age-normalized thresholds."
  },
  "error_code": null,
  "error_message": null,
  "recoverable": true
}
```

#### Response (Error)

```json
{
  "session_id": "sess_1705881234567",
  "pipeline_version": "2.0.0",
  "timestamp": "2026-01-22T00:02:30Z",
  "processing_time_ms": 15.0,
  "status": "failed",
  "stages": [
    {
      "stage": "failed",
      "stage_index": 0,
      "message": "Validation failed",
      "error": "E_INP_002: Invalid task data structure"
    }
  ],
  "risk_assessment": null,
  "features": null,
  "recommendations": [],
  "explainability": null,
  "error_code": "E_INP_002",
  "error_message": "Invalid task data structure",
  "recoverable": true,
  "retry_after_ms": null
}
```

---

### POST `/api/cognitive/validate`

Validate session data without full processing (dry run).

#### Request

Same as `/analyze`

#### Response

```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Task reaction_time_v1: High variability in response times"],
  "task_validity": {
    "reaction_time_v1": true,
    "n_back_2": true
  }
}
```

---

### GET `/api/cognitive/health`

Health check endpoint.

#### Response

```json
{
  "status": "ok",
  "service": "cognitive-pipeline",
  "version": "2.0.0",
  "uptime_seconds": 3600,
  "last_request_at": "2026-01-22T00:00:00Z"
}
```

---

### GET `/api/cognitive/schema`

Returns the expected request/response schema for integration.

---

## Status Codes

| Code | Meaning | Recoverable |
|------|---------|-------------|
| 200 | Success (full or partial) | Yes |
| 400 | Validation error | Yes |
| 413 | Request body too large | Yes |
| 429 | Rate limit exceeded | Yes (after delay) |
| 500 | Internal processing error | No |
| 502 | Backend unavailable | Yes |

---

## Error Codes

| Code | Layer | Description |
|------|-------|-------------|
| E_HTTP_001 | HTTP | Invalid request format |
| E_HTTP_002 | HTTP | Missing required field |
| E_INP_001 | Input | Empty session data |
| E_INP_002 | Input | Invalid task data structure |
| E_INP_003 | Input | Session timestamp mismatch |
| E_INP_004 | Input | Suspicious timing data detected |
| E_INP_005 | Input | Invalid session_id format |
| E_INP_006 | Input | Task events not monotonic |
| E_INP_007 | Input | Insufficient events for analysis |
| E_FEAT_001 | Feature | Feature extraction failed |
| E_FEAT_002 | Feature | Unknown task type |
| E_FEAT_003 | Feature | Insufficient valid trials |
| E_CLIN_001 | Clinical | Risk model convergence failure |
| E_CLIN_002 | Clinical | No valid domain scores |

---

## Event Types

| Event Type | Description | Payload |
|------------|-------------|---------|
| `test_start` | Test begins | `{}` or `{"n": 2}` for N-Back |
| `stimulus_shown` | Visual stimulus presented | `{"trial": 0}` |
| `response_received` | Valid user response | `{"trial": 0, "rt": 250}` |
| `response_early` | Anticipatory response | `{"trial": 0}` |
| `trial_result` | Trial outcome | `{"result": "hit|miss|false_alarm|correct_rejection"}` |
| `test_end` | Test completes | `{"completed": true}` |

---

## Rate Limits

- **Standard**: 60 requests/minute per session
- **Burst**: 10 requests/second

---

## Retry Semantics

1. On `recoverable: true`, client may retry
2. If `retry_after_ms` is set, wait that duration
3. Maximum 3 retries recommended
4. Use exponential backoff

---

## Versioning

API version is included in response as `pipeline_version`. Breaking changes will increment major version.

Current: `2.0.0`
