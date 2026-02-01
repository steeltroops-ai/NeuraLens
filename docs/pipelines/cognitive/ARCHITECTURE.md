# Cognitive Assessment Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Cognitive Assessment (Digital Testing) |
| Version | 2.1.0 |
| Last Updated | 2026-01-22 |
| Clinical Accuracy Target | 82%+ |
| Domains Assessed | Memory, Attention, Executive, Processing Speed, Inhibition |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|                                                                   |
|  [Test Selection]  [Interactive Tests]  [Timer]  [Results]        |
|         |                  |               |          |           |
|         v                  v               v          |           |
|  +---------------+  +------------------+  +--------+  |           |
|  | Domain Toggle |  | Test Components  |  | Precision|           |
|  | - Memory      |  | - Word List      |  | Timer   |  |           |
|  | - Attention   |  | - Reaction Time  |  | (ms)    |  |           |
|  | - Executive   |  | - Trail Making   |  +--------+  |           |
|  | - Speed       |  | - Stroop Test    |              |           |
|  +---------------+  +------------------+              |           |
|         |                  |                          |           |
|         +--------+---------+                          |           |
|                  |                                    |           |
|                  v                                    |           |
|  +------------------------------------------+         |           |
|  |          Test Results JSON               |         |           |
|  |  - Raw scores per test                   |         |           |
|  |  - Response times                        |         |           |
|  |  - Error counts                          |         |           |
|  +------------------------------------------+         |           |
|                  |                                    |           |
+------------------------------------------------------------------+
                   |                                    ^
                   | HTTPS POST /api/cognitive/analyze  |
                   v                                    |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - Test results validation               |                     |
|  |  - Score range verification              |                     |
|  |  - Patient info validation               |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |         SCORING LAYER                    |                     |
|  |  analyzer.py                             |                     |
|  |                                          |                     |
|  |  +----------------------------------+    |                     |
|  |  | Memory Scoring                   |    |                     |
|  |  | - Learning curve analysis        |    |                     |
|  |  | - Retention calculation          |    |                     |
|  |  | - Recognition discriminability   |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Attention Scoring                |    |                     |
|  |  | - RT mean and variability        |    |                     |
|  |  | - Commission/omission errors     |    |                     |
|  |  | - Sustained attention decay      |    |                     |
|  |  +----------------------------------+    |                     |
|  |                 |                        |                     |
|  |  +----------------------------------+    |                     |
|  |  | Executive Scoring                |    |                     |
|  |  | - Trail Making B/A ratio         |    |                     |
|  |  | - Stroop interference            |    |                     |
|  |  | - Cognitive flexibility          |    |                     |
|  |  +----------------------------------+    |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |       NORMALIZATION LAYER                |                     |
|  |  normative_data.py                       |                     |
|  |  - Age adjustment                        |                     |
|  |  - Education adjustment                  |                     |
|  |  - Percentile calculation                |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |       PATTERN ANALYSIS                   |                     |
|  |  patterns.py                             |                     |
|  |  - Cognitive profile analysis            |                     |
|  |  - Domain comparison                     |                     |
|  |  - Clinical pattern matching             |                     |
|  +------------------------------------------+                     |
|                  |                                                |
|                  v                                                |
|  +------------------------------------------+                     |
|  |           OUTPUT LAYER                   |                     |
|  |  - Domain scores                          |                     |
|  |  - Risk assessment                        |                     |
|  |  - Recommendations                        |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
```

---

## 2. Cognitive Test Battery

### 2.1 Test Specifications
```python
TEST_BATTERY = {
    "memory": {
        "word_list_learning": {
            "description": "Learn 10 words across 3 trials",
            "duration_minutes": 3,
            "metrics": ["trial1_correct", "trial2_correct", "trial3_correct"],
            "max_score": 30,
            "clinical_basis": "CERAD Word List"
        },
        "delayed_recall": {
            "description": "Recall words after 5-minute delay",
            "duration_minutes": 1,
            "metrics": ["delayed_correct"],
            "max_score": 10,
            "delay_minutes": 5
        },
        "recognition": {
            "description": "Identify learned vs new words",
            "duration_minutes": 2,
            "metrics": ["hits", "false_positives"],
            "max_score": 10
        }
    },
    "attention": {
        "simple_reaction_time": {
            "description": "Press button when stimulus appears",
            "duration_minutes": 2,
            "metrics": ["mean_rt_ms", "sd_rt_ms"],
            "trials": 30
        },
        "go_nogo": {
            "description": "Respond to target, inhibit to non-target",
            "duration_minutes": 3,
            "metrics": ["accuracy", "commission_errors", "omission_errors"],
            "go_percentage": 0.75
        },
        "continuous_performance": {
            "description": "Monitor for target pattern",
            "duration_minutes": 5,
            "metrics": ["hits", "vigilance_decrement"]
        }
    },
    "executive": {
        "trail_making_a": {
            "description": "Connect numbers 1-25 in sequence",
            "duration_minutes": 1,
            "metrics": ["completion_time_sec", "errors"],
            "cutoff_time": 150
        },
        "trail_making_b": {
            "description": "Connect alternating numbers-letters",
            "duration_minutes": 2,
            "metrics": ["completion_time_sec", "errors"],
            "cutoff_time": 300
        },
        "stroop": {
            "description": "Name ink color, ignore word meaning",
            "duration_minutes": 2,
            "metrics": ["congruent_ms", "incongruent_ms", "interference"]
        }
    },
    "processing_speed": {
        "symbol_digit": {
            "description": "Match symbols to digits using key",
            "duration_seconds": 90,
            "metrics": ["correct_matches"],
            "max_score": 110
        },
        "coding": {
            "description": "Copy symbol patterns",
            "duration_minutes": 2,
            "metrics": ["symbols_completed"]
        }
    }
}
```

### 2.2 Domain Scoring Specifications
```python
DOMAIN_SCORES = {
    "memory": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["learning", "retention", "recognition"],
        "clinical_significance": "Alzheimer's, MCI, amnesia"
    },
    "attention": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["processing_speed", "sustained", "inhibitory"],
        "clinical_significance": "ADHD, delirium, TBI"
    },
    "executive": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["flexibility", "inhibition", "planning"],
        "clinical_significance": "Frontal dysfunction, vascular dementia"
    },
    "processing_speed": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["simple_speed", "complex_speed"],
        "clinical_significance": "Normal aging, white matter disease"
    },
    "language": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["fluency", "naming"],
        "clinical_significance": "Aphasia, semantic dementia"
    },
    "visuospatial": {
        "unit": "score",
        "normal_range": (0.70, 1.00),
        "abnormal_threshold": 0.50,
        "components": ["construction", "perception"],
        "clinical_significance": "Posterior cortical atrophy"
    }
}
```

---

## 3. Scoring Algorithms

### 3.1 Memory Scoring
```python
def score_memory(results: dict) -> dict:
    """
    Score memory test results using clinical criteria
    
    Components:
    1. Total learning (sum of 3 trials, max 30)
    2. Learning slope (improvement across trials)
    3. Delayed recall (retention)
    4. Recognition discriminability (hits - false positives)
    """
    
    # Learning trials (max 30 = 10 words x 3 trials)
    trial1 = results.get('trial1_correct', 0)
    trial2 = results.get('trial2_correct', 0)
    trial3 = results.get('trial3_correct', 0)
    
    total_learning = trial1 + trial2 + trial3
    learning_score = total_learning / 30.0
    
    # Learning slope (improvement per trial)
    slope = (trial3 - trial1) / 2.0
    
    # Delayed recall (max 10)
    delayed = results.get('delayed_correct', 0)
    delayed_score = delayed / 10.0
    
    # Retention percentage (if trial3 > 0)
    retention = delayed / max(trial3, 1)
    
    # Recognition discriminability
    hits = results.get('recognition_hits', 0) / 10.0
    false_positives = results.get('recognition_false_positives', 0) / 10.0
    discriminability = hits - false_positives
    
    # Combined weighted score
    memory_score = (
        learning_score * 0.30 +
        delayed_score * 0.40 +  # Delayed recall weighted highest (clinical importance)
        max(0, discriminability) * 0.30
    )
    
    return {
        'score': round(memory_score, 3),
        'total_learning': total_learning,
        'learning_slope': round(slope, 2),
        'delayed_recall': delayed,
        'retention_percent': round(retention * 100, 1),
        'discriminability': round(discriminability, 3),
        'components': {
            'learning': round(learning_score, 3),
            'retention': round(delayed_score, 3),
            'recognition': round(max(0, discriminability), 3)
        }
    }
```

### 3.2 Attention Scoring
```python
def score_attention(results: dict) -> dict:
    """
    Score attention test results
    
    Components:
    1. Simple RT - processing speed baseline
    2. Go/No-Go - response inhibition
    3. CPT - sustained attention
    """
    
    # Simple Reaction Time
    mean_rt = results.get('simple_rt_mean_ms', 300)
    sd_rt = results.get('simple_rt_sd_ms', 50)
    
    # RT score (faster is better, normalized)
    # Expected normal: 200-350ms
    rt_score = max(0, min(1, (400 - mean_rt) / 200))
    
    # RT variability (lower is better)
    variability_score = max(0, min(1, (100 - sd_rt) / 100))
    
    # Go/No-Go
    accuracy = results.get('go_nogo_accuracy', 0.9)
    commission = results.get('commission_errors', 0)
    omission = results.get('omission_errors', 0)
    
    # Inhibitory control (fewer commissions = better)
    inhibition_score = max(0, 1 - commission / 10)
    
    # Combined attention score
    attention_score = (
        rt_score * 0.25 +
        variability_score * 0.15 +
        accuracy * 0.30 +
        inhibition_score * 0.30
    )
    
    return {
        'score': round(attention_score, 3),
        'mean_rt_ms': mean_rt,
        'rt_variability': round(sd_rt, 1),
        'accuracy': round(accuracy, 3),
        'commission_rate': round(commission / 30 if commission else 0, 3),
        'omission_rate': round(omission / 30 if omission else 0, 3),
        'components': {
            'processing_speed': round(rt_score, 3),
            'sustained': round(accuracy, 3),
            'inhibitory': round(inhibition_score, 3)
        }
    }
```

### 3.3 Executive Function Scoring
```python
def score_executive(results: dict) -> dict:
    """
    Score executive function tests
    
    Components:
    1. Trail Making B/A ratio (cognitive flexibility)
    2. Stroop interference (inhibition)
    3. Combined executive score
    """
    
    # Trail Making
    trail_a = results.get('trail_a_time_sec', 30)
    trail_b = results.get('trail_b_time_sec', 70)
    trail_a_errors = results.get('trail_a_errors', 0)
    trail_b_errors = results.get('trail_b_errors', 0)
    
    # B/A ratio (normal ~2.0-3.0, higher = worse flexibility)
    ba_ratio = trail_b / max(trail_a, 1)
    
    # Trail Making scores (faster is better)
    trail_a_score = max(0, min(1, (60 - trail_a) / 60))
    trail_b_score = max(0, min(1, (150 - trail_b) / 150))
    
    # Cognitive flexibility (lower ratio is better)
    flexibility_score = max(0, min(1, (4 - ba_ratio) / 3))
    
    # Stroop
    congruent = results.get('stroop_congruent_ms', 600)
    incongruent = results.get('stroop_incongruent_ms', 900)
    
    # Stroop interference effect
    interference = incongruent - congruent
    interference_ratio = incongruent / max(congruent, 1)
    
    # Inhibition score (smaller interference = better)
    # Normal interference: 100-200ms
    inhibition_score = max(0, min(1, (300 - interference) / 300))
    
    # Combined executive score
    executive_score = (
        flexibility_score * 0.40 +
        inhibition_score * 0.35 +
        (trail_a_score + trail_b_score) / 2 * 0.25
    )
    
    return {
        'score': round(executive_score, 3),
        'trail_a_time': trail_a,
        'trail_b_time': trail_b,
        'ba_ratio': round(ba_ratio, 2),
        'stroop_interference_ms': round(interference, 0),
        'stroop_ratio': round(interference_ratio, 2),
        'components': {
            'cognitive_flexibility': round(flexibility_score, 3),
            'inhibition': round(inhibition_score, 3),
            'processing_speed': round((trail_a_score + trail_b_score) / 2, 3)
        }
    }
```

---

## 4. Normative Data & Age Adjustment

### 4.1 Normative Tables
```python
# Based on published normative studies (NACC, CERAD, etc.)
NORMATIVE_DATA = {
    'memory': {
        # age_group: (mean, standard_deviation)
        '50-54': (0.88, 0.08),
        '55-59': (0.85, 0.10),
        '60-64': (0.82, 0.11),
        '65-69': (0.78, 0.12),
        '70-74': (0.74, 0.13),
        '75-79': (0.70, 0.14),
        '80-84': (0.65, 0.15),
        '85+': (0.60, 0.16)
    },
    'attention': {
        '50-54': (0.90, 0.07),
        '55-59': (0.88, 0.08),
        '60-64': (0.85, 0.09),
        '65-69': (0.82, 0.10),
        '70-74': (0.78, 0.11),
        '75-79': (0.74, 0.12),
        '80-84': (0.70, 0.13),
        '85+': (0.65, 0.14)
    },
    'executive': {
        '50-54': (0.87, 0.09),
        '55-59': (0.84, 0.10),
        '60-64': (0.80, 0.11),
        '65-69': (0.76, 0.12),
        '70-74': (0.72, 0.13),
        '75-79': (0.68, 0.14),
        '80-84': (0.63, 0.15),
        '85+': (0.58, 0.16)
    },
    'processing_speed': {
        '50-54': (0.92, 0.06),
        '55-59': (0.88, 0.08),
        '60-64': (0.83, 0.10),
        '65-69': (0.78, 0.11),
        '70-74': (0.72, 0.12),
        '75-79': (0.66, 0.13),
        '80-84': (0.60, 0.14),
        '85+': (0.54, 0.15)
    }
}

def get_age_group(age: int) -> str:
    """Get age group string for normative lookup"""
    if age >= 85:
        return '85+'
    decade_start = (age // 5) * 5
    return f'{decade_start}-{decade_start + 4}'

def age_adjust_score(raw_score: float, age: int, domain: str) -> dict:
    """
    Convert raw score to age-adjusted percentile
    
    Args:
        raw_score: Domain score (0-1)
        age: Patient age
        domain: Cognitive domain name
    
    Returns:
        dict with z-score, percentile, interpretation
    """
    from scipy.stats import norm
    
    age_group = get_age_group(age)
    norms = NORMATIVE_DATA.get(domain, {}).get(age_group, (0.75, 0.12))
    
    mean, sd = norms
    
    # Z-score
    z_score = (raw_score - mean) / sd
    
    # Percentile
    percentile = int(norm.cdf(z_score) * 100)
    
    # Interpretation
    if percentile < 5:
        interpretation = "severely_impaired"
    elif percentile < 16:
        interpretation = "impaired"
    elif percentile < 25:
        interpretation = "low_normal"
    elif percentile <= 75:
        interpretation = "normal"
    elif percentile <= 90:
        interpretation = "high_normal"
    else:
        interpretation = "superior"
    
    return {
        'raw_score': raw_score,
        'z_score': round(z_score, 2),
        'percentile': min(99, max(1, percentile)),
        'age_group': age_group,
        'interpretation': interpretation
    }
```

---

## 5. Clinical Pattern Analysis

### 5.1 Pattern Matching
```python
CLINICAL_PATTERNS = {
    "alzheimers_pattern": {
        "description": "Memory >> Executive > Attention",
        "pattern": {
            "memory": {"threshold": 0.60, "direction": "below"},
            "executive": {"threshold": 0.70, "direction": "below", "relative_to_memory": "better"}
        },
        "recommendation": "Consider Alzheimer's disease evaluation"
    },
    "vascular_pattern": {
        "description": "Executive >> Memory, Processing Speed affected",
        "pattern": {
            "executive": {"threshold": 0.55, "direction": "below"},
            "processing_speed": {"threshold": 0.60, "direction": "below"},
            "memory": {"threshold": 0.65, "direction": "above"}  # Relatively preserved
        },
        "recommendation": "Consider vascular cognitive impairment evaluation, MRI recommended"
    },
    "subcortical_pattern": {
        "description": "Processing Speed disproportionately affected",
        "pattern": {
            "processing_speed": {"threshold": 0.50, "direction": "below"},
            "executive": {"threshold": 0.70, "direction": "at_or_above"}
        },
        "recommendation": "Rule out depression, consider subcortical pathology"
    },
    "mci_amnestic": {
        "description": "Memory impaired, other domains intact",
        "pattern": {
            "memory": {"threshold": 0.60, "direction": "below"},
            "attention": {"threshold": 0.70, "direction": "above"},
            "executive": {"threshold": 0.70, "direction": "above"}
        },
        "recommendation": "Meets criteria for amnestic MCI, monitor every 6 months"
    }
}

def detect_clinical_pattern(domain_scores: dict) -> list:
    """
    Match domain scores against clinical patterns
    """
    detected_patterns = []
    
    for pattern_name, pattern_def in CLINICAL_PATTERNS.items():
        matches = True
        criteria = pattern_def["pattern"]
        
        for domain, criterion in criteria.items():
            score = domain_scores.get(domain, {}).get("score", 0.75)
            threshold = criterion["threshold"]
            direction = criterion["direction"]
            
            if direction == "below" and score >= threshold:
                matches = False
            elif direction == "above" and score < threshold:
                matches = False
            elif direction == "at_or_above" and score < threshold:
                matches = False
        
        if matches:
            detected_patterns.append({
                "pattern": pattern_name,
                "description": pattern_def["description"],
                "recommendation": pattern_def["recommendation"]
            })
    
    return detected_patterns
```

---

## 6. Risk Calculation

### 6.1 Cognitive Risk Score
```python
def calculate_cognitive_risk(domain_scores: dict, age: int) -> dict:
    """
    Calculate overall cognitive risk score
    """
    # Weight domains by clinical importance
    domain_weights = {
        'memory': 0.30,
        'executive': 0.25,
        'attention': 0.20,
        'processing_speed': 0.15,
        'language': 0.05,
        'visuospatial': 0.05
    }
    
    weighted_score = 0
    total_weight = 0
    
    for domain, weight in domain_weights.items():
        if domain in domain_scores:
            score = domain_scores[domain].get('score', 0.75)
            weighted_score += score * weight
            total_weight += weight
    
    if total_weight > 0:
        cognitive_score = weighted_score / total_weight
    else:
        cognitive_score = 0.75  # Default
    
    # Convert to risk score (0-100, higher = more risk)
    risk_score = (1 - cognitive_score) * 100
    
    # Categorize
    if risk_score < 20:
        category = "normal"
    elif risk_score < 35:
        category = "mild_concern"
    elif risk_score < 55:
        category = "moderate_impairment"
    else:
        category = "significant_impairment"
    
    return {
        'cognitive_score': round(cognitive_score, 3),
        'risk_score': round(risk_score, 1),
        'category': category,
        'age_adjusted': True
    }
```

---

## 7. Technology Stack

### 7.1 Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# Scoring
numpy>=1.24.0
scipy>=1.11.0  # For normative calculations

# No ML - all rule-based scoring
```

### 7.2 Frontend Test Components
```txt
# Timer precision
performance.now() API

# Canvas (Trail Making)
HTML5 Canvas API

# Audio (if verbal fluency)
Web Audio API
```

---

## 8. File Structure

```
backend/app/pipelines/cognitive/
├── __init__.py
├── config.py                   # Pipeline configuration (VERSION, thresholds)
├── schemas.py                  # Pydantic models (CognitiveSessionInput, etc.)
├── router.py                   # FastAPI endpoints
│
├── input/                      # Input Layer
│   ├── __init__.py
│   └── validator.py            # Session/task validation
│
├── features/                   # Feature Extraction
│   ├── __init__.py
│   └── extractor.py            # Task-specific feature extraction
│
├── analysis/                   # NEW: Research-grade Analysis
│   ├── __init__.py
│   ├── statistics.py           # SDT, MAD, RT analysis (PhD-grade)
│   └── normative.py            # Age-adjusted norms (Tombaugh, Wechsler, etc.)
│
├── clinical/                   # Clinical Scoring
│   ├── __init__.py
│   └── risk_scorer.py          # Risk assessment with normative comparison
│
├── output/                     # Output Formatting
│   ├── __init__.py
│   └── formatter.py            # Response formatting
│
├── reporting/                  # NEW: Report Generation
│   ├── __init__.py
│   └── pdf_generator.py        # Clinical PDF reports (reportlab)
│
├── core/                       # Core Orchestration
│   ├── __init__.py
│   └── service.py              # Pipeline orchestrator
│
└── errors/                     # Error Handling
    ├── __init__.py
    └── codes.py                # Error definitions

docs/pipelines/cognitive/
├── ARCHITECTURE.md             # This document
├── API_SPECIFICATION.md        # API contract
└── RESEARCH_METHODOLOGY.md     # NEW: PhD-grade methods documentation
```

---

## 9. Clinical References

### Core References
1. **Weintraub et al. (2009)** - "The Alzheimer's Disease Centers' Uniform Data Set (UDS)"
2. **Nasreddine et al. (2005)** - "The Montreal Cognitive Assessment (MoCA)"
3. **Reitan (1958)** - "Trail Making Test: Manual for Administration"
4. **Stroop (1935)** - "Studies of interference in serial verbal reactions"
5. **NACC** - National Alzheimer's Coordinating Center Normative Database

### v2.1.0 Research Methodology References
6. **Leys et al. (2013)** - MAD-based outlier detection for RT data
7. **Stanislaw & Todorov (1999)** - Signal Detection Theory (d-prime calculation)
8. **Hautus (1995)** - Log-linear correction for extreme hit/FA rates
9. **Tombaugh (2004)** - Trail Making Test normative data
10. **Scarpina & Tagini (2017)** - Stroop effect meta-analysis
11. **Wechsler (2008)** - WAIS-IV Digit Symbol norms
12. **Jaeggi et al. (2010)** - N-back working memory validation
13. **Verbruggen & Logan (2008)** - Go/No-Go inhibition paradigm
14. **Ratcliff (1993)** - Ex-Gaussian RT distribution modeling

---

## 10. Version History

| Version | Date | Changes |
|---------|------|--------|
| 1.0.0 | 2025-10-01 | Initial pipeline |
| 2.0.0 | 2026-01-17 | Production-grade refactor |
| 2.1.0 | 2026-01-22 | PhD-grade research upgrade: SDT, MAD, normative data, PDF reports |
