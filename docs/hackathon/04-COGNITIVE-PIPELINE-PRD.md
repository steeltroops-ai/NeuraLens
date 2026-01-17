# MediLens Cognitive Assessment Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P1 - High (Interactive Demo) |
| Est. Dev Time | 5 hours |
| Clinical Validation | Standardized cognitive tests |

---

## 1. Overview

### Purpose
Evaluate cognitive function through standardized digital tests for:
- **Memory** (immediate, delayed, recognition)
- **Attention** (sustained, selective, divided)
- **Executive Function** (planning, inhibition, flexibility)
- **Processing Speed** (reaction time, throughput)
- **Language** (verbal fluency, naming)
- **Visuospatial** (construction, mental rotation)

### Clinical Basis
Digital cognitive testing correlates highly (r=0.85+) with paper-based neuropsychological assessments. Early cognitive changes can precede dementia diagnosis by 5-10 years, making screening valuable for early intervention.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Scoring** | Rule-based algorithms | No ML needed |
| **Normative Data** | Published age norms | Age-adjusted scores |
| **Timer** | High-precision JS | Accurate RT measurement |
| **Statistics** | NumPy/SciPy | Statistical analysis |

### Key Insight
Cognitive testing is **rule-based scoring**, not machine learning. We implement standardized tests and compare to published normative data.

### Installation
```bash
pip install numpy scipy  # That's all!
```

---

## 3. Test Battery

### Memory Tests

| Test | Description | Duration | Metrics |
|------|-------------|----------|---------|
| **Word List Learning** | Learn 10 words, 3 trials | 3 min | Words recalled per trial |
| **Delayed Recall** | Recall words after delay | 1 min | Words recalled |
| **Recognition** | Identify learned vs new words | 2 min | Hits, false positives |

### Attention Tests

| Test | Description | Duration | Metrics |
|------|-------------|----------|---------|
| **Simple Reaction Time** | Press on stimulus | 2 min | Mean RT, variability |
| **Go/No-Go** | Respond to target only | 3 min | Accuracy, commission errors |
| **Continuous Performance** | Monitor for target pattern | 5 min | Hits, vigilance decrement |

### Executive Function Tests

| Test | Description | Duration | Metrics |
|------|-------------|----------|---------|
| **Trail Making A** | Connect 1-2-3-4... | 1 min | Completion time |
| **Trail Making B** | Connect 1-A-2-B-3-C... | 2 min | Completion time, errors |
| **Stroop Test** | Name color, ignore word | 2 min | Interference score |
| **Tower Task** | Plan moves to goal | 3 min | Moves, planning time |

### Processing Speed

| Test | Description | Duration | Metrics |
|------|-------------|----------|---------|
| **Symbol Digit** | Match symbols to digits | 90 sec | Correct matches |
| **Coding** | Copy symbol patterns | 2 min | Symbols completed |

---

## 4. Biomarkers Specification

### Primary Biomarkers (6 Domain Scores)

| # | Domain | Normal Range | Abnormal | Scoring Basis |
|---|--------|--------------|----------|---------------|
| 1 | **Memory** | 0.70-1.00 | <0.50 | Recall accuracy |
| 2 | **Attention** | 0.70-1.00 | <0.50 | RT + accuracy combined |
| 3 | **Executive** | 0.70-1.00 | <0.50 | Time + errors |
| 4 | **Processing Speed** | 0.70-1.00 | <0.50 | Items completed |
| 5 | **Language** | 0.70-1.00 | <0.50 | Words generated |
| 6 | **Visuospatial** | 0.70-1.00 | <0.50 | Accuracy score |

### Clinical Interpretation

| Pattern | Possible Condition | Recommendation |
|---------|-------------------|----------------|
| Memory ↓↓, Executive ↓ | Alzheimer's type | Neurology referral |
| Executive ↓↓, Attention ↓ | Vascular/frontal | MRI recommended |
| All domains ↓ mildly | MCI | Monitor 6 months |
| Processing speed ↓↓ only | Subcortical | Rule out depression |

---

## 5. API Specification

### Endpoint
```
POST /api/cognitive/analyze
Content-Type: application/json
```

### Request
```json
{
  "session_id": "cog_session_123",
  "patient_info": {
    "age": 68,
    "education_years": 16,
    "primary_language": "english"
  },
  "test_battery": ["memory", "attention", "executive"],
  "test_results": {
    "memory": {
      "trial1_correct": 5,
      "trial2_correct": 7,
      "trial3_correct": 8,
      "delayed_correct": 6,
      "recognition_hits": 9,
      "recognition_false_positives": 1
    },
    "attention": {
      "simple_rt_mean_ms": 285,
      "simple_rt_sd_ms": 45,
      "go_nogo_accuracy": 0.92,
      "commission_errors": 2,
      "omission_errors": 1
    },
    "executive": {
      "trail_a_time_sec": 32,
      "trail_a_errors": 0,
      "trail_b_time_sec": 68,
      "trail_b_errors": 1,
      "stroop_congruent_ms": 620,
      "stroop_incongruent_ms": 850
    }
  },
  "difficulty": "standard"
}
```

### Response
```json
{
  "success": true,
  "session_id": "cog_session_123",
  "timestamp": "2026-01-17T14:30:00Z",
  "processing_time_ms": 125,
  
  "overall_assessment": {
    "cognitive_score": 0.82,
    "risk_score": 18.0,
    "category": "normal",
    "confidence": 0.88,
    "age_adjusted": true
  },
  
  "domain_scores": {
    "memory": {
      "raw_score": 0.78,
      "age_adjusted_score": 0.82,
      "percentile": 55,
      "status": "normal",
      "components": {
        "learning_slope": 1.5,
        "delayed_retention": 0.75,
        "recognition_discriminability": 0.90
      }
    },
    "attention": {
      "raw_score": 0.85,
      "age_adjusted_score": 0.88,
      "percentile": 68,
      "status": "normal",
      "components": {
        "processing_speed": 0.82,
        "sustained_attention": 0.88,
        "inhibitory_control": 0.85
      }
    },
    "executive": {
      "raw_score": 0.79,
      "age_adjusted_score": 0.83,
      "percentile": 58,
      "status": "normal",
      "components": {
        "cognitive_flexibility": 0.78,
        "inhibition": 0.80,
        "processing_speed": 0.79
      }
    }
  },
  
  "detailed_metrics": {
    "memory": {
      "total_learning": 20,
      "learning_slope": 1.5,
      "delayed_recall": 6,
      "retention_percent": 75,
      "recognition_hits": 9,
      "false_positive_rate": 0.1
    },
    "attention": {
      "mean_rt_ms": 285,
      "rt_variability": 45,
      "accuracy": 0.92,
      "commission_rate": 0.04,
      "omission_rate": 0.02
    },
    "executive": {
      "trail_a_time": 32,
      "trail_b_time": 68,
      "trail_b_a_ratio": 2.13,
      "stroop_interference": 230,
      "stroop_ratio": 1.37
    }
  },
  
  "normative_comparison": {
    "age_group": "65-69",
    "education_matched": true,
    "reference_population": "NACC normative sample"
  },
  
  "recommendations": [
    "Cognitive function within normal limits for age and education",
    "Memory performance in average range",
    "Executive function intact",
    "Continue mentally stimulating activities",
    "Recommend annual cognitive screening"
  ],
  
  "clinical_notes": "All assessed domains within normal limits. Learning curve and retention percentage suggest intact memory encoding and consolidation. Executive function performance appropriate for age."
}
```

---

## 6. Scoring Algorithms

### Memory Scoring
```python
def score_memory(results: dict) -> dict:
    """Score memory test results"""
    
    # Learning trials (max 30 = 10 words x 3 trials)
    total_learning = (
        results['trial1_correct'] + 
        results['trial2_correct'] + 
        results['trial3_correct']
    )
    learning_score = total_learning / 30
    
    # Learning slope (improvement across trials)
    slope = (results['trial3_correct'] - results['trial1_correct']) / 2
    
    # Delayed recall (max 10)
    delayed_score = results['delayed_correct'] / 10
    
    # Retention percentage
    retention = results['delayed_correct'] / max(results['trial3_correct'], 1)
    
    # Recognition discriminability
    hits = results['recognition_hits'] / 10
    fa = results['recognition_false_positives'] / 10
    discriminability = hits - fa
    
    # Combined score (weighted)
    memory_score = (
        learning_score * 0.30 +
        delayed_score * 0.40 +  # Delayed recall weighted highest
        discriminability * 0.30
    )
    
    return {
        'score': memory_score,
        'total_learning': total_learning,
        'learning_slope': slope,
        'delayed_recall': results['delayed_correct'],
        'retention_percent': retention * 100,
        'discriminability': discriminability
    }
```

### Age Adjustment
```python
# Normative data (example for memory score)
MEMORY_NORMS = {
    # age_group: (mean, sd)
    '55-59': (0.85, 0.10),
    '60-64': (0.82, 0.11),
    '65-69': (0.78, 0.12),
    '70-74': (0.74, 0.13),
    '75-79': (0.70, 0.14),
    '80-84': (0.65, 0.15),
}

def age_adjust_score(raw_score: float, age: int, domain: str) -> float:
    """Convert raw score to age-adjusted percentile"""
    
    age_group = f"{(age // 5) * 5}-{(age // 5) * 5 + 4}"
    norms = MEMORY_NORMS.get(age_group, (0.75, 0.12))
    
    # Z-score
    z = (raw_score - norms[0]) / norms[1]
    
    # Convert to percentile (using scipy.stats.norm.cdf)
    from scipy.stats import norm
    percentile = norm.cdf(z) * 100
    
    return percentile
```

---

## 7. Frontend Integration

### Required UI Components

#### 1. Test Selection
- Domain toggles (Memory, Attention, etc.)
- Difficulty selector (easy/standard/hard)
- Estimated time display
- Start button

#### 2. Interactive Tests
- **Word List**: Display words -> Input recall
- **Reaction Time**: Wait -> Click button
- **Trail Making**: Canvas with numbered circles
- **Stroop**: Color words with conflicting colors

#### 3. Results Display
- Radar chart (6 domains)
- Domain score cards
- Percentile comparisons
- Age-adjusted indicators
- Recommendations

### Trail Making Test Implementation
```javascript
const TrailMakingTest = () => {
  const [circles, setCircles] = useState([]);
  const [currentTarget, setCurrentTarget] = useState(1);
  const [path, setPath] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [errors, setErrors] = useState(0);
  
  const handleCircleClick = (label) => {
    if (label === currentTarget || label === getExpectedLabel()) {
      setPath([...path, label]);
      setCurrentTarget(currentTarget + 1);
    } else {
      setErrors(errors + 1);
      // Visual feedback for error
    }
  };
  
  const getExpectedLabel = () => {
    // For Trail B: 1, A, 2, B, 3, C...
    const index = currentTarget - 1;
    if (index % 2 === 0) {
      return Math.floor(index / 2) + 1;
    } else {
      return String.fromCharCode(65 + Math.floor(index / 2));
    }
  };
  
  return (
    <canvas onClick={handleCircleClick}>
      {circles.map(c => <Circle key={c.label} {...c} />)}
    </canvas>
  );
};
```

---

## 8. Implementation Checklist

### Backend
- [ ] Test result validation
- [ ] Memory scoring algorithm
- [ ] Attention scoring algorithm
- [ ] Executive function scoring
- [ ] Processing speed scoring
- [ ] Age normalization
- [ ] Percentile calculation
- [ ] Risk score computation
- [ ] Pattern analysis
- [ ] Recommendation generation

### Frontend
- [ ] Test selection interface
- [ ] Word list memory test
- [ ] Reaction time test
- [ ] Go/No-Go test
- [ ] Trail Making canvas
- [ ] Stroop test
- [ ] Timer component
- [ ] Results radar chart
- [ ] Domain score cards
- [ ] Percentile visualization

---

## 9. Clinical References

1. Weintraub et al. (2009) - "The Alzheimer's Disease Centers' Uniform Data Set (UDS)"
2. Nasreddine et al. (2005) - "The Montreal Cognitive Assessment (MoCA)"
3. Reitan (1958) - "Trail Making Test"
4. Stroop (1935) - "Studies of interference in serial verbal reactions"

---

## 10. Files

```
app/pipelines/cognitive/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # Scoring algorithms
├── normative_data.py   # Age/education norms
└── patterns.py         # Pattern interpretation
```
