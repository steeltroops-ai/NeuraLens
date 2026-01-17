# Cognitive Pipeline - Product Requirements Document

## Agent Assignment: COGNITIVE-AGENT-04
## Branch: `feature/cognitive-pipeline-fix`
## Priority: P1 (Important for Complete Demo)

---

## Overview

The Cognitive Testing Pipeline assesses memory, attention, and executive function to detect Mild Cognitive Impairment (MCI). This pipeline is interactive and gamified, making it engaging for demo videos.

**Demo Appeal**:
- Interactive memory games
- Real-time reaction time measurement
- Clear domain-specific results (Memory, Attention, Executive)

---

## Current Architecture

### Backend Files

```
backend/app/pipelines/cognitive/
  |-- __init__.py     (31 bytes)
  |-- analyzer.py     (15,438 bytes) - Cognitive scoring
  |-- router.py       (9,096 bytes)  - FastAPI routes
```

### Frontend Files

```
frontend/src/app/dashboard/cognitive/
  |-- page.tsx            - Main cognitive page
  |-- _components/        - Cognitive-specific components
```

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| CG-F01 | Memory test - word list recall | P0 | Needs implementation |
| CG-F02 | Attention test - reaction time game | P0 | Needs implementation |
| CG-F03 | Executive function - pattern matching | P1 | Needs implementation |
| CG-F04 | Age-adjusted scoring | P0 | Needs implementation |
| CG-F05 | Domain-specific scores | P0 | Needs implementation |
| CG-F06 | Composite cognitive index | P0 | Needs implementation |
| CG-F07 | Progress tracking across tests | P1 | Not started |
| CG-F08 | Practice trials before tests | P1 | Not started |

---

## Agent Task Breakdown

### Step 1: Implement Memory Test Backend (2 hours)

**File**: `backend/app/pipelines/cognitive/analyzer.py`

**Tasks**:
1. Score immediate word recall (10 words shown, count recalled)
2. Score delayed recall (after 5 minutes)
3. Calculate recognition accuracy (correct vs false positives)
4. Normalize scores to 0.0-1.0

**Implementation**:
```python
def score_memory_test(test_results: dict) -> dict:
    """Score memory test components"""
    
    # Immediate recall: 10 words shown, how many recalled correctly
    immediate_score = test_results["immediate_recall_count"] / 10.0
    
    # Delayed recall: same list after 5-10 minute delay
    delayed_score = test_results["delayed_recall_count"] / 10.0
    
    # Recognition: shown 20 words (10 original + 10 distractors)
    true_positives = test_results["recognition_correct"]
    false_positives = test_results["recognition_incorrect"]
    recognition_score = (true_positives - false_positives * 0.5) / 10.0
    recognition_score = max(0, min(1, recognition_score))
    
    # Learning curve: improvement across trials
    trial_scores = test_results.get("trial_scores", [])
    if len(trial_scores) >= 3:
        learning_slope = (trial_scores[-1] - trial_scores[0]) / len(trial_scores)
        learning_score = min(1, max(0, 0.5 + learning_slope * 5))
    else:
        learning_score = 0.5
    
    return {
        "immediate_recall": round(immediate_score, 2),
        "delayed_recall": round(delayed_score, 2),
        "recognition": round(recognition_score, 2),
        "learning_curve": round(learning_score, 2),
        "domain_score": round(
            0.3 * immediate_score + 
            0.35 * delayed_score + 
            0.2 * recognition_score +
            0.15 * learning_score, 
            2
        )
    }
```

### Step 2: Implement Attention Test Backend (2 hours)

**File**: `backend/app/pipelines/cognitive/analyzer.py`

**Tasks**:
1. Score reaction time (average and variability)
2. Calculate sustained attention (performance over time)
3. Measure selective attention (correct vs incorrect responses)
4. Add divided attention task

**Implementation**:
```python
def score_attention_test(test_results: dict) -> dict:
    """Score attention test components"""
    
    reaction_times = test_results["reaction_times_ms"]
    
    # Mean reaction time (normalize: 200ms = 1.0, 600ms = 0.0)
    mean_rt = sum(reaction_times) / len(reaction_times)
    rt_score = max(0, min(1, 1 - (mean_rt - 200) / 400))
    
    # RT variability (lower is better)
    rt_std = (sum((rt - mean_rt)**2 for rt in reaction_times) / len(reaction_times)) ** 0.5
    variability_score = max(0, min(1, 1 - rt_std / 100))
    
    # Sustained attention: compare first half vs second half
    first_half = reaction_times[:len(reaction_times)//2]
    second_half = reaction_times[len(reaction_times)//2:]
    sustained_diff = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
    sustained_score = max(0, min(1, 1 - sustained_diff / 100))
    
    # Accuracy (correct responses / total stimuli)
    accuracy = test_results["correct_responses"] / test_results["total_stimuli"]
    
    return {
        "reaction_time_ms": round(mean_rt, 1),
        "rt_variability": round(rt_std, 1),
        "sustained_attention": round(sustained_score, 2),
        "accuracy": round(accuracy, 2),
        "domain_score": round(
            0.25 * rt_score +
            0.25 * variability_score +
            0.25 * sustained_score +
            0.25 * accuracy,
            2
        )
    }
```

### Step 3: Implement Executive Function Backend (1.5 hours)

**File**: `backend/app/pipelines/cognitive/analyzer.py`

**Tasks**:
1. Score planning tasks (e.g., Tower of London)
2. Score inhibition (e.g., Stroop-like interference)
3. Score flexibility (e.g., set-shifting)
4. Include working memory component

**Implementation**:
```python
def score_executive_test(test_results: dict) -> dict:
    """Score executive function components"""
    
    # Planning: moves to solve vs optimal moves
    planning_score = test_results["optimal_moves"] / max(1, test_results["actual_moves"])
    planning_score = min(1, planning_score)
    
    # Inhibition: incongruent trial accuracy
    inhibition_score = test_results["incongruent_correct"] / test_results["incongruent_total"]
    
    # Flexibility: successful switches / total switches
    flexibility_score = test_results["successful_switches"] / max(1, test_results["total_switches"])
    
    # Working memory: n-back accuracy
    working_memory_score = test_results.get("nback_accuracy", 0.7)
    
    return {
        "planning": round(planning_score, 2),
        "inhibition": round(inhibition_score, 2),
        "flexibility": round(flexibility_score, 2),
        "working_memory": round(working_memory_score, 2),
        "domain_score": round(
            0.25 * planning_score +
            0.30 * inhibition_score +
            0.25 * flexibility_score +
            0.20 * working_memory_score,
            2
        )
    }
```

### Step 4: Add Age-Adjusted Scoring (1 hour)

**File**: `backend/app/pipelines/cognitive/analyzer.py`

**Tasks**:
1. Define age-based normative data
2. Calculate z-scores against age norms
3. Convert to percentile ranks

**Implementation**:
```python
# Age-based normative data (simplified)
AGE_NORMS = {
    "memory": {
        (18, 30): {"mean": 0.85, "std": 0.10},
        (31, 50): {"mean": 0.80, "std": 0.12},
        (51, 65): {"mean": 0.72, "std": 0.14},
        (66, 80): {"mean": 0.65, "std": 0.16},
        (81, 100): {"mean": 0.55, "std": 0.18}
    },
    "attention": {
        (18, 30): {"mean": 0.90, "std": 0.08},
        (31, 50): {"mean": 0.85, "std": 0.10},
        (51, 65): {"mean": 0.78, "std": 0.12},
        (66, 80): {"mean": 0.70, "std": 0.14},
        (81, 100): {"mean": 0.60, "std": 0.16}
    },
    "executive": {
        (18, 30): {"mean": 0.88, "std": 0.09},
        (31, 50): {"mean": 0.82, "std": 0.11},
        (51, 65): {"mean": 0.75, "std": 0.13},
        (66, 80): {"mean": 0.68, "std": 0.15},
        (81, 100): {"mean": 0.58, "std": 0.17}
    }
}

def get_age_adjusted_score(raw_score: float, domain: str, age: int) -> dict:
    """Convert raw score to age-adjusted z-score and percentile"""
    
    # Find age bracket
    for (min_age, max_age), norms in AGE_NORMS[domain].items():
        if min_age <= age <= max_age:
            z_score = (raw_score - norms["mean"]) / norms["std"]
            percentile = norm.cdf(z_score) * 100  # From scipy.stats
            break
    else:
        z_score = 0
        percentile = 50
    
    return {
        "raw_score": raw_score,
        "z_score": round(z_score, 2),
        "percentile": round(percentile, 1)
    }
```

### Step 5: Build Frontend Tests (2 hours)

**File**: `frontend/src/app/dashboard/cognitive/page.tsx`

**Tasks**:
1. Create memory word list display (show 10 words for 30 seconds)
2. Create recall input interface
3. Create reaction time game (click when target appears)
4. Show real-time feedback

---

## API Contract

### POST /api/v1/cognitive/analyze

**Request**:
```json
{
  "session_id": "uuid",
  "demographics": {
    "age": 62,
    "education_years": 16
  },
  "test_results": {
    "memory": {
      "immediate_recall_count": 7,
      "delayed_recall_count": 5,
      "recognition_correct": 8,
      "recognition_incorrect": 1,
      "trial_scores": [5, 6, 7]
    },
    "attention": {
      "reaction_times_ms": [320, 345, 298, 410, 335, 380, 355, 312],
      "correct_responses": 38,
      "total_stimuli": 40
    },
    "executive": {
      "optimal_moves": 12,
      "actual_moves": 15,
      "incongruent_correct": 28,
      "incongruent_total": 32,
      "successful_switches": 14,
      "total_switches": 16
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 22.0,
    "risk_category": "low",
    "confidence": 0.91,
    "domain_scores": {
      "memory": {
        "raw_score": 0.78,
        "z_score": 0.5,
        "percentile": 69,
        "components": {
          "immediate_recall": 0.70,
          "delayed_recall": 0.50,
          "recognition": 0.75,
          "learning_curve": 0.60
        }
      },
      "attention": {
        "raw_score": 0.82,
        "z_score": 0.3,
        "percentile": 62,
        "components": {
          "reaction_time_ms": 344,
          "rt_variability": 38.2,
          "sustained_attention": 0.85,
          "accuracy": 0.95
        }
      },
      "executive": {
        "raw_score": 0.80,
        "z_score": 0.4,
        "percentile": 65,
        "components": {
          "planning": 0.80,
          "inhibition": 0.875,
          "flexibility": 0.875,
          "working_memory": 0.70
        }
      }
    },
    "composite_score": 0.80,
    "interpretation": "Cognitive performance within normal limits for age 62. All domains show age-appropriate function.",
    "recommendations": [
      "Continue regular cognitive activities",
      "Maintain cardiovascular health",
      "Follow-up assessment in 12 months"
    ],
    "processing_time_ms": 85
  }
}
```

---

## Cognitive Test Design

### Memory Test - Word List Recall

```
Test Structure:
1. Show 10 common words, one at a time (2 seconds each)
2. Immediately ask to type recalled words (60 seconds)
3. Do interference task (attention test)
4. Delayed recall after 5 minutes (60 seconds)
5. Recognition: show 20 words, identify original 10

Word List Example:
["apple", "chair", "sunset", "pencil", "ocean", 
 "garden", "mirror", "kitchen", "blanket", "forest"]
```

### Attention Test - Reaction Time

```
Test Structure:
1. Show fixation cross
2. After random delay (0.5-2s), show target (green circle)
3. User clicks as fast as possible
4. 40 trials total, ~3 minutes

Scoring:
- Mean RT < 300ms = excellent
- Mean RT 300-400ms = good
- Mean RT 400-500ms = moderate
- Mean RT > 500ms = impaired
```

### Executive Function - Card Sorting

```
Test Structure:
1. Sort cards by rule (color, shape, or number)
2. Rule changes without warning
3. Count successful switches vs perseverative errors

Simplified Version for Demo:
- Show pattern, user selects matching card
- Rule changes every ~5 trials
```

---

## Verification Checklist

- [ ] Memory test word display works
- [ ] Memory recall input captures responses
- [ ] Attention test measures reaction time accurately
- [ ] Reaction time displayed in real-time
- [ ] Executive function test switches rules
- [ ] Age-adjusted scoring applied
- [ ] Domain scores calculated correctly
- [ ] Radar chart visualization works
- [ ] Progress bar shows test completion
- [ ] Results interpretation is clear

---

## Demo Script

For the hackathon video, demonstrate:

1. "Now let's assess cognitive function"
2. Show memory test: "Here are 10 words to remember"
3. Show reaction time game: "Click when you see the green circle"
4. Show results: "Domain scores for memory, attention, and executive function"
5. Point to radar chart: "This shows a comprehensive cognitive profile"
6. Highlight age adjustment: "Scores are normalized for the user's age group"

---

## Estimated Time

| Task | Hours |
|------|-------|
| Memory test backend | 2.0 |
| Attention test backend | 2.0 |
| Executive function backend | 1.5 |
| Age-adjusted scoring | 1.0 |
| Frontend test UI | 2.0 |
| Testing | 1.5 |
| **Total** | **10.0 hours** |
