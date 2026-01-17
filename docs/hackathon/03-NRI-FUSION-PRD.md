# MediLens NRI Fusion Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P0 - Critical (Core Feature) |
| Est. Dev Time | 4 hours |
| Clinical Validation | Multi-modal fusion approach |

---

## 1. Overview

### Purpose
Combine risk assessments from all individual pipelines (Speech, Retinal, Cardiology, Radiology, Cognitive, Motor) into a unified **Neurological Risk Index (NRI)** using weighted Bayesian fusion with confidence estimation.

### Clinical Basis
Multi-modal assessment is more reliable than single-modality testing. Research shows that combining biomarkers from different physiological systems improves diagnostic accuracy by 15-25% compared to individual tests.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Fusion Algorithm** | Weighted Average | Interpretable combination |
| **Confidence** | Bayesian Estimation | Uncertainty quantification |
| **Normalization** | Z-score + Min-Max | Standardize inputs |
| **Visualization** | Radar Chart | Multi-axis display |

### Installation
```bash
pip install numpy scipy  # Core math only
```

---

## 3. Modality Weights

### Clinical Weight Distribution

| Modality | Weight | Rationale |
|----------|--------|-----------|
| **Retinal** | 22% | Direct vascular observation, strong neuro correlation |
| **Cognitive** | 20% | Direct cognitive function measurement |
| **Speech** | 18% | Early Parkinson's/AD biomarkers |
| **Motor** | 15% | Movement disorder indicators |
| **Cardiology** | 15% | Cardiovascular-neurological link |
| **Radiology** | 10% | Secondary pulmonary indicators |

**Total: 100%**

### Weight Adjustment by Confidence
```python
adjusted_weight = base_weight * modality_confidence
```

---

## 4. API Specification

### Endpoint
```
POST /api/nri/calculate
Content-Type: application/json
```

### Request
```json
{
  "session_id": "combined_session_123",
  "patient_info": {
    "age": 65,
    "sex": "male",
    "medical_history": ["hypertension", "type2_diabetes"]
  },
  "modalities": ["speech", "retinal", "cognitive", "motor"],
  "modality_scores": {
    "speech": 28.5,
    "retinal": 22.0,
    "cognitive": 18.5,
    "motor": 24.0
  },
  "modality_confidences": {
    "speech": 0.87,
    "retinal": 0.91,
    "cognitive": 0.88,
    "motor": 0.85
  },
  "modality_details": {
    "speech": {
      "jitter": 0.025,
      "voice_tremor": 0.08
    },
    "retinal": {
      "cup_disc_ratio": 0.28,
      "av_ratio": 0.68
    }
  }
}
```

### Response
```json
{
  "success": true,
  "session_id": "combined_session_123",
  "timestamp": "2026-01-17T14:15:00Z",
  "processing_time_ms": 45,
  
  "nri_assessment": {
    "score": 23.4,
    "category": "low",
    "confidence": 0.88,
    "percentile": 35
  },
  
  "modality_contributions": [
    {
      "modality": "retinal",
      "raw_score": 22.0,
      "confidence": 0.91,
      "base_weight": 0.22,
      "adjusted_weight": 0.200,
      "contribution": 4.40,
      "status": "normal"
    },
    {
      "modality": "cognitive",
      "raw_score": 18.5,
      "confidence": 0.88,
      "base_weight": 0.20,
      "adjusted_weight": 0.176,
      "contribution": 3.26,
      "status": "normal"
    },
    {
      "modality": "speech",
      "raw_score": 28.5,
      "confidence": 0.87,
      "base_weight": 0.18,
      "adjusted_weight": 0.157,
      "contribution": 4.47,
      "status": "low_risk"
    },
    {
      "modality": "motor",
      "raw_score": 24.0,
      "confidence": 0.85,
      "base_weight": 0.15,
      "adjusted_weight": 0.128,
      "contribution": 3.07,
      "status": "normal"
    }
  ],
  
  "missing_modalities": ["cardiology", "radiology"],
  "coverage": 0.75,
  
  "trend": {
    "direction": "stable",
    "change_from_last": 0.0,
    "historical_scores": []
  },
  
  "risk_factors": [
    {
      "factor": "Speech biomarkers slightly elevated",
      "modality": "speech",
      "severity": "mild",
      "recommendation": "Monitor voice changes over time"
    }
  ],
  
  "recommendations": [
    "Overall neurological risk is low",
    "All assessed modalities within acceptable ranges",
    "Continue routine health monitoring",
    "Consider completing cardiology and radiology assessments for comprehensive evaluation",
    "Schedule follow-up NRI assessment in 12 months"
  ],
  
  "clinical_summary": "Multi-modal assessment using 4 of 6 modalities indicates low overall neurological risk. Speech biomarkers show minimal elevation within acceptable range. No urgent clinical action required."
}
```

---

## 5. Fusion Algorithm

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModalityContribution:
    modality: str
    raw_score: float
    confidence: float
    base_weight: float
    adjusted_weight: float
    contribution: float

class NRICalculator:
    """Multi-modal Neurological Risk Index Calculator"""
    
    BASE_WEIGHTS = {
        'retinal': 0.22,
        'cognitive': 0.20,
        'speech': 0.18,
        'motor': 0.15,
        'cardiology': 0.15,
        'radiology': 0.10
    }
    
    def calculate(
        self,
        modality_scores: Dict[str, float],
        modality_confidences: Dict[str, float]
    ) -> Tuple[float, float, List[ModalityContribution]]:
        """
        Calculate NRI score with confidence-weighted fusion
        
        Args:
            modality_scores: Dict of modality -> risk score (0-100)
            modality_confidences: Dict of modality -> confidence (0-1)
        
        Returns:
            Tuple of (nri_score, overall_confidence, contributions)
        """
        
        contributions = []
        weighted_sum = 0.0
        total_weight = 0.0
        
        for modality, score in modality_scores.items():
            if modality not in self.BASE_WEIGHTS:
                continue
            
            base_weight = self.BASE_WEIGHTS[modality]
            confidence = modality_confidences.get(modality, 0.8)
            
            # Adjust weight by confidence
            adjusted_weight = base_weight * confidence
            contribution = score * adjusted_weight
            
            weighted_sum += contribution
            total_weight += adjusted_weight
            
            contributions.append(ModalityContribution(
                modality=modality,
                raw_score=score,
                confidence=confidence,
                base_weight=base_weight,
                adjusted_weight=adjusted_weight,
                contribution=contribution / 100 * adjusted_weight * 100
            ))
        
        # Calculate NRI
        if total_weight > 0:
            nri_score = weighted_sum / total_weight
        else:
            nri_score = 50.0  # Default if no valid modalities
        
        # Calculate overall confidence
        coverage = len(modality_scores) / len(self.BASE_WEIGHTS)
        avg_confidence = np.mean(list(modality_confidences.values()))
        overall_confidence = avg_confidence * (0.7 + 0.3 * coverage)
        
        return nri_score, overall_confidence, contributions
    
    def categorize_risk(self, score: float) -> str:
        """Categorize NRI score into risk level"""
        if score < 25:
            return "low"
        elif score < 50:
            return "moderate"
        elif score < 75:
            return "high"
        else:
            return "critical"
```

---

## 6. Frontend Integration

### Required UI Components

#### 1. NRI Score Display
- Large central gauge (0-100)
- Color-coded zones (green/yellow/orange/red)
- Confidence indicator
- Category badge

#### 2. Radar Chart
- 6 axes (one per modality)
- Filled area showing risk profile
- Normal range overlay
- Interactive hover details

#### 3. Contribution Breakdown
- Horizontal bar chart
- Sorted by contribution
- Weight annotation
- Status indicators

#### 4. Recommendations Panel
- Priority-sorted list
- Actionable items
- Urgency indicators

### Visual Design
```
+--------------------------------------------------+
|                    NRI SCORE                      |
|                                                   |
|         [===========O=============]               |
|               23.4 / 100                          |
|              [LOW RISK]                           |
|           Confidence: 88%                         |
+--------------------------------------------------+
|                                                   |
|              [RADAR CHART]                        |
|                                                   |
|          Speech     Retinal                       |
|              \       /                            |
|               \     /                             |
|        Motor---[*]---Cognitive                    |
|               /     \                             |
|              /       \                            |
|        Cardio     Radiology                       |
|            (missing) (missing)                    |
|                                                   |
+--------------------------------------------------+
|  MODALITY CONTRIBUTIONS                           |
|  Speech     [===========] 4.47  (18%)            |
|  Retinal    [==========]  4.40  (22%)            |
|  Cognitive  [========]    3.26  (20%)            |
|  Motor      [=======]     3.07  (15%)            |
+--------------------------------------------------+
|  RECOMMENDATIONS                                  |
|  [i] Overall neurological risk is low            |
|  [!] Consider cardiology assessment              |
|  [i] Schedule follow-up in 12 months             |
+--------------------------------------------------+
```

---

## 7. Risk Categories

| Score | Category | Color | Action | Timeline |
|-------|----------|-------|--------|----------|
| 0-25 | Low | Green | Routine monitoring | 12 months |
| 25-50 | Moderate | Yellow | Enhanced surveillance | 6 months |
| 50-75 | High | Orange | Specialist referral | 1 month |
| 75-100 | Critical | Red | Urgent intervention | Immediate |

---

## 8. Implementation Checklist

### Backend
- [ ] Input validation (scores 0-100, confidence 0-1)
- [ ] Weighted fusion calculation
- [ ] Confidence estimation
- [ ] Risk categorization
- [ ] Contribution breakdown
- [ ] Missing modality handling
- [ ] Trend analysis (if historical data)
- [ ] Recommendation generation
- [ ] Clinical summary generation

### Frontend
- [ ] NRI gauge component
- [ ] Radar chart (Recharts or Chart.js)
- [ ] Contribution bar chart
- [ ] Risk category badge
- [ ] Missing modality indicators
- [ ] Trend line chart (if historical)
- [ ] Recommendations list
- [ ] Export/share functionality

---

## 9. Clinical References

1. Van Horn et al. (2014) - "Multimodal neuroimaging markers of cognitive decline"
2. Jiang et al. (2021) - "Multi-modal machine learning for early detection of cognitive impairment"
3. Weiner et al. (2013) - "The Alzheimer's Disease Neuroimaging Initiative"

---

## 10. Files

```
app/pipelines/nri/
├── __init__.py
├── router.py           # FastAPI endpoints
├── analyzer.py         # NRICalculator implementation
└── recommendations.py  # Recommendation generation
```
