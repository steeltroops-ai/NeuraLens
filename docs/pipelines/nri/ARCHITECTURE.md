# NRI Fusion Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | NRI Fusion (Neurological Risk Index) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Purpose | Multi-modal score fusion |
| Clinical Basis | Weighted Bayesian combination of 6 modalities |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    PIPELINE INPUT SOURCES                         |
+------------------------------------------------------------------+
|                                                                   |
|  [Speech]  [Retinal]  [Cardiology]  [Radiology]  [Cognitive]  [Motor]
|     |          |           |             |            |          |
|     v          v           v             v            v          v
|   28.5        22.0        15.0         12.5         18.5       24.0
|  (87%)       (91%)       (85%)        (88%)        (88%)      (85%)
|  Risk        Risk        Risk         Risk         Risk       Risk
|  Score       Score       Score        Score        Score      Score
|                                                                   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    NRI FUSION ENGINE                              |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------------------------+                     |
|  |         WEIGHT ASSIGNMENT                |                     |
|  |                                          |                     |
|  |  Retinal:     22% (vascular observation) |                     |
|  |  Cognitive:   20% (direct function test) |                     |
|  |  Speech:      18% (early neuro markers)  |                     |
|  |  Motor:       15% (movement indicators)  |                     |
|  |  Cardiology:  15% (neuro-cardio link)    |                     |
|  |  Radiology:   10% (pulmonary secondary)  |                     |
|  |                                          |                     |
|  +------------------------------------------+                     |
|                    |                                              |
|                    v                                              |
|  +------------------------------------------+                     |
|  |       CONFIDENCE ADJUSTMENT              |                     |
|  |                                          |                     |
|  |  adjusted_weight = base_weight *         |                     |
|  |                    modality_confidence   |                     |
|  |                                          |                     |
|  |  Example:                                |                     |
|  |  Speech: 0.18 * 0.87 = 0.157             |                     |
|  |  Retinal: 0.22 * 0.91 = 0.200            |                     |
|  |                                          |                     |
|  +------------------------------------------+                     |
|                    |                                              |
|                    v                                              |
|  +------------------------------------------+                     |
|  |       WEIGHTED FUSION                    |                     |
|  |                                          |                     |
|  |  NRI = SUM(score_i * adjusted_weight_i)  |                     |
|  |        / SUM(adjusted_weight_i)          |                     |
|  |                                          |                     |
|  |  Overall_Confidence =                    |                     |
|  |    avg(confidences) * (0.7 + 0.3*coverage)|                     |
|  |                                          |                     |
|  +------------------------------------------+                     |
|                    |                                              |
|                    v                                              |
|  +------------------------------------------+                     |
|  |       RISK CATEGORIZATION                |                     |
|  |                                          |                     |
|  |  0-25:  LOW (Green)                      |                     |
|  |  25-50: MODERATE (Yellow)                |                     |
|  |  50-75: HIGH (Orange)                    |                     |
|  |  75-100: CRITICAL (Red)                  |                     |
|  |                                          |                     |
|  +------------------------------------------+                     |
|                                                                   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    OUTPUT LAYER                                   |
+------------------------------------------------------------------+
|                                                                   |
|  {                                                                |
|    "nri_score": 23.4,                                             |
|    "category": "low",                                             |
|    "confidence": 0.88,                                            |
|    "modality_contributions": [...],                               |
|    "missing_modalities": ["cardiology", "radiology"],             |
|    "coverage": 0.67,                                              |
|    "recommendations": [...]                                       |
|  }                                                                |
|                                                                   |
+------------------------------------------------------------------+
```

---

## 2. Multi-Modal Fusion Theory

### 2.1 Clinical Rationale
Multi-modal assessment is more reliable than single-modality testing:
- Different modalities capture different aspects of neurological health
- Cross-validation between modalities increases diagnostic confidence
- Missing a single modality doesn't invalidate the assessment
- Research shows 15-25% improved accuracy with multi-modal fusion

### 2.2 Weight Distribution (Evidence-Based)
```python
BASE_WEIGHTS = {
    # Modality: (weight, rationale)
    'retinal': (0.22, "Direct vascular observation, strong neuro correlation"),
    'cognitive': (0.20, "Direct cognitive function measurement"),
    'speech': (0.18, "Early Parkinson's/AD biomarkers in voice"),
    'motor': (0.15, "Movement disorder indicators"),
    'cardiology': (0.15, "Cardiovascular-neurological link"),
    'radiology': (0.10, "Secondary pulmonary/cardiac indicators"),
}
# Total: 100%
```

---

## 3. Input Layer Specification

### 3.1 Input Schema
```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PatientInfo(BaseModel):
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[str] = Field(None, pattern="^(male|female|other)$")
    medical_history: Optional[List[str]] = []

class NRIRequest(BaseModel):
    session_id: str
    patient_info: Optional[PatientInfo] = None
    modalities: List[str]  # Which modalities are included
    modality_scores: Dict[str, float]  # modality -> risk score (0-100)
    modality_confidences: Dict[str, float]  # modality -> confidence (0-1)
    modality_details: Optional[Dict[str, Dict]] = None  # Optional raw biomarkers

# Example:
example_request = {
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
    }
}
```

### 3.2 Validation Rules
```python
VALIDATION_RULES = {
    "score_range": (0, 100),
    "confidence_range": (0, 1),
    "min_modalities": 1,
    "valid_modalities": ["speech", "retinal", "cognitive", "motor", "cardiology", "radiology"],
    "require_confidence": True
}
```

---

## 4. Fusion Algorithm Implementation

### 4.1 Core NRI Calculator
```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ModalityContribution:
    modality: str
    raw_score: float
    confidence: float
    base_weight: float
    adjusted_weight: float
    contribution: float
    status: str

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
        
        Algorithm:
        1. Adjust weights by modality confidence
        2. Calculate weighted sum of risk scores
        3. Normalize by total adjusted weight
        4. Calculate overall confidence including coverage
        
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
            contribution_value = score * adjusted_weight
            
            weighted_sum += contribution_value
            total_weight += adjusted_weight
            
            # Determine status
            if score < 25:
                status = "normal"
            elif score < 50:
                status = "low_risk"
            elif score < 75:
                status = "moderate_risk"
            else:
                status = "high_risk"
            
            contributions.append(ModalityContribution(
                modality=modality,
                raw_score=score,
                confidence=confidence,
                base_weight=base_weight,
                adjusted_weight=round(adjusted_weight, 3),
                contribution=round(contribution_value / total_weight if total_weight > 0 else 0, 2),
                status=status
            ))
        
        # Calculate NRI score
        if total_weight > 0:
            nri_score = weighted_sum / total_weight
        else:
            nri_score = 50.0  # Default if no valid modalities
        
        # Calculate overall confidence with coverage factor
        coverage = len(modality_scores) / len(self.BASE_WEIGHTS)
        avg_confidence = np.mean(list(modality_confidences.values())) if modality_confidences else 0.5
        overall_confidence = avg_confidence * (0.7 + 0.3 * coverage)
        
        # Sort contributions by value
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        
        return round(nri_score, 2), round(overall_confidence, 3), contributions
    
    def categorize_risk(self, score: float) -> Tuple[str, str, str]:
        """
        Categorize NRI score into risk level
        
        Returns:
            (category, color, clinical_action)
        """
        if score < 25:
            return ("low", "green", "Routine monitoring (12 months)")
        elif score < 50:
            return ("moderate", "yellow", "Enhanced surveillance (6 months)")
        elif score < 75:
            return ("high", "orange", "Specialist referral (1 month)")
        else:
            return ("critical", "red", "Urgent intervention (immediate)")
    
    def calculate_percentile(self, score: float) -> int:
        """
        Estimate percentile based on normative data
        (Simplified - would use real distribution in production)
        """
        # Assuming normal distribution with mean=30, std=15
        from scipy.stats import norm
        percentile = norm.cdf(score, loc=30, scale=15) * 100
        return int(min(99, max(1, percentile)))
```

### 4.2 Risk Factor Analysis
```python
def analyze_risk_factors(
    contributions: List[ModalityContribution],
    modality_details: Dict[str, Dict] = None
) -> List[Dict]:
    """
    Generate risk factor analysis from contributions
    
    Identifies:
    - Primary risk contributors
    - Concerning patterns
    - Actionable recommendations
    """
    risk_factors = []
    
    for contrib in contributions:
        if contrib.status in ("moderate_risk", "high_risk"):
            factor = {
                "factor": f"{contrib.modality.title()} assessment elevated",
                "modality": contrib.modality,
                "severity": "moderate" if contrib.status == "moderate_risk" else "high",
                "score": contrib.raw_score,
                "recommendation": get_modality_recommendation(contrib.modality, contrib.raw_score)
            }
            risk_factors.append(factor)
    
    # Check for multi-domain patterns
    if modality_details:
        patterns = detect_patterns(modality_details)
        risk_factors.extend(patterns)
    
    return risk_factors

def get_modality_recommendation(modality: str, score: float) -> str:
    """Get specific recommendation for each modality"""
    recommendations = {
        "speech": "Consider neurological speech evaluation",
        "retinal": "Schedule comprehensive eye examination",
        "cognitive": "Consider neuropsychological assessment",
        "motor": "Evaluate for movement disorders",
        "cardiology": "Consider cardiology consultation",
        "radiology": "Follow up on pulmonary findings"
    }
    return recommendations.get(modality, "Consult specialist")

def detect_patterns(modality_details: Dict[str, Dict]) -> List[Dict]:
    """
    Detect clinical patterns across modalities
    
    Examples:
    - Parkinson's: motor tremor + speech jitter + cognitive exec dysfunction
    - Alzheimer's: memory deficit + speech pauses + retinal RNFL thinning
    """
    patterns = []
    
    # Check for Parkinson's pattern
    speech = modality_details.get("speech", {})
    motor = modality_details.get("motor", {})
    
    if (speech.get("voice_tremor", 0) > 0.15 and 
        motor.get("tremor_score", 0) > 0.20):
        patterns.append({
            "factor": "Multi-modal tremor pattern detected",
            "modality": "cross-modal",
            "severity": "moderate",
            "recommendation": "Consider Parkinson's disease screening"
        })
    
    # Check for Alzheimer's pattern
    cognitive = modality_details.get("cognitive", {})
    if (speech.get("pause_ratio", 0) > 0.30 and
        cognitive.get("memory_score", 1.0) < 0.60):
        patterns.append({
            "factor": "Memory and language pattern detected",
            "modality": "cross-modal",
            "severity": "moderate",
            "recommendation": "Consider MCI/Alzheimer's evaluation"
        })
    
    return patterns
```

---

## 5. Recommendation Generation

### 5.1 Recommendation Engine
```python
def generate_recommendations(
    nri_score: float,
    category: str,
    contributions: List[ModalityContribution],
    missing_modalities: List[str],
    risk_factors: List[Dict]
) -> List[str]:
    """
    Generate prioritized recommendations based on NRI analysis
    """
    recommendations = []
    
    # Overall assessment
    if category == "low":
        recommendations.append("Overall neurological risk is low")
    elif category == "moderate":
        recommendations.append("Moderate neurological risk detected - enhanced monitoring advised")
    elif category == "high":
        recommendations.append("Elevated neurological risk - specialist consultation recommended")
    else:
        recommendations.append("Critical neurological risk - urgent intervention required")
    
    # Modality-specific recommendations
    for contrib in contributions:
        if contrib.status == "high_risk":
            recommendations.append(
                f"Priority: Address {contrib.modality} findings (score: {contrib.raw_score})"
            )
        elif contrib.status == "moderate_risk":
            recommendations.append(
                f"Monitor: {contrib.modality.title()} showing mild elevation"
            )
    
    # Missing modality recommendations
    if missing_modalities:
        if len(missing_modalities) <= 2:
            recommendations.append(
                f"Consider completing: {', '.join(missing_modalities)} for comprehensive evaluation"
            )
        else:
            recommendations.append(
                "Additional modality assessments recommended for complete profile"
            )
    
    # Timeline-based recommendation
    if category == "low":
        recommendations.append("Schedule follow-up NRI assessment in 12 months")
    elif category == "moderate":
        recommendations.append("Schedule follow-up NRI assessment in 6 months")
    elif category == "high":
        recommendations.append("Schedule follow-up NRI assessment in 1-3 months")
    else:
        recommendations.append("Immediate clinical attention required")
    
    return recommendations
```

---

## 6. Output Layer Specification

### 6.1 Response Schema
```python
class NRIAssessment(BaseModel):
    score: float
    category: str
    confidence: float
    percentile: int

class ModalityContributionResponse(BaseModel):
    modality: str
    raw_score: float
    confidence: float
    base_weight: float
    adjusted_weight: float
    contribution: float
    status: str

class TrendData(BaseModel):
    direction: str  # "stable", "improving", "declining"
    change_from_last: float
    historical_scores: List[float]

class NRIResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: str
    processing_time_ms: int
    
    nri_assessment: NRIAssessment
    modality_contributions: List[ModalityContributionResponse]
    
    missing_modalities: List[str]
    coverage: float
    
    trend: Optional[TrendData]
    risk_factors: List[Dict]
    recommendations: List[str]
    clinical_summary: str
```

### 6.2 Clinical Summary Generation
```python
def generate_clinical_summary(
    nri_score: float,
    category: str,
    contributions: List[ModalityContribution],
    missing_modalities: List[str]
) -> str:
    """
    Generate natural language clinical summary
    """
    # Modality count
    assessed = len(contributions)
    total = assessed + len(missing_modalities)
    
    # Base summary
    summary = f"Multi-modal assessment using {assessed} of {total} modalities "
    summary += f"indicates {category} overall neurological risk (NRI: {nri_score}/100). "
    
    # Top contributors
    if contributions:
        top = contributions[0]
        if top.status in ("moderate_risk", "high_risk"):
            summary += f"{top.modality.title()} assessment shows largest contribution to risk. "
        else:
            summary += f"All assessed modalities within acceptable ranges. "
    
    # Clinical action
    if category == "low":
        summary += "No urgent clinical action required."
    elif category == "moderate":
        summary += "Enhanced monitoring recommended."
    else:
        summary += "Clinical follow-up advised."
    
    return summary
```

---

## 7. Visualization Specifications

### 7.1 Radar Chart Data Format
```json
{
  "radarData": [
    {"axis": "Speech", "value": 28.5, "normalMax": 25},
    {"axis": "Retinal", "value": 22.0, "normalMax": 25},
    {"axis": "Cognitive", "value": 18.5, "normalMax": 25},
    {"axis": "Motor", "value": 24.0, "normalMax": 25},
    {"axis": "Cardiology", "value": null, "normalMax": 25},
    {"axis": "Radiology", "value": null, "normalMax": 25}
  ]
}
```

### 7.2 NRI Gauge Configuration
```javascript
const gaugeConfig = {
  min: 0,
  max: 100,
  zones: [
    { min: 0, max: 25, color: '#22c55e', label: 'Low' },
    { min: 25, max: 50, color: '#eab308', label: 'Moderate' },
    { min: 50, max: 75, color: '#f97316', label: 'High' },
    { min: 75, max: 100, color: '#ef4444', label: 'Critical' }
  ],
  needle: {
    value: 23.4,
    color: '#1e293b'
  }
};
```

---

## 8. Technology Stack

### 8.1 Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# Numerical
numpy>=1.24.0
scipy>=1.11.0  # For percentile calculations

# No ML dependencies - pure calculation
```

### 8.2 Frontend Visualization
```txt
# Charts
recharts (radar chart, gauge)
chart.js (alternative)

# UI
lucide-react (icons)
```

---

## 9. File Structure

```
app/pipelines/nri/
├── __init__.py
├── ARCHITECTURE.md         # This document
├── router.py               # FastAPI endpoints
├── calculator.py           # NRICalculator class
├── recommendations.py      # Recommendation engine
├── patterns.py             # Cross-modal pattern detection
└── models.py               # Pydantic schemas
```

---

## 10. Clinical References

1. **Van Horn et al. (2014)** - "Multimodal neuroimaging markers of cognitive decline"
2. **Jiang et al. (2021)** - "Multi-modal machine learning for early detection of cognitive impairment"
3. **ADNI Study** - Alzheimer's Disease Neuroimaging Initiative
4. **Weiner et al. (2013)** - "The Alzheimer's Disease Neuroimaging Initiative: A review"
