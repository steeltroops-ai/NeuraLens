# Speech Pipeline Frontend-Backend Integration & AI Explanation Rules
## Complete Analysis for Research-Grade Speech Biomarker System

---

## Table of Contents
1. [Current State Analysis](#1-current-state-analysis)
2. [Complete Biomarker Mapping](#2-complete-biomarker-mapping)
3. [API Contract Specification](#3-api-contract-specification)
4. [AI Explanation Rules](#4-ai-explanation-rules)
5. [Frontend Upgrade Requirements](#5-frontend-upgrade-requirements)
6. [Implementation Plan](#6-implementation-plan)

---

## 1. Current State Analysis

### Backend Capabilities (v3.0)
The research-grade speech pipeline now extracts **40+ features** organized into:

| Category | Features | Status |
|----------|----------|--------|
| Acoustic (Parselmouth) | Jitter (4 variants), Shimmer (5 variants), HNR, CPPS, F0, Formants | ✅ Implemented |
| Prosodic | Speech rate, Pause metrics, Tremor analysis, Intensity dynamics | ✅ Implemented |
| Composite Novel | NII, VFMT, ACE, RPCS, FCR | ✅ Implemented |
| Clinical Scoring | Condition probabilities, Confidence intervals, Risk levels | ✅ Implemented |

### Frontend Current State
The frontend currently displays **9 biomarkers**:
1. Jitter (%) - frequency variation
2. Shimmer (%) - amplitude variation
3. HNR (dB) - harmonics-to-noise ratio
4. CPPS (dB) - cepstral peak prominence (optional field)
5. Speech Rate (syl/s)
6. Pause Ratio
7. Fluency Score
8. Voice Tremor
9. Articulation Clarity (FCR proxy)
10. Prosody Variation (F0 std)

### Gap Analysis
| Backend Feature | Frontend Display | Status |
|-----------------|------------------|--------|
| jitter_local | jitter | ✅ Connected |
| shimmer_local | shimmer | ✅ Connected |
| hnr | hnr | ✅ Connected |
| cpps | cpps | ✅ Connected |
| speech_rate | speech_rate | ✅ Connected |
| pause_ratio | pause_ratio | ✅ Connected |
| tremor_score | voice_tremor | ✅ Connected |
| fcr | articulation_clarity | ✅ Connected |
| std_f0 | prosody_variation | ✅ Connected |
| **nii** | Not displayed | ❌ Missing |
| **vfmt_ratio** | Not displayed | ❌ Missing |
| **ace** | Not displayed | ❌ Missing |
| **rpcs** | Not displayed | ❌ Missing |
| **mean_f0** | Not displayed | ❌ Missing |
| **condition_risks** | Not displayed | ❌ Missing |
| **confidence_interval** | Not displayed | ❌ Missing |

---

## 2. Complete Biomarker Mapping

### Tier 1: Primary Clinical Biomarkers (Always Display)

| Biomarker | Backend Key | Unit | Normal Range | Clinical Meaning | Display Priority |
|-----------|-------------|------|--------------|------------------|------------------|
| **Jitter** | `jitter_local` | % | 0.00-1.04 | Voice pitch stability. Elevated in Parkinson's, laryngeal pathology | 1 |
| **Shimmer** | `shimmer_local` | % | 0.00-3.81 | Voice amplitude stability. Elevated in vocal fold disorders | 2 |
| **HNR** | `hnr` | dB | 20.0-30.0 | Voice clarity. Low = breathy, strained voice | 3 |
| **CPPS** | `cpps` | dB | 14.0-30.0 | Gold standard for voice quality. Low = dysphonia | 4 |
| **Speech Rate** | `speech_rate` | syl/s | 3.5-6.5 | Speaking speed. Low = cognitive/motor impairment | 5 |
| **Voice Tremor** | `tremor_score` | idx | 0.0-0.15 | Motor instability. Elevated in PD, essential tremor | 6 |

### Tier 2: Secondary Clinical Biomarkers (Show in Expanded View)

| Biomarker | Backend Key | Unit | Normal Range | Clinical Meaning |
|-----------|-------------|------|--------------|------------------|
| **Pause Ratio** | `pause_ratio` | ratio | 0.0-0.25 | Silence proportion. High = cognitive decline, word-finding difficulty |
| **FCR** | `fcr` | ratio | 0.9-1.1 | Articulation precision. High = vowel centralization, dysarthria |
| **F0 Variation** | `std_f0` | Hz | 20.0-100.0 | Prosodic richness. Low = monotone (depression, PD) |
| **Mean F0** | `mean_f0` | Hz | M: 85-180, F: 165-255 | Fundamental frequency. Age/sex dependent |

### Tier 3: Research-Grade Novel Biomarkers (Advanced View)

| Biomarker | Backend Key | Unit | Range | Clinical Meaning | Research Status |
|-----------|-------------|------|-------|------------------|-----------------|
| **NII** | `nii` | idx | 0.0-1.0 | Neuromotor Instability Index. Combines tremor, jitter, shimmer, F0 | Novel - Research |
| **VFMT** | `vfmt_ratio` | ratio | varies | Vocal Fold Micro-Tremor. Detects subclinical tremor | Novel - Research |
| **ACE** | `ace` | bits | 0.0-1.0 | Articulatory Coordination Entropy. High = uncoordinated | Novel - Research |
| **RPCS** | `rpcs` | coherence | 0.0-1.0 | Respiratory-Phonatory Coupling. Low = motor disorder | Novel - Research |

### Tier 4: Condition Risk Probabilities

| Condition | Backend Key | Probability | Contributing Factors |
|-----------|-------------|-------------|---------------------|
| Parkinson's Disease | `condition_risks[].parkinsons` | 0-100% | tremor_score, jitter, speech_rate, nii |
| Cognitive Decline | `condition_risks[].cognitive_decline` | 0-100% | pause_ratio, speech_rate, f0_cv |
| Depression | `condition_risks[].depression` | 0-100% | f0_cv, speech_rate, intensity_std |
| Dysarthria | `condition_risks[].dysarthria` | 0-100% | fcr, hnr, shimmer, cpps |

---

## 3. API Contract Specification

### Enhanced API Response Schema

```typescript
interface EnhancedSpeechAnalysisResponse {
    // Session & Metadata
    session_id: string;
    timestamp: string;
    processing_time: number;
    status: 'completed' | 'partial' | 'error';
    
    // Scores with Uncertainty
    risk_score: number;           // 0-1
    confidence: number;           // 0-1
    quality_score: number;        // 0-1
    confidence_interval: [number, number];  // 95% CI for risk
    
    // Primary Biomarkers (MUST have these)
    biomarkers: {
        jitter: BiomarkerResult;
        shimmer: BiomarkerResult;
        hnr: BiomarkerResult;
        cpps: BiomarkerResult;
        speech_rate: BiomarkerResult;
        voice_tremor: BiomarkerResult;
        pause_ratio: BiomarkerResult;
        articulation_clarity: BiomarkerResult;  // FCR
        prosody_variation: BiomarkerResult;     // std_f0
        fluency_score: BiomarkerResult;
    };
    
    // Extended Biomarkers (Research Grade)
    extended_biomarkers?: {
        mean_f0: BiomarkerResult;
        f0_range: BiomarkerResult;
        nii: BiomarkerResult;        // Neuromotor Instability Index
        vfmt: BiomarkerResult;       // Vocal Fold Micro-Tremor
        ace: BiomarkerResult;        // Articulatory Coordination Entropy
        rpcs: BiomarkerResult;       // Respiratory-Phonatory Coupling
    };
    
    // Condition Risk Assessment
    condition_risks: ConditionRisk[];
    
    // Clinical Output
    recommendations: string[];
    clinical_notes?: string;
    requires_review: boolean;
    review_reason?: string;
    
    // File Info
    file_info?: FileInfo;
}

interface ConditionRisk {
    condition: string;
    probability: number;           // 0-1
    confidence: number;            // Model confidence
    confidence_interval: [number, number];
    risk_level: 'low' | 'moderate' | 'high' | 'critical';
    contributing_factors: string[];
}

interface BiomarkerResult {
    value: number;
    unit: string;
    normal_range: [number, number];
    is_estimated: boolean;
    confidence: number | null;
    status?: 'normal' | 'borderline' | 'abnormal';
    z_score?: number;
    percentile?: number;
}
```

---

## 4. AI Explanation Rules

The voice AI assistant MUST follow these rules when explaining speech analysis results to users.

### 4.1 General Communication Principles

```yaml
RULE_001_TONE:
  description: "Professional but accessible medical communication"
  guidelines:
    - Use plain language, avoid jargon without explanation
    - Never diagnose - only indicate "patterns consistent with" or "may warrant"
    - Always emphasize this is a screening tool, not diagnostic
    - Express uncertainty appropriately
    - Be reassuring for normal results, gentle for concerning results

RULE_002_STRUCTURE:
  description: "Consistent explanation structure"
  format:
    1. Overall Summary (1-2 sentences)
    2. Key Findings (top 3 biomarkers)
    3. What This Means (clinical interpretation)
    4. Recommendations (next steps)
    5. Disclaimer (always include)

RULE_003_UNCERTAINTY:
  description: "Always communicate uncertainty"
  requirements:
    - Include confidence percentage in explanation
    - Mention if any measurements are estimated
    - Note quality score if below 80%
    - Explain confidence intervals for high-risk findings
```

### 4.2 Biomarker Explanation Templates

```yaml
# JITTER
jitter_explanation:
  normal: >
    Your voice pitch stability is excellent at {value}%, which is within the 
    healthy range (under 1.04%). This suggests smooth vocal fold vibration.
  borderline: >
    Your voice shows slightly elevated pitch variation at {value}%. 
    This is borderline and may be due to fatigue, nervousness, or a mild 
    cold. Consider re-recording when relaxed.
  abnormal: >
    Your voice pitch variation is elevated at {value}% (normal is under 1.04%). 
    This pattern can be associated with vocal strain or neurological conditions 
    affecting voice control. A follow-up with a speech pathologist may be helpful.

# SHIMMER
shimmer_explanation:
  normal: >
    Your voice amplitude stability is healthy at {value}%, indicating steady 
    breath support and vocal fold closure.
  borderline: >
    Slight amplitude variation detected at {value}%. This could indicate 
    mild vocal fatigue or subtle changes in breath control.
  abnormal: >
    Elevated amplitude variation at {value}% suggests potential vocal fold 
    irregularities. This may warrant evaluation by an ENT specialist.

# HNR (Harmonics-to-Noise Ratio)
hnr_explanation:
  normal: >
    Your voice clarity is excellent at {value} dB, indicating clear, 
    resonant speech with minimal breathiness.
  borderline: >
    Voice clarity at {value} dB is slightly reduced. This could be due to 
    a cold, allergies, or voice fatigue.
  abnormal: >
    Your voice shows increased breathiness or hoarseness (HNR: {value} dB). 
    This pattern may indicate vocal fold changes worth discussing with a 
    healthcare provider.

# CPPS (Cepstral Peak Prominence)
cpps_explanation:
  normal: >
    Your overall voice quality score (CPPS: {value} dB) is excellent, 
    indicating strong, clear phonation.
  abnormal: >
    Your CPPS score of {value} dB is below optimal. CPPS is considered 
    the most reliable measure of voice quality. Lower values may indicate 
    dysphonia or voice disorders.

# SPEECH RATE
speech_rate_explanation:
  normal: >
    Your speaking rate of {value} syllables/second is within the healthy 
    range (3.5-6.5), indicating normal motor speech function.
  slow: >
    Your speaking rate of {value} syllables/second is slower than typical. 
    This can occur with fatigue, careful speech, or in some neurological 
    conditions. Context matters - deliberate slow speech is not concerning.
  fast: >
    Your speaking rate is faster than average at {value} syllables/second. 
    This may simply reflect your natural speaking style.

# VOICE TREMOR
tremor_explanation:
  normal: >
    No significant voice tremor detected ({value}%). Your vocal control 
    appears stable.
  borderline: >
    Slight voice tremor detected at {value}%. This is common with 
    nervousness or caffeine intake and usually not concerning.
  abnormal: >
    Voice tremor detected at {value}%, which is higher than typical. 
    Voice tremor can be associated with essential tremor, Parkinson's 
    disease, or other neurological conditions. A neurological evaluation 
    may be beneficial.

# PAUSE RATIO
pause_ratio_explanation:
  normal: >
    Your speech flow is smooth with a healthy pause ratio of {value}%.
  abnormal: >
    Your speech contains more pauses than typical ({value}%). 
    Increased pauses can be associated with word-finding difficulties, 
    fatigue, or cognitive changes. Consider discussing with your physician 
    if this is a change from your baseline.

# NII (Novel - Neuromotor Instability Index)
nii_explanation:
  research_context: >
    The Neuromotor Instability Index (NII) is a research-grade composite 
    measure combining multiple voice biomarkers. Your score of {value} 
    {interpretation}. This is an experimental metric being studied for 
    early detection of motor disorders.
```

### 4.3 Risk Level Communication

```yaml
RISK_LEVEL_COMMUNICATION:
  low:
    message: >
      Your voice analysis shows low neurological risk indicators. 
      All measured biomarkers are within healthy ranges.
    tone: Reassuring, brief
    action: Continue routine monitoring

  moderate:
    message: >
      Some voice biomarkers are outside typical ranges. While this 
      doesn't indicate a problem, it may warrant attention if you 
      notice other changes.
    tone: Informative, not alarming
    action: Follow-up in 3-6 months

  high:
    message: >
      Your voice analysis shows patterns that warrant clinical attention. 
      Several biomarkers indicate possible changes in speech motor control. 
      We recommend consulting with a healthcare provider.
    tone: Serious but supportive
    action: Schedule specialist consultation

  critical:
    message: >
      Your voice analysis shows significant abnormalities in multiple 
      biomarkers. We strongly recommend prompt evaluation by a healthcare 
      professional. This is a screening result, not a diagnosis.
    tone: Urgent but calm
    action: Seek medical attention promptly
```

### 4.4 Condition-Specific Explanations

```yaml
CONDITION_EXPLANATIONS:
  parkinsons:
    detected_pattern: >
      Your voice shows patterns sometimes seen in early Parkinson's disease: 
      {contributing_factors}. The probability estimate is {probability}% 
      (confidence: {confidence}%). 
      
      IMPORTANT: This is a screening indicator, not a diagnosis. Many 
      conditions can produce similar voice patterns. A neurological 
      evaluation can provide clarity.
    
    what_to_tell_doctor: >
      Tell your doctor: "My voice screening showed patterns that may be 
      worth investigating - specifically {contributing_factors}. I'd like 
      to discuss whether further evaluation is appropriate."

  cognitive_decline:
    detected_pattern: >
      Your speech pattern shows increased pauses and slower rate, which 
      can be associated with cognitive changes. The estimated probability 
      is {probability}%.
      
      NOTE: These patterns can also result from fatigue, medication, or 
      simply careful speech. Consider re-testing after rest.

  depression:
    detected_pattern: >
      Your voice shows reduced prosodic variation (more monotone speech) 
      and slower rate. These can be mood-related voice changes. 
      
      If you've been feeling down or experiencing mood changes, speaking 
      with a healthcare provider about your mental health may be helpful.

  dysarthria:
    detected_pattern: >
      Voice analysis indicates possible motor speech changes affecting 
      articulation clarity. This pattern ({probability}% likelihood) 
      suggests a speech-language pathology evaluation could be beneficial.
```

### 4.5 Quality and Disclaimer Rules

```yaml
QUALITY_WARNINGS:
  low_quality: >
    Audio quality score: {quality}%. This affects measurement reliability. 
    For best results, re-record in a quiet environment, speaking clearly 
    at a normal volume for at least 10 seconds.

  estimated_values: >
    Some measurements are estimated due to audio limitations. These are 
    marked with lower confidence scores.

MANDATORY_DISCLAIMER: >
  IMPORTANT DISCLAIMER: This analysis is for informational screening 
  purposes only and is NOT a medical diagnosis. Voice biomarkers can be 
  affected by many factors including recording conditions, fatigue, 
  medications, and temporary illness. Always consult a qualified healthcare 
  provider for medical advice, diagnosis, or treatment. This tool is 
  intended to support, not replace, professional medical judgment.
```

### 4.6 Complete Explanation Generation Algorithm

```python
def generate_explanation(results: EnhancedSpeechAnalysisResponse) -> str:
    """
    Generate AI explanation following the rules above.
    """
    explanation = []
    
    # 1. OPENING SUMMARY
    risk_level = get_risk_level(results.risk_score)
    explanation.append(f"""
## Voice Analysis Summary

Your voice has been analyzed using {len(BIOMARKER_CONFIGS)} clinically-validated 
biomarkers. Overall risk indicator: **{risk_level.upper()}** 
(Score: {results.risk_score * 100:.0f}/100, Confidence: {results.confidence * 100:.0f}%)
""")
    
    # 2. KEY FINDINGS (Top 3 abnormal or notable)
    abnormal = get_abnormal_biomarkers(results.biomarkers)
    if abnormal:
        explanation.append("\n### Key Findings\n")
        for bio in abnormal[:3]:
            explanation.append(f"- **{bio.label}**: {format_biomarker(bio)}\n")
    else:
        explanation.append("\n### Key Findings\nAll biomarkers within normal ranges.\n")
    
    # 3. CONDITION RISKS (if any elevated)
    elevated_conditions = [c for c in results.condition_risks if c.probability > 0.15]
    if elevated_conditions:
        explanation.append("\n### Patterns Detected\n")
        for cond in elevated_conditions:
            explanation.append(get_condition_explanation(cond))
    
    # 4. RECOMMENDATIONS
    explanation.append("\n### Next Steps\n")
    for rec in results.recommendations:
        explanation.append(f"- {rec}\n")
    
    # 5. TECHNICAL DETAILS (collapsible)
    explanation.append("\n### Detailed Biomarkers\n")
    for config in BIOMARKER_CONFIGS:
        bio = results.biomarkers[config.key]
        status = get_status(bio)
        explanation.append(f"- {config.label}: {bio.value:.2f} {bio.unit} [{status}]\n")
    
    # 6. MANDATORY DISCLAIMER
    explanation.append(f"\n---\n{MANDATORY_DISCLAIMER}")
    
    return "".join(explanation)
```

---

## 5. Frontend Upgrade Requirements

### 5.1 Type Updates (speech-enhanced.ts)

```typescript
// Add extended biomarkers to EnhancedBiomarkers interface
export interface EnhancedBiomarkers {
    // Existing 10 biomarkers...
    
    // NEW: Research-grade biomarkers
    mean_f0?: BiomarkerResult;
    f0_range?: BiomarkerResult;
    nii?: BiomarkerResult;           // Neuromotor Instability Index
    vfmt?: BiomarkerResult;          // Vocal Fold Micro-Tremor
    ace?: BiomarkerResult;           // Articulatory Coordination Entropy
    rpcs?: BiomarkerResult;          // Respiratory-Phonatory Coupling
}

// Add condition risks
export interface ConditionRisk {
    condition: string;
    probability: number;
    confidence: number;
    confidence_interval: [number, number];
    risk_level: 'low' | 'moderate' | 'high' | 'critical';
    contributing_factors: string[];
}

// Update response
export interface EnhancedSpeechAnalysisResponse {
    // ... existing fields ...
    
    // NEW fields
    confidence_interval?: [number, number];
    condition_risks?: ConditionRisk[];
    clinical_notes?: string;
    requires_review?: boolean;
    review_reason?: string;
}
```

### 5.2 New UI Components Required

1. **ConditionRiskCard.tsx** - Display condition probabilities with contributing factors
2. **ResearchBiomarkersPanel.tsx** - Toggle for research-grade biomarkers
3. **ClinicalNotesSection.tsx** - Display clinical notes and review status
4. **UncertaintyDisplay.tsx** - Show confidence intervals visually
5. **AIExplanationPanel.tsx** - Display AI-generated explanation

### 5.3 BIOMARKER_CONFIGS Update

```typescript
export const BIOMARKER_CONFIGS: BiomarkerDisplayConfig[] = [
    // Existing 9 biomarkers...
    
    // NEW: Research biomarkers
    {
        key: 'nii',
        label: 'Neuromotor Instability',
        description: 'Composite index combining tremor, jitter, shimmer (Research)',
        icon: 'brain',
        higherIsBetter: false,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
        isResearch: true,
    },
    {
        key: 'vfmt',
        label: 'Vocal Fold Micro-Tremor',
        description: 'Subclinical tremor detection (Research)',
        icon: 'activity',
        higherIsBetter: false,
        formatValue: (v) => v.toFixed(3),
        isResearch: true,
    },
    {
        key: 'ace',
        label: 'Articulation Entropy',
        description: 'Articulatory coordination measure (Research)',
        icon: 'git-branch',
        higherIsBetter: false,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
        isResearch: true,
    },
    {
        key: 'rpcs',
        label: 'Respiratory-Phonatory Coupling',
        description: 'Breathing-voice synchronization (Research)',
        icon: 'wind',
        higherIsBetter: true,
        formatValue: (v) => `${(v * 100).toFixed(0)}%`,
        isResearch: true,
    },
    {
        key: 'mean_f0',
        label: 'Fundamental Frequency',
        description: 'Average voice pitch',
        icon: 'music-2',
        higherIsBetter: false,
        formatValue: (v, u) => `${v.toFixed(0)} ${u}`,
    },
];
```

---

## 6. Implementation Plan

### Phase 1: Backend API Enhancement (Priority: HIGH)
- [x] Implement research-grade feature extractors
- [x] Implement clinical risk scoring with uncertainty
- [ ] Update router to return extended_biomarkers and condition_risks
- [ ] Add clinical notes to response

### Phase 2: Frontend Types & API (Priority: HIGH)
- [ ] Update speech-enhanced.ts with new fields
- [ ] Update api/services.ts to handle new response format
- [ ] Add fallback handling for missing optional fields

### Phase 3: UI Components (Priority: MEDIUM)
- [ ] Create ConditionRiskCard component
- [ ] Create ResearchBiomarkersPanel with toggle
- [ ] Update SpeechResultsPanel to show all biomarkers
- [ ] Add uncertainty visualization

### Phase 4: AI Explanation Integration (Priority: HIGH)
- [ ] Create AIExplanationPanel component
- [ ] Implement explanation generation following rules
- [ ] Integrate with voice synthesis for spoken explanation
- [ ] Add "Explain to me" button

### Phase 5: Testing & Validation (Priority: HIGH)
- [ ] End-to-end testing with real audio samples
- [ ] Validate all biomarkers display correctly
- [ ] Test AI explanation quality
- [ ] Clinical review of explanation templates

---

## Appendix A: Full Biomarker Reference

| # | Biomarker | Backend Key | Type | Unit | Normal Range | Clinical Relevance |
|---|-----------|-------------|------|------|--------------|-------------------|
| 1 | Jitter (local) | jitter_local | % | % | 0-1.04 | Voice stability |
| 2 | Jitter (RAP) | jitter_rap | % | % | 0-0.68 | Cycle-to-cycle variation |
| 3 | Jitter (PPQ5) | jitter_ppq5 | % | % | 0-0.84 | 5-point average variation |
| 4 | Shimmer (local) | shimmer_local | % | % | 0-3.81 | Amplitude stability |
| 5 | Shimmer (APQ3) | shimmer_apq3 | % | % | 0-2.5 | 3-point amplitude |
| 6 | Shimmer (APQ5) | shimmer_apq5 | % | % | 0-3.0 | 5-point amplitude |
| 7 | HNR | hnr | dB | dB | 20-30 | Voice clarity |
| 8 | CPPS | cpps | dB | dB | 14-30 | Voice quality gold standard |
| 9 | Mean F0 | mean_f0 | Hz | Hz | 85-255 | Pitch (sex-dependent) |
| 10 | Std F0 | std_f0 | Hz | Hz | 20-100 | Prosodic variation |
| 11 | F0 Range | f0_range | Hz | Hz | 50-200 | Pitch range |
| 12 | F1 Mean | f1_mean | Hz | Hz | 200-800 | First formant |
| 13 | F2 Mean | f2_mean | Hz | Hz | 800-2500 | Second formant |
| 14 | Speech Rate | speech_rate | syl/s | syl/s | 3.5-6.5 | Speaking speed |
| 15 | Pause Ratio | pause_ratio | ratio | - | 0-0.25 | Silence proportion |
| 16 | Tremor Score | tremor_score | idx | - | 0-0.15 | Voice tremor |
| 17 | Tremor Freq | tremor_dominant_freq | Hz | Hz | - | Tremor frequency |
| 18 | FCR | fcr | ratio | - | 0.9-1.1 | Articulation clarity |
| 19 | NII | nii | idx | - | 0-0.30 | Neuromotor instability |
| 20 | VFMT | vfmt_ratio | ratio | - | - | Micro-tremor |
| 21 | ACE | ace | bits | - | 0-1.0 | Articulation entropy |
| 22 | RPCS | rpcs | coherence | - | 0.5-1.0 | Respiratory coupling |

---

## Appendix B: Voice Pipeline Rules Summary

```yaml
# Rules for AI voice assistant when explaining speech analysis

RULE_NEVER:
  - Never provide a diagnosis
  - Never state certainty about medical conditions
  - Never ignore the disclaimer
  - Never use alarming language unnecessarily
  - Never fake or interpolate missing biomarker values

RULE_ALWAYS:
  - Always explain in plain language
  - Always mention confidence/uncertainty
  - Always include the disclaimer
  - Always suggest consulting professionals for concerning results
  - Always respect real data - display actual values from backend

RULE_PRIORITIZE:
  - Show abnormal biomarkers first
  - Highlight clinically significant findings
  - Provide actionable recommendations
  - Explain what each biomarker means for the user
```

---

*Document Version: 1.0*
*Created: 2026-01-19*
*Status: Implementation Ready*
