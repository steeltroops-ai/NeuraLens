# Research-Grade Speech Analysis Pipeline for Health Biomarker Detection

## Document Information
| Field | Value |
|-------|-------|
| Version | 3.0.0 |
| Date | 2026-01-19 |
| Authors | Multidisciplinary Team: Clinical Neurology, Speech Pathology, Biomedical Engineering, ML Research |
| Classification | Research-Grade Medical Device Pipeline |
| Regulatory Target | FDA Class II (510(k)), CE Mark Class IIa |

---

# SECTION 1: PROBLEM UNDERSTANDING AND SCOPE

## 1.1 Problem Statement

**Primary Objective**: Develop a clinically validated, research-grade speech analysis pipeline capable of detecting and quantifying neurological, respiratory, psychiatric, and cognitive biomarkers from voice recordings with sufficient accuracy for clinical decision support.

**Clinical Gap**: Early detection of neurodegenerative diseases (Parkinson's, Alzheimer's, ALS) currently relies on subjective clinical assessments that detect changes only after significant neuronal loss (60-80%). Voice-based biomarkers can potentially detect subclinical changes 5-10 years earlier.

## 1.2 Medical Conditions Detectable from Speech

### Tier 1: Strong Scientific Evidence (Meta-analyses, RCTs)

| Condition | Key Speech Markers | Sensitivity/Specificity | Evidence Level |
|-----------|-------------------|------------------------|----------------|
| **Parkinson's Disease** | Jitter, shimmer, voice tremor (4-6Hz), hypophonia, monopitch | 85-95% / 80-90% | Level I |
| **Alzheimer's/MCI** | Pause patterns, lexical diversity, speech rate decline | 75-85% / 70-85% | Level I-II |
| **Depression (MDD)** | Prosody flattening, speech rate, vocal energy | 70-80% / 65-80% | Level I |
| **Dysarthria** | FCR, articulation rate, consonant precision | 85-95% / 85-95% | Level I |
| **Laryngeal Pathology** | CPPS, HNR, jitter, shimmer | 90-95% / 85-95% | Level I |

### Tier 2: Moderate Evidence (Case-control studies)

| Condition | Key Speech Markers | Evidence Level |
|-----------|-------------------|----------------|
| **ALS/Motor Neuron Disease** | Articulatory breakdown, hypernasality, speaking rate | Level II-III |
| **Multiple Sclerosis** | Scanning speech, prosodic abnormalities | Level II-III |
| **Essential Tremor** | Voice tremor (5-12Hz), vocal instability | Level II |
| **Anxiety Disorders** | Speech rate variability, filled pauses | Level II-III |
| **Schizophrenia** | Poverty of speech, latency, prosodic deficits | Level II |

### Tier 3: Emerging/Speculative

| Area | Status |
|------|--------|
| COVID-19 respiratory assessment | Early research, promising |
| Stroke recovery monitoring | Preliminary studies |
| Concussion/TBI | Emerging evidence |
| Fatigue/Cognitive load | Consumer wellness only |
| Stress detection | Consumer wellness only |

## 1.3 Clinical vs Consumer Use Cases

| Aspect | Clinical Use | Consumer Wellness |
|--------|-------------|-------------------|
| **Regulatory** | FDA 510(k), CE Mark required | No medical claims |
| **Accuracy** | >85% sensitivity, >80% specificity | Trend monitoring only |
| **Output** | Risk stratification, referral recommendation | "Insights", not diagnoses |
| **Validation** | Clinical trials, external validation | User studies |
| **Liability** | Medical device liability | General software |

## 1.4 Target Patient Populations

1. **Primary Screening**: Adults 50+ for neurodegenerative disease risk
2. **Parkinson's Monitoring**: Diagnosed PD patients tracking progression
3. **Cognitive Decline Tracking**: MCI/early AD longitudinal monitoring
4. **Voice Disorder Clinics**: Dysphonia, vocal cord pathology assessment
5. **Mental Health Integration**: Depression screening adjunct

## 1.5 Accuracy Requirements for Clinical Relevance

| Use Case | Minimum Sensitivity | Minimum Specificity | PPV Target |
|----------|--------------------|--------------------|------------|
| Screening (rule-out) | 90% | 70% | 20%+ |
| Diagnostic support | 85% | 85% | 50%+ |
| Monitoring/Progression | 80% (change detection) | 80% | N/A |
| Research biomarker | 75% | 75% | N/A |

## 1.6 Ethical and Regulatory Constraints

### Medical Device Classification
- **FDA**: Class II (Special Controls) - requires 510(k) premarket notification
- **EU MDR**: Class IIa - software intended for diagnosis/monitoring
- **HIPAA**: PHI protection mandatory for US deployment
- **GDPR**: Explicit consent, right to erasure for EU

### Bias and Fairness Requirements
- Validation across age, sex, accent, language, socioeconomic status
- Documented performance across demographic subgroups
- Transparency in training data composition

### Key Ethical Considerations
- No standalone diagnosis - always clinical decision support
- Clear communication of uncertainty
- Right to human review
- Protection against insurance/employment discrimination

---

# SECTION 2: FULL PIPELINE ARCHITECTURE

## 2.1 Architecture Diagram (Textual)

```
+==============================================================================+
|                           DATA ACQUISITION LAYER                              |
+==============================================================================+
|  [Microphone]          [Environment]           [Speech Tasks]                 |
|  - 16kHz+ sample rate  - <40dB ambient         - Sustained vowels /a/         |
|  - 16-bit PCM          - No reverberation      - Rainbow Passage reading      |
|  - Flat response       - Consistent distance   - Diadochokinetic /pa-ta-ka/   |
|                                                - Free conversational speech   |
+==============================================================================+
                                    |
                                    v
+==============================================================================+
|                          PREPROCESSING LAYER                                  |
+==============================================================================+
|  +----------------+   +----------------+   +------------------+               |
|  | Format Convert |-->| Resample 16kHz |-->| Mono Conversion  |               |
|  +----------------+   +----------------+   +------------------+               |
|           |                                                                   |
|           v                                                                   |
|  +----------------+   +----------------+   +------------------+               |
|  | Noise Reduction|-->| VAD/Silence    |-->| Normalization    |               |
|  | (Spectral Sub) |   | Trim           |   | (Peak/-3dB)      |               |
|  +----------------+   +----------------+   +------------------+               |
|           |                                                                   |
|           v                                                                   |
|  +------------------+   +------------------+                                  |
|  | Segmentation     |-->| Speaker          |                                  |
|  | (VAD, Phoneme)   |   | Adaptation       |                                  |
|  +------------------+   +------------------+                                  |
+==============================================================================+
                                    |
                                    v
+==============================================================================+
|                       FEATURE EXTRACTION LAYER                                |
+==============================================================================+
|  +---------------------+  +---------------------+  +---------------------+    |
|  | ACOUSTIC FEATURES   |  | PROSODIC FEATURES   |  | ARTICULATORY        |    |
|  | - Jitter (5 types)  |  | - F0 mean/std/range |  | - FCR (Formant      |    |
|  | - Shimmer (6 types) |  | - F0 contour        |  |   Centralization)   |    |
|  | - HNR               |  | - Intensity contour |  | - VSA (Vowel Space) |    |
|  | - CPPS (smoothed)   |  | - Speech rate       |  | - F1/F2 transitions |    |
|  | - Voice breaks      |  | - Pause patterns    |  | - Consonant         |    |
|  +---------------------+  +---------------------+  |   precision         |    |
|                                                    +---------------------+    |
|  +---------------------+  +---------------------+  +---------------------+    |
|  | SPECTRAL BIOMARKERS |  | VOICE QUALITY       |  | TEMPORAL STABILITY  |    |
|  | - MFCC (13 or 40)   |  | - Breathiness       |  | - Jitter/Shimmer    |    |
|  | - Spectral centroid |  | - Roughness         |  |   coefficient var   |    |
|  | - Spectral tilt     |  | - Strain            |  | - F0 stability      |    |
|  | - Formant freqs     |  | - Asthenia          |  | - Micro-tremor      |    |
|  | - Spectral flux     |  | - GRBAS automated   |  |   (4-12Hz bands)    |    |
|  +---------------------+  +---------------------+  +---------------------+    |
+==============================================================================+
                                    |
                                    v
+==============================================================================+
|                    REPRESENTATION LEARNING LAYER                              |
+==============================================================================+
|  +----------------------+   +----------------------+                          |
|  | Wav2Vec 2.0 / HuBERT |   | Whisper Embeddings   |                          |
|  | - Raw waveform       |   | - Encoder hidden     |                          |
|  | - Self-supervised    |   |   states             |                          |
|  | - 768-dim embeddings |   | - 512/768-dim        |                          |
|  +----------------------+   +----------------------+                          |
|                                    |                                          |
|                                    v                                          |
|  +----------------------------------------------------------+                 |
|  | FUSION LAYER                                              |                 |
|  | - Concatenate: Handcrafted + Learned features             |                 |
|  | - Dimension: ~200 handcrafted + 768 learned = ~1000       |                 |
|  | - Optional: Attention-based fusion                        |                 |
|  +----------------------------------------------------------+                 |
+==============================================================================+
                                    |
                                    v
+==============================================================================+
|                      MODELING & INFERENCE LAYER                               |
+==============================================================================+
|  +------------------------+   +------------------------+                      |
|  | DISEASE-SPECIFIC HEADS |   | UNCERTAINTY ESTIMATION |                      |
|  | - Parkinson's head     |   | - Monte Carlo Dropout  |                      |
|  | - Alzheimer's head     |   | - Ensemble variance    |                      |
|  | - Depression head      |   | - Calibration (Platt)  |                      |
|  | - Dysarthria head      |   | - Conformal prediction |                      |
|  +------------------------+   +------------------------+                      |
|                                    |                                          |
|                                    v                                          |
|  +----------------------------------------------------------+                 |
|  | MULTITASK LEARNING                                        |                 |
|  | - Shared encoder (transformer/conformer)                  |                 |
|  | - Task-specific output heads                              |                 |
|  | - Auxiliary tasks: age, sex prediction (debiasing)        |                 |
|  +----------------------------------------------------------+                 |
+==============================================================================+
                                    |
                                    v
+==============================================================================+
|                        CLINICAL OUTPUT LAYER                                  |
+==============================================================================+
|  +-----------------------+  +-----------------------+  +--------------------+ |
|  | INTERPRETABLE SCORES  |  | RISK STRATIFICATION   |  | LONGITUDINAL       | |
|  | - Per-biomarker       |  | - Low/Moderate/High   |  | - Baseline compare | |
|  | - Deviation from norm |  | - Condition probs     |  | - Trend detection  | |
|  | - Clinical thresholds |  | - Decision thresholds |  | - Change alerts    | |
|  +-----------------------+  +-----------------------+  +--------------------+ |
|                                    |                                          |
|                                    v                                          |
|  +----------------------------------------------------------+                 |
|  | CONFIDENCE & UNCERTAINTY                                  |                 |
|  | - 95% confidence intervals on all scores                  |                 |
|  | - Signal quality confidence modifier                      |                 |
|  | - "Unable to assess" for low-quality samples              |                 |
|  +----------------------------------------------------------+                 |
+==============================================================================+
```

## 2.2 Data Flow Specification

### Stage 1: Acquisition
```
Input: Raw audio (multiple formats)
Output: AudioMetadata {duration, sample_rate, channels, format}
Validation: Size <10MB, Duration 3-60s, Signal >-40dB
```

### Stage 2: Preprocessing
```
Input: Raw audio bytes
Output: ProcessedAudio {
    samples: float32[-1, 1],
    sample_rate: 16000,
    voiced_segments: [(start, end), ...],
    snr_db: float
}
```

### Stage 3: Feature Extraction
```
Input: ProcessedAudio
Output: FeatureVector {
    acoustic: {jitter, shimmer, hnr, cpps, ...},
    prosodic: {f0_mean, f0_std, speech_rate, pause_ratio, ...},
    articulatory: {fcr, vsa, formants, ...},
    spectral: {mfcc[40], spectral_centroid, ...},
    temporal: {tremor_score, stability_index, ...}
}
```

### Stage 4: Representation Learning
```
Input: ProcessedAudio
Output: LearnedEmbeddings {
    wav2vec: float[768],
    whisper: float[512],
    fused: float[1280]
}
```

### Stage 5: Inference
```
Input: FeatureVector + LearnedEmbeddings
Output: PredictionResult {
    condition_scores: {parkinson: (0.23, 0.95CI), ...},
    risk_level: "moderate",
    confidence: 0.87,
    calibrated: true
}
```

### Stage 6: Clinical Output
```
Input: PredictionResult + PatientContext
Output: ClinicalReport {
    risk_assessment: RiskAssessment,
    biomarkers: Dict[str, BiomarkerResult],
    recommendations: List[str],
    confidence_intervals: Dict[str, (float, float)],
    longitudinal_comparison: Optional[TrendAnalysis]
}
```

## 2.3 Module Interfaces

```python
# Core interfaces (Python Protocol definitions)

class IAudioValidator(Protocol):
    def validate(self, audio: bytes, metadata: dict) -> ValidationResult: ...

class IPreprocessor(Protocol):
    def process(self, audio: bytes) -> ProcessedAudio: ...

class IFeatureExtractor(Protocol):
    def extract(self, audio: ProcessedAudio) -> FeatureVector: ...

class IEmbeddingModel(Protocol):
    def encode(self, audio: ProcessedAudio) -> np.ndarray: ...

class IConditionClassifier(Protocol):
    def predict(self, features: FeatureVector, embeddings: np.ndarray) -> PredictionResult: ...

class IClinicalMapper(Protocol):
    def map_to_clinical(self, prediction: PredictionResult, context: PatientContext) -> ClinicalReport: ...
```

---

# SECTION 3: BIOMARKERS AND PHYSIOLOGICAL MAPPING

## 3.1 Comprehensive Biomarker Table

| Condition | Biomarker | Physiological Mechanism | Measurable Feature | Normal Range | Clinical Threshold |
|-----------|-----------|------------------------|-------------------|--------------|-------------------|
| **PARKINSON'S DISEASE** |
| | Jitter (local) | Dopamine depletion -> laryngeal muscle rigidity | Cycle-to-cycle F0 perturbation | <1.04% | >2.0% |
| | Shimmer (local) | Vocal fold stiffness, incomplete closure | Cycle-to-cycle amplitude perturbation | <3.81% | >5.0% |
| | Voice Tremor | Resting tremor affecting laryngeal muscles | 4-6Hz modulation in F0/amplitude | <0.15 score | >0.25 |
| | Hypophonia | Reduced respiratory drive, chest wall rigidity | Mean intensity, dynamic range | >60dB SPL | <50dB SPL |
| | Monopitch | Bradykinesia affecting pitch modulation | F0 standard deviation | 20-100Hz | <10Hz |
| | Reduced Articulation | Orofacial bradykinesia | FCR, articulation rate | FCR<1.2 | FCR>1.5 |
| **ALZHEIMER'S/MCI** |
| | Pause Ratio | Word-finding difficulty, semantic memory decline | Non-speech duration / total duration | 10-25% | >40% |
| | Speech Rate | Cognitive slowing, planning deficits | Syllables per second | 3.5-6.5 syl/s | <2.5 syl/s |
| | Lexical Diversity | Semantic memory degradation | Type-token ratio, vocabulary richness | TTR>0.5 | TTR<0.3 |
| | Filled Pauses | Executive dysfunction, retrieval failure | "um", "uh" frequency per 100 words | <5 per 100 | >15 per 100 |
| | Syntactic Complexity | Language network degeneration | Mean length of utterance, clause embedding | MLU>10 | MLU<6 |
| **DEPRESSION (MDD)** |
| | Prosodic Flattening | Psychomotor retardation, anhedonia | F0 coefficient of variation | CV>20% | CV<10% |
| | Reduced Energy | Fatigue, lack of motivation | Mean vocal intensity, energy contour | Normal range | <-6dB from norm |
| | Speech Rate Slowing | Psychomotor retardation | Words/syllables per minute | 120-180 wpm | <90 wpm |
| | Phonation Time | Reduced social engagement, fatigue | Speaking time ratio in conversation | 40-60% | <20% |
| **DYSARTHRIA** |
| | FCR (Formant Centralization) | Reduced articulatory range, weakness | (F2u+F2a+F1i+F1u)/(F2i+F1a) | 0.9-1.1 | >1.2 |
| | VSA (Vowel Space Area) | Articulatory undershoot | Triangle area (F1/F2 of /a/,/i/,/u/) | >200,000 Hz^2 | <100,000 Hz^2 |
| | Consonant Precision | Oral motor imprecision | Burst accuracy, VOT variability | >90% | <70% |
| | HNR | Incomplete glottal closure, breathiness | Harmonics-to-noise ratio | 20-30dB | <12dB |
| **LARYNGEAL PATHOLOGY** |
| | CPPS | Aperiodicity from mass lesions, edema | Cepstral peak prominence (smoothed) | 14-30dB | <11dB |
| | Jitter/Shimmer | Irregular vibration from pathology | Combined perturbation measures | See above | See above |
| | Maximum Phonation Time | Glottal inefficiency | Sustained /a/ duration | >15s | <10s |
| **ALS/MOTOR NEURON** |
| | Speaking Rate Decline | Bulbar weakness progression | Words per minute over time | Stable | >15% decline/year |
| | Hypernasality | Velopharyngeal incompetence | Nasalance ratio | <30% | >50% |
| | Articulatory Breakdown | Tongue weakness | Diadochokinetic rate (DDK) | >5 reps/s | <3 reps/s |

## 3.2 Novel Composite Biomarkers

### 3.2.1 Neuromotor Instability Index (NII)

**Purpose**: Unified score for motor control dysfunction across tremor, jitter, and coordination.

```python
def neuromotor_instability_index(features: dict) -> float:
    """
    Composite index combining multiple motor instability markers.
    
    Components:
    1. Voice tremor power (4-6Hz band)
    2. Jitter instability (coefficient of variation of jitter)
    3. Shimmer instability (coefficient of variation of shimmer)
    4. F0 trajectory smoothness (first derivative variance)
    """
    tremor_component = normalize(features['tremor_score'], 0, 0.5)
    jitter_cv = features['jitter_std'] / (features['jitter_mean'] + 1e-6)
    shimmer_cv = features['shimmer_std'] / (features['shimmer_mean'] + 1e-6)
    f0_roughness = np.std(np.diff(features['f0_contour']))
    
    nii = (0.35 * tremor_component + 
           0.25 * normalize(jitter_cv, 0, 1) +
           0.20 * normalize(shimmer_cv, 0, 1) +
           0.20 * normalize(f0_roughness, 0, 50))
    
    return nii  # 0-1, higher = more instability
```

### 3.2.2 Vocal Fold Micro-Tremor Metric (VFMT)

**Purpose**: Detect subtle, subclinical tremor before clinical manifestation.

```python
def vocal_fold_microtremor(f0_contour: np.ndarray, sr: int = 100) -> dict:
    """
    Quantify micro-tremor in voice using high-resolution spectral analysis.
    
    Clinical rationale: Micro-tremor (low amplitude, 4-8Hz) may precede 
    clinically visible tremor by years in prodromal PD.
    """
    # High-resolution FFT (zero-padded)
    n_fft = 2048
    f0_centered = f0_contour - np.mean(f0_contour)
    spectrum = np.abs(np.fft.rfft(f0_centered, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Tremor bands
    micro_band = (freqs >= 4) & (freqs <= 8)
    control_band = (freqs >= 15) & (freqs <= 25)
    
    tremor_power = np.sum(spectrum[micro_band]**2)
    control_power = np.sum(spectrum[control_band]**2) + 1e-10
    
    return {
        'microtremor_ratio': tremor_power / control_power,
        'peak_tremor_freq': freqs[micro_band][np.argmax(spectrum[micro_band])],
        'tremor_bandwidth': estimate_bandwidth(spectrum, freqs, micro_band)
    }
```

### 3.2.3 Articulatory Coordination Entropy (ACE)

**Purpose**: Measure the predictability/coordination of articulatory movements.

```python
def articulatory_coordination_entropy(formants: dict) -> float:
    """
    Compute entropy of formant transitions as proxy for articulatory coordination.
    
    High entropy = unpredictable, uncoordinated movements (dysarthria)
    Low entropy = smooth, coordinated articulation
    """
    # Formant velocities
    f1_vel = np.diff(formants['f1_contour'])
    f2_vel = np.diff(formants['f2_contour'])
    
    # Joint histogram (2D distribution of F1/F2 velocities)
    hist, _, _ = np.histogram2d(f1_vel, f2_vel, bins=20, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    
    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    # Normalize to 0-1 (higher = more disordered)
    max_entropy = np.log2(400)  # 20x20 bins
    return entropy / max_entropy
```

### 3.2.4 Respiratory-Phonatory Coupling Score (RPCS)

**Purpose**: Assess coordination between breathing and phonation.

```python
def respiratory_phonatory_coupling(audio: np.ndarray, sr: int) -> float:
    """
    Measure synchronization between respiratory envelope and phonation.
    
    Clinical relevance: Decoupling indicates respiratory muscle weakness
    or poor motor planning (ALS, PD, respiratory disorders).
    """
    # Extract amplitude envelope (proxy for respiratory effort)
    envelope = np.abs(scipy.signal.hilbert(audio))
    envelope_smooth = scipy.signal.savgol_filter(envelope, 1001, 3)
    
    # Extract instantaneous F0 (phonatory signal)
    f0 = extract_f0(audio, sr)
    
    # Resample to same rate
    f0_resampled = scipy.signal.resample(f0, len(envelope_smooth))
    
    # Compute phase coherence
    phase_env = np.angle(scipy.signal.hilbert(envelope_smooth))
    phase_f0 = np.angle(scipy.signal.hilbert(f0_resampled))
    
    coherence = np.abs(np.mean(np.exp(1j * (phase_env - phase_f0))))
    
    return coherence  # 0-1, higher = better coupling
```

### 3.2.5 Cognitive-Linguistic Latency Markers (CLLM)

**Purpose**: Quantify cognitive processing delays manifested in speech timing.

```python
def cognitive_linguistic_latency(transcription: dict) -> dict:
    """
    Extract latency markers indicative of cognitive processing load.
    
    Based on word-level timestamps from ASR (Whisper).
    """
    words = transcription['words']
    
    latencies = {
        'response_latency': words[0]['start'],  # Time to first word
        'inter_word_gaps': [],
        'pre_content_pauses': [],  # Pauses before content words
        'hesitation_rate': 0
    }
    
    content_words = {'NOUN', 'VERB', 'ADJ', 'ADV'}  # Would use POS tagger
    
    for i in range(1, len(words)):
        gap = words[i]['start'] - words[i-1]['end']
        latencies['inter_word_gaps'].append(gap)
        
        if words[i].get('pos') in content_words and gap > 0.2:
            latencies['pre_content_pauses'].append(gap)
    
    latencies['mean_gap'] = np.mean(latencies['inter_word_gaps'])
    latencies['gap_variability'] = np.std(latencies['inter_word_gaps'])
    latencies['content_pause_ratio'] = len(latencies['pre_content_pauses']) / max(1, len(words))
    
    return latencies
```

---

# SECTION 4: MODELS AND LIBRARIES

## 4.1 Pretrained Model Recommendations

### 4.1.1 Self-Supervised Speech Embeddings

| Model | Architecture | Embedding Dim | Training Data | Best For | Compute |
|-------|-------------|---------------|---------------|----------|---------|
| **Wav2Vec 2.0 Large** | Transformer | 1024 | 60k hrs | General speech representation | High |
| **HuBERT Large** | Transformer | 1024 | 60k hrs | Acoustic patterns, pathology | High |
| **WavLM Large** | Transformer | 1024 | 94k hrs | Noisy/paralinguistic tasks | High |
| **Whisper Encoder** | Transformer | 768/1024 | 680k hrs | Multilingual, transcription | Medium |
| **ECAPA-TDNN** | TDNN+SE | 192 | VoxCeleb | Speaker embeddings, voice quality | Low |

**Primary Recommendation**: **WavLM Large** for pathology detection
- Best performance on paralinguistic tasks (emotion, pathology)
- Robust to noise (important for real-world deployment)
- Pre-trained with denoising objective

**Secondary**: **Whisper Encoder** for linguistic features
- Provides both transcription AND hidden state embeddings
- Multilingual support critical for diverse populations

### 4.1.2 Task-Specific Architectures

```
RECOMMENDED ARCHITECTURE STACK
===============================

                    Raw Audio (16kHz)
                           |
            +--------------+--------------+
            |                             |
            v                             v
    +---------------+             +---------------+
    | WavLM Large   |             | Whisper Small |
    | (Frozen)      |             | (Frozen)      |
    +---------------+             +---------------+
            |                             |
            | 1024-dim                    | 768-dim + transcription
            |                             |
            v                             v
    +---------------+             +---------------+
    | Handcrafted   |             | Linguistic    |
    | Features      |             | Features      |
    | (Parselmouth) |             | (from text)   |
    +---------------+             +---------------+
            |                             |
            | ~100 features               | ~50 features
            |                             |
            +-------------+---------------+
                          |
                          v
                  +---------------+
                  | Fusion Layer  |
                  | (Attention or |
                  |  Concat+MLP)  |
                  +---------------+
                          |
                          v
                  +---------------+
                  | Shared Trunk  |
                  | Conformer x4  |
                  +---------------+
                          |
        +-----------------+-----------------+
        |        |        |        |        |
        v        v        v        v        v
    +------+ +------+ +------+ +------+ +------+
    |  PD  | |  AD  | | Dep  | | Dys  | | Aux  |
    | Head | | Head | | Head | | Head | | Head |
    +------+ +------+ +------+ +------+ +------+
        |        |        |        |        |
        v        v        v        v        v
    P(PD)    P(AD)   P(Dep)  P(Dys)  Age,Sex
```

### 4.1.3 Conformer vs Transformer Selection

| Aspect | Transformer | Conformer | Recommendation |
|--------|-------------|-----------|----------------|
| Local patterns | Weak | Strong (Conv) | **Conformer** for acoustic |
| Long-range | Strong | Strong | Tie |
| Compute | High | Higher | Transformers if GPU-limited |
| Speech tasks | Good | Better | **Conformer** preferred |

**Selection**: Conformer for condition heads (better for acoustic patterns)

### 4.1.4 Uncertainty Estimation

```python
class UncertaintyHead(nn.Module):
    """
    Probabilistic output head with calibrated uncertainty.
    """
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes * 2)  # Mean + log_var
        self.temperature = nn.Parameter(torch.ones(1))  # Platt scaling
        
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> dict:
        # Monte Carlo Dropout would be applied externally
        out = self.fc(x)
        mean, log_var = out.chunk(2, dim=-1)
        
        # Reparameterization for aleatoric uncertainty
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(n_samples, *mean.shape)
        samples = mean + eps * std
        
        # Calibrated probabilities
        probs = F.softmax(samples / self.temperature, dim=-1)
        
        return {
            'mean_prob': probs.mean(dim=0),
            'epistemic_uncertainty': probs.std(dim=0),  # From MC dropout
            'aleatoric_uncertainty': std.mean(),  # From learned variance
            'calibrated': True
        }
```

## 4.2 Libraries and Tools

### 4.2.1 Core Stack

| Component | Library | Justification |
|-----------|---------|---------------|
| **Acoustic Analysis** | `parselmouth` (Praat) | Gold standard, clinically validated algorithms |
| **Audio I/O** | `torchaudio` | GPU-accelerated, PyTorch integration |
| **Spectral Features** | `librosa` | Comprehensive, well-documented |
| **Extended Features** | `openSMILE` | eGeMAPS feature set (standardized for affect) |
| **VAD** | `pyannote.audio` | State-of-the-art, neural VAD |
| **Diarization** | `pyannote.audio` | Speaker separation if needed |
| **ASR** | `whisper` (OpenAI) | Best accuracy, word timestamps |
| **Deep Learning** | `speechbrain` | Pre-built recipes for speech pathology |
| **Experiment Tracking** | `wandb` / `mlflow` | Reproducibility for clinical validation |

### 4.2.2 Feature Extraction Configuration

```python
# Recommended OpenSMILE configuration (eGeMAPS)
OPENSMILE_CONFIG = {
    "feature_set": "eGeMAPSv02",  # Extended Geneva Minimalistic
    "feature_level": "functionals",  # Summary statistics per utterance
    "features_include": [
        "F0semitoneFrom27.5Hz_sma3nz_*",  # Pitch
        "jitterLocal_sma3nz_*",            # Jitter
        "shimmerLocaldB_sma3nz_*",         # Shimmer
        "HNRdBACF_sma3nz_*",               # HNR
        "logRelF0-H1-H2_sma3nz_*",         # Spectral slope
        "logRelF0-H1-A3_sma3nz_*",         # Harmonic structure
        "F1frequency_sma3nz_*",            # Formants
        "F2frequency_sma3nz_*",
        "F3frequency_sma3nz_*",
        "loudness_sma3_*",                 # Intensity
        "spectralFlux_sma3_*",             # Temporal dynamics
        "mfcc1-4_sma3_*"                   # MFCCs
    ]
}

# Parselmouth extraction (clinical grade)
PARSELMOUTH_CONFIG = {
    "pitch": {
        "time_step": 0.01,
        "pitch_floor": 75,
        "pitch_ceiling": 500,
        "method": "cc"  # Cross-correlation (more robust)
    },
    "jitter": ["local", "rap", "ppq5", "ddp"],
    "shimmer": ["local", "apq3", "apq5", "apq11", "dda"],
    "hnr": {"time_step": 0.01, "min_pitch": 75},
    "formants": {
        "time_step": 0.01,
        "max_formant": 5500,  # 5000 for males, 5500 for females
        "num_formants": 5
    }
}
```

## 4.3 Training Strategy

### 4.3.1 Pretraining -> Fine-tuning Pipeline

```
Stage 1: Feature Extractor Pretraining (Optional)
-------------------------------------------------
- Use self-supervised learning on unlabeled medical speech
- Contrastive learning on healthy vs pathological voice
- Dataset: Large unlabeled clinical voice recordings

Stage 2: Multitask Pretraining
------------------------------
- Train shared encoder on multiple conditions simultaneously
- Auxiliary tasks: age prediction, sex prediction, accent classification
- Purpose: Learn robust, generalizable representations

Stage 3: Condition-Specific Fine-tuning
---------------------------------------
- Freeze shared encoder (or low learning rate)
- Train condition-specific heads
- Use class-weighted loss for imbalanced datasets

Stage 4: Calibration
--------------------
- Hold-out calibration set (10% of validation)
- Platt scaling or isotonic regression
- Ensure predicted probabilities match actual frequencies
```

### 4.3.2 Training Hyperparameters

```python
TRAINING_CONFIG = {
    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 1e-4,  # Fine-tuning
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    
    # Scheduler
    "scheduler": "cosine_with_restarts",
    "num_cycles": 3,
    
    # Regularization
    "dropout": 0.3,  # Higher for uncertainty estimation
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    
    # Training
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10,
    
    # Loss
    "loss": "focal_loss",  # For class imbalance
    "focal_gamma": 2.0,
    
    # Validation
    "val_frequency": "epoch",
    "metrics": ["auroc", "sensitivity", "specificity", "calibration_error"]
}
```

---

# SECTION 5: DATASET STRATEGY

## 5.1 Existing Medical Speech Datasets

### 5.1.1 Parkinson's Disease

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| **mPower** | 65k recordings | Sustained vowels, walking, tapping | Synapse (open) |
| **PC-GITA** | 100 PD + 100 HC | Sentences, DDK, monologue | Request |
| **Italian PD** | 28 PD + 28 HC | Multiple tasks | Request |
| **MDVR-KCL** | 48 PD + 20 HC | Reading passages | Open |

### 5.1.2 Alzheimer's / Cognitive Decline

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| **ADReSS/ADReSSo** | 156 subjects | Cookie Theft description | Challenge |
| **Pitt Corpus** | 552 subjects | Spontaneous speech | DementiaBank |
| **I-CONECT** | Longitudinal | Video calls | Request |

### 5.1.3 Depression / Mental Health

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| **DAIC-WOZ** | 189 subjects | Clinical interviews | Request |
| **AVEC Depression** | ~300 subjects | Video + audio | Challenge |
| **eRisk** | Variable | Social media (text) | Challenge |

### 5.1.4 Voice Disorders / Dysphonia

| Dataset | Size | Content | Access |
|---------|------|---------|--------|
| **PVQD** | 296 subjects | Sustained vowels | Open |
| **Saarbruecken** | 2000+ subjects | Vowels, sentences | Open |
| **KayPentax** | Commercial | Clinical recordings | Purchase |

## 5.2 Population Diversity Requirements

```python
DEMOGRAPHIC_REQUIREMENTS = {
    "age": {
        "18-39": 0.20,  # Minimum proportion
        "40-59": 0.30,
        "60-79": 0.35,
        "80+": 0.15
    },
    "sex": {
        "male": 0.45,
        "female": 0.45,
        "other": 0.10  # Target, may be lower
    },
    "ethnicity": {
        "white": 0.40,
        "black": 0.15,
        "asian": 0.15,
        "hispanic": 0.15,
        "other": 0.15
    },
    "accent": {
        "native_english": 0.50,
        "non_native": 0.50
    },
    "socioeconomic": {
        "diverse_education_levels": True,
        "diverse_occupation": True
    }
}
```

## 5.3 Label Quality Standards

### 5.3.1 Diagnostic Criteria

| Condition | Required Diagnosis | Gold Standard |
|-----------|-------------------|---------------|
| Parkinson's | Movement disorder specialist | UK Brain Bank Criteria |
| Alzheimer's | Neurologist + neuropsych | NIA-AA Criteria |
| Depression | Psychiatrist | DSM-5 + PHQ-9 score |
| Dysarthria | Speech pathologist | Perceptual + instrumental |

### 5.3.2 Annotation Protocol

```
1. Primary labeler: Clinical specialist
2. Secondary labeler: Trained research assistant
3. Disagreement resolution: Panel review
4. Inter-rater reliability target: Cohen's kappa > 0.8
5. Severity scoring: Standardized scales (UPDRS, CDR, HAM-D)
```

## 5.4 Augmentation Strategies

### 5.4.1 Medically Safe Augmentations

| Augmentation | Safe | Risk | Notes |
|--------------|------|------|-------|
| **Time stretch (0.9-1.1x)** | Yes | Low | Preserves pathology |
| **Pitch shift (+/-2 semitones)** | Caution | Medium | May alter jitter/shimmer interpretation |
| **Noise addition (low SNR)** | Yes | Low | Improves robustness |
| **Room impulse response** | Yes | Low | Realistic variation |
| **SpecAugment (masking)** | Yes | Low | For learned embeddings only |
| **Speed perturbation (>1.2x)** | No | High | Alters speech rate biomarker |
| **Synthetic voice (TTS)** | No | High | Does not contain pathology |
| **Voice conversion** | No | High | Removes pathological features |

### 5.4.2 Augmentation Pipeline

```python
class MedicalSafeAugmentor:
    """
    Augmentation pipeline validated for medical speech.
    """
    def __init__(self):
        self.augmentations = [
            AddGaussianNoise(min_snr=15, max_snr=30, p=0.3),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            AddBackgroundNoise(sounds_path="noise_samples/", p=0.2),
            RoomSimulator(p=0.2),
            # NO pitch shift, speed change, or voice conversion
        ]
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            audio = aug(audio)
        return audio
```

## 5.5 Dataset Sourcing Plan

```
Phase 1: Existing Public Datasets (Months 1-3)
----------------------------------------------
- Acquire mPower, PC-GITA, ADReSS, DAIC-WOZ, PVQD
- Harmonize formats, sampling rates, labels
- Establish baseline performance

Phase 2: Clinical Partnerships (Months 3-12)
--------------------------------------------
- Partner with 3+ medical centers
- IRB approval for prospective collection
- Target: 500 patients per condition
- Include longitudinal follow-up

Phase 3: Diverse Population Expansion (Months 12-24)
---------------------------------------------------
- Multi-site, multi-country collection
- Focus on underrepresented demographics
- Accent/language diversity
- Target: 2000+ patients total

Phase 4: Continuous Collection (Ongoing)
----------------------------------------
- Integrate with clinical workflow
- Federated learning from deployed systems
- Continuous model updates
```

---

# SECTION 6: VALIDATION AND CLINICAL CREDIBILITY

## 6.1 Evaluation Metrics

### 6.1.1 Primary Metrics

| Metric | Formula | Target | Purpose |
|--------|---------|--------|---------|
| **Sensitivity (Recall)** | TP / (TP + FN) | >85% | Screening (don't miss cases) |
| **Specificity** | TN / (TN + FP) | >80% | Avoid false alarms |
| **AUROC** | Area under ROC | >0.85 | Overall discrimination |
| **AUPRC** | Area under PR curve | >0.50 | Important for imbalanced data |
| **PPV (Precision)** | TP / (TP + FP) | >30% | Clinical utility |
| **NPV** | TN / (TN + FN) | >95% | Rule-out confidence |

### 6.1.2 Calibration Metrics

| Metric | Formula | Target | Purpose |
|--------|---------|--------|---------|
| **ECE** | Expected Calibration Error | <0.05 | Probability accuracy |
| **MCE** | Maximum Calibration Error | <0.15 | Worst-case calibration |
| **Brier Score** | Mean squared error | <0.15 | Overall probability quality |

### 6.1.3 Reliability Metrics

| Metric | Method | Target | Purpose |
|--------|--------|--------|---------|
| **Test-Retest Reliability** | ICC (same subject, 1 week apart) | >0.80 | Measurement stability |
| **Inter-Rater Agreement** | Cohen's kappa | >0.75 | Annotation quality |

## 6.2 Validation Framework

### 6.2.1 Internal Validation

```
5-Fold Stratified Cross-Validation
-----------------------------------
- Stratify by: condition, age, sex
- Report: mean +/- std across folds
- No data leakage between folds

Nested Cross-Validation (for hyperparameter tuning)
--------------------------------------------------
- Outer: 5-fold for evaluation
- Inner: 3-fold for hyperparameter selection
- Report: unbiased performance estimate
```

### 6.2.2 External Validation

```
Geographic Validation
---------------------
- Train: US/UK datasets
- Test: European, Asian datasets
- Purpose: Generalization across healthcare systems

Temporal Validation
-------------------
- Train: Data collected 2020-2023
- Test: Data collected 2024+
- Purpose: Robustness to distribution shift

Device Validation
-----------------
- Train: Clinical microphones
- Test: Consumer devices (phones, laptops)
- Purpose: Real-world deployment readiness
```

### 6.2.3 Clinical Trial-Style Validation

```
Phase I: Technical Validation (n=100)
-------------------------------------
- Primary: Algorithm accuracy vs reference standard
- Secondary: Processing time, failure rate
- Setting: Controlled clinical environment

Phase II: Clinical Validation (n=500)
-------------------------------------
- Primary: Sensitivity, specificity vs clinical diagnosis
- Secondary: Added value over clinical assessment alone
- Setting: Multiple clinical sites

Phase III: Prospective Validation (n=2000)
-----------------------------------------
- Primary: Impact on clinical outcomes (earlier diagnosis)
- Secondary: Cost-effectiveness, patient acceptance
- Setting: Real-world clinical practice
- Design: Randomized controlled trial (if possible)
```

## 6.3 Bias and Fairness Testing

### 6.3.1 Subgroup Analysis

```python
FAIRNESS_SUBGROUPS = [
    "age_group",      # 18-39, 40-59, 60-79, 80+
    "sex",            # Male, Female
    "ethnicity",      # White, Black, Asian, Hispanic, Other
    "accent",         # Native, Non-native
    "education",      # High school, College, Graduate
    "device_type",    # Clinical, Smartphone, Laptop
    "noise_level",    # Low (<30dB), Medium (30-50dB), High (>50dB)
]

def fairness_report(predictions: np.ndarray, labels: np.ndarray, 
                    subgroups: pd.DataFrame) -> dict:
    """
    Generate comprehensive fairness report.
    """
    report = {}
    for subgroup in FAIRNESS_SUBGROUPS:
        group_metrics = {}
        for group_value in subgroups[subgroup].unique():
            mask = subgroups[subgroup] == group_value
            group_metrics[group_value] = {
                'auroc': roc_auc_score(labels[mask], predictions[mask]),
                'sensitivity': recall_score(labels[mask], predictions[mask] > 0.5),
                'specificity': recall_score(1-labels[mask], predictions[mask] <= 0.5),
                'n_samples': mask.sum()
            }
        report[subgroup] = group_metrics
        
        # Compute disparity metrics
        aurocs = [m['auroc'] for m in group_metrics.values()]
        report[f"{subgroup}_disparity"] = max(aurocs) - min(aurocs)
        
    return report
```

### 6.3.2 Fairness Criteria

| Criterion | Definition | Target |
|-----------|------------|--------|
| **Demographic Parity** | P(positive\|group) equal across groups | Difference <10% |
| **Equalized Odds** | TPR and FPR equal across groups | Difference <10% |
| **Calibration** | Predicted prob = actual prob per group | ECE <0.05 per group |

## 6.4 Acceptance Criteria for Deployment

```python
DEPLOYMENT_CRITERIA = {
    # Performance
    "auroc_overall": {"minimum": 0.80, "target": 0.85},
    "sensitivity_overall": {"minimum": 0.80, "target": 0.90},
    "specificity_overall": {"minimum": 0.75, "target": 0.85},
    
    # Calibration
    "expected_calibration_error": {"maximum": 0.10, "target": 0.05},
    
    # Fairness
    "max_subgroup_auroc_disparity": {"maximum": 0.10},
    "max_subgroup_sensitivity_disparity": {"maximum": 0.15},
    
    # Reliability
    "test_retest_icc": {"minimum": 0.75, "target": 0.85},
    
    # Usability
    "processing_time_seconds": {"maximum": 5.0, "target": 2.0},
    "failure_rate": {"maximum": 0.05, "target": 0.02},
    
    # External validation
    "external_auroc_drop": {"maximum": 0.10},  # vs internal
}
```

---

# SECTION 7: FAILURE ANALYSIS AND RISK MITIGATION

## 7.1 Confounding Factors

### 7.1.1 Speaker Characteristics

| Confounder | Impact | Mitigation |
|------------|--------|------------|
| **Age** | Jitter/shimmer naturally increase with age | Age-normalized z-scores |
| **Sex** | F0 range differs; formant frequencies differ | Sex-specific normalization |
| **Smoking History** | Increases shimmer, reduces HNR | Include as covariate; document |
| **Accent/Dialect** | Affect prosody, formant positions | Diverse training data; accent adaptation |
| **Native Language** | Prosodic patterns vary by L1 | Multilingual models; language detection |
| **Vocal Training** | Singers may have atypical control | Occupational history as covariate |
| **Medication** | Antidepressants, beta-blockers affect voice | Document medication status |
| **Time of Day** | Morning hoarseness common | Standardize recording time |
| **Caffeine/Hydration** | Affect vocal fold hydration | Document or control |

### 7.1.2 Recording Conditions

| Confounder | Impact | Mitigation |
|------------|--------|------------|
| **Microphone Type** | Frequency response varies | Device calibration; transfer learning |
| **Microphone Distance** | Affects intensity, noise ratio | Distance normalization; instruction |
| **Background Noise** | Contaminates all features | Noise estimation; SNR threshold |
| **Reverberation** | Smears spectral features | Dereverberation; room correction |
| **Compression (MP3)** | Alters high-frequency content | Prefer lossless; document codec |
| **Clipping** | Distorts amplitude features | Detect and reject; re-record |

### 7.1.3 Task Variability

| Confounder | Impact | Mitigation |
|------------|--------|------------|
| **Reading vs Spontaneous** | Cognitive load differs | Standardize task types |
| **Passage Familiarity** | Practice effects | Use unfamiliar passages |
| **Emotional State** | Affects prosody, rate | Neutral instruction; document affect |
| **Fatigue** | Increases with session length | Short recordings; time limit |

## 7.2 Adversarial Failures

### 7.2.1 Known Attack Vectors

```
1. Voice Modification Software
   - Risk: Patient uses pitch correction, noise removal
   - Detection: Analyze for unnatural smoothness, artifacts
   
2. Malingering (Fake Symptoms)
   - Risk: Patient deliberately speaks abnormally
   - Mitigation: Consistency checks across multiple recordings
   
3. Concealment (Hide Symptoms)
   - Risk: Patient consciously compensates
   - Detection: Subtle biomarkers less under voluntary control
   
4. Audio Replay Attacks
   - Risk: Playback of different person's recording
   - Mitigation: Liveness detection (if speaker verification needed)
```

### 7.2.2 Detection Mechanisms

```python
class ArtifactDetector:
    """
    Detect recording artifacts and manipulation.
    """
    def detect(self, audio: np.ndarray, sr: int) -> dict:
        return {
            'clipping': self._detect_clipping(audio),
            'codec_artifacts': self._detect_compression(audio, sr),
            'unnaturally_smooth': self._detect_oversmooothing(audio, sr),
            'splicing': self._detect_discontinuities(audio, sr),
            'synthetic': self._detect_deepfake(audio, sr),
            'quality_score': self._overall_quality(audio, sr)
        }
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Proportion of samples at max/min amplitude."""
        threshold = 0.99 * np.max(np.abs(audio))
        return np.mean(np.abs(audio) > threshold)
    
    def _detect_compression(self, audio: np.ndarray, sr: int) -> bool:
        """Detect lossy compression artifacts in high frequencies."""
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        high_freq_power = np.sum(spectrum[freqs > 15000])
        total_power = np.sum(spectrum)
        # Lossy codecs often cut high frequencies
        return high_freq_power / total_power < 0.001
```

## 7.3 Distribution Shift Scenarios

| Shift Type | Example | Detection | Response |
|------------|---------|-----------|----------|
| **Covariate Shift** | New microphone in clinic | Feature distribution monitoring | Recalibration |
| **Prior Shift** | PD prevalence differs in new site | Label distribution tracking | Threshold adjustment |
| **Concept Shift** | New medication changes voice patterns | Outcome monitoring | Alert + retrain |
| **Subpopulation Shift** | New pediatric population | Age distribution monitoring | Reject + develop new model |

## 7.4 Overfitting Risks

### 7.4.1 Dataset Artifacts

```
- Recording device signatures (microphone fingerprinting)
- Site-specific acoustic environments
- Transcriptionist speaking styles (for read speech)
- Label leakage through metadata
- Class imbalance memorization

Mitigation:
- Leave-one-site-out validation
- Leave-one-device-out validation
- Adversarial training to remove device/site effects
- Rigorous data splitting by patient, not by recording
```

### 7.4.2 Monitoring Strategies

```python
class ModelMonitor:
    """
    Production monitoring for speech analysis pipeline.
    """
    def __init__(self, reference_distribution: dict):
        self.reference = reference_distribution
        self.alerts = []
        
    def check_input_distribution(self, features: dict) -> list:
        """Alert on input feature drift."""
        alerts = []
        for feature, value in features.items():
            ref_mean = self.reference[feature]['mean']
            ref_std = self.reference[feature]['std']
            z_score = abs(value - ref_mean) / ref_std
            if z_score > 3:
                alerts.append({
                    'type': 'input_drift',
                    'feature': feature,
                    'z_score': z_score,
                    'severity': 'warning' if z_score < 5 else 'critical'
                })
        return alerts
    
    def check_output_calibration(self, predictions: list, outcomes: list) -> dict:
        """Track calibration over time."""
        # Compute rolling calibration error
        bins = np.linspace(0, 1, 11)
        calibration_errors = []
        for i in range(len(bins) - 1):
            mask = (predictions >= bins[i]) & (predictions < bins[i+1])
            if mask.sum() > 0:
                expected = np.mean(predictions[mask])
                observed = np.mean(outcomes[mask])
                calibration_errors.append(abs(expected - observed))
        return {
            'ece': np.mean(calibration_errors),
            'mce': np.max(calibration_errors),
            'calibrated': np.mean(calibration_errors) < 0.10
        }
```

## 7.5 Human-in-the-Loop Design

```
Mandatory Review Triggers
=========================

1. High risk score (>70) -> Clinician review before report
2. Low confidence (<0.5) -> Flag for manual verification
3. Contradictory biomarkers -> Expert panel review
4. Near decision threshold -> Dual assessment required
5. Unusual feature patterns -> Quality check before acceptance

Clinician Interface Requirements
================================

- All raw biomarkers visible (not just composite score)
- Uncertainty quantification prominently displayed
- Access to audio playback
- Ability to override/annotate
- Feedback mechanism for model improvement
```

## 7.6 Risk Register

| Risk ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---------|------|------------|--------|------------|-------|
| R001 | False negative (missed disease) | Medium | High | High sensitivity design; mandatory follow-up | Clinical |
| R002 | False positive (unnecessary referral) | Medium | Medium | Calibrated specificity; multi-stage screening | Clinical |
| R003 | Microphone variability degrades accuracy | High | High | Device calibration; transfer learning | Engineering |
| R004 | Accent bias reduces accuracy for minorities | Medium | High | Diverse training data; subgroup testing | Ethics |
| R005 | Adversarial input manipulation | Low | Medium | Artifact detection; liveness checks | Security |
| R006 | Model drift over time | Medium | High | Continuous monitoring; retraining pipeline | ML Ops |
| R007 | Regulatory non-compliance | Low | Critical | Regulatory consulting; documentation | Legal |
| R008 | Privacy breach (voice data) | Low | Critical | Encryption; access controls; anonymization | Security |
| R009 | System downtime in clinical workflow | Low | High | Redundancy; graceful degradation | Engineering |
| R010 | Clinician over-reliance on AI | Medium | High | Training; clear uncertainty display | Clinical |

---

# SECTION 8: IMPROVEMENT ROADMAP

## 8.1 Current System Weaknesses

1. **Single Modality**: Voice-only limits accuracy ceiling
2. **Cross-sectional Only**: No longitudinal baseline modeling
3. **Fixed Thresholds**: Not personalized to individual
4. **Limited Language Support**: English-centric
5. **Device Sensitivity**: Performance varies by microphone
6. **No Real-time**: Batch processing only

## 8.2 Staged Roadmap

### Phase 1: Foundation (Months 1-6)
**Goal**: Robust single-modality pipeline with clinical validation

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| M1.1 | Implement core pipeline | Working prototype |
| M1.2 | Parselmouth integration | Clinical-grade features |
| M1.3 | Wav2Vec/Whisper embeddings | Deep representations |
| M1.4 | Basic uncertainty estimation | Confidence intervals |
| M1.5 | Internal validation (5-fold CV) | Performance report |
| M1.6 | Fairness audit | Subgroup analysis |

### Phase 2: Clinical Hardening (Months 6-12)
**Goal**: External validation and regulatory readiness

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| M2.1 | Multi-site data collection | 500+ patients |
| M2.2 | External validation | Generalization report |
| M2.3 | Device variability study | Calibration protocol |
| M2.4 | Regulatory documentation | 510(k) submission prep |
| M2.5 | Clinical interface | Clinician dashboard |
| M2.6 | Human-in-loop workflow | Review protocols |

### Phase 3: Advanced Capabilities (Months 12-24)
**Goal**: Multimodal fusion and personalization

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| M3.1 | Video integration (facial) | Lip movement features |
| M3.2 | Breathing pattern analysis | Respiratory biomarkers |
| M3.3 | Wearable integration | Heart rate, movement |
| M3.4 | Longitudinal tracking | Personal baseline |
| M3.5 | Personalized thresholds | Adaptive algorithms |
| M3.6 | Multilingual expansion | 5+ languages |

### Phase 4: Next-Generation (Months 24-36)
**Goal**: Cutting-edge research integration

| Milestone | Description | Deliverable |
|-----------|-------------|-------------|
| M4.1 | Federated learning | Privacy-preserving updates |
| M4.2 | Continuous learning | Automatic adaptation |
| M4.3 | Real-time streaming | Live analysis |
| M4.4 | Explainable AI | Feature attribution |
| M4.5 | Predictive modeling | Progression forecasting |
| M4.6 | Digital twin | Patient-specific models |

## 8.3 Multimodal Fusion Strategy

```
+------------------+     +------------------+     +------------------+
|  AUDIO STREAM    |     |  VIDEO STREAM    |     |  SENSOR STREAM   |
|  (Voice)         |     |  (Face/Lips)     |     |  (Wearables)     |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
| Speech Features  |     | Visual Features  |     | Physiological    |
| - Acoustic       |     | - Lip aperture   |     | - Heart rate     |
| - Prosodic       |     | - Facial tremor  |     | - Respiration    |
| - Linguistic     |     | - Micro-express  |     | - Movement       |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                    +------------------------+
                    | MULTIMODAL FUSION      |
                    | - Early (feature)      |
                    | - Late (decision)      |
                    | - Hybrid (attention)   |
                    +------------------------+
                                 |
                                 v
                    +------------------------+
                    | ENHANCED PREDICTION    |
                    | - Higher accuracy      |
                    | - Better robustness    |
                    | - Missing modality     |
                    |   tolerance            |
                    +------------------------+
```

## 8.4 Personalized Baseline Modeling

```python
class PersonalizedBaseline:
    """
    Adaptive baseline for longitudinal tracking.
    """
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.history = []
        self.baseline = None
        self.adaptation_rate = 0.1
        
    def update(self, features: dict, timestamp: datetime):
        """Add new observation and update baseline."""
        self.history.append({
            'features': features,
            'timestamp': timestamp
        })
        
        if len(self.history) >= 3:
            # Compute stable baseline from first N recordings
            if self.baseline is None:
                self.baseline = self._compute_initial_baseline()
            else:
                # Slow adaptation (don't track disease progression as normal)
                # Only update if within normal variation
                if self._is_within_normal_variation(features):
                    self._exponential_update(features)
    
    def get_deviation(self, features: dict) -> dict:
        """Compute personalized deviation from baseline."""
        if self.baseline is None:
            return {'status': 'insufficient_history'}
        
        deviations = {}
        for key, value in features.items():
            baseline_mean = self.baseline[key]['mean']
            baseline_std = self.baseline[key]['std']
            z_score = (value - baseline_mean) / (baseline_std + 1e-6)
            deviations[key] = {
                'value': value,
                'baseline': baseline_mean,
                'z_score': z_score,
                'significant': abs(z_score) > 2
            }
        return deviations
```

---

# SECTION 9: IMPLEMENTATION BLUEPRINT

## 9.1 Folder Structure

```
neuralens/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application entry
│   │   ├── config.py                    # Global configuration
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py                # API route aggregator
│   │   │   └── middleware.py            # Auth, logging, rate limiting
│   │   │
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── speech/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ARCHITECTURE.md      # Technical specification
│   │   │   │   ├── router.py            # Speech API endpoints
│   │   │   │   ├── schemas.py           # Pydantic models
│   │   │   │   ├── service.py           # Business logic orchestrator
│   │   │   │   ├── validator.py         # Input validation
│   │   │   │   ├── preprocessor.py      # Audio preprocessing
│   │   │   │   ├── features/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── acoustic.py      # Parselmouth features
│   │   │   │   │   ├── prosodic.py      # Pitch/intensity features
│   │   │   │   │   ├── linguistic.py    # Whisper + NLP features
│   │   │   │   │   ├── embeddings.py    # Wav2Vec/HuBERT
│   │   │   │   │   └── composite.py     # Novel biomarkers
│   │   │   │   ├── models/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── classifier.py    # Condition classifiers
│   │   │   │   │   ├── uncertainty.py   # Uncertainty estimation
│   │   │   │   │   └── calibration.py   # Probability calibration
│   │   │   │   ├── clinical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── risk_calculator.py
│   │   │   │   │   ├── normative_data.py
│   │   │   │   │   └── recommendations.py
│   │   │   │   ├── monitoring/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── drift_detector.py
│   │   │   │   │   ├── quality_checker.py
│   │   │   │   │   └── logging.py
│   │   │   │   └── config.py            # Pipeline-specific config
│   │   │   │
│   │   │   └── retinal/                 # Other pipelines...
│   │   │
│   │   ├── models/                      # Shared ML model utilities
│   │   │   ├── __init__.py
│   │   │   ├── loader.py                # Model loading/caching
│   │   │   └── registry.py              # Model version registry
│   │   │
│   │   ├── services/                    # Shared services
│   │   │   ├── __init__.py
│   │   │   ├── storage.py               # File storage
│   │   │   ├── cache.py                 # Result caching
│   │   │   └── audit.py                 # Audit logging
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── audio.py                 # Audio utilities
│   │       └── exceptions.py            # Custom exceptions
│   │
│   ├── ml/                              # ML training code (separate from serving)
│   │   ├── training/
│   │   │   ├── train_classifier.py
│   │   │   ├── hyperparameter_search.py
│   │   │   └── cross_validation.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── fairness.py
│   │   │   └── calibration.py
│   │   └── data/
│   │       ├── loaders.py
│   │       ├── augmentation.py
│   │       └── preprocessing.py
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── data/
│   │       └── fixtures/
│   │
│   ├── models/                          # Trained model artifacts
│   │   ├── speech/
│   │   │   ├── classifier_v1.0.pt
│   │   │   ├── calibrator_v1.0.pkl
│   │   │   └── model_card.md
│   │   └── embeddings/
│   │       └── wavlm_finetuned.pt
│   │
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/                            # Next.js frontend
│   └── ...
│
├── docs/
│   ├── api/                             # API documentation
│   ├── clinical/                        # Clinical validation docs
│   ├── regulatory/                      # FDA/CE documentation
│   │   ├── 510k_submission/
│   │   ├── risk_analysis/
│   │   └── clinical_evidence/
│   └── research/
│       └── SPEECH_PIPELINE.md           # This document
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── deploy.py
│   └── monitor.py
│
└── data/                                # Data storage (gitignored)
    ├── raw/
    ├── processed/
    └── models/
```

## 9.2 API Design

### 9.2.1 Endpoints

```python
# Speech Analysis API

POST /api/v1/speech/analyze
    Request: multipart/form-data
        audio_file: File (required)
        patient_id: str (optional, for longitudinal)
        metadata: JSON (optional)
    Response: SpeechAnalysisResponse

GET /api/v1/speech/biomarkers/{session_id}
    Response: BiomarkerDetails

GET /api/v1/speech/history/{patient_id}
    Response: LongitudinalAnalysis

POST /api/v1/speech/batch
    Request: List of audio files
    Response: BatchAnalysisResult

# Model Management API

GET /api/v1/models/speech/version
    Response: ModelVersionInfo

POST /api/v1/models/speech/reload
    Response: ReloadStatus

# Monitoring API

GET /api/v1/health
    Response: HealthCheck

GET /api/v1/metrics
    Response: PrometheusMetrics

GET /api/v1/monitoring/drift
    Response: DriftReport
```

### 9.2.2 Response Schemas

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class BiomarkerStatus(str, Enum):
    NORMAL = "normal"
    BORDERLINE = "borderline"
    ABNORMAL = "abnormal"

class BiomarkerResult(BaseModel):
    value: float
    unit: str
    normal_range: tuple[float, float]
    status: BiomarkerStatus
    z_score: Optional[float] = None
    percentile: Optional[int] = None
    confidence_interval: Optional[tuple[float, float]] = None

class ConditionProbability(BaseModel):
    condition: str
    probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)

class RiskAssessment(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    condition_probabilities: List[ConditionProbability]
    calibrated: bool = True

class QualityMetrics(BaseModel):
    signal_quality: float = Field(ge=0, le=1)
    snr_db: float
    clipping_detected: bool
    duration_seconds: float
    processing_warnings: List[str] = []

class SpeechAnalysisResponse(BaseModel):
    success: bool
    session_id: str
    timestamp: datetime
    processing_time_ms: int
    
    risk_assessment: RiskAssessment
    biomarkers: Dict[str, BiomarkerResult]
    quality_metrics: QualityMetrics
    
    recommendations: List[str]
    clinical_notes: str
    
    # Longitudinal (if patient_id provided)
    longitudinal_comparison: Optional[dict] = None
    
    # Uncertainty
    model_version: str
    uncertainty_note: str = "Results should be interpreted by a qualified healthcare provider"
```

## 9.3 Model Serving Strategy

### 9.3.1 Architecture

```
                    Load Balancer (NGINX)
                           |
           +---------------+---------------+
           |               |               |
        API Pod 1       API Pod 2       API Pod N
           |               |               |
           +---------------+---------------+
                           |
                    Model Server (Triton/TorchServe)
                           |
           +---------------+---------------+
           |               |               |
        GPU 1           GPU 2           GPU N
        (Whisper)       (WavLM)         (Classifier)
```

### 9.3.2 Caching Strategy

```python
CACHING_CONFIG = {
    # Feature caching (Redis)
    "feature_cache": {
        "enabled": True,
        "ttl_seconds": 3600,  # 1 hour
        "key_pattern": "features:{audio_hash}"
    },
    
    # Embedding caching (same audio = same embeddings)
    "embedding_cache": {
        "enabled": True,
        "ttl_seconds": 86400,  # 24 hours
        "key_pattern": "embeddings:{audio_hash}:{model_version}"
    },
    
    # Model caching (in-memory)
    "model_cache": {
        "whisper": {"preload": True, "device": "cuda:0"},
        "wavlm": {"preload": True, "device": "cuda:1"},
        "classifier": {"preload": True, "device": "cuda:0"}
    }
}
```

## 9.4 Logging and Auditability

### 9.4.1 Audit Log Schema

```python
class AuditLog(BaseModel):
    """
    Immutable audit log for regulatory compliance.
    """
    # Identifiers
    log_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    patient_id: Optional[str]
    
    # Timing
    timestamp: datetime
    processing_start: datetime
    processing_end: datetime
    
    # Input
    input_hash: str  # SHA-256 of audio
    input_metadata: dict
    
    # Processing
    pipeline_version: str
    model_versions: Dict[str, str]
    feature_extraction_log: dict
    
    # Output
    risk_score: float
    risk_level: str
    condition_probabilities: dict
    biomarkers_summary: dict
    
    # Quality
    quality_metrics: dict
    warnings: List[str]
    errors: List[str]
    
    # Review
    requires_review: bool
    review_reason: Optional[str]
    reviewed_by: Optional[str]
    review_timestamp: Optional[datetime]
    review_override: Optional[dict]
```

### 9.4.2 Logging Configuration

```python
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(timestamp)s %(level)s %(name)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/speech_pipeline.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "json"
        },
        "audit": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/audit.log",
            "maxBytes": 104857600,  # 100MB
            "backupCount": 100,
            "formatter": "json"
        }
    },
    "loggers": {
        "speech_pipeline": {"level": "INFO", "handlers": ["console", "file"]},
        "audit": {"level": "INFO", "handlers": ["audit"]}
    }
}
```

## 9.5 Regulatory Readiness Checklist

### 9.5.1 FDA 510(k) Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Device classification determination | [ ] | Letter from FDA |
| Predicate device identification | [ ] | Comparison table |
| Substantial equivalence argument | [ ] | Technical comparison |
| Software documentation (IEC 62304) | [ ] | SRS, SDS, test docs |
| Risk management (ISO 14971) | [ ] | Risk analysis file |
| Clinical evidence | [ ] | Validation study |
| Labeling | [ ] | IFU, labels |
| Quality management (21 CFR 820) | [ ] | QMS documentation |

### 9.5.2 Technical Documentation

| Document | Description | Owner |
|----------|-------------|-------|
| Software Requirements Specification (SRS) | Functional requirements | Engineering |
| Software Design Specification (SDS) | Architecture, interfaces | Engineering |
| Software Test Plan | Verification strategy | QA |
| Software Test Report | Test results | QA |
| Traceability Matrix | Requirements -> tests | QA |
| Risk Analysis Report | FMEA, hazard analysis | Clinical/Engineering |
| Clinical Validation Protocol | Study design | Clinical |
| Clinical Validation Report | Study results | Clinical |
| Intended Use Statement | Legal indication | Regulatory |
| Instructions for Use (IFU) | User documentation | Technical Writing |

### 9.5.3 Quality System Requirements

```
Quality Management System (QMS) Elements
========================================

1. Design Controls (21 CFR 820.30)
   - Design input documentation
   - Design output documentation
   - Design review records
   - Design verification records
   - Design validation records
   - Design transfer procedures
   - Design change control

2. Document Control (21 CFR 820.40)
   - Document approval procedures
   - Version control (Git)
   - Change history tracking

3. Software Validation (21 CFR 820.70)
   - Unit testing (>80% coverage)
   - Integration testing
   - System testing
   - User acceptance testing
   - Regression testing

4. CAPA (21 CFR 820.100)
   - Issue tracking system
   - Root cause analysis procedures
   - Corrective action records

5. Complaints (21 CFR 820.198)
   - Customer feedback system
   - Adverse event reporting (MDR)
```

---

# SECTION 10: REFERENCES

## Clinical Literature

1. Tsanas A, et al. (2012). "Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests." IEEE TBME.

2. Konig A, et al. (2015). "Automatic speech analysis for the assessment of patients with predementia and Alzheimer's disease." Alzheimer's & Dementia.

3. Fraser KC, et al. (2016). "Linguistic features identify Alzheimer's disease in narrative speech." J Alzheimer's Disease.

4. Godino-Llorente JI, et al. (2017). "Acoustic analysis of voice in neurological diseases." J Voice.

5. Orozco-Arroyave JR, et al. (2016). "Automatic detection of Parkinson's disease in running speech spoken in three different languages." JASA.

6. Cummins N, et al. (2015). "A review of depression and suicide risk assessment using speech analysis." Speech Communication.

7. Maryn Y, et al. (2010). "The Acoustic Voice Quality Index." J Voice. (CPPS development)

## Technical References

8. Baevski A, et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS.

9. Hsu WN, et al. (2021). "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM TASLP.

10. Radford A, et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." ICML. (Whisper)

11. Eyben F, et al. (2016). "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)." IEEE TAC.

## Regulatory Guidance

12. FDA. (2021). "Clinical Decision Support Software: Guidance for Industry."

13. FDA. (2022). "Software as a Medical Device (SaMD): Clinical Evaluation."

14. IMDRF. (2017). "Software as a Medical Device: Possible Framework for Risk Categorization."

---

# Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 3.0.0 | 2026-01-19 | Research Team | Complete rewrite for research-grade pipeline |
| 2.0.0 | 2026-01-17 | Engineering | Initial architecture document |

**Review Status**: Draft - Pending Clinical Review

**Classification**: Research Document - Not for Clinical Use

---

*This document represents a comprehensive research blueprint for developing a medically valid speech analysis pipeline. Implementation should be conducted under appropriate clinical oversight and regulatory guidance.*
