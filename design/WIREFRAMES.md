# NeuroLens-X Wireframes & User Flow

## 🎯 **USER JOURNEY OVERVIEW**

### **Primary User Flow**
1. **Landing** → Assessment Introduction
2. **Assessment** → Multi-Modal Data Collection
3. **Processing** → Real-Time Analysis Feedback
4. **Results** → NRI Score & Clinical Interpretation
5. **Report** → Professional PDF Generation
6. **Dashboard** → Historical Tracking (Future)

---

## 🏠 **LANDING PAGE WIREFRAME**

```
┌─────────────────────────────────────────────────────────────┐
│ [NeuroLens-X Logo]                    [Menu] [Get Started]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│           HERO SECTION                                      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │    Early Neurological Risk Detection               │   │
│   │    ═══════════════════════════════                 │   │
│   │                                                     │   │
│   │    Detect neurological decline 5-10 years          │   │
│   │    before symptoms appear through AI-powered       │   │
│   │    multi-modal assessment                           │   │
│   │                                                     │   │
│   │    [Start Assessment] [Learn More]                  │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│           FEATURES OVERVIEW                                 │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│   │ [🎤 Icon]   │ │ [👁️ Icon]   │ │ [📊 Icon]   │         │
│   │ Speech      │ │ Retinal     │ │ Risk        │         │
│   │ Analysis    │ │ Imaging     │ │ Assessment  │         │
│   │             │ │             │ │             │         │
│   │ Voice       │ │ Vascular    │ │ Personal    │         │
│   │ biomarkers  │ │ patterns    │ │ factors     │         │
│   └─────────────┘ └─────────────┘ └─────────────┘         │
│                                                             │
│           CLINICAL VALIDATION                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ ✓ 85% Sensitivity  ✓ 90% Specificity               │   │
│   │ ✓ WCAG 2.1 AA     ✓ HIPAA Compliant               │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 **ASSESSMENT FLOW WIREFRAMES**

### **Step 1: Assessment Introduction**
```
┌─────────────────────────────────────────────────────────────┐
│ [← Back]              NeuroLens-X              [Progress 1/4]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                NEUROLOGICAL RISK ASSESSMENT                 │
│                ═══════════════════════════════               │
│                                                             │
│   This assessment combines multiple modalities to           │
│   evaluate your neurological health risk:                   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 🎤 Speech Analysis        (~2 minutes)             │   │
│   │    Record a short speech sample                    │   │
│   │                                                     │   │
│   │ 👁️ Retinal Imaging       (~1 minute)              │   │
│   │    Upload fundus photograph                        │   │
│   │                                                     │   │
│   │ 📊 Risk Assessment       (~2 minutes)              │   │
│   │    Complete health questionnaire                   │   │
│   │                                                     │   │
│   │ 🧠 NRI Calculation       (~30 seconds)             │   │
│   │    Generate unified risk score                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ⚠️ Important: This is a screening tool, not a           │
│      diagnostic device. Consult healthcare providers       │
│      for medical decisions.                                │
│                                                             │
│                    [Begin Assessment]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Step 2: Speech Analysis**
```
┌─────────────────────────────────────────────────────────────┐
│ [← Back]              NeuroLens-X              [Progress 2/4]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    SPEECH ANALYSIS                          │
│                    ═══════════════                          │
│                                                             │
│   Please read the following passage aloud:                  │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │ "The quick brown fox jumps over the lazy dog.      │   │
│   │  Peter Piper picked a peck of pickled peppers.     │   │
│   │  She sells seashells by the seashore."             │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │              [🎤 Record Audio]                      │   │
│   │                                                     │   │
│   │     ●●●●●●●●●●○○○○○○○○○○  [00:45 / 01:30]           │   │
│   │                                                     │   │
│   │     [🔄 Re-record]  [▶️ Play]  [✓ Use Recording]    │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   💡 Tips:                                                 │
│   • Speak at normal pace and volume                        │
│   • Ensure quiet environment                               │
│   • Hold device 6-12 inches from mouth                     │
│                                                             │
│                      [Continue]                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Step 3: Retinal Imaging**
```
┌─────────────────────────────────────────────────────────────┐
│ [← Back]              NeuroLens-X              [Progress 3/4]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                   RETINAL IMAGING                           │
│                   ══════════════                           │
│                                                             │
│   Upload a fundus photograph (retinal image):               │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │              [📁 Upload Image]                      │   │
│   │                                                     │   │
│   │         Drag & drop or click to select              │   │
│   │                                                     │   │
│   │    Supported formats: JPG, PNG, TIFF               │   │
│   │    Maximum size: 10MB                               │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   OR                                                        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │              [📷 Use Demo Image]                    │   │
│   │                                                     │   │
│   │         Try with sample retinal image               │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ℹ️ Note: Fundus photographs are typically taken by       │
│      eye care professionals. Demo images available for     │
│      testing purposes.                                     │
│                                                             │
│                      [Continue]                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Step 4: Risk Assessment Questionnaire**
```
┌─────────────────────────────────────────────────────────────┐
│ [← Back]              NeuroLens-X              [Progress 4/4]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                   RISK ASSESSMENT                           │
│                   ═══════════════                           │
│                                                             │
│   Please complete the following health questionnaire:       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Demographics                                        │   │
│   │ ─────────────                                       │   │
│   │ Age: [____] years                                   │   │
│   │ Sex: ○ Male  ○ Female  ○ Other                      │   │
│   │ Ethnicity: [Dropdown ▼]                            │   │
│   │                                                     │   │
│   │ Medical History                                     │   │
│   │ ───────────────                                     │   │
│   │ ☐ Hypertension                                      │   │
│   │ ☐ Diabetes                                          │   │
│   │ ☐ High Cholesterol                                  │   │
│   │ ☐ Heart Disease                                     │   │
│   │ ☐ Stroke History                                    │   │
│   │                                                     │   │
│   │ Family History                                      │   │
│   │ ──────────────                                      │   │
│   │ ☐ Alzheimer's Disease                               │   │
│   │ ☐ Parkinson's Disease                               │   │
│   │ ☐ Dementia                                          │   │
│   │ ☐ Stroke                                            │   │
│   │                                                     │   │
│   │ Lifestyle Factors                                   │   │
│   │ ─────────────────                                   │   │
│   │ Exercise: ○ Regular  ○ Occasional  ○ Sedentary     │   │
│   │ Smoking: ○ Never  ○ Former  ○ Current              │   │
│   │ Alcohol: ○ None  ○ Moderate  ○ Heavy               │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│                   [Calculate Risk]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ **PROCESSING SCREEN**

```
┌─────────────────────────────────────────────────────────────┐
│                     NeuroLens-X                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                                                             │
│                ANALYZING YOUR DATA                          │
│                ══════════════════                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │    🎤 Speech Analysis        ✓ Complete            │   │
│   │       Processing voice patterns...                  │   │
│   │                                                     │   │
│   │    👁️ Retinal Analysis       ⟳ Processing...       │   │
│   │       Analyzing vascular patterns...               │   │
│   │       ████████████░░░░░░░░░░ 67%                   │   │
│   │                                                     │   │
│   │    📊 Risk Calculation      ⏳ Pending             │   │
│   │       Calculating baseline risk...                 │   │
│   │                                                     │   │
│   │    🧠 NRI Fusion            ⏳ Pending             │   │
│   │       Combining all modalities...                  │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│              Estimated time remaining: 45 seconds           │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ 🔒 Your data is processed locally in your browser  │   │
│   │    for maximum privacy and security                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 **RESULTS DISPLAY**

```
┌─────────────────────────────────────────────────────────────┐
│ [🏠 Home]             NeuroLens-X              [📄 Report]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                 ASSESSMENT RESULTS                          │
│                 ══════════════════                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │              NEURO-RISK INDEX                       │   │
│   │                                                     │   │
│   │                    78/100                           │   │
│   │                                                     │   │
│   │     ████████████████████████████████████░░░░░░░░    │   │
│   │                                                     │   │
│   │               HIGH RISK                             │   │
│   │          Confidence: ±8%                           │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   DETAILED BREAKDOWN                                        │
│   ──────────────────                                        │
│                                                             │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│   │ 🎤 Speech   │ │ 👁️ Retinal  │ │ 📊 Risk     │         │
│   │             │ │             │ │             │         │
│   │   72/100    │ │   68/100    │ │   75/100    │         │
│   │             │ │             │ │             │         │
│   │ Moderate    │ │ Moderate    │ │ High        │         │
│   │ Risk        │ │ Risk        │ │ Risk        │         │
│   │             │ │             │ │             │         │
│   │ • Tremor    │ │ • Vessel    │ │ • Age       │         │
│   │   detected  │ │   changes   │ │ • Family    │         │
│   │ • Pause     │ │ • Mild      │ │   history   │         │
│   │   patterns  │ │   tortuosity│ │ • Diabetes  │         │
│   └─────────────┘ └─────────────┘ └─────────────┘         │
│                                                             │
│   RECOMMENDATIONS                                           │
│   ───────────────                                           │
│   🏥 Specialist referral recommended within 30 days        │
│   📅 Schedule neurological evaluation                       │
│   🔄 Follow-up assessment in 6 months                       │
│                                                             │
│   [📄 Download Report] [📧 Share Results] [🔄 New Test]    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📱 **MOBILE RESPONSIVE DESIGN**

### **Mobile Assessment Flow**
```
┌─────────────────────┐
│ NeuroLens-X    [☰] │
├─────────────────────┤
│                     │
│   SPEECH ANALYSIS   │
│   ═══════════════   │
│                     │
│ Read this passage:  │
│                     │
│ ┌─────────────────┐ │
│ │ "The quick...   │ │
│ │  brown fox..."  │ │
│ └─────────────────┘ │
│                     │
│ ┌─────────────────┐ │
│ │   [🎤 Record]   │ │
│ │                 │ │
│ │ ●●●●●○○○○○      │ │
│ │ 00:45 / 01:30   │ │
│ │                 │ │
│ │ [🔄] [▶️] [✓]   │ │
│ └─────────────────┘ │
│                     │
│    [Continue]       │
│                     │
└─────────────────────┘
```

---

## ♿ **ACCESSIBILITY CONSIDERATIONS**

### **Screen Reader Support**
- All interactive elements have proper ARIA labels
- Form fields include descriptive labels and help text
- Progress indicators announce completion status
- Results include detailed text descriptions

### **Keyboard Navigation**
- Tab order follows logical flow
- All interactive elements are keyboard accessible
- Skip links for main content areas
- Focus indicators clearly visible

### **Visual Accessibility**
- High contrast color combinations (4.5:1 minimum)
- Text scaling up to 200% without horizontal scrolling
- Color is not the only means of conveying information
- Alternative text for all images and icons

### **Motor Accessibility**
- Large touch targets (44px minimum)
- Generous spacing between interactive elements
- No time-based interactions required
- Alternative input methods supported

---

*These wireframes ensure intuitive user flow with professional clinical aesthetics and comprehensive accessibility support.*
