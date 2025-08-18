# NeuroLens-X Component Designs

## 🧩 **COMPONENT LIBRARY SPECIFICATIONS**

### **Design Principles**
- **Clinical Professionalism**: Healthcare-grade interface components
- **Accessibility First**: WCAG 2.1 AA+ compliance built-in
- **Consistency**: Unified design language across all components
- **Performance**: Optimized for fast rendering and smooth interactions

---

## 🎯 **CORE COMPONENTS**

### **1. NRI Score Display Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│              NEURO-RISK INDEX (NRI)                     │
│              ═══════════════════════                     │
│                                                         │
│                      78                                 │
│                     ────                                │
│                     100                                 │
│                                                         │
│    ████████████████████████████████████░░░░░░░░░░░░     │
│                                                         │
│                  HIGH RISK                              │
│               Confidence: ±8%                           │
│                                                         │
│    🔴 Specialist referral recommended                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface NRIScoreProps {
  score: number;           // 0-100
  confidence: number;      // ±percentage
  category: 'low' | 'moderate' | 'high' | 'critical';
  animated?: boolean;      // Animate score reveal
  showRecommendation?: boolean;
}

// Color mapping
const riskColors = {
  low: '#10B981',      // Green
  moderate: '#F59E0B', // Amber
  high: '#F97316',     // Orange
  critical: '#EF4444'  // Red
};
```

#### **Accessibility Features**
- ARIA live region for score announcements
- High contrast color ratios (7:1)
- Screen reader friendly descriptions
- Keyboard navigation support

---

### **2. Assessment Progress Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│ Step 2 of 4: Speech Analysis                           │
│ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                         │
│ ✓ Introduction    ⟳ Speech    ○ Retinal    ○ Risk     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface ProgressProps {
  currentStep: number;
  totalSteps: number;
  steps: Array<{
    id: string;
    label: string;
    status: 'completed' | 'current' | 'pending';
  }>;
}

// Status indicators
const statusIcons = {
  completed: '✓',
  current: '⟳',
  pending: '○'
};
```

---

### **3. Audio Recording Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                [🎤 Record Audio]                        │
│                                                         │
│       ●●●●●●●●●●●●●●●○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○     │
│                                                         │
│              00:45 / 01:30                              │
│                                                         │
│        [🔄 Re-record]  [▶️ Play]  [✓ Accept]           │
│                                                         │
│  🔊 Volume: ████████░░  📶 Quality: Good               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface AudioRecorderProps {
  maxDuration: number;     // seconds
  onRecordingComplete: (blob: Blob) => void;
  onError: (error: string) => void;
  showWaveform?: boolean;
  showQualityIndicator?: boolean;
}

// Recording states
type RecordingState = 'idle' | 'recording' | 'paused' | 'completed';
```

---

### **4. File Upload Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│              [📁 Upload Retinal Image]                  │
│                                                         │
│            Drag & drop or click to select               │
│                                                         │
│         Supported: JPG, PNG, TIFF (Max 10MB)           │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                                                 │   │
│  │              [📷 Use Demo Image]                │   │
│  │                                                 │   │
│  │           Try with sample data                  │   │
│  │                                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface FileUploadProps {
  acceptedTypes: string[];
  maxSize: number;         // bytes
  onFileSelect: (file: File) => void;
  onError: (error: string) => void;
  showDemoOption?: boolean;
  demoFiles?: Array<{ name: string; url: string; }>;
}
```

---

### **5. Processing Status Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│               ANALYZING YOUR DATA                       │
│               ══════════════════                       │
│                                                         │
│  🎤 Speech Analysis        ✓ Complete (2.3s)          │
│  👁️ Retinal Analysis       ⟳ Processing... 67%        │
│  📊 Risk Calculation      ⏳ Pending                   │
│  🧠 NRI Fusion            ⏳ Pending                   │
│                                                         │
│         Estimated time remaining: 45 seconds            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 🔒 Processing locally for maximum privacy       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface ProcessingStatusProps {
  tasks: Array<{
    id: string;
    label: string;
    icon: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress?: number;      // 0-100
    duration?: number;      // seconds
  }>;
  estimatedTimeRemaining?: number;
}
```

---

### **6. Results Breakdown Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                DETAILED BREAKDOWN                       │
│                ══════════════════                       │
│                                                         │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│ │ 🎤 Speech   │ │ 👁️ Retinal  │ │ 📊 Risk     │       │
│ │             │ │             │ │             │       │
│ │   72/100    │ │   68/100    │ │   75/100    │       │
│ │             │ │             │ │             │       │
│ │ ████████░░  │ │ ███████░░░  │ │ ████████░░  │       │
│ │             │ │             │ │             │       │
│ │ Moderate    │ │ Moderate    │ │ High        │       │
│ │ Risk        │ │ Risk        │ │ Risk        │       │
│ │             │ │             │ │             │       │
│ │ Key Findings│ │ Key Findings│ │ Key Factors │       │
│ │ • Tremor    │ │ • Vessel    │ │ • Age: 65   │       │
│ │   detected  │ │   changes   │ │ • Family    │       │
│ │ • Pause     │ │ • Mild      │ │   history   │       │
│ │   patterns  │ │   tortuosity│ │ • Diabetes  │       │
│ │             │ │             │ │             │       │
│ │ [View Details] │ [View Details] │ [View Details] │   │
│ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface ResultsBreakdownProps {
  results: Array<{
    modality: 'speech' | 'retinal' | 'risk' | 'motor';
    score: number;
    category: 'low' | 'moderate' | 'high' | 'critical';
    findings: string[];
    confidence: number;
    processingTime: number;
  }>;
  onViewDetails: (modality: string) => void;
}
```

---

### **7. Clinical Recommendations Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                  RECOMMENDATIONS                        │
│                  ═══════════════                        │
│                                                         │
│  🏥 IMMEDIATE ACTIONS                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • Schedule neurological evaluation within 30    │   │
│  │   days with a qualified specialist              │   │
│  │                                                 │   │
│  │ • Bring this report to your appointment         │   │
│  │                                                 │   │
│  │ • Discuss family history and current symptoms   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  📅 FOLLOW-UP CARE                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • Repeat assessment in 6 months                 │   │
│  │                                                 │   │
│  │ • Monitor for new symptoms                      │   │
│  │                                                 │   │
│  │ • Maintain healthy lifestyle habits             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ⚠️ IMPORTANT DISCLAIMER                                │
│  This is a screening tool, not a diagnostic device.     │
│  Always consult healthcare professionals for medical    │
│  decisions and treatment plans.                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface RecommendationsProps {
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
  immediateActions: string[];
  followUpCare: string[];
  timeframe: string;
  showDisclaimer?: boolean;
}
```

---

### **8. Report Generation Component**

#### **Visual Design**
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│              CLINICAL REPORT READY                      │
│              ══════════════════════                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │                                                 │   │
│  │    📄 NeuroLens-X Assessment Report             │   │
│  │                                                 │   │
│  │    Patient: [Anonymized ID]                     │   │
│  │    Date: March 15, 2024                         │   │
│  │    NRI Score: 78/100 (High Risk)                │   │
│  │                                                 │   │
│  │    [📥 Download PDF]                            │   │
│  │                                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  SHARING OPTIONS                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 📧 Email to healthcare provider                 │   │
│  │ 🔗 Generate secure sharing link                 │   │
│  │ 📱 Save to health app                           │   │
│  │ 🖨️ Print for appointment                        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  🔒 Your data remains private and secure               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **Technical Specifications**
```typescript
interface ReportGenerationProps {
  assessmentData: AssessmentResults;
  patientId: string;
  onDownload: () => void;
  onShare: (method: 'email' | 'link' | 'health-app' | 'print') => void;
  showSharingOptions?: boolean;
}
```

---

## 🎨 **COMPONENT STYLING SYSTEM**

### **Button Variants**
```css
/* Primary Action Button */
.btn-primary {
  background: linear-gradient(135deg, #3B82F6, #2563EB);
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.25);
  transition: all 0.2s ease;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.35);
}

/* Secondary Button */
.btn-secondary {
  background: var(--surface-secondary);
  color: var(--text-primary);
  border: 1px solid var(--neutral-700);
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 500;
}

/* Icon Button */
.btn-icon {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface-tertiary);
  border: none;
  color: var(--text-secondary);
}
```

### **Card Variants**
```css
/* Standard Card */
.card {
  background: var(--surface-primary);
  border: 1px solid var(--neutral-800);
  border-radius: 1rem;
  padding: 1.5rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Glass Morphism Card */
.card-glass {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 1rem;
  padding: 1.5rem;
}

/* Results Card */
.card-results {
  background: var(--surface-primary);
  border: 2px solid var(--primary-500);
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 8px 25px -3px rgba(59, 130, 246, 0.2);
}
```

### **Progress Indicators**
```css
/* Linear Progress Bar */
.progress-bar {
  width: 100%;
  height: 0.5rem;
  background: var(--neutral-800);
  border-radius: 0.25rem;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10B981, #F59E0B, #EF4444);
  border-radius: 0.25rem;
  transition: width 0.5s ease;
}

/* Circular Progress */
.progress-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: conic-gradient(
    from 0deg,
    var(--primary-500) 0deg,
    var(--primary-500) calc(var(--progress) * 3.6deg),
    var(--neutral-800) calc(var(--progress) * 3.6deg),
    var(--neutral-800) 360deg
  );
}
```

---

## ♿ **ACCESSIBILITY IMPLEMENTATION**

### **ARIA Labels and Roles**
```typescript
// Component accessibility props
interface AccessibilityProps {
  'aria-label'?: string;
  'aria-describedby'?: string;
  'aria-live'?: 'polite' | 'assertive' | 'off';
  role?: string;
  tabIndex?: number;
}

// Example implementation
<div
  role="progressbar"
  aria-label="Assessment progress"
  aria-valuenow={currentStep}
  aria-valuemin={1}
  aria-valuemax={totalSteps}
  aria-describedby="progress-description"
>
  Step {currentStep} of {totalSteps}
</div>
```

### **Keyboard Navigation**
```css
/* Focus indicators */
.focus-visible {
  outline: 2px solid var(--primary-500);
  outline-offset: 2px;
  border-radius: 0.25rem;
}

/* Skip links */
.skip-link {
  position: absolute;
  top: -40px;
  left: 6px;
  background: var(--primary-500);
  color: white;
  padding: 8px;
  text-decoration: none;
  border-radius: 4px;
  z-index: 1000;
}

.skip-link:focus {
  top: 6px;
}
```

### **Screen Reader Support**
```typescript
// Screen reader announcements
const announceToScreenReader = (message: string) => {
  const announcement = document.createElement('div');
  announcement.setAttribute('aria-live', 'polite');
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;
  document.body.appendChild(announcement);
  
  setTimeout(() => {
    document.body.removeChild(announcement);
  }, 1000);
};
```

---

## 📱 **RESPONSIVE DESIGN PATTERNS**

### **Mobile-First Components**
```css
/* Mobile base styles */
.component {
  padding: 1rem;
  font-size: 1rem;
}

/* Tablet and up */
@media (min-width: 768px) {
  .component {
    padding: 1.5rem;
    font-size: 1.125rem;
  }
}

/* Desktop and up */
@media (min-width: 1024px) {
  .component {
    padding: 2rem;
    font-size: 1.25rem;
  }
}
```

### **Touch-Friendly Interactions**
```css
/* Minimum touch target size */
.touch-target {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Hover states for non-touch devices */
@media (hover: hover) {
  .interactive:hover {
    background-color: var(--surface-tertiary);
  }
}
```

---

*These component designs ensure clinical-grade professionalism with modern aesthetics, comprehensive accessibility, and optimal user experience across all devices.*
