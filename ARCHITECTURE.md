# MediLens Architecture

> Clinical-grade AI diagnostic platform architecture documentation

---

## 1. System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        Mobile[Mobile Device]
    end
    
    subgraph "Frontend - Vercel"
        Next[Next.js 16 App]
        UI[React Components]
        API_Routes[API Routes /api/*]
    end
    
    subgraph "Backend - HuggingFace Spaces"
        FastAPI[FastAPI Server]
        Pipelines[AI Pipelines]
        Models[ML Models]
    end
    
    subgraph "External Services"
        Cerebras[Cerebras Cloud<br/>Llama 3.3 70B]
        AWS[AWS Polly<br/>Text-to-Speech]
        Clerk[Clerk Auth]
    end
    
    Browser --> Next
    Mobile --> Next
    Next --> API_Routes
    API_Routes --> FastAPI
    FastAPI --> Pipelines
    Pipelines --> Models
    FastAPI --> Cerebras
    FastAPI --> AWS
    Next --> Clerk
```

---

## 2. Deployment Pipeline

```mermaid
graph LR
    subgraph "Development"
        Code[Source Code]
        Git[Git Repository]
    end
    
    subgraph "CI/CD"
        GH[GitHub Actions]
        Build[Build & Test]
    end
    
    subgraph "Production"
        Vercel[Vercel<br/>Frontend]
        HF[HuggingFace Spaces<br/>Backend Docker]
    end
    
    Code --> Git
    Git --> GH
    GH --> Build
    Build --> Vercel
    Build --> HF
    
    Vercel -.->|HTTPS| HF
```

**Deployment Flow:**
- Push to `main` triggers Vercel & HuggingFace builds
- Frontend: Next.js SSR on Vercel Edge
- Backend: Docker container on HuggingFace Spaces (port 7860)

---

## 3. Frontend Architecture

```mermaid
graph TB
    subgraph "Next.js App Router"
        Layout[Root Layout]
        Dashboard[Dashboard Layout]
        Pages[Page Components]
    end
    
    subgraph "Dashboard Pages"
        Retinal[/retinal]
        Speech[/speech]
        Cardio[/cardiology]
        Radio[/radiology]
        Derm[/dermatology]
        Motor[/motor]
        Cognitive[/cognitive]
        Multi[/multimodal]
        NRI[/nri-fusion]
    end
    
    subgraph "Shared Components"
        Sidebar[DashboardSidebar]
        Header[DashboardHeader]
        StatusBar[PipelineStatusBar]
        Chatbot[MedicalChatbot]
        Explain[ExplanationPanel]
    end
    
    subgraph "API Routes"
        Health[/api/health]
        Analyze[/api/*/analyze]
        ExplainAPI[/api/explain]
        Voice[/api/voice]
    end
    
    Layout --> Dashboard
    Dashboard --> Sidebar
    Dashboard --> Header
    Dashboard --> Pages
    Dashboard --> StatusBar
    Dashboard --> Chatbot
    
    Pages --> Retinal
    Pages --> Speech
    Pages --> Cardio
    Pages --> Radio
    Pages --> Derm
    Pages --> Motor
    Pages --> Cognitive
    Pages --> Multi
    Pages --> NRI
    
    Pages --> Explain
    Pages --> Analyze
    Explain --> ExplainAPI
    ExplainAPI --> Voice
```

**Frontend Stack:**
- Next.js 16 with App Router
- React 19, TypeScript
- Framer Motion animations
- Clerk authentication
- TailwindCSS styling

---

## 4. Backend Architecture

```mermaid
graph TB
    subgraph "FastAPI Application"
        Main[main.py<br/>CORS, Routes]
        Router[API Router]
    end
    
    subgraph "Pipeline Layer"
        Retinal[Retinal Pipeline]
        Speech[Speech Pipeline]
        Cardio[Cardiology Pipeline]
        Radio[Radiology Pipeline]
        Explain[Explain Pipeline]
        Voice[Voice Pipeline]
    end
    
    subgraph "Core Services"
        Orchestrator[Pipeline Orchestrator]
        Service[Core Service]
        Validator[Input Validator]
    end
    
    subgraph "ML Layer"
        Preprocessing[Preprocessing]
        Features[Feature Extraction]
        Analysis[Analysis Engine]
        Clinical[Clinical Assessment]
    end
    
    subgraph "Output Layer"
        Formatter[Response Formatter]
        Visualization[Visualization]
        Report[Report Generator]
    end
    
    Main --> Router
    Router --> Retinal
    Router --> Speech
    Router --> Cardio
    Router --> Radio
    Router --> Explain
    Router --> Voice
    
    Retinal --> Orchestrator
    Speech --> Orchestrator
    Cardio --> Orchestrator
    
    Orchestrator --> Validator
    Orchestrator --> Preprocessing
    Preprocessing --> Features
    Features --> Analysis
    Analysis --> Clinical
    Clinical --> Formatter
    Formatter --> Visualization
```

**Backend Stack:**
- FastAPI with async/await
- PyTorch, TensorFlow
- OpenCV, Parselmouth
- SQLAlchemy ORM

---

## 5. Pipeline Architecture (Standard)

Each pipeline follows this layered structure:

```mermaid
graph TB
    subgraph "Router Layer"
        Router[router.py<br/>/api/{pipeline}/*]
    end
    
    subgraph "Core Layer"
        Orchestrator[orchestrator.py]
        Service[service.py]
    end
    
    subgraph "Input Layer"
        Validator[validator.py]
        Parser[parser.py]
    end
    
    subgraph "Preprocessing Layer"
        Normalizer[normalizer.py]
        QualityGate[quality_gate.py]
    end
    
    subgraph "Features Layer"
        Extractor[extractor.py]
        Biomarkers[biomarkers.py]
    end
    
    subgraph "Analysis Layer"
        Analyzer[analyzer.py]
        Classifier[classifier.py]
    end
    
    subgraph "Clinical Layer"
        RiskScorer[risk_scorer.py]
        Recommendations[recommendations.py]
    end
    
    subgraph "Explanation Layer"
        Rules[rules.py]
        Templates[templates.py]
    end
    
    subgraph "Output Layer"
        Formatter[formatter.py]
        Visualization[visualization.py]
    end
    
    Router --> Orchestrator
    Orchestrator --> Validator
    Validator --> Normalizer
    Normalizer --> Extractor
    Extractor --> Analyzer
    Analyzer --> RiskScorer
    RiskScorer --> Rules
    Rules --> Formatter
```

---

## 6. Retinal Pipeline

```mermaid
graph LR
    subgraph "Input"
        Upload[Fundus Image<br/>JPEG/PNG]
    end
    
    subgraph "Preprocessing"
        Resize[Resize 512x512]
        Normalize[Normalize RGB]
        Enhance[CLAHE Enhancement]
    end
    
    subgraph "Analysis"
        Segment[Vessel Segmentation]
        Detect[Lesion Detection]
        Grade[DR Grading 0-4]
    end
    
    subgraph "Biomarkers"
        VD[Vessel Density]
        Hemorrhage[Hemorrhages]
        Exudates[Exudates]
        MA[Microaneurysms]
    end
    
    subgraph "Output"
        Risk[Risk Score 0-100]
        Heatmap[Attention Heatmap]
        Report[Clinical Report]
    end
    
    Upload --> Resize --> Normalize --> Enhance
    Enhance --> Segment --> Detect --> Grade
    Grade --> VD & Hemorrhage & Exudates & MA
    VD & Hemorrhage & Exudates & MA --> Risk --> Heatmap --> Report
```

---

## 7. Speech Pipeline

```mermaid
graph LR
    subgraph "Input"
        Audio[Audio File<br/>WAV/MP3/WebM]
    end
    
    subgraph "Preprocessing"
        Convert[Convert to WAV]
        Denoise[Noise Reduction]
        VAD[Voice Activity Detection]
    end
    
    subgraph "Feature Extraction"
        F0[Fundamental Frequency]
        Jitter[Jitter %]
        Shimmer[Shimmer %]
        HNR[Harmonics-to-Noise]
        Formants[Formants F1-F3]
        MFCC[MFCCs]
        CPPS[Cepstral Peak]
        Speech[Speech Rate]
        Pause[Pause Patterns]
    end
    
    subgraph "Analysis"
        Biomarker[Biomarker Scoring]
        Clinical[Clinical Assessment]
        Risk[Risk Classification]
    end
    
    subgraph "Output"
        Score[Risk Score 0-100]
        Radar[Radar Chart]
        Trends[Trend Analysis]
    end
    
    Audio --> Convert --> Denoise --> VAD
    VAD --> F0 & Jitter & Shimmer & HNR & Formants & MFCC & CPPS & Speech & Pause
    F0 & Jitter & Shimmer & HNR --> Biomarker
    Formants & MFCC & CPPS --> Biomarker
    Speech & Pause --> Biomarker
    Biomarker --> Clinical --> Risk --> Score --> Radar --> Trends
```

---

## 8. Cardiology Pipeline

```mermaid
graph LR
    subgraph "Input"
        ECG[ECG Signal<br/>CSV/EDF]
    end
    
    subgraph "Preprocessing"
        Filter[Bandpass Filter]
        Baseline[Baseline Correction]
        QRS[QRS Detection]
    end
    
    subgraph "Features"
        Intervals[PR/QT/QRS Intervals]
        HRV[Heart Rate Variability]
        Morphology[Wave Morphology]
    end
    
    subgraph "Analysis"
        Rhythm[Rhythm Classification]
        Arrhythmia[Arrhythmia Detection]
        Ischemia[Ischemia Detection]
    end
    
    subgraph "Output"
        Risk[Cardiac Risk Score]
        ECGPlot[ECG Visualization]
        Report[Clinical Report]
    end
    
    ECG --> Filter --> Baseline --> QRS
    QRS --> Intervals & HRV & Morphology
    Intervals & HRV & Morphology --> Rhythm --> Arrhythmia --> Ischemia
    Ischemia --> Risk --> ECGPlot --> Report
```

---

## 9. AI Explanation Pipeline

```mermaid
graph TB
    subgraph "Input"
        Results[Pipeline Results]
        Context[Patient Context]
    end
    
    subgraph "Rule Loading"
        RuleLoader[rule_loader.py]
        PipelineRules[Pipeline-Specific Rules]
    end
    
    subgraph "Prompt Building"
        PromptBuilder[prompt_builder.py]
        SystemPrompt[System Prompt]
        UserPrompt[User Prompt + Results]
    end
    
    subgraph "LLM Generation"
        Cerebras[Cerebras Cloud]
        Llama[Llama 3.3 70B]
        Stream[Streaming Response]
    end
    
    subgraph "Voice Synthesis"
        Polly[AWS Polly]
        Audio[MP3 Audio]
    end
    
    subgraph "Output"
        Text[Explanation Text]
        Voice[Voice Audio]
    end
    
    Results --> RuleLoader
    Context --> RuleLoader
    RuleLoader --> PipelineRules --> PromptBuilder
    PromptBuilder --> SystemPrompt & UserPrompt
    SystemPrompt & UserPrompt --> Cerebras --> Llama --> Stream
    Stream --> Text
    Text --> Polly --> Audio --> Voice
```

---

## 10. Data Flow (End-to-End)

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend (Vercel)
    participant A as API Routes
    participant B as Backend (HF)
    participant P as Pipeline
    participant L as LLM (Cerebras)
    participant V as Voice (Polly)
    
    U->>F: Upload Medical Image
    F->>A: POST /api/{pipeline}/analyze
    A->>B: Forward to FastAPI
    B->>P: Process Pipeline
    P->>P: Preprocess → Features → Analysis
    P-->>B: Results + Biomarkers
    B-->>A: JSON Response
    A-->>F: Display Results
    
    F->>A: POST /api/explain
    A->>B: Request Explanation
    B->>L: Stream LLM Request
    L-->>B: Streaming Text
    B-->>A: SSE Stream
    A-->>F: Display Explanation
    
    F->>A: POST /api/voice
    A->>B: TTS Request
    B->>V: Text to Speech
    V-->>B: MP3 Audio
    B-->>A: Base64 Audio
    A-->>F: Play Audio
```

---

## 11. Security Architecture

```mermaid
graph TB
    subgraph "Authentication"
        Clerk[Clerk Auth]
        JWT[JWT Tokens]
        Session[Session Management]
    end
    
    subgraph "API Security"
        CORS[CORS Policy]
        RateLimit[Rate Limiting]
        Validation[Input Validation]
    end
    
    subgraph "Data Security"
        HTTPS[HTTPS/TLS]
        NoStore[No PHI Storage]
        Ephemeral[Ephemeral Processing]
    end
    
    subgraph "Compliance"
        HIPAA[HIPAA Guidelines]
        Privacy[Privacy First]
        Audit[Audit Logging]
    end
    
    Clerk --> JWT --> Session
    CORS --> RateLimit --> Validation
    HTTPS --> NoStore --> Ephemeral
    HIPAA --> Privacy --> Audit
```

---

## 12. Folder Structure

```
NeuraLens/
├── frontend/                 # Next.js Application
│   ├── src/
│   │   ├── app/             # App Router Pages
│   │   │   ├── dashboard/   # Dashboard Pages
│   │   │   └── api/         # API Routes
│   │   ├── components/      # Shared Components
│   │   └── data/            # Static Data
│   └── public/              # Static Assets
│
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── main.py          # Entry Point
│   │   ├── routers/         # API Routers
│   │   └── pipelines/       # AI Pipelines
│   │       ├── retinal/     # Retinal Analysis
│   │       ├── speech/      # Speech Analysis
│   │       ├── cardiology/  # Cardiology Analysis
│   │       ├── radiology/   # Radiology Analysis
│   │       ├── explain/     # AI Explanations
│   │       └── voice/       # Voice Synthesis
│   └── Dockerfile           # Docker Config
│
└── docs/                     # Documentation
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Next.js App Router | Server components, streaming, edge runtime |
| FastAPI | Async support, auto-docs, type safety |
| Layered Pipelines | Separation of concerns, testability |
| Cerebras Llama 3.3 | Fast inference, medical knowledge |
| AWS Polly | Reliable TTS, natural voices |
| Docker on HF Spaces | Free hosting, GPU support |
| Vercel Edge | Global CDN, serverless |

---

*Architecture Version: 1.0 | Last Updated: January 2026*
