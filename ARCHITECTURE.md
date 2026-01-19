# MediLens Architecture

> Clinical-grade AI diagnostic platform architecture documentation

---

## 1. System Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Browser["Web Browser"]
        Mobile["Mobile Device"]
    end
    
    subgraph Frontend["Frontend - Vercel"]
        Next["Next.js 16 App"]
        UI["React Components"]
        API_Routes["API Routes"]
    end
    
    subgraph Backend["Backend - HuggingFace"]
        FastAPI["FastAPI Server"]
        Pipelines["AI Pipelines"]
        Models["ML Models"]
    end
    
    subgraph External["External Services"]
        Cerebras["Cerebras Cloud"]
        AWS["AWS Polly TTS"]
        Clerk["Clerk Auth"]
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
    subgraph Dev["Development"]
        Code["Source Code"]
        Git["Git Repository"]
    end
    
    subgraph CICD["CI/CD"]
        GH["GitHub Actions"]
        Build["Build & Test"]
    end
    
    subgraph Prod["Production"]
        Vercel["Vercel Frontend"]
        HF["HuggingFace Backend"]
    end
    
    Code --> Git
    Git --> GH
    GH --> Build
    Build --> Vercel
    Build --> HF
    Vercel -.-> HF
```

**Deployment Flow:**
- Push to `main` triggers Vercel & HuggingFace builds
- Frontend: Next.js SSR on Vercel Edge
- Backend: Docker container on HuggingFace Spaces (port 7860)

---

## 3. Frontend Architecture

```mermaid
graph TB
    subgraph AppRouter["Next.js App Router"]
        Layout["Root Layout"]
        Dashboard["Dashboard Layout"]
        Pages["Page Components"]
    end
    
    subgraph DashPages["Dashboard Pages"]
        Retinal["Retinal Page"]
        Speech["Speech Page"]
        Cardio["Cardiology Page"]
        Radio["Radiology Page"]
        Derm["Dermatology Page"]
        Motor["Motor Page"]
        Cognitive["Cognitive Page"]
        Multi["MultiModal Page"]
        NRI["NRI Fusion Page"]
    end
    
    subgraph Components["Shared Components"]
        Sidebar["DashboardSidebar"]
        Header["DashboardHeader"]
        StatusBar["PipelineStatusBar"]
        Chatbot["MedicalChatbot"]
        Explain["ExplanationPanel"]
    end
    
    subgraph Routes["API Routes"]
        Health["Health Check"]
        Analyze["Analyze Endpoints"]
        ExplainAPI["Explain API"]
        Voice["Voice API"]
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
    subgraph FastAPIApp["FastAPI Application"]
        Main["main.py Entry"]
        Router["API Router"]
    end
    
    subgraph PipelineLayer["Pipeline Layer"]
        RetinalP["Retinal Pipeline"]
        SpeechP["Speech Pipeline"]
        CardioP["Cardiology Pipeline"]
        RadioP["Radiology Pipeline"]
        ExplainP["Explain Pipeline"]
        VoiceP["Voice Pipeline"]
    end
    
    subgraph CoreServices["Core Services"]
        Orchestrator["Pipeline Orchestrator"]
        Service["Core Service"]
        Validator["Input Validator"]
    end
    
    subgraph MLLayer["ML Layer"]
        Preprocessing["Preprocessing"]
        Features["Feature Extraction"]
        Analysis["Analysis Engine"]
        Clinical["Clinical Assessment"]
    end
    
    subgraph OutputLayer["Output Layer"]
        Formatter["Response Formatter"]
        Visualization["Visualization"]
        Report["Report Generator"]
    end
    
    Main --> Router
    Router --> RetinalP
    Router --> SpeechP
    Router --> CardioP
    Router --> RadioP
    Router --> ExplainP
    Router --> VoiceP
    
    RetinalP --> Orchestrator
    SpeechP --> Orchestrator
    CardioP --> Orchestrator
    
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
    subgraph RouterLayer["Router Layer"]
        RouterFile["router.py"]
    end
    
    subgraph CoreLayer["Core Layer"]
        OrchestratorFile["orchestrator.py"]
        ServiceFile["service.py"]
    end
    
    subgraph InputLayer["Input Layer"]
        ValidatorFile["validator.py"]
        ParserFile["parser.py"]
    end
    
    subgraph PreprocessLayer["Preprocessing Layer"]
        Normalizer["normalizer.py"]
        QualityGate["quality_gate.py"]
    end
    
    subgraph FeaturesLayer["Features Layer"]
        Extractor["extractor.py"]
        Biomarkers["biomarkers.py"]
    end
    
    subgraph AnalysisLayer["Analysis Layer"]
        Analyzer["analyzer.py"]
        Classifier["classifier.py"]
    end
    
    subgraph ClinicalLayer["Clinical Layer"]
        RiskScorer["risk_scorer.py"]
        Recommendations["recommendations.py"]
    end
    
    subgraph ExplanationLayer["Explanation Layer"]
        Rules["rules.py"]
        Templates["templates.py"]
    end
    
    subgraph OutputLayerP["Output Layer"]
        FormatterFile["formatter.py"]
        VisualizationFile["visualization.py"]
    end
    
    RouterFile --> OrchestratorFile
    OrchestratorFile --> ValidatorFile
    ValidatorFile --> Normalizer
    Normalizer --> Extractor
    Extractor --> Analyzer
    Analyzer --> RiskScorer
    RiskScorer --> Rules
    Rules --> FormatterFile
```

---

## 6. Retinal Pipeline

```mermaid
graph LR
    subgraph Input1["Input"]
        Upload["Fundus Image"]
    end
    
    subgraph Preprocess1["Preprocessing"]
        Resize["Resize 512x512"]
        Normalize["Normalize RGB"]
        Enhance["CLAHE Enhancement"]
    end
    
    subgraph Analysis1["Analysis"]
        Segment["Vessel Segmentation"]
        Detect["Lesion Detection"]
        Grade["DR Grading 0-4"]
    end
    
    subgraph Biomarkers1["Biomarkers"]
        VD["Vessel Density"]
        Hemorrhage["Hemorrhages"]
        Exudates["Exudates"]
        MA["Microaneurysms"]
    end
    
    subgraph Output1["Output"]
        Risk["Risk Score 0-100"]
        Heatmap["Attention Heatmap"]
        ReportR["Clinical Report"]
    end
    
    Upload --> Resize --> Normalize --> Enhance
    Enhance --> Segment --> Detect --> Grade
    Grade --> VD
    Grade --> Hemorrhage
    Grade --> Exudates
    Grade --> MA
    VD --> Risk
    Hemorrhage --> Risk
    Exudates --> Risk
    MA --> Risk
    Risk --> Heatmap --> ReportR
```

---

## 7. Speech Pipeline

```mermaid
graph LR
    subgraph Input2["Input"]
        Audio["Audio File"]
    end
    
    subgraph Preprocess2["Preprocessing"]
        Convert["Convert to WAV"]
        Denoise["Noise Reduction"]
        VAD["Voice Detection"]
    end
    
    subgraph Features2["Feature Extraction"]
        F0["Fundamental Frequency"]
        Jitter["Jitter"]
        Shimmer["Shimmer"]
        HNR["Harmonics Noise"]
        Formants["Formants"]
        MFCC["MFCCs"]
    end
    
    subgraph Analysis2["Analysis"]
        Biomarker["Biomarker Scoring"]
        ClinicalS["Clinical Assessment"]
        RiskS["Risk Classification"]
    end
    
    subgraph Output2["Output"]
        Score["Risk Score 0-100"]
        Radar["Radar Chart"]
        Trends["Trend Analysis"]
    end
    
    Audio --> Convert --> Denoise --> VAD
    VAD --> F0
    VAD --> Jitter
    VAD --> Shimmer
    VAD --> HNR
    VAD --> Formants
    VAD --> MFCC
    F0 --> Biomarker
    Jitter --> Biomarker
    Shimmer --> Biomarker
    HNR --> Biomarker
    Biomarker --> ClinicalS --> RiskS --> Score --> Radar --> Trends
```

---

## 8. Cardiology Pipeline

```mermaid
graph LR
    subgraph Input3["Input"]
        ECG["ECG Signal"]
    end
    
    subgraph Preprocess3["Preprocessing"]
        Filter["Bandpass Filter"]
        Baseline["Baseline Correction"]
        QRS["QRS Detection"]
    end
    
    subgraph Features3["Features"]
        Intervals["PR QT QRS Intervals"]
        HRV["Heart Rate Variability"]
        Morphology["Wave Morphology"]
    end
    
    subgraph Analysis3["Analysis"]
        Rhythm["Rhythm Classification"]
        Arrhythmia["Arrhythmia Detection"]
        Ischemia["Ischemia Detection"]
    end
    
    subgraph Output3["Output"]
        RiskC["Cardiac Risk Score"]
        ECGPlot["ECG Visualization"]
        ReportC["Clinical Report"]
    end
    
    ECG --> Filter --> Baseline --> QRS
    QRS --> Intervals
    QRS --> HRV
    QRS --> Morphology
    Intervals --> Rhythm
    HRV --> Rhythm
    Morphology --> Rhythm
    Rhythm --> Arrhythmia --> Ischemia
    Ischemia --> RiskC --> ECGPlot --> ReportC
```

---

## 9. AI Explanation Pipeline

```mermaid
graph TB
    subgraph Input4["Input"]
        Results["Pipeline Results"]
        Context["Patient Context"]
    end
    
    subgraph RuleLoading["Rule Loading"]
        RuleLoader["rule_loader.py"]
        PipelineRules["Pipeline Rules"]
    end
    
    subgraph PromptBuild["Prompt Building"]
        PromptBuilder["prompt_builder.py"]
        SystemPrompt["System Prompt"]
        UserPrompt["User Prompt"]
    end
    
    subgraph LLMGen["LLM Generation"]
        CerebrasLLM["Cerebras Cloud"]
        Llama["Llama 3.3 70B"]
        Stream["Streaming Response"]
    end
    
    subgraph VoiceSynth["Voice Synthesis"]
        Polly["AWS Polly"]
        AudioOut["MP3 Audio"]
    end
    
    subgraph Output4["Output"]
        Text["Explanation Text"]
        VoiceOut["Voice Audio"]
    end
    
    Results --> RuleLoader
    Context --> RuleLoader
    RuleLoader --> PipelineRules --> PromptBuilder
    PromptBuilder --> SystemPrompt
    PromptBuilder --> UserPrompt
    SystemPrompt --> CerebrasLLM
    UserPrompt --> CerebrasLLM
    CerebrasLLM --> Llama --> Stream
    Stream --> Text
    Text --> Polly --> AudioOut --> VoiceOut
```

---

## 10. Data Flow (End-to-End)

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as API Routes
    participant B as Backend
    participant P as Pipeline
    participant L as LLM
    participant V as Voice

    U->>F: Upload Medical Image
    F->>A: POST analyze
    A->>B: Forward to FastAPI
    B->>P: Process Pipeline
    P->>P: Preprocess Features Analysis
    P-->>B: Results Biomarkers
    B-->>A: JSON Response
    A-->>F: Display Results
    
    F->>A: POST explain
    A->>B: Request Explanation
    B->>L: Stream LLM Request
    L-->>B: Streaming Text
    B-->>A: SSE Stream
    A-->>F: Display Explanation
    
    F->>A: POST voice
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
    subgraph Auth["Authentication"]
        ClerkAuth["Clerk Auth"]
        JWT["JWT Tokens"]
        Session["Session Mgmt"]
    end
    
    subgraph APISec["API Security"]
        CORS["CORS Policy"]
        RateLimit["Rate Limiting"]
        Validation["Input Validation"]
    end
    
    subgraph DataSec["Data Security"]
        HTTPS["HTTPS TLS"]
        NoStore["No PHI Storage"]
        Ephemeral["Ephemeral Processing"]
    end
    
    subgraph Compliance["Compliance"]
        HIPAA["HIPAA Guidelines"]
        Privacy["Privacy First"]
        Audit["Audit Logging"]
    end
    
    ClerkAuth --> JWT --> Session
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
