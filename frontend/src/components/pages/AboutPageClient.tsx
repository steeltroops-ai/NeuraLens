"use client";

import { motion } from "framer-motion";
import {
  Code,
  Database,
  Cpu,
  Layers,
  GitBranch,
  Terminal,
  Shield,
  Zap,
  Lock,
  Server,
  Workflow,
  Share2,
  FileCode,
  Box,
  Brain,
  ChevronRight,
  ExternalLink,
  CheckCircle2,
  ArrowRight,
  Microscope,
  Activity,
  Eye,
  Mic,
  Heart,
  Stethoscope,
  Award,
  Users,
  TrendingUp,
  BarChart3,
  Globe,
  Sparkles,
  Target,
  Rocket,
  Loader2,
} from "lucide-react";
import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";

// Mermaid is now dynamically imported to reduce initial bundle (~340KB savings)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MermaidAPI = any;

// Mermaid Component with lazy loading
const MermaidDiagram = ({ chart, id }: { chart: string; id: string }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const loadAndRender = async () => {
      try {
        // Dynamic import of mermaid
        const mermaid: MermaidAPI = await import("mermaid").then(
          (m) => m.default,
        );

        // Initialize mermaid (only once)
        mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "loose",
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", system-ui, sans-serif',
          themeVariables: {
            primaryColor: "#3B82F6",
            primaryTextColor: "#fff",
            primaryBorderColor: "#2563EB",
            lineColor: "#52525b",
            secondaryColor: "#18181b",
            tertiaryColor: "#27272a",
          },
        });

        if (!isMounted || !ref.current) return;

        const result = await mermaid.render(id, chart);

        if (isMounted && ref.current) {
          ref.current.innerHTML = result.svg;
          setIsLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setError("Failed to load diagram");
          setIsLoading(false);
        }
      }
    };

    loadAndRender();

    return () => {
      isMounted = false;
    };
  }, [chart, id]);

  if (error) {
    return (
      <div className="flex items-center justify-center p-6 text-zinc-500 text-sm">
        {error}
      </div>
    );
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/50">
          <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        </div>
      )}
      <div ref={ref} className="overflow-x-auto p-6 flex justify-center" />
    </div>
  );
};

// Animation variants following UI/UX guidelines
const fadeInUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.08,
    },
  },
};

const scaleIn = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] },
};

// Enhanced Architecture Diagrams
const systemArchDiagram = `
graph TB
    subgraph Client["🖥️ Client Layer - Next.js 15"]
        UI[React Server Components]
        Auth[Clerk Authentication]
        State[Zustand State Management]
    end

    subgraph API["⚡ API Gateway Layer"]
        Router[App Router + Route Handlers]
        Val[Zod Schema Validation]
        Rate[Rate Limiting & Throttling]
        Cache[Redis Cache Layer]
    end

    subgraph ML["🧠 ML Inference Services"]
        Orch[Orchestration Engine]
        Vision[Vision Models<br/>EfficientNet, ResNet, ViT]
        Audio[Audio Models<br/>wav2vec 2.0, HuBERT]
        Tabular[Structured Data<br/>XGBoost, TabNet]
        Ensemble[Ensemble Voting]
    end

    subgraph Data["💾 Data & Storage Layer"]
        DB[(PostgreSQL<br/>Primary Database)]
        Blob[(S3 Blob Storage<br/>Medical Images)]
        Vector[(Vector DB<br/>Embeddings)]
        Queue[(Redis Queue<br/>Async Jobs)]
    end

    UI --> Auth
    UI --> State
    UI --> Router
    Router --> Val
    Val --> Rate
    Rate --> Cache
    Cache --> Orch
    Orch --> Vision
    Orch --> Audio
    Orch --> Tabular
    Vision --> Ensemble
    Audio --> Ensemble
    Tabular --> Ensemble
    Ensemble --> DB
    Orch --> Blob
    Orch --> Queue
    Vision --> Vector

    style Client fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff
    style API fill:#581c87,stroke:#a855f7,stroke-width:2px,color:#fff
    style ML fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style Data fill:#7c2d12,stroke:#f59e0b,stroke-width:2px,color:#fff
`;

const pipelineDiagram = `
sequenceDiagram
    autonumber
    participant User as 👤 Healthcare Provider
    participant FE as 🖥️ Frontend (Next.js)
    participant API as ⚡ API Gateway
    participant ML as 🧠 ML Inference Engine
    participant DB as 💾 PostgreSQL
    participant S3 as 📦 S3 Storage

    User->>FE: Upload Medical Data<br/>(Image/Audio/DICOM)
    FE->>FE: Client Validation<br/>Format & Size Check
    FE->>FE: Image Compression<br/>& Preprocessing
    FE->>API: POST /api/v1/analyze<br/>(Multipart FormData)
    API->>API: Authentication Check<br/>(JWT Validation)
    API->>API: Rate Limit Check<br/>(Redis)
    API->>S3: Upload to Secure Bucket<br/>(Encrypted at Rest)
    API->>DB: Log Request Metadata<br/>(User, Timestamp, Type)
    API->>ML: Dispatch Async Job<br/>(Queue: Redis)
    
    activate ML
    ML->>S3: Fetch Medical Data
    ML->>ML: Preprocessing Pipeline<br/>(Normalization, Resize)
    ML->>ML: Model Inference<br/>(Ensemble: 3-5 Models)
    ML->>ML: Post-processing<br/>(Heatmap, Annotations)
    ML->>ML: Confidence Scoring<br/>(Calibration)
    ML-->>API: Return Results<br/>(JSON + Confidence)
    deactivate ML
    
    API->>DB: Store Analysis Results<br/>(Encrypted)
    API->>DB: Update Request Status<br/>(Completed)
    API-->>FE: Stream Results<br/>(Server-Sent Events)
    FE->>FE: Render Visualization<br/>(Charts, Heatmaps)
    FE->>User: Display Clinical Report<br/>(PDF Export Available)

    Note over User,S3: ⏱️ Total Pipeline: <500ms (excluding ML inference)
    Note over ML: 🎯 ML Inference: 200-800ms depending on modality
`;

// Hero Section - Compact and focused
function HeroSection() {
  return (
    <section className="relative -mt-14 pt-14 overflow-hidden bg-black selection:bg-white/20">
      {/* Spotlight Effect */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-blue-500/[0.03] blur-[100px] rounded-full pointer-events-none" />

      <div className="relative z-10 mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-16 sm:py-20 lg:py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          {/* Badge */}
          <div className="inline-flex items-center gap-2 rounded-full bg-blue-900/20 border border-blue-800/50 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-blue-300 mb-6">
            <Terminal className="h-3 w-3" />
            <span>Technical Overview</span>
          </div>

          {/* Headline */}
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold leading-tight tracking-tight text-white mb-4">
            AI-Powered Medical Diagnostics
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-400">
              Platform Architecture
            </span>
          </h1>

          {/* Description */}
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] leading-relaxed text-zinc-400 mb-8">
            Technical overview of MediLens' architecture, ML pipelines, and
            infrastructure. Built for developers and technical stakeholders.
          </p>

          {/* Stats - Compact inline */}
          <div className="flex flex-wrap items-center justify-center gap-6 mb-8">
            {[
              { value: "11", label: "Modules" },
              { value: "<200ms", label: "Latency" },
              { value: "95%+", label: "Accuracy" },
              { value: "99.9%", label: "Uptime" },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl font-semibold text-white">
                  {stat.value}
                </div>
                <div className="text-[11px] text-zinc-500 uppercase tracking-wider">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 rounded-full bg-white px-6 py-2.5 text-[13px] font-medium text-black hover:bg-zinc-200 transition-colors"
            >
              Try Platform
              <ArrowRight className="h-4 w-4" />
            </Link>
            <a
              href="#architecture"
              className="inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-white/5 px-6 py-2.5 text-[13px] font-medium text-white hover:bg-white/10 transition-colors"
            >
              View Architecture
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Platform Overview Section - All 11 Modules
function PlatformOverviewSection() {
  const modules = [
    {
      id: "retinal",
      name: "RetinaScan AI",
      specialty: "Ophthalmology",
      icon: Eye,
      color: "text-cyan-400",
      bg: "bg-cyan-900/20",
      border: "border-cyan-800/30",
      diagnoses: ["Diabetic Retinopathy", "Glaucoma", "AMD", "Cataracts"],
      models: "EfficientNet, ResNet, Vision Transformers",
      datasets: "APTOS 2019, MESSIDOR, IDRiD",
      status: "available",
    },
    {
      id: "radiology",
      name: "ChestXplorer AI",
      specialty: "Radiology",
      icon: Activity,
      color: "text-blue-400",
      bg: "bg-blue-900/20",
      border: "border-blue-800/30",
      diagnoses: ["Pneumonia", "COVID-19", "TB", "Lung Cancer"],
      models: "DenseNet-121, ResNet-50, YOLO",
      datasets: "NIH ChestX-ray14, CheXpert",
      status: "available",
    },
    {
      id: "dermatology",
      name: "SkinSense AI",
      specialty: "Dermatology",
      icon: Sparkles,
      color: "text-rose-400",
      bg: "bg-rose-900/20",
      border: "border-rose-800/30",
      diagnoses: ["Melanoma", "Basal Cell Carcinoma", "Eczema", "Psoriasis"],
      models: "EfficientNet, Inception-v3, MobileNet",
      datasets: "ISIC (50K+ images), HAM10000",
      status: "available",
    },
    {
      id: "pathology",
      name: "HistoVision AI",
      specialty: "Pathology",
      icon: Microscope,
      color: "text-purple-400",
      bg: "bg-purple-900/20",
      border: "border-purple-800/30",
      diagnoses: ["Cancer Detection", "Malaria", "Leukemia", "Tumor Grading"],
      models: "U-Net, ResNet/VGG, YOLO",
      datasets: "PatchCamelyon, NIH Malaria",
      status: "coming-soon",
    },
    {
      id: "cardiology",
      name: "CardioPredict AI",
      specialty: "Cardiology",
      icon: Heart,
      color: "text-red-400",
      bg: "bg-red-900/20",
      border: "border-red-800/30",
      diagnoses: ["Arrhythmia", "AFib", "MI", "Heart Failure"],
      models: "1D CNN, LSTM/GRU, Transformers",
      datasets: "MIT-BIH, PTB Diagnostic ECG",
      status: "coming-soon",
    },
    {
      id: "neurology",
      name: "NeuroScan AI",
      specialty: "Neurology",
      icon: Brain,
      color: "text-indigo-400",
      bg: "bg-indigo-900/20",
      border: "border-indigo-800/30",
      diagnoses: ["Brain Tumors", "Alzheimer's", "Stroke", "MS Lesions"],
      models: "3D U-Net, ResNet-3D, V-Net",
      datasets: "BraTS, ADNI, OASIS",
      status: "coming-soon",
    },
    {
      id: "pulmonology",
      name: "RespiRate AI",
      specialty: "Pulmonology",
      icon: Stethoscope,
      color: "text-teal-400",
      bg: "bg-teal-900/20",
      border: "border-teal-800/30",
      diagnoses: ["COPD", "Asthma", "Sleep Apnea", "Cystic Fibrosis"],
      models: "wav2vec, LSTM, Computer Vision",
      datasets: "ICBHI Respiratory Sound DB",
      status: "coming-soon",
    },
    {
      id: "speech",
      name: "SpeechMD AI",
      specialty: "Speech & Cognitive",
      icon: Mic,
      color: "text-blue-400",
      bg: "bg-blue-900/20",
      border: "border-blue-800/30",
      diagnoses: ["Parkinson's", "Aphasia", "Dementia", "Autism"],
      models: "wav2vec 2.0, HuBERT, Whisper",
      datasets: "mPower, DementiaBank, AVEC",
      status: "available",
    },
    {
      id: "diabetic-foot",
      name: "FootCare AI",
      specialty: "Diabetic Foot Care",
      icon: Target,
      color: "text-orange-400",
      bg: "bg-orange-900/20",
      border: "border-orange-800/30",
      diagnoses: [
        "Diabetic Foot Ulcers",
        "Wound Classification",
        "Infection Risk",
      ],
      models: "Mask R-CNN, ResNet",
      datasets: "DFUC2020, Medetec Wound DB",
      status: "coming-soon",
    },
    {
      id: "orthopedics",
      name: "BoneScan AI",
      specialty: "Orthopedics",
      icon: Activity,
      color: "text-zinc-400",
      bg: "bg-zinc-800/50",
      border: "border-zinc-700/30",
      diagnoses: ["Bone Fractures", "Osteoporosis", "Arthritis", "Scoliosis"],
      models: "YOLO, ResNet",
      datasets: "MURA, RSNA Bone Age",
      status: "coming-soon",
    },
    {
      id: "omni",
      name: "OmniMed AI",
      specialty: "Multi-Modal Fusion",
      icon: Layers,
      color: "text-violet-400",
      bg: "bg-violet-900/20",
      border: "border-violet-800/30",
      diagnoses: ["Holistic Diagnosis", "Risk Scoring", "Treatment Pathways"],
      models: "CLIP-style, Multi-modal Transformers",
      datasets: "Multi-source EHR Data",
      status: "coming-soon",
    },
  ];

  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12 sm:mb-16"
        >
          <div className="flex items-center gap-2 mb-3 sm:mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-blue-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">
              Platform Overview
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-3 sm:mb-4 tracking-tight">
            11 Specialized AI Diagnostic Modules
          </h2>
          <p className="max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            Each module is built with specialty-specific AI models, trained on
            validated medical datasets, and designed for clinical-grade
            accuracy. From ophthalmology to multi-modal fusion.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6"
        >
          {modules.map((module, index) => (
            <motion.div
              key={module.id}
              variants={fadeInUp}
              className={`group relative rounded-xl bg-zinc-900 border ${module.border} p-6 hover:bg-zinc-900/80 transition-all duration-300`}
            >
              {/* Status Badge */}
              <div className="absolute top-4 right-4">
                {module.status === "available" ? (
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-green-900/20 px-2.5 py-1 text-[10px] font-medium text-green-400 tracking-wide border border-green-900/30">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                    LIVE
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded-full bg-zinc-800 px-2.5 py-1 text-[10px] font-medium text-zinc-500 tracking-wide border border-zinc-700">
                    SOON
                  </span>
                )}
              </div>

              {/* Icon */}
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-lg ${module.bg} mb-4`}
              >
                <module.icon
                  className={`h-6 w-6 ${module.color}`}
                  strokeWidth={1.5}
                />
              </div>

              {/* Content */}
              <h3 className="text-[15px] font-medium text-white mb-1">
                {module.name}
              </h3>
              <p className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider mb-4">
                {module.specialty}
              </p>

              {/* Diagnoses */}
              <div className="mb-4">
                <p className="text-[10px] font-medium text-zinc-600 uppercase tracking-wider mb-2">
                  Diagnoses
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {module.diagnoses.slice(0, 3).map((diagnosis, idx) => (
                    <span
                      key={idx}
                      className="inline-block rounded px-2 py-1 text-[10px] font-medium text-zinc-400 bg-zinc-800/50 border border-zinc-800"
                    >
                      {diagnosis}
                    </span>
                  ))}
                  {module.diagnoses.length > 3 && (
                    <span className="inline-block rounded px-2 py-1 text-[10px] font-medium text-zinc-500 bg-zinc-900 border border-zinc-800">
                      +{module.diagnoses.length - 3}
                    </span>
                  )}
                </div>
              </div>

              {/* Technical Details */}
              <div className="pt-4 border-t border-zinc-800/50 space-y-2">
                <div>
                  <p className="text-[10px] font-medium text-zinc-600 uppercase tracking-wider mb-1">
                    Models
                  </p>
                  <p className="text-[11px] text-zinc-500 font-mono">
                    {module.models}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-medium text-zinc-600 uppercase tracking-wider mb-1">
                    Datasets
                  </p>
                  <p className="text-[11px] text-zinc-500 font-mono">
                    {module.datasets}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// Architecture Section with Interactive Diagrams
function ArchitectureSection() {
  const [activeTab, setActiveTab] = useState<"system" | "pipeline">("system");

  return (
    <section
      id="architecture"
      className="bg-black py-16 sm:py-20 lg:py-24 border-t border-zinc-900"
    >
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12 sm:mb-16"
        >
          <div className="flex items-center gap-2 mb-3 sm:mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-violet-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">
              System Architecture
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-3 sm:mb-4 tracking-tight">
            Modern, Scalable Architecture
          </h2>
          <p className="max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            MediLens adopts a modular, microservices-based architecture that
            separates client-side presentation from high-performance ML
            inference services, ensuring scalability and maintainability.
          </p>
        </motion.div>

        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl overflow-hidden">
          {/* Tab Navigation */}
          <div className="flex border-b border-zinc-800">
            <button
              onClick={() => setActiveTab("system")}
              className={`flex-1 px-4 sm:px-6 py-3 sm:py-4 text-[13px] sm:text-[14px] font-medium transition-all duration-200 ${
                activeTab === "system"
                  ? "text-white bg-zinc-900 border-b-2 border-blue-500"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50"
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                <Layers className="h-4 w-4" />
                System Overview
              </span>
            </button>
            <button
              onClick={() => setActiveTab("pipeline")}
              className={`flex-1 px-4 sm:px-6 py-3 sm:py-4 text-[13px] sm:text-[14px] font-medium transition-all duration-200 ${
                activeTab === "pipeline"
                  ? "text-white bg-zinc-900 border-b-2 border-violet-500"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50"
              }`}
            >
              <span className="flex items-center justify-center gap-2">
                <Workflow className="h-4 w-4" />
                Inference Pipeline
              </span>
            </button>
          </div>

          {/* Diagram Content */}
          <div className="p-4 sm:p-8 bg-black/50 min-h-[500px]">
            {activeTab === "system" ? (
              <div className="w-full">
                <MermaidDiagram chart={systemArchDiagram} id="system-diagram" />

                {/* Architecture Highlights */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 mt-8">
                  <div className="p-4 sm:p-5 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors">
                    <div className="flex items-center gap-2 mb-3 text-blue-400 font-medium font-mono text-[13px]">
                      <Layers className="h-4 w-4" /> Client Layer
                    </div>
                    <p className="text-[12px] text-zinc-400 leading-relaxed">
                      Next.js 15 App Router with React Server Components for
                      optimal First Contentful Paint (FCP) and SEO. Clerk
                      handles authentication with JWT tokens.
                    </p>
                  </div>
                  <div className="p-4 sm:p-5 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors">
                    <div className="flex items-center gap-2 mb-3 text-violet-400 font-medium font-mono text-[13px]">
                      <Server className="h-4 w-4" /> API Gateway
                    </div>
                    <p className="text-[12px] text-zinc-400 leading-relaxed">
                      Route Handlers provide rate limiting, Zod schema
                      validation, and authentication guards. Redis caching layer
                      reduces database load.
                    </p>
                  </div>
                  <div className="p-4 sm:p-5 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:border-zinc-700 transition-colors">
                    <div className="flex items-center gap-2 mb-3 text-emerald-400 font-medium font-mono text-[13px]">
                      <Brain className="h-4 w-4" /> ML Services
                    </div>
                    <p className="text-[12px] text-zinc-400 leading-relaxed">
                      Python-based microservices running PyTorch/TensorFlow
                      models on GPU instances. Ensemble voting for improved
                      accuracy.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="w-full">
                <MermaidDiagram chart={pipelineDiagram} id="pipeline-diagram" />

                {/* Pipeline Features */}
                <div className="mt-8 p-4 sm:p-6 rounded-lg bg-zinc-900/50 border border-zinc-800 max-w-4xl mx-auto">
                  <h4 className="text-white text-[14px] font-semibold mb-4 flex items-center gap-2">
                    <Zap className="h-4 w-4 text-amber-400" />
                    Key Pipeline Features
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {[
                      "Asynchronous Inference Dispatch",
                      "End-to-End Encryption (At rest & transit)",
                      "Automatic Image Standardization",
                      "Ensemble Voting for High Accuracy",
                      "Real-time Progress Streaming (SSE)",
                      "Confidence Score Calibration",
                    ].map((feature, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 text-[12px] text-zinc-400"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                        {feature}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

// Technology Stack Section
function TechStackSection() {
  const stack = [
    {
      domain: "Frontend",
      tools: [
        "Next.js 15 (App Router)",
        "React 18 (Server Components)",
        "TypeScript (Strict Mode)",
        "Tailwind CSS",
        "Framer Motion",
        "Clerk Authentication",
        "Zustand State Management",
      ],
      icon: Code,
      color: "text-blue-400",
      bg: "bg-blue-900/10",
      border: "border-blue-800/30",
    },
    {
      domain: "AI / ML Core",
      tools: [
        "PyTorch Lightning",
        "TensorFlow 2.x",
        "Hugging Face Transformers",
        "OpenCV (Image Processing)",
        "Scikit-learn",
        "ONNX Runtime",
      ],
      icon: Brain,
      color: "text-violet-400",
      bg: "bg-violet-900/10",
      border: "border-violet-800/30",
    },
    {
      domain: "Backend & Infrastructure",
      tools: [
        "FastAPI (Python 3.10+)",
        "PostgreSQL (Async SQLAlchemy)",
        "Redis (Cache & Queue)",
        "Docker & Kubernetes",
        "AWS S3 (Blob Storage)",
        "Nginx (Load Balancer)",
      ],
      icon: Server,
      color: "text-amber-400",
      bg: "bg-amber-900/10",
      border: "border-amber-800/30",
    },
    {
      domain: "DevOps & Testing",
      tools: [
        "GitHub Actions (CI/CD)",
        "Vitest (Unit Tests)",
        "Playwright (E2E Tests)",
        "ESLint & Prettier",
        "Sentry (Error Tracking)",
        "Prometheus & Grafana",
      ],
      icon: GitBranch,
      color: "text-pink-400",
      bg: "bg-pink-900/10",
      border: "border-pink-800/30",
    },
  ];

  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12 sm:mb-16"
        >
          <div className="flex items-center gap-2 mb-3 sm:mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-amber-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">
              Technology Stack
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-3 sm:mb-4 tracking-tight">
            Built on Industry Standards
          </h2>
          <p className="max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            Leveraging battle-tested, open-source technologies for reliability,
            performance, and scalability.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          {stack.map((item, i) => (
            <motion.div
              key={item.domain}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              viewport={{ once: true }}
              className={`p-6 rounded-xl border ${item.border} bg-zinc-900/20 hover:bg-zinc-900/40 transition-all duration-300`}
            >
              <div
                className={`w-12 h-12 rounded-lg ${item.bg} flex items-center justify-center mb-4`}
              >
                <item.icon
                  className={`h-6 w-6 ${item.color}`}
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="text-[15px] font-semibold text-white mb-4">
                {item.domain}
              </h3>
              <ul className="space-y-2">
                {item.tools.map((tool) => (
                  <li
                    key={tool}
                    className="text-[12px] text-zinc-400 flex items-start gap-2"
                  >
                    <span className="w-1 h-1 rounded-full bg-zinc-600 mt-1.5 flex-shrink-0" />
                    <span>{tool}</span>
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-12 sm:mt-16 p-6 sm:p-8 rounded-2xl bg-gradient-to-br from-zinc-900 to-zinc-900/50 border border-zinc-800"
        >
          <h3 className="text-[16px] font-semibold text-white mb-6 flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            Performance Benchmarks
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { metric: "API Response", value: "<200ms", desc: "P95 latency" },
              {
                metric: "ML Inference",
                value: "200-800ms",
                desc: "Per modality",
              },
              { metric: "Uptime", value: "99.9%", desc: "SLA guarantee" },
              {
                metric: "Throughput",
                value: "100+",
                desc: "Concurrent requests",
              },
            ].map((item, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl sm:text-3xl font-semibold text-white mb-1">
                  {item.value}
                </div>
                <div className="text-[12px] font-medium text-zinc-400 mb-1">
                  {item.metric}
                </div>
                <div className="text-[10px] text-zinc-600">{item.desc}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Development Roadmap Section
function RoadmapSection() {
  const phases = [
    {
      number: "01",
      title: "Foundation",
      status: "completed",
      timeline: "Q1 2024",
      items: [
        "Platform Infrastructure Setup",
        "User Authentication & RBAC",
        "Secure File Upload Pipeline",
        "Database Schema Design",
      ],
      icon: Rocket,
      position: "top",
    },
    {
      number: "02",
      title: "Initial Modules",
      status: "active",
      timeline: "Q2-Q3 2024",
      items: [
        "RetinaScan AI (Ophthalmology)",
        "ChestXplorer AI (Radiology)",
        "SkinSense AI (Dermatology)",
        "SpeechMD AI (Speech Analysis)",
      ],
      icon: Brain,
      position: "bottom",
    },
    {
      number: "03",
      title: "Expansion",
      status: "planned",
      timeline: "Q4 2024 - Q1 2025",
      items: [
        "CardioPredict AI (Cardiology)",
        "NeuroScan AI (Neurology)",
        "HistoVision AI (Pathology)",
        "Multi-modal Fusion Engine",
      ],
      icon: Layers,
      position: "top",
    },
    {
      number: "04",
      title: "Enterprise Features",
      status: "planned",
      timeline: "Q2 2025+",
      items: [
        "EMR/EHR Integration (Epic, Cerner)",
        "Telemedicine Suite",
        "Population Health Analytics",
        "FHIR API Compliance",
      ],
      icon: Globe,
      position: "bottom",
    },
  ];

  return (
    <section className="bg-black py-16 sm:py-20 lg:py-24 border-t border-zinc-900 overflow-hidden">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-16 text-center"
        >
          <div className="flex items-center justify-center gap-2 mb-3">
            <span className="flex h-1.5 w-1.5 rounded-full bg-green-500" />
            <span className="text-[10px] font-medium text-zinc-400 uppercase tracking-widest">
              Development Roadmap
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-3 tracking-tight">
            Strategic Implementation Plan
          </h2>
          <p className="max-w-2xl mx-auto text-[14px] text-zinc-400 leading-relaxed">
            From foundational infrastructure to full clinical ecosystem
            integration.
          </p>
        </motion.div>

        {/* Roadmap Cards - Desktop */}
        <div className="hidden lg:block">
          <div className="grid grid-cols-4 gap-6">
            {phases.map((phase, i) => (
              <motion.div
                key={phase.number}
                initial={{ opacity: 0, y: phase.position === "top" ? -20 : 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.15, duration: 0.5 }}
                viewport={{ once: true }}
                className={phase.position === "bottom" ? "mt-32" : ""}
              >
                <div className="relative p-5 rounded-xl border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900/70 hover:border-zinc-700 transition-all duration-300">
                  <div className="absolute -top-3 -left-3 w-9 h-9 rounded-lg bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 flex items-center justify-center">
                    <span className="text-[13px] font-bold text-zinc-400 font-mono">
                      {phase.number}
                    </span>
                  </div>
                  <div className="flex items-start justify-between mb-3 pt-1">
                    <div>
                      <h3 className="text-[15px] font-semibold text-white mb-1">
                        {phase.title}
                      </h3>
                      <p className="text-[11px] text-zinc-500 font-mono">
                        {phase.timeline}
                      </p>
                    </div>
                    <phase.icon
                      className="h-4 w-4 text-zinc-500 flex-shrink-0 ml-2"
                      strokeWidth={1.5}
                    />
                  </div>
                  <div className="mb-3">
                    {phase.status === "completed" && (
                      <span className="inline-flex items-center gap-1.5 text-[9px] bg-green-900/30 text-green-400 px-2.5 py-1 rounded-full border border-green-800/50 font-medium uppercase tracking-wider">
                        <CheckCircle2 className="h-2.5 w-2.5" strokeWidth={2} />
                        Completed
                      </span>
                    )}
                    {phase.status === "active" && (
                      <span className="inline-flex items-center gap-1.5 text-[9px] bg-blue-900/30 text-blue-400 px-2.5 py-1 rounded-full border border-blue-800/50 font-medium uppercase tracking-wider">
                        <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
                        In Progress
                      </span>
                    )}
                    {phase.status === "planned" && (
                      <span className="inline-flex items-center text-[9px] bg-zinc-800/50 text-zinc-500 px-2.5 py-1 rounded-full border border-zinc-700/50 font-medium uppercase tracking-wider">
                        Planned
                      </span>
                    )}
                  </div>
                  <ul className="space-y-2">
                    {phase.items.map((item, idx) => (
                      <li
                        key={idx}
                        className="text-[12px] text-zinc-400 flex items-start gap-2 leading-relaxed"
                      >
                        <ChevronRight
                          className="h-3.5 w-3.5 text-zinc-600 flex-shrink-0 mt-0.5"
                          strokeWidth={2}
                        />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mobile/Tablet - Vertical Timeline */}
        <div className="lg:hidden relative max-w-2xl mx-auto">
          {/* Vertical Line */}
          <div className="absolute left-[19px] top-0 bottom-0 w-[2px] bg-gradient-to-b from-green-500 via-blue-500 to-zinc-700" />

          <div className="space-y-8">
            {phases.map((phase, i) => (
              <motion.div
                key={phase.number}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                viewport={{ once: true }}
                className="relative flex gap-6"
              >
                {/* Status Dot */}
                <div className="relative flex-shrink-0">
                  <div
                    className={`w-10 h-10 rounded-full border-4 border-black flex items-center justify-center ${
                      phase.status === "completed"
                        ? "bg-green-500"
                        : phase.status === "active"
                          ? "bg-blue-500"
                          : "bg-zinc-700"
                    }`}
                  >
                    {phase.status === "completed" && (
                      <CheckCircle2
                        className="h-5 w-5 text-white"
                        strokeWidth={2.5}
                      />
                    )}
                    {phase.status === "active" && (
                      <div className="h-2 w-2 rounded-full bg-white animate-pulse" />
                    )}
                    {phase.status === "planned" && (
                      <div className="h-2 w-2 rounded-full bg-zinc-400" />
                    )}
                  </div>
                </div>

                {/* Content Card */}
                <div className="flex-1 pb-8">
                  <div className="relative p-5 rounded-xl border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900/70 hover:border-zinc-700 transition-all duration-300">
                    {/* Phase Number Badge */}
                    <div className="absolute -top-3 -left-3 w-9 h-9 rounded-lg bg-gradient-to-br from-zinc-800 to-zinc-900 border border-zinc-700 flex items-center justify-center">
                      <span className="text-[13px] font-bold text-zinc-400 font-mono">
                        {phase.number}
                      </span>
                    </div>

                    {/* Header */}
                    <div className="flex items-start justify-between mb-3 pt-1">
                      <div>
                        <h3 className="text-[15px] font-semibold text-white mb-1">
                          {phase.title}
                        </h3>
                        <p className="text-[11px] text-zinc-500 font-mono">
                          {phase.timeline}
                        </p>
                      </div>
                      <phase.icon
                        className="h-4 w-4 text-zinc-500 flex-shrink-0 ml-2"
                        strokeWidth={1.5}
                      />
                    </div>

                    {/* Status Badge */}
                    <div className="mb-3">
                      {phase.status === "completed" && (
                        <span className="inline-flex items-center gap-1.5 text-[9px] bg-green-900/30 text-green-400 px-2.5 py-1 rounded-full border border-green-800/50 font-medium uppercase tracking-wider">
                          <CheckCircle2
                            className="h-2.5 w-2.5"
                            strokeWidth={2}
                          />
                          Completed
                        </span>
                      )}
                      {phase.status === "active" && (
                        <span className="inline-flex items-center gap-1.5 text-[9px] bg-blue-900/30 text-blue-400 px-2.5 py-1 rounded-full border border-blue-800/50 font-medium uppercase tracking-wider">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
                          In Progress
                        </span>
                      )}
                      {phase.status === "planned" && (
                        <span className="inline-flex items-center text-[9px] bg-zinc-800/50 text-zinc-500 px-2.5 py-1 rounded-full border border-zinc-700/50 font-medium uppercase tracking-wider">
                          Planned
                        </span>
                      )}
                    </div>

                    {/* Items */}
                    <ul className="space-y-2">
                      {phase.items.map((item, idx) => (
                        <li
                          key={idx}
                          className="text-[12px] text-zinc-400 flex items-start gap-2 leading-relaxed"
                        >
                          <ChevronRight
                            className="h-3.5 w-3.5 text-zinc-600 flex-shrink-0 mt-0.5"
                            strokeWidth={2}
                          />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// Security & Compliance Section
function SecuritySection() {
  const features = [
    {
      icon: Lock,
      title: "HIPAA Compliant",
      description:
        "Full healthcare data protection with end-to-end encryption, audit logging, and access controls.",
      color: "text-blue-400",
      bg: "bg-blue-900/10",
    },
    {
      icon: Shield,
      title: "Data Encryption",
      description:
        "AES-256 encryption at rest, TLS 1.3 in transit. All medical data stored in encrypted S3 buckets.",
      color: "text-green-400",
      bg: "bg-green-900/10",
    },
    {
      icon: Users,
      title: "Role-Based Access",
      description:
        "Granular RBAC with JWT authentication. Separate permissions for patients, providers, and admins.",
      color: "text-violet-400",
      bg: "bg-violet-900/10",
    },
    {
      icon: FileCode,
      title: "Audit Trails",
      description:
        "Complete audit logging of all diagnostic actions, data access, and system events for compliance.",
      color: "text-amber-400",
      bg: "bg-amber-900/10",
    },
  ];

  return (
    <section className="bg-zinc-950 py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12 sm:mb-16 text-center"
        >
          <div className="flex items-center justify-center gap-2 mb-3 sm:mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-red-500" />
            <span className="text-[10px] sm:text-[11px] font-medium text-zinc-400 uppercase tracking-widest">
              Security & Compliance
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-3 sm:mb-4 tracking-tight">
            Enterprise-Grade Security
          </h2>
          <p className="mx-auto max-w-2xl text-[14px] sm:text-[15px] text-zinc-400 leading-relaxed">
            Built with healthcare compliance at the core. Every layer of the
            stack implements industry-standard security practices and regulatory
            requirements.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.5 }}
              viewport={{ once: true }}
              className="p-6 rounded-xl bg-zinc-900/30 border border-zinc-800 hover:border-zinc-700 transition-colors"
            >
              <div
                className={`w-12 h-12 rounded-lg ${feature.bg} flex items-center justify-center mb-4`}
              >
                <feature.icon
                  className={`h-6 w-6 ${feature.color}`}
                  strokeWidth={1.5}
                />
              </div>
              <h3 className="text-[14px] font-semibold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-[12px] text-zinc-400 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Compliance Badges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-12 sm:mt-16 flex flex-wrap items-center justify-center gap-6 sm:gap-8"
        >
          {[
            { label: "HIPAA Compliant", icon: Shield },
            { label: "SOC 2 Type II", icon: Award },
            { label: "GDPR Ready", icon: Globe },
            { label: "ISO 27001", icon: Lock },
          ].map((badge, i) => (
            <div key={i} className="flex items-center gap-2 text-zinc-500">
              <badge.icon className="h-4 w-4" strokeWidth={1.5} />
              <span className="text-[12px] font-medium">{badge.label}</span>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  return (
    <section className="bg-black py-16 sm:py-20 lg:py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-white mb-4 tracking-tight">
            Ready to Experience MediLens?
          </h2>
          <p className="text-[14px] sm:text-[15px] text-zinc-400 mb-8 sm:mb-10 max-w-2xl mx-auto">
            Explore our AI-powered diagnostic platform or get in touch to
            discuss enterprise deployment and integration options.
          </p>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 sm:gap-4">
            <Link
              href="/dashboard"
              className="group inline-flex items-center justify-center gap-2 rounded-full bg-white px-6 sm:px-8 py-3 sm:py-3.5 text-[13px] sm:text-[14px] font-medium text-black transition-all duration-200 hover:bg-zinc-200"
            >
              Start Diagnosing
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-zinc-800 bg-white/5 px-6 sm:px-8 py-3 sm:py-3.5 text-[13px] sm:text-[14px] font-medium text-white transition-all duration-200 hover:bg-white/10"
            >
              <Globe className="h-4 w-4 text-zinc-400" />
              View Homepage
            </Link>
          </div>

          {/* Additional Links */}
          <div className="mt-12 sm:mt-16 flex flex-wrap justify-center gap-6 sm:gap-8">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-[12px] text-zinc-500 hover:text-white transition-colors"
            >
              <GitBranch className="h-4 w-4" />
              <span>GitHub Repository</span>
              <ExternalLink className="h-3 w-3 opacity-50" />
            </a>
            <a
              href="#"
              className="inline-flex items-center gap-2 text-[12px] text-zinc-500 hover:text-white transition-colors"
            >
              <FileCode className="h-4 w-4" />
              <span>API Documentation</span>
              <ExternalLink className="h-3 w-3 opacity-50" />
            </a>
            <a
              href="#"
              className="inline-flex items-center gap-2 text-[12px] text-zinc-500 hover:text-white transition-colors"
            >
              <Box className="h-4 w-4" />
              <span>Technical Whitepaper</span>
              <ExternalLink className="h-3 w-3 opacity-50" />
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Main Export Component
export default function AboutPageClient() {
  return (
    <>
      <HeroSection />
      <PlatformOverviewSection />
      <ArchitectureSection />
      <TechStackSection />
      <RoadmapSection />
      <SecuritySection />
      <CTASection />
    </>
  );
}
