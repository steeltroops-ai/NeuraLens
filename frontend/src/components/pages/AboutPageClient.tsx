'use client';

import { motion } from 'framer-motion';
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
  ExternalLink
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'inherit',
});

// Mermaid Component
const MermaidDiagram = ({ chart, id }: { chart: string; id: string }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      mermaid.render(id, chart).then((result) => {
        if (ref.current) {
          ref.current.innerHTML = result.svg;
        }
      });
    }
  }, [chart, id]);

  return <div ref={ref} className="overflow-x-auto p-4 flex justify-center mix-blend-lighten" />;
};

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

// Architecture Diagrams
const systemArchDiagram = `
graph TD
    subgraph Client ["Client Layer (Next.js 14)"]
        UI[User Interface]
        Auth[Clerk Auth]
        Store[Zustand State]
    end

    subgraph API ["API Layer (Server Actions/Route Handlers)"]
        Router[App Router]
        Val[Zod Validation]
        Rate[Rate Limiting]
    end

    subgraph Service ["Microservices / ML Pipeline"]
        Orch[Orchestrator]
        Vision[Vision Transformer]
        NLP[wav2vec/BERT]
        Tabular[XGBoost/TabNet]
    end

    subgraph Data ["Data Layer"]
        DB[(PostgreSQL)]
        Blob[(S3/Blob Storage)]
        Vector[(Pinecone/Vector DB)]
    end

    UI --> Auth
    UI --> Store
    UI -- JSON/FormData --> Router
    Router --> Val
    Val --> Rate
    Rate --> Orch
    Orch --> Vision
    Orch --> NLP
    Orch --> Tabular
    Orch --> DB
    Orch --> Blob
    Vision --> Vector
`;

const pipelineDiagram = `
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as API Gateway
    participant ML as ML Inference
    participant DB as Database

    U->>FE: Upload Diagnostic Data (Image/Audio)
    FE->>FE: Client-side Validation & Compression
    FE->>API: Secure Upload (Signed URL)
    API->>DB: Log Request Meta
    API->>ML: Dispatch Inference Job (Async)
    activate ML
    ML->>ML: Preprocessing (Norm/Resize)
    ML->>ML: Model Inference (Ensemble)
    ML->>ML: Post-processing (Heatmap Gen)
    ML-->>API: Return Results + Confidence
    deactivate ML
    API->>DB: Store Results (Encrypted)
    API-->>FE: Return Clinical Report JSON
    FE->>U: Render Visualization & Report
`;

function HeroSection() {
  return (
    <section className="relative pt-32 pb-20 -mt-14 overflow-hidden bg-black selection:bg-white/20">
      {/* Background Effects */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-blue-500/[0.04] blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:32px_32px]" />

      <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 rounded-full bg-blue-900/20 border border-blue-800/50 px-3 py-1 text-[11px] font-medium text-blue-300 mb-8"
        >
          <Terminal className="h-3 w-3" />
          <span>TECHNICAL WHITE PAPER</span>
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl sm:text-5xl lg:text-7xl font-semibold tracking-tight text-white mb-6"
        >
          Engineering the <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-violet-500">
            Future of Diagnostics
          </span>
        </motion.h1>

        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mx-auto max-w-2xl text-slate-400 text-lg leading-relaxed mb-10"
        >
          A deep dive into the architecture, machine learning pipelines, and cloud infrastructure powering MediLens.
          Designed for developers, researchers, and technical stakeholders.
        </motion.p>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  const [activeTab, setActiveTab] = useState<'system' | 'pipeline'>('system');

  return (
    <section className="bg-zinc-950 py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2 className="text-3xl font-semibold text-white mb-4">System Architecture</h2>
          <p className="text-zinc-400 max-w-2xl">
            MediLens adopts a modern, modular architecture separating the client-side presentation
            from high-performance inference services.
          </p>
        </div>

        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl overflow-hidden">
          <div className="flex border-b border-zinc-800">
            <button
              onClick={() => setActiveTab('system')}
              className={`px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'system' ? 'text-white border-b-2 border-blue-500 bg-zinc-900' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              System Overview
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-6 py-4 text-sm font-medium transition-colors ${
                activeTab === 'pipeline' ? 'text-white border-b-2 border-violet-500 bg-zinc-900' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Inference Pipeline
            </button>
          </div>
          
          <div className="p-8 bg-black/50 min-h-[500px] flex items-center justify-center">
            {activeTab === 'system' ? (
              <div className="w-full">
                <MermaidDiagram chart={systemArchDiagram} id="system-diagram" />
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                  <div className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
                    <div className="flex items-center gap-2 mb-2 text-blue-400 font-medium font-mono text-sm">
                      <Layers className="h-4 w-4" /> Client
                    </div>
                    <p className="text-xs text-zinc-500">Next.js 14 App Router with React Server Components for optimal FCP and SEO.</p>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
                    <div className="flex items-center gap-2 mb-2 text-violet-400 font-medium font-mono text-sm">
                      <Server className="h-4 w-4" /> API
                    </div>
                    <p className="text-xs text-zinc-500">Route Handlers act as a gateway, providing rate limiting, validation, and auth guards.</p>
                  </div>
                  <div className="p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
                    <div className="flex items-center gap-2 mb-2 text-emerald-400 font-medium font-mono text-sm">
                      <Brain className="h-4 w-4" /> Inference
                    </div>
                    <p className="text-xs text-zinc-500">Python-based microservices running PyTorch/TensorFlow models on GPU instances.</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="w-full">
                 <MermaidDiagram chart={pipelineDiagram} id="pipeline-diagram" />
                 <div className="mt-8 p-4 rounded-lg bg-zinc-900/50 border border-zinc-800 max-w-3xl mx-auto">
                    <h4 className="text-white text-sm font-medium mb-2">Key Pipeline Features</h4>
                    <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-zinc-400">
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/>Asynchronous Inference Dispatch</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/>End-to-End Encryption (At rest & transit)</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/>Automatic Image Standardization</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-green-500 rounded-full"/>Ensemble Voting for High Accuracy</li>
                    </ul>
                 </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function TechStackSection() {
  const stack = [
    {
      domain: "Frontend",
      tools: ["Next.js 14", "React 18", "TypeScript", "Tailwind CSS", "Framer Motion", "Lucide React", "Clerk Auth"],
      icon: Code,
      color: "text-blue-400",
      bg: "bg-blue-900/10"
    },
    {
      domain: "AI / ML Core",
      tools: ["PyTorch Lightning", "TensorFlow 2.0", "Hugging Face Transformers", "OpenCV", "Scikit-learn"],
      icon: Brain,
      color: "text-violet-400",
      bg: "bg-violet-900/10"
    },
    {
      domain: "Backend & Infra",
      tools: ["FastAPI", "PostgreSQL", "Redis (Queue)", "Docker", "Kubernetes", "AWS S3"],
      icon: Server,
      color: "text-amber-400",
      bg: "bg-amber-900/10"
    },
    {
      domain: "DevOps & QA",
      tools: ["GitHub Actions", "Jest / Vitest", "Playwright", "ESLint", "Prettier"],
      icon: GitBranch,
      color: "text-pink-400",
      bg: "bg-pink-900/10"
    }
  ];

  return (
    <section className="bg-black py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2 className="text-3xl font-semibold text-white mb-4">Technology Stack</h2>
          <p className="text-zinc-400">Built on industry-standard, scalable open-source technologies.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stack.map((item, i) => (
            <motion.div
              key={item.domain}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              viewport={{ once: true }}
              className="p-6 rounded-xl border border-zinc-800 bg-zinc-900/20 hover:bg-zinc-900/40 transition-colors"
            >
              <div className={`w-12 h-12 rounded-lg ${item.bg} flex items-center justify-center mb-4`}>
                <item.icon className={`h-6 w-6 ${item.color}`} />
              </div>
              <h3 className="text-lg font-medium text-white mb-4">{item.domain}</h3>
              <ul className="space-y-2">
                {item.tools.map(tool => (
                  <li key={tool} className="text-sm text-zinc-400 flex items-center gap-2">
                    <span className="w-1 h-1 rounded-full bg-zinc-600" />
                    {tool}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function RoadmapSection() {
  const phases = [
    {
      id: "Phase 1",
      title: "Foundation",
      status: "completed",
      items: ["Platform Infrastructure", "Auth & RBAC", "Secure File Pipeline"]
    },
    {
      id: "Phase 2",
      title: "Initial Modules",
      status: "active",
      items: ["RetinaScan AI (Ophthalmology)", "ChestXplorer AI (Radiology)", "SkinSense AI (Dermatology)"]
    },
    {
      id: "Phase 3",
      title: "Expansion",
      status: "planned",
      items: ["Cardiology Module", "Pathology Module", "Multi-modal Fusion"]
    },
    {
      id: "Phase 4",
      title: "Advanced Features",
      status: "planned",
      items: ["EMR/EHR Integration", "Telemedicine Suite", "Population Analytics"]
    }
  ];

  return (
    <section className="bg-zinc-950 py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        <div className="mb-16">
          <h2 className="text-3xl font-semibold text-white mb-4">Implementation Roadmap</h2>
          <p className="text-zinc-400">Strategic execution plan from concept to full clinical ecosystem.</p>
        </div>

        <div className="relative">
          {/* Connecting Line */}
          <div className="absolute top-8 left-0 w-full h-0.5 bg-zinc-800 hidden lg:block" />
          
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {phases.map((phase, i) => (
              <motion.div
                key={phase.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="relative z-10"
              >
                {/* Dot */}
                <div className={`w-4 h-4 rounded-full border-2 mb-4 hidden lg:block ${
                  phase.status === 'completed' ? 'bg-green-500 border-green-900' :
                  phase.status === 'active' ? 'bg-blue-500 border-blue-900 animate-pulse' :
                  'bg-zinc-950 border-zinc-700'
                }`} />

                <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-900/30">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs font-mono text-zinc-500">{phase.id}</span>
                    {phase.status === 'completed' && <span className="text-[10px] bg-green-900/20 text-green-400 px-2 py-0.5 rounded border border-green-900/30">DONE</span>}
                    {phase.status === 'active' && <span className="text-[10px] bg-blue-900/20 text-blue-400 px-2 py-0.5 rounded border border-blue-900/30">IN PROGRESS</span>}
                  </div>
                  <h3 className="text-lg font-medium text-white mb-4">{phase.title}</h3>
                  <ul className="space-y-3">
                    {phase.items.map(item => (
                      <li key={item} className="text-sm text-zinc-400 flex items-start gap-2">
                        <ChevronRight className="h-4 w-4 text-zinc-600 shrink-0 mt-0.5" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>

  );
}

function DocLinksSection() {
  return (
    <section className="bg-black py-20 border-t border-zinc-900">
      <div className="mx-auto max-w-4xl px-4 text-center">
        <h2 className="text-2xl font-semibold text-white mb-6">Further Reading</h2>
        <div className="flex flex-wrap justify-center gap-4">
            {/* Note: In a real app these might link to internal doc pages or external Wikis. 
                For now we assume there's a docs section or repo link. */}
            <a href="#" className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-white hover:bg-zinc-800 transition-all">
                <FileCode className="h-4 w-4" />
                <span>Concept Document</span>
                <ExternalLink className="h-3 w-3 opacity-50" />
            </a>
            <a href="#" className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-white hover:bg-zinc-800 transition-all">
                <Box className="h-4 w-4" />
                <span>API Reference</span>
                <ExternalLink className="h-3 w-3 opacity-50" />
            </a>
        </div>
      </div>
    </section>
  );
}

export default function AboutPageClient() {
  return (
    <>
      <HeroSection />
      <ArchitectureSection />
      <TechStackSection />
      <RoadmapSection />
      <DocLinksSection />
    </>
  );
}
