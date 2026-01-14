'use client';

import { motion } from 'framer-motion';
import {
  Shield,
  Zap,
  ArrowRight,
  CheckCircle,
  Users,
  Award,
  Sparkles,
  FileText,
  Lock,
  Activity,
  Brain,
  Eye,
  Mic,
  Microscope,
  ChevronRight,
  Globe,
  Server,
  Play,
  BarChart3,
  Layers,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SignedIn, SignedOut } from '@clerk/nextjs';
import {
  getAvailableModules,
  getComingSoonModules,
  TOTAL_MODULES_COUNT,
  DiagnosticModule,
} from '@/data/diagnostic-modules';

// Smooth animation variants
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

// Color palette for modules - professional blue-based scheme
const moduleColors: Record<string, { gradient: string; bg: string; text: string; border: string }> = {
  speech: { gradient: 'from-blue-500 to-blue-600', bg: 'bg-blue-900/20', text: 'text-blue-400', border: 'group-hover:border-blue-500/30' },
  retinal: { gradient: 'from-cyan-500 to-cyan-600', bg: 'bg-cyan-900/20', text: 'text-cyan-400', border: 'group-hover:border-cyan-500/30' },
  motor: { gradient: 'from-violet-500 to-violet-600', bg: 'bg-violet-900/20', text: 'text-violet-400', border: 'group-hover:border-violet-500/30' },
  cognitive: { gradient: 'from-amber-500 to-amber-600', bg: 'bg-amber-900/20', text: 'text-amber-400', border: 'group-hover:border-amber-500/30' },
  radiology: { gradient: 'from-sky-500 to-sky-600', bg: 'bg-sky-900/20', text: 'text-sky-400', border: 'group-hover:border-sky-500/30' },
  dermatology: { gradient: 'from-rose-500 to-rose-600', bg: 'bg-rose-900/20', text: 'text-rose-400', border: 'group-hover:border-rose-500/30' },
  pathology: { gradient: 'from-purple-500 to-purple-600', bg: 'bg-purple-900/20', text: 'text-purple-400', border: 'group-hover:border-purple-500/30' },
  cardiology: { gradient: 'from-red-500 to-red-600', bg: 'bg-red-900/20', text: 'text-red-400', border: 'group-hover:border-red-500/30' },
  neurology: { gradient: 'from-indigo-500 to-indigo-600', bg: 'bg-indigo-900/20', text: 'text-indigo-400', border: 'group-hover:border-indigo-500/30' },
  pulmonology: { gradient: 'from-teal-500 to-teal-600', bg: 'bg-teal-900/20', text: 'text-teal-400', border: 'group-hover:border-teal-500/30' },
  'diabetic-foot': { gradient: 'from-orange-500 to-orange-600', bg: 'bg-orange-900/20', text: 'text-orange-400', border: 'group-hover:border-orange-500/30' },
  orthopedics: { gradient: 'from-zinc-500 to-zinc-600', bg: 'bg-zinc-800/50', text: 'text-zinc-300', border: 'group-hover:border-zinc-500/30' },
};

// Hero Section
function HeroSection() {
  const [currentFeature, setCurrentFeature] = useState(0);
  const features = [
    { name: 'RetinaScan AI', specialty: 'Ophthalmology', color: 'text-zinc-300' },
    { name: 'ChestXplorer AI', specialty: 'Radiology', color: 'text-zinc-300' },
    { name: 'SkinSense AI', specialty: 'Dermatology', color: 'text-red-400' },
    { name: 'CardioPredict AI', specialty: 'Cardiology', color: 'text-red-400' },
    { name: 'NeuroScan AI', specialty: 'Neurology', color: 'text-zinc-300' },
    { name: 'SpeechMD AI', specialty: 'Speech Analysis', color: 'text-zinc-300' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature(prev => (prev + 1) % features.length);
    }, 2800);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <section className="relative min-h-screen -mt-14 pt-14 flex items-center overflow-hidden bg-black selection:bg-white/20">
      {/* Spotlight Effect - Subtle top center glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-red-500/[0.04] blur-[120px] rounded-full pointer-events-none" />

      {/* Very faint grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />

      <div className="relative z-10 mx-auto max-w-6xl px-4 sm:px-6 py-24">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center gap-2.5 rounded-full bg-zinc-900/50 border border-zinc-800 px-3 py-1 text-[11px] font-medium text-zinc-300 mb-8"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-500 opacity-20"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
            </span>
            <span className="tracking-wide uppercase font-semibold text-[10px] text-zinc-400">{TOTAL_MODULES_COUNT} Diagnostic Modules</span>
          </motion.div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-semibold leading-[1.05] tracking-tight mb-8 text-white">
            AI-Powered Diagnostics
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-rose-600">
              for Modern Healthcare.
            </span>
          </h1>

          {/* Dynamic feature */}
          <div className="h-7 flex items-center justify-center mb-6">
            <motion.div
              key={currentFeature}
              initial={{ opacity: 0, filter: 'blur(4px)' }}
              animate={{ opacity: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, filter: 'blur(4px)' }}
              transition={{ duration: 0.4 }}
              className="flex items-center gap-3 text-[15px]"
            >
              <span className="font-medium text-white">
                {features[currentFeature]?.name}
              </span>
              <span className="h-4 w-[1px] bg-zinc-800" />
              <span className="text-zinc-500 font-mono tracking-tight text-[13px] uppercase">{features[currentFeature]?.specialty}</span>
            </motion.div>
          </div>

          {/* Description */}
          <p className="mx-auto max-w-2xl text-[15px] sm:text-[16px] leading-relaxed text-slate-400 mb-10">
            MediLens unifies AI diagnostics across ophthalmology, radiology, dermatology,
            cardiology, neurology, and pathology — enabling faster, more accurate clinical decisions
            with validated models and real-time processing.
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              href="/dashboard"
              className="group inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-full bg-white px-8 py-3.5 text-[14px] font-medium text-black transition-all duration-200 hover:bg-zinc-200"
            >
              Start Diagnosing
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/about"
              className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-full border border-zinc-800 bg-white/5 px-8 py-3.5 text-[14px] font-medium text-white transition-all duration-200 hover:bg-white/10"
            >
              <Play className="h-4 w-4 text-zinc-400" />
              Watch Demo
            </Link>
          </div>

          {/* Trust badges */}
          <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-4 opacity-60 grayscale hover:grayscale-0 transition-all duration-500">
            {[
              { icon: Shield, label: 'HIPAA Compliant' },
              { icon: Award, label: '95%+ Accuracy' },
              { icon: Zap, label: '< 200ms Latency' },
              { icon: Globe, label: 'Global Standard' },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-2">
                <item.icon className="h-4 w-4 text-zinc-400" />
                <span className="text-[12px] font-medium text-zinc-500">{item.label}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Bottom fade - smooth transition to next section */}
      {/* Bottom fade - smooth transition to next section */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent" />

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-zinc-500">
        <span className="text-[11px] font-medium tracking-wide">SCROLL</span>
        <div className="w-[1px] h-12 bg-gradient-to-b from-zinc-800 to-transparent">
          <motion.div
            className="w-full h-1/2 bg-white"
            animate={{ y: [0, 24, 0], opacity: [0, 1, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </div>
    </section>
  );
}

// Module Card
function DiagnosticModuleCard({ module }: { module: DiagnosticModule }) {
  const isAvailable = module.status === 'available';
  const colors = moduleColors[module.id] ?? { gradient: 'from-blue-500 to-indigo-600', bg: 'bg-blue-900/20', text: 'text-blue-400', border: 'group-hover:border-blue-500/30' };

  return (
    <motion.div
      variants={fadeInUp}
      whileHover={isAvailable ? { y: -6, transition: { duration: 0.25 } } : {}}
      className={`group relative rounded-xl bg-zinc-900 border overflow-hidden transition-all duration-300 ${isAvailable
        ? `cursor-pointer border-zinc-800 ${colors.border}`
        : 'border-zinc-900 opacity-40'
        }`}
    >
      {/* Top gradient bar - Subtle glow */}
      <div className={`h-[1px] w-full bg-gradient-to-r ${colors.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

      <div className="p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-5">
          <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${colors.bg} ${colors.text} transition-colors duration-300`}>
            <module.icon className="h-5 w-5" strokeWidth={1.5} />
          </div>
          {isAvailable ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-zinc-800 px-2.5 py-1 text-[10px] font-medium text-white tracking-wide border border-zinc-700 group-hover:border-zinc-600">
              <span className={`h-1.5 w-1.5 rounded-full animate-pulse ${colors.bg.replace('/20', '')}`} />
              LIVE
            </span>
          ) : (
            <span className="inline-flex items-center rounded-full bg-zinc-900 px-2.5 py-1 text-[10px] font-medium text-zinc-600 tracking-wide border border-zinc-800">
              SOON
            </span>
          )}
        </div>

        {/* Content */}
        <h3 className="text-[15px] font-medium text-white mb-2 group-hover:text-white transition-colors">{module.name}</h3>
        <p className="text-[13px] text-zinc-400 leading-relaxed mb-5">{module.description}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mb-4">
          {module.diagnoses.slice(0, 3).map((diagnosis, idx) => (
            <span
              key={idx}
              className="inline-block rounded px-2 py-1 text-[10px] font-medium text-zinc-400 bg-zinc-800/50 border border-zinc-800 group-hover:border-zinc-700 transition-colors"
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

        {/* Methods */}
        <div className="pt-4 border-t border-zinc-800/50 flex justify-between items-center bg-transparent group-hover:border-zinc-700/50 transition-colors">
             <p className="text-[11px] text-zinc-500 font-mono group-hover:text-zinc-400 transition-colors">{module.howItWorks}</p>
             {/* Hover arrow */}
            {isAvailable && (
              <ChevronRight className={`h-4 w-4 text-zinc-600 group-hover:text-white transition-colors transform group-hover:translate-x-1`} />
            )}
        </div>
      </div>

      {isAvailable && (
        <Link href={module.route} className="absolute inset-0 z-10" aria-label={`Go to ${module.name}`} />
      )}
    </motion.div>
  );
}

// Available Modules Section
function AvailableNowSection() {
  const availableModules = getAvailableModules();

  return (
    <section className="bg-black py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-widest">Available Now</span>
          </div>
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            Clinical-Grade Modules
          </h2>
          <p className="max-w-xl text-[15px] text-zinc-400 leading-relaxed">
            Four validated AI diagnostic modules with real-time processing,
            clinical-grade accuracy, and seamless workflow integration.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          {availableModules.map((module) => (
            <DiagnosticModuleCard key={module.id} module={module} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// Stats Section
function StatsSection() {
  const stats = [
    { value: '< 200ms', label: 'Response Time', icon: Zap },
    { value: '95%+', label: 'Accuracy Rate', icon: BarChart3 },
    { value: '11', label: 'Specialties', icon: Layers },
    { value: '99.9%', label: 'Uptime SLA', icon: Server },
  ];

  return (
    <section className="bg-black py-16 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="text-center"
            >
              <div className={`mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-zinc-900 border border-zinc-800`}>
                <stat.icon className="h-5 w-5 text-white" strokeWidth={1.5} />
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-white mb-1 tracking-tight">{stat.value}</div>
              <div className="text-[12px] font-medium text-zinc-500 uppercase tracking-wider">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// How It Works Section
function HowItWorksSection() {
  const steps = [
    {
      number: '01',
      title: 'Select Specialty',
      description: 'Choose from 11 AI modules: ophthalmology, radiology, cardiology, dermatology, neurology, and more.',
      icon: Brain,
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
      border: 'group-hover:border-blue-500/30',
    },
    {
      number: '02',
      title: 'Upload Data',
      description: 'Provide retinal images, X-rays, ECG signals, audio recordings, or complete interactive assessments.',
      icon: FileText,
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
      border: 'group-hover:border-violet-500/30',
    },
    {
      number: '03',
      title: 'AI Analysis',
      description: 'Specialty-trained models (EfficientNet, wav2vec, transformers) process data in under 200ms.',
      icon: Zap,
      color: 'text-amber-400',
      bg: 'bg-amber-900/20',
      border: 'group-hover:border-amber-500/30',
    },
    {
      number: '04',
      title: 'Clinical Report',
      description: 'Receive detailed findings with confidence scores, heatmaps, and actionable recommendations.',
      icon: Activity,
      color: 'text-cyan-400',
      bg: 'bg-cyan-900/20',
      border: 'group-hover:border-cyan-500/30',
    },
  ];

  return (
    <section className="bg-zinc-950 py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            Workflow Integration
          </h2>
          <p className="mx-auto max-w-xl text-[15px] text-zinc-400">
            From data upload to clinical insights in four simple steps.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((step, index) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group relative bg-zinc-900/50 rounded-xl p-8 border border-zinc-800 transition-all duration-300 hover:bg-zinc-900 ${step.border}`}
            >
              <div className={`mb-6 flex h-12 w-12 items-center justify-center rounded-lg ${step.bg} border border-zinc-800`}>
                <step.icon className={`h-5 w-5 ${step.color}`} strokeWidth={1.5} />
              </div>
              <div className="text-[10px] font-mono text-zinc-600 mb-3 uppercase tracking-wider">Step {step.number}</div>
              <h3 className="text-[16px] font-medium text-white mb-2">{step.title}</h3>
              <p className="text-[13px] text-zinc-500 leading-relaxed">{step.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Coming Soon Section
function ComingSoonSection() {
  const comingSoonModules = getComingSoonModules();

  return (
    <section className="bg-black py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="flex h-1.5 w-1.5 rounded-full bg-zinc-700" />
            <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-widest">In Development</span>
          </div>
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            Expanding Capabilities
          </h2>
          <p className="max-w-xl text-[15px] text-zinc-400 leading-relaxed">
            Radiology, dermatology, cardiology, pathology, pulmonology — each built with
            specialty-specific AI models trained on validated medical datasets.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          {comingSoonModules.map((module) => (
            <DiagnosticModuleCard key={module.id} module={module} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}


// Technology Section
// Technology Section
// Technology Section
function TechnologySection() {
  const technologies = [
    {
      category: 'Vision Models',
      items: ['EfficientNet', 'ResNet-50', 'Vision Transformers', 'U-Net', 'YOLO'],
      icon: Eye,
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
      border: 'group-hover:border-blue-500/30',
    },
    {
      category: 'Audio Models',
      items: ['wav2vec 2.0', 'HuBERT', 'OpenAI Whisper', 'LSTM/GRU'],
      icon: Mic,
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
      border: 'group-hover:border-violet-500/30',
    },
    {
      category: 'Signal Processing',
      items: ['1D CNN', 'Transformers', 'Time-series Analysis'],
      icon: Activity,
      color: 'text-emerald-400',
      bg: 'bg-emerald-900/20',
      border: 'group-hover:border-emerald-500/30',
    },
    {
      category: 'Multi-Modal',
      items: ['CLIP-style', 'Cross-modal Transformers', 'Ensemble'],
      icon: Brain,
      color: 'text-amber-400',
      bg: 'bg-amber-900/20',
      border: 'group-hover:border-amber-500/30',
    },
  ];

  return (
    <section className="bg-black py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            State-of-the-Art AI Models
          </h2>
          <p className="mx-auto max-w-xl text-[15px] text-zinc-400">
            Each module uses specialty-specific architectures trained on validated medical datasets.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group rounded-xl bg-zinc-900/50 border border-zinc-800 p-6 hover:bg-zinc-900 transition-all duration-300 ${tech.border}`}
            >
              <div className={`mb-5 flex h-10 w-10 items-center justify-center rounded-lg ${tech.bg} border border-white/5`}>
                <tech.icon className={`h-5 w-5 ${tech.color}`} strokeWidth={1.5} />
              </div>
              <h3 className="text-[15px] font-medium text-white mb-4">{tech.category}</h3>
              <div className="flex flex-wrap gap-1.5">
                {tech.items.map((item, i) => (
                  <span key={i} className="text-[11px] text-zinc-400 bg-black/50 border border-zinc-800 rounded px-2 py-1 group-hover:border-zinc-700 transition-colors">
                    {item}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Clinical Standards Section
// Clinical Standards Section
function ClinicalStandardsSection() {
  const standards = [
    {
      icon: Lock,
      title: 'HIPAA Compliant',
      description: 'End-to-end encryption, audit logging, and full healthcare data protection.',
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
      border: 'group-hover:border-blue-500/30',
    },
    {
      icon: Award,
      title: 'Clinical Validation',
      description: 'Models validated against NIH, ISIC, PhysioNet, and other medical datasets.',
      color: 'text-amber-400',
      bg: 'bg-amber-900/20',
      border: 'group-hover:border-amber-500/30',
    },
    {
      icon: FileText,
      title: 'HL7 FHIR Export',
      description: 'Generate reports compatible with EMR/EHR systems and clinical standards.',
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
      border: 'group-hover:border-violet-500/30',
    },
    {
      icon: Server,
      title: 'Enterprise Ready',
      description: 'Scalable infrastructure with 99.9% uptime and concurrent processing.',
      color: 'text-cyan-400',
      bg: 'bg-cyan-900/20',
      border: 'group-hover:border-cyan-500/30',
    },
  ];

  return (
    <section className="bg-zinc-950 py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            Security & Compliance
          </h2>
          <p className="mx-auto max-w-xl text-[15px] text-zinc-400">
            Enterprise-grade infrastructure built for healthcare.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {standards.map((standard, index) => (
            <motion.div
              key={standard.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group rounded-xl bg-black border border-zinc-800 p-6 hover:border-zinc-700 transition-all duration-300 ${standard.border}`}
            >
              <div className={`mb-5 flex h-10 w-10 items-center justify-center rounded-lg ${standard.bg} border border-white/5`}>
                <standard.icon className={`h-5 w-5 ${standard.color}`} />
              </div>
              <h3 className="text-[15px] font-medium text-white mb-2">{standard.title}</h3>
              <p className="text-[13px] text-zinc-400 leading-relaxed">{standard.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Target Users Section
// Target Users Section
function TargetUsersSection() {
  const users = [
    {
      title: 'Healthcare Professionals',
      description: 'Doctors, specialists, and clinicians seeking AI-assisted diagnostic support for faster, more accurate decisions.',
      icon: Users,
      color: 'text-blue-400',
      bg: 'bg-blue-900/20',
      border: 'group-hover:border-blue-500/30',
    },
    {
      title: 'Medical Researchers',
      description: 'Access validated AI models and datasets for clinical research, trials, and academic studies.',
      icon: Microscope,
      color: 'text-violet-400',
      bg: 'bg-violet-900/20',
      border: 'group-hover:border-violet-500/30',
    },
    {
      title: 'Healthcare Organizations',
      description: 'Enterprise deployment with EMR integration, population health analytics, and multi-site support.',
      icon: Globe,
      color: 'text-cyan-400',
      bg: 'bg-cyan-900/20',
      border: 'group-hover:border-cyan-500/30',
    },
  ];

  return (
    <section className="bg-black py-24 border-t border-zinc-900">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-semibold text-white mb-4 tracking-tight">
            Designed for Professionals
          </h2>
          <p className="mx-auto max-w-xl text-[15px] text-zinc-400">
            Tailored workflows for every healthcare role.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {users.map((user, index) => (
            <motion.div
              key={user.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group text-center p-6 rounded-2xl border border-transparent transition-all duration-300 ${user.border}`}
            >
              <div className={`mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl ${user.bg} border border-white/5`}>
                <user.icon className={`h-6 w-6 ${user.color}`} strokeWidth={1.5} />
              </div>
              <h3 className="text-[15px] font-medium text-white mb-2">{user.title}</h3>
              <p className="text-[13px] text-zinc-500 leading-relaxed">{user.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  return (
    <section className="bg-zinc-950 border-t border-zinc-900 py-24">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 text-center">
        <h2 className="text-3xl sm:text-4xl font-semibold text-white mb-6 tracking-tight">
          Ready to Modernize Your Practice?
        </h2>
        <p className="text-[16px] text-zinc-400 mb-10 max-w-2xl mx-auto leading-relaxed">
          Join thousands of clinicians using MediLens for faster, more accurate diagnostics.
        </p>

        <SignedIn>
          <div className="flex justify-center">
            <Link
              href="/dashboard"
              className="inline-flex h-12 items-center justify-center rounded-full bg-white px-8 text-[14px] font-medium text-black transition-colors hover:bg-zinc-200"
            >
              Go to Dashboard
            </Link>
          </div>
        </SignedIn>

        <SignedOut>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/dashboard"
              className="group inline-flex h-11 items-center justify-center gap-2 rounded-full bg-white px-8 text-[14px] font-medium text-black transition-all hover:bg-zinc-200"
            >
              Get Started Free
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              href="/about"
              className="inline-flex h-11 items-center justify-center rounded-full border border-zinc-700 bg-transparent px-8 text-[14px] font-medium text-white transition-colors hover:bg-zinc-900 hover:border-zinc-500"
            >
              View Documentation
            </Link>
          </div>
        </SignedOut>
      </div>
    </section>
  );
}

// Main Component
export function HomePageClient() {
  return (
    <>
      <HeroSection />
      <AvailableNowSection />
      <StatsSection />
      <HowItWorksSection />
      <ComingSoonSection />
      <TechnologySection />
      <ClinicalStandardsSection />
      <TargetUsersSection />
      <CTASection />
    </>
  );
}
