'use client';

import { motion } from 'framer-motion';
import {
  Brain,
  Shield,
  Clock,
  Zap,
  ArrowRight,
  CheckCircle,
  Users,
  Award,
  Sparkles,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  getAvailableModules,
  getComingSoonModules,
  TOTAL_MODULES_COUNT,
  DiagnosticModule,
} from '@/data/diagnostic-modules';

// Animation variants for smooth Apple-like transitions (MediLens Design System)
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

// Hero Section Component with MediLens Branding
function HeroSection() {
  const [currentFeature, setCurrentFeature] = useState(0);
  const availableModules = getAvailableModules();
  const features = availableModules.map(m => m.name);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature(prev => (prev + 1) % features.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [features.length]);

  return (
    <section className="relative flex min-h-[calc(100vh-80px)] items-center justify-center overflow-hidden bg-gradient-to-br from-white via-[#E8F4FF] to-[#F2F2F7]">
      {/* Background geometric elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute left-1/4 top-1/4 h-32 w-32 sm:h-64 sm:w-64 rounded-full bg-[#007AFF]/10 blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 h-48 w-48 sm:h-96 sm:w-96 rounded-full bg-[#5AC8FA]/10 blur-3xl" />
        <div className="absolute right-1/3 top-1/3 h-24 w-24 sm:h-48 sm:w-48 rounded-full bg-[#34C759]/10 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8 py-12 sm:py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="space-y-6 sm:space-y-8"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center gap-2 rounded-full bg-[#007AFF]/10 px-4 py-2 text-sm font-medium text-[#007AFF]"
          >
            <Sparkles className="h-4 w-4" />
            <span>AI-Powered Medical Diagnostics</span>
          </motion.div>

          {/* Main headline */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold leading-tight tracking-tight text-[#000000]">
            Welcome to{' '}
            <span className="bg-gradient-to-r from-[#007AFF] to-[#5AC8FA] bg-clip-text text-transparent">
              MediLens
            </span>
          </h1>

          {/* Dynamic feature showcase */}
          <div className="flex h-10 sm:h-12 items-center justify-center">
            <motion.p
              key={currentFeature}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              className="text-lg sm:text-xl md:text-2xl font-medium text-[#3C3C43]"
            >
              {features[currentFeature]} • AI-Powered Detection
            </motion.p>
          </div>

          {/* Subtitle */}
          <p className="mx-auto max-w-2xl text-base sm:text-lg leading-relaxed text-[#3C3C43]">
            A centralized platform where doctors and patients can access multiple AI-powered
            diagnostic tools for various medical conditions — all in one place.
          </p>

          {/* CTA Buttons - Apple-style compact */}
          <div className="flex flex-col items-center justify-center gap-3 pt-8 sm:flex-row">
            <Link
              href="/assessment"
              className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-full bg-[#007AFF] px-6 py-3 text-base font-medium text-white transition-all duration-200 hover:bg-[#0062CC] active:scale-[0.98]"
            >
              Start Assessment
              <ArrowRight className="h-4 w-4" />
            </Link>

            <Link
              href="/dashboard"
              className="inline-flex w-full sm:w-auto items-center justify-center gap-2 rounded-full border border-[#D1D1D6] bg-white px-6 py-3 text-base font-medium text-[#000000] transition-all duration-200 hover:bg-[#F2F2F7] active:scale-[0.98]"
            >
              View Dashboard
            </Link>
          </div>

          {/* Trust indicators */}
          <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-8 pt-10 text-[#8E8E93]">
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              <span className="text-sm font-medium">HIPAA Compliant</span>
            </div>
            <div className="flex items-center gap-2">
              <Award className="h-5 w-5" />
              <span className="text-sm font-medium">Clinically Validated</span>
            </div>
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              <span className="text-sm font-medium">{TOTAL_MODULES_COUNT} Diagnostic Modules</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// Diagnostic Module Card Component
function DiagnosticModuleCard({ module }: { module: DiagnosticModule }) {
  const isAvailable = module.status === 'available';

  return (
    <motion.div
      variants={fadeInUp}
      whileHover={isAvailable ? { y: -4, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] } } : {}}
      className={`relative rounded-2xl border p-6 transition-all duration-300 ${isAvailable
        ? 'cursor-pointer border-black/5 bg-white shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1)] hover:shadow-[0_10px_20px_-5px_rgba(0,0,0,0.1)]'
        : 'border-[#E5E5EA] bg-[#F9F9F9] opacity-75'
        }`}
    >
      {/* Status Badge */}
      <div className="absolute right-4 top-4">
        {isAvailable ? (
          <span className="inline-flex items-center rounded-full bg-[#34C759]/10 px-3 py-1 text-xs font-medium text-[#34C759]">
            Available
          </span>
        ) : (
          <span className="inline-flex items-center rounded-full bg-[#8E8E93]/10 px-3 py-1 text-xs font-medium text-[#8E8E93]">
            Coming Soon
          </span>
        )}
      </div>

      {/* Icon */}
      <div
        className={`mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br ${module.gradient}`}
      >
        <module.icon className="h-7 w-7 text-white" />
      </div>

      {/* Content */}
      <h3 className="mb-2 text-lg font-semibold text-[#000000]">{module.name}</h3>
      <p className="mb-4 text-sm leading-relaxed text-[#3C3C43]">{module.description}</p>

      {/* Diagnoses */}
      <div className="mb-4">
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-[#8E8E93]">Diagnoses</p>
        <div className="flex flex-wrap gap-1.5">
          {module.diagnoses.slice(0, 3).map((diagnosis, idx) => (
            <span
              key={idx}
              className="inline-block rounded-full bg-[#F2F2F7] px-2.5 py-1 text-xs text-[#3C3C43]"
            >
              {diagnosis}
            </span>
          ))}
          {module.diagnoses.length > 3 && (
            <span className="inline-block rounded-full bg-[#F2F2F7] px-2.5 py-1 text-xs text-[#8E8E93]">
              +{module.diagnoses.length - 3} more
            </span>
          )}
        </div>
      </div>

      {/* How it works */}
      <p className="text-xs text-[#8E8E93]">
        <span className="font-medium">How:</span> {module.howItWorks}
      </p>

      {/* Link overlay for available modules */}
      {isAvailable && (
        <Link href={module.route} className="absolute inset-0 z-10" aria-label={`Go to ${module.name}`}>
          <span className="sr-only">Go to {module.name}</span>
        </Link>
      )}
    </motion.div>
  );
}


// Available Now Section - 4 Current Modules
function AvailableNowSection() {
  const availableModules = getAvailableModules();

  return (
    <section className="bg-white py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mb-12 text-center"
        >
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-[#34C759]/10 px-4 py-2 text-sm font-medium text-[#34C759]">
            <CheckCircle className="h-4 w-4" />
            <span>Available Now</span>
          </div>
          <h2 className="mb-4 text-3xl sm:text-4xl font-bold tracking-tight text-[#000000]">
            Start Your Assessment Today
          </h2>
          <p className="mx-auto max-w-2xl text-base sm:text-lg text-[#3C3C43]">
            Four specialized AI-powered diagnostic modules ready for clinical use.
            Each module has been validated and optimized for accuracy.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4"
        >
          {availableModules.map((module) => (
            <DiagnosticModuleCard key={module.id} module={module} />
          ))}
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-12 text-center"
        >
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 rounded-full bg-[#007AFF] px-6 py-3 text-base font-medium text-white transition-all duration-200 hover:bg-[#0062CC] active:scale-[0.98]"
          >
            Access All Modules
            <ArrowRight className="h-4 w-4" />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}

// Coming Soon Section - Upcoming Modules
function ComingSoonSection() {
  const comingSoonModules = getComingSoonModules();

  return (
    <section className="bg-[#F2F2F7] py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mb-12 text-center"
        >
          <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-[#AF52DE]/10 px-4 py-2 text-sm font-medium text-[#AF52DE]">
            <Sparkles className="h-4 w-4" />
            <span>Coming Soon</span>
          </div>
          <h2 className="mb-4 text-3xl sm:text-4xl font-bold tracking-tight text-[#000000]">
            Expanding Our Diagnostic Suite
          </h2>
          <p className="mx-auto max-w-2xl text-base sm:text-lg text-[#3C3C43]">
            We're continuously developing new AI-powered diagnostic modules to cover
            more medical specialties including radiology, dermatology, cardiology, and more.
          </p>
        </motion.div>

        <motion.div
          variants={staggerContainer}
          initial="initial"
          whileInView="animate"
          viewport={{ once: true }}
          className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4"
        >
          {comingSoonModules.map((module) => (
            <DiagnosticModuleCard key={module.id} module={module} />
          ))}
        </motion.div>
      </div>
    </section>
  );
}

// Performance Stats Section
function PerformanceStats() {
  const stats = [
    { icon: Clock, value: '< 200ms', label: 'Response Time' },
    { icon: Shield, value: '95%+', label: 'Accuracy Rate' },
    { icon: Users, value: '10,000+', label: 'Assessments' },
    { icon: Zap, value: '24/7', label: 'Availability' },
  ];

  return (
    <section className="bg-white py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mb-12 text-center"
        >
          <h2 className="mb-4 text-3xl sm:text-4xl font-bold tracking-tight text-[#000000]">
            Trusted by Healthcare Professionals
          </h2>
          <p className="mx-auto max-w-2xl text-base sm:text-lg text-[#3C3C43]">
            Our platform delivers consistent, reliable results that healthcare providers
            depend on for critical diagnostic decisions.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
              className="text-center"
            >
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-[#007AFF]/10">
                <stat.icon className="h-8 w-8 text-[#007AFF]" />
              </div>
              <div className="mb-2 text-2xl sm:text-3xl md:text-4xl font-bold text-[#000000]">{stat.value}</div>
              <div className="text-sm sm:text-base font-medium text-[#3C3C43]">{stat.label}</div>
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
      title: 'Select Module',
      description: 'Choose from our suite of AI-powered diagnostic tools based on your clinical needs.',
    },
    {
      number: '02',
      title: 'Upload Data',
      description: 'Provide the required input — images, audio recordings, or interactive assessments.',
    },
    {
      number: '03',
      title: 'AI Analysis',
      description: 'Our validated AI models process your data with clinical-grade accuracy.',
    },
    {
      number: '04',
      title: 'Get Results',
      description: 'Receive comprehensive reports with actionable insights and recommendations.',
    },
  ];

  return (
    <section className="bg-[#F2F2F7] py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="mb-12 text-center"
        >
          <h2 className="mb-4 text-3xl sm:text-4xl font-bold tracking-tight text-[#000000]">
            How MediLens Works
          </h2>
          <p className="mx-auto max-w-2xl text-base sm:text-lg text-[#3C3C43]">
            A simple, streamlined workflow designed for busy healthcare professionals.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, index) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
              className="relative"
            >
              <div className="mb-4 text-5xl font-bold text-[#007AFF]/20">{step.number}</div>
              <h3 className="mb-2 text-lg font-semibold text-[#000000]">{step.title}</h3>
              <p className="text-sm sm:text-base text-[#3C3C43]">{step.description}</p>
              {index < steps.length - 1 && (
                <div className="absolute right-0 top-8 hidden h-0.5 w-8 bg-[#E5E5EA] lg:block" />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}


// Main HomePageClient Component
export function HomePageClient() {
  return (
    <>
      <HeroSection />
      <AvailableNowSection />
      <ComingSoonSection />
      <HowItWorksSection />
      <PerformanceStats />
    </>
  );
}
