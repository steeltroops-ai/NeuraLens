'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  Brain,
  ArrowRight,
  Activity,
  Shield,
  Clock,
  Mic,
  Eye,
  Hand,
  Zap,
  Menu,
  X,
} from 'lucide-react';

export default function HomePage() {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Navigation Header */}
      <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-2">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold text-slate-900">
                NeuroLens-X
              </span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden items-center space-x-8 md:flex">
              <a
                href="/validation"
                className="text-slate-600 transition-colors hover:text-slate-900"
              >
                Validation
              </a>
              <a
                href="/dashboard"
                className="text-slate-600 transition-colors hover:text-slate-900"
              >
                Dashboard
              </a>
              <button
                onClick={() => (window.location.href = '/assessment')}
                className="rounded-lg bg-blue-600 px-6 py-2 font-medium text-white transition-colors hover:bg-blue-700"
              >
                Start Assessment
              </button>
            </div>

            {/* Mobile Menu Button */}
            <div className="md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-slate-600 hover:text-slate-900"
              >
                {mobileMenuOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden">
              <div className="space-y-1 border-t border-slate-200 bg-white px-2 pb-3 pt-2 sm:px-3">
                <a
                  href="/validation"
                  className="block px-3 py-2 text-slate-600 hover:text-slate-900"
                >
                  Validation
                </a>
                <a
                  href="/dashboard"
                  className="block px-3 py-2 text-slate-600 hover:text-slate-900"
                >
                  Dashboard
                </a>
                <button
                  onClick={() => (window.location.href = '/assessment')}
                  className="block w-full rounded-lg bg-blue-600 px-3 py-2 text-left font-medium text-white"
                >
                  Start Assessment
                </button>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -right-32 -top-40 h-80 w-80 rounded-full bg-gradient-to-br from-blue-400/20 to-purple-400/20 blur-3xl"></div>
          <div className="absolute -bottom-40 -left-32 h-80 w-80 rounded-full bg-gradient-to-br from-purple-400/20 to-pink-400/20 blur-3xl"></div>
        </div>

        <div className="relative mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8 lg:py-32">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-16 space-y-8"
            >
              <h1 className="text-5xl font-bold leading-tight text-slate-900 sm:text-6xl lg:text-7xl">
                Advanced{' '}
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Neurological
                </span>
                <br />
                Assessment Platform
              </h1>
              <p className="mx-auto max-w-4xl text-xl leading-relaxed text-slate-600 sm:text-2xl">
                Real-time AI-powered screening for neurological conditions with
                <span className="font-semibold text-blue-600">
                  {' '}
                  95%+ accuracy
                </span>{' '}
                and
                <span className="font-semibold text-purple-600">
                  {' '}
                  sub-100ms processing
                </span>
              </p>
            </motion.div>

            {/* Key Features */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="mb-16 flex flex-wrap justify-center gap-6"
            >
              <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                <Clock className="h-5 w-5 text-blue-600" />
                <span className="font-medium text-slate-700">
                  Real-time Processing
                </span>
              </div>
              <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                <Shield className="h-5 w-5 text-green-600" />
                <span className="font-medium text-slate-700">
                  95%+ Accuracy
                </span>
              </div>
              <div className="flex items-center space-x-2 rounded-full border border-slate-200 bg-white/60 px-6 py-3 backdrop-blur-sm">
                <Activity className="h-5 w-5 text-purple-600" />
                <span className="font-medium text-slate-700">
                  Multi-Modal Analysis
                </span>
              </div>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="mb-20 flex flex-col items-center justify-center gap-6 sm:flex-row"
            >
              <button
                onClick={() => (window.location.href = '/assessment')}
                className="group flex w-full transform items-center justify-center space-x-3 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-10 py-4 text-lg font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:from-blue-700 hover:to-purple-700 hover:shadow-xl sm:w-auto"
              >
                <span>Start Assessment</span>
                <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
              </button>

              <button
                onClick={() => (window.location.href = '/dashboard')}
                className="w-full rounded-xl border border-slate-200 bg-white px-10 py-4 text-lg font-semibold text-slate-900 shadow-sm transition-all duration-300 hover:border-slate-300 hover:bg-slate-50 hover:shadow-md sm:w-auto"
              >
                View Dashboard
              </button>
            </motion.div>

            {/* Assessment Types - Clean Grid */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="grid grid-cols-2 gap-6 lg:grid-cols-4"
            >
              <AssessmentCard
                icon={<Mic className="h-8 w-8 text-white" />}
                title="Speech Analysis"
                description="Voice pattern analysis"
                processingTime="11.7ms"
                color="from-blue-500 to-blue-600"
              />
              <AssessmentCard
                icon={<Eye className="h-8 w-8 text-white" />}
                title="Retinal Imaging"
                description="Fundus analysis"
                processingTime="145ms"
                color="from-green-500 to-green-600"
              />
              <AssessmentCard
                icon={<Hand className="h-8 w-8 text-white" />}
                title="Motor Function"
                description="Movement patterns"
                processingTime="42ms"
                color="from-purple-500 to-purple-600"
              />
              <AssessmentCard
                icon={<Brain className="h-8 w-8 text-white" />}
                title="Cognitive Tests"
                description="Memory & attention"
                processingTime="38ms"
                color="from-indigo-500 to-indigo-600"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 py-16 text-white">
        <div className="mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8">
          <div className="mb-6 flex items-center justify-center space-x-3">
            <div className="rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 p-2">
              <Brain className="h-6 w-6 text-white" />
            </div>
            <span className="text-2xl font-bold">NeuroLens-X</span>
          </div>
          <p className="mb-6 text-lg text-slate-400">
            Advanced neurological assessment platform powered by real-time AI
            analysis.
          </p>
          <div className="flex items-center justify-center space-x-6 text-slate-400">
            <span>© 2024 NeuroLens</span>
            <span>•</span>
            <span>All rights reserved</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Assessment Card Component
function AssessmentCard({
  icon,
  title,
  description,
  processingTime,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  processingTime: string;
  color: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white/60 p-6 backdrop-blur-sm transition-all duration-300 hover:border-slate-300 hover:shadow-lg">
      <div
        className={`bg-gradient-to-r p-4 ${color} mx-auto mb-4 w-fit rounded-lg`}
      >
        {icon}
      </div>
      <h3 className="mb-2 text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mb-4 text-slate-600">{description}</p>
      <div className="text-sm font-medium text-green-600">
        {processingTime} processing
      </div>
    </div>
  );
}
