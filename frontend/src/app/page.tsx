'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { Activity, Shield, Clock, Zap, X, Camera, Loader } from 'lucide-react';
import { Layout } from '@/components/layout';
import {
  ErrorBoundary,
  AnatomicalLoadingSkeleton,
  NetworkStatus,
} from '@/components/ErrorBoundary';
import {
  SpeechWaveform,
  RetinalEye,
  HandKinematics,
  BrainNeural,
  NRIFusion,
  MultiModalNetwork,
} from '@/components/visuals/AnimatedGeometry';

export default function HomePage() {
  const router = useRouter();

  const handleStartAssessment = () => {
    router.push('/dashboard');
  };

  const handleOpenDashboard = () => {
    router.push('/dashboard');
  };

  const handleLearnMore = () => {
    router.push('/about');
  };

  return (
    <ErrorBoundary>
      <NetworkStatus />
      <Layout showHeader={true} showFooter={false} containerized={false}>
        <div className="min-h-screen bg-white">
          {/* Hero Section */}
          <section className="relative overflow-hidden bg-white">
            {/* Subtle Neural Grid Background */}
            <div className="absolute inset-0 opacity-[0.02]">
              <div
                className="h-full w-full"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='200' height='200' xmlns='http://www.w3.org/2000/svg'%3E%3Cdefs%3E%3Cpattern id='neural-grid' x='0' y='0' width='60' height='60' patternUnits='userSpaceOnUse'%3E%3Ccircle cx='30' cy='30' r='1.5' fill='%231e3a8a' opacity='0.4'/%3E%3Cline x1='30' y1='30' x2='50' y2='10' stroke='%231e3a8a' stroke-width='0.5' opacity='0.3'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23neural-grid)'/%3E%3C/svg%3E")`,
                  backgroundSize: '200px 200px',
                }}
              />
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
                    <span style={{ color: '#1D1D1F' }}>Neuralens</span>
                  </h1>
                  <p className="text-2xl font-medium text-slate-700 sm:text-3xl">
                    Early Detection, Better Outcomes.
                  </p>
                  <p className="mx-auto max-w-4xl text-lg leading-relaxed text-slate-600 sm:text-xl">
                    Transforming neurological health with AI-powered insights
                    for millions.
                  </p>
                </motion.div>

                {/* Key Features */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="mb-16 flex flex-wrap justify-center gap-6"
                >
                  <div className="flex items-center space-x-2 rounded-full border border-slate-100 bg-white/85 px-6 py-3 shadow-sm backdrop-blur-md">
                    <Clock className="h-5 w-5" style={{ color: '#007AFF' }} />
                    <span className="font-medium text-slate-800">
                      Sub-100ms Processing
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 rounded-full border border-slate-100 bg-white/85 px-6 py-3 shadow-sm backdrop-blur-md">
                    <Shield className="h-5 w-5" style={{ color: '#007AFF' }} />
                    <span className="font-medium text-slate-800">
                      90%+ Clinical Accuracy
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 rounded-full border border-slate-100 bg-white/85 px-6 py-3 shadow-sm backdrop-blur-md">
                    <Activity
                      className="h-5 w-5"
                      style={{ color: '#007AFF' }}
                    />
                    <span className="font-medium text-slate-800">
                      4-Modal AI Analysis
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 rounded-full border border-slate-100 bg-white/85 px-6 py-3 shadow-sm backdrop-blur-md">
                    <Zap className="h-5 w-5" style={{ color: '#007AFF' }} />
                    <span className="font-medium text-slate-800">
                      HIPAA Compliant
                    </span>
                  </div>
                </motion.div>

                {/* CTA Buttons */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                  className="mb-20 flex flex-col items-center justify-center gap-4 sm:flex-row"
                >
                  <button
                    onClick={handleStartAssessment}
                    className="hover:scale-98 rounded-xl px-6 py-3 text-base font-semibold text-white transition-all duration-150 focus:ring-2 focus:ring-blue-500 active:scale-95"
                    style={{
                      backgroundColor: '#007AFF',
                      backdropFilter: 'blur(20px)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
                    }}
                    onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                      e.currentTarget.style.backgroundColor = '#0056CC';
                    }}
                    onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                      e.currentTarget.style.backgroundColor = '#007AFF';
                    }}
                  >
                    Start Test
                  </button>

                  <button
                    onClick={handleOpenDashboard}
                    className="hover:scale-98 rounded-xl px-6 py-3 text-base font-semibold transition-all duration-150 focus:ring-2 focus:ring-blue-500 active:scale-95"
                    style={{
                      backgroundColor: 'rgba(142, 142, 147, 0.2)',
                      color: '#1D1D1F',
                      backdropFilter: 'blur(20px)',
                      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
                    }}
                    onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                      e.currentTarget.style.backgroundColor =
                        'rgba(142, 142, 147, 0.3)';
                    }}
                    onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                      e.currentTarget.style.backgroundColor =
                        'rgba(142, 142, 147, 0.2)';
                    }}
                  >
                    Dashboard
                  </button>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Consolidated Overview Section */}
          <section className="py-20" style={{ backgroundColor: '#F5F5F7' }}>
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="mx-auto max-w-6xl"
              >
                <div className="grid grid-cols-1 gap-12 lg:grid-cols-3">
                  {/* What is NeuraLens */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.1 }}
                    className="space-y-4 text-center"
                  >
                    <h2
                      className="text-2xl font-bold"
                      style={{ color: '#1D1D1F' }}
                    >
                      What is NeuraLens?
                    </h2>
                    <p className="text-lg leading-relaxed text-slate-600">
                      A pioneering platform harnessing speech, retinal, motor,
                      and cognitive analysis to detect neurological risks early,
                      empowering better health decisions with AI-powered
                      precision.
                    </p>
                  </motion.div>

                  {/* The Impact */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                    className="space-y-4 text-center"
                  >
                    <h2
                      className="text-2xl font-bold"
                      style={{ color: '#1D1D1F' }}
                    >
                      The Impact
                    </h2>
                    <p className="text-lg leading-relaxed text-slate-600">
                      1 in 6 adults experience neurological conditions
                      worldwide. Early detection improves outcomes
                      significantly. Advanced AI accuracy enables precise
                      multi-modal assessment for better health decisions.
                    </p>
                  </motion.div>

                  {/* Advanced Features */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.3 }}
                    className="space-y-4 text-center"
                  >
                    <h2
                      className="text-2xl font-bold"
                      style={{ color: '#1D1D1F' }}
                    >
                      Advanced AI Assessment Features
                    </h2>
                    <p className="text-lg leading-relaxed text-slate-600">
                      Comprehensive neurological evaluation through cutting-edge
                      multi-modal analysis
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Speech Analysis Section - White Background */}
          <section className="relative bg-white py-24">
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    Speech Analysis
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Model:
                      </span>{' '}
                      Whisper-tiny (OpenAI) optimized for neurological biomarker
                      extraction
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Purpose:
                      </span>{' '}
                      Speech-to-text with MFCC, prosody, and fluency analysis
                      for early detection
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Source:
                      </span>{' '}
                      Hugging Face Transformers, converted to ONNX for
                      WebAssembly
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Accuracy:
                      </span>{' '}
                      90%+ validated on DementiaBank dataset
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Latency:
                      </span>{' '}
                      &lt;100ms real-time processing with client-side inference
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Detects Alzheimer's through speech fluency, pause
                      patterns, and vocal tremor analysis
                    </p>
                  </div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <SpeechWaveform />
                  </ErrorBoundary>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Retinal Analysis Section - Apple Gray Background */}
          <section
            className="relative py-24"
            style={{ backgroundColor: '#F5F5F7' }}
          >
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <RetinalEye />
                  </ErrorBoundary>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    Retinal Analysis
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Model:
                      </span>{' '}
                      EfficientNet-B0 fine-tuned on APTOS 2019 diabetic
                      retinopathy dataset
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Purpose:
                      </span>{' '}
                      Vascular analysis, cup-disc ratio measurement, and
                      neurological risk assessment
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Source:
                      </span>{' '}
                      timm library, converted to ONNX for browser deployment
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Accuracy:
                      </span>{' '}
                      85%+ precision on retinal biomarker detection
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Latency:
                      </span>{' '}
                      &lt;150ms with GPU acceleration and intelligent caching
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Non-invasive stroke risk assessment through retinal vessel
                      analysis and optic nerve evaluation
                    </p>
                  </div>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Motor Assessment Section - White Background */}
          <section className="relative bg-white py-24">
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    Motor Assessment
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Model:
                      </span>{' '}
                      LSTM-based tremor analysis with MediaPipe hand tracking
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Purpose:
                      </span>{' '}
                      Finger tapping analysis, tremor detection, and movement
                      coordination assessment
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Source:
                      </span>{' '}
                      TensorFlow.js with MediaPipe integration for real-time
                      hand tracking
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Accuracy:
                      </span>{' '}
                      88%+ on Parkinson's disease detection benchmarks
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Latency:
                      </span>{' '}
                      &lt;50ms real-time movement analysis with webcam input
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Parkinson's disease detection through finger tapping,
                      tremor analysis, and movement coordination
                    </p>
                  </div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <HandKinematics />
                  </ErrorBoundary>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Cognitive Evaluation Section - Apple Gray Background */}
          <section
            className="relative py-24"
            style={{ backgroundColor: '#F5F5F7' }}
          >
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <BrainNeural />
                  </ErrorBoundary>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    Cognitive Evaluation
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Model:
                      </span>{' '}
                      Decision Tree ensemble with Random Forest for cognitive
                      assessment
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Purpose:
                      </span>{' '}
                      Working memory, attention span, and executive function
                      evaluation
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Source:
                      </span>{' '}
                      Scikit-learn with custom cognitive game metrics and
                      reaction time analysis
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Accuracy:
                      </span>{' '}
                      92%+ on dementia screening and cognitive decline detection
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Latency:
                      </span>{' '}
                      &lt;30ms for real-time cognitive performance scoring
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Dementia screening through working memory, attention span,
                      and executive function assessment
                    </p>
                  </div>
                </motion.div>
              </div>
            </div>
          </section>

          {/* NRI Fusion Section - White Background */}
          <section className="relative bg-white py-24">
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    NRI Fusion Engine
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Model:
                      </span>{' '}
                      XGBoost ensemble with SHAP explainability for multi-modal
                      fusion
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Purpose:
                      </span>{' '}
                      Neurological Risk Index calculation from speech, retinal,
                      motor, and cognitive biomarkers
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Source:
                      </span>{' '}
                      XGBoost with SHAP integration for transparent AI
                      decision-making
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Accuracy:
                      </span>{' '}
                      94%+ on comprehensive neurological risk assessment
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Latency:
                      </span>{' '}
                      &lt;5ms for real-time risk score calculation and
                      explanation
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Comprehensive neurological risk profiling through
                      AI-powered integration of all assessment modalities
                    </p>
                  </div>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <NRIFusion />
                  </ErrorBoundary>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Multi-Modal Assessment Section - Apple Gray Background */}
          <section
            className="relative py-24"
            style={{ backgroundColor: '#F5F5F7' }}
          >
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8 }}
                  className="relative h-64"
                >
                  <ErrorBoundary
                    fallback={<AnatomicalLoadingSkeleton className="h-64" />}
                  >
                    <MultiModalNetwork />
                  </ErrorBoundary>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="space-y-6"
                >
                  <h2 className="mb-6 text-4xl font-bold text-slate-900">
                    Multi-Modal Assessment
                  </h2>
                  <div className="space-y-4 text-slate-600">
                    <p>
                      <span className="font-semibold text-slate-900">
                        Overview:
                      </span>{' '}
                      Integrated pipeline combining all assessment modalities
                      for comprehensive neurological evaluation
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Integration:
                      </span>{' '}
                      Real-time fusion of speech, retinal, motor, and cognitive
                      biomarkers
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Benefits:
                      </span>{' '}
                      Enhanced robustness through cross-modal validation and
                      correlation analysis
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Output:
                      </span>{' '}
                      Consolidated NRI score with SHAP-based explainability and
                      confidence intervals
                    </p>
                    <p>
                      <span className="font-semibold text-slate-900">
                        Clinical Value:
                      </span>{' '}
                      Comprehensive risk assessment enabling early intervention
                      and personalized care
                    </p>
                    <p className="text-sm italic text-slate-500">
                      Holistic neurological assessment combining multiple
                      biomarkers for superior diagnostic accuracy
                    </p>
                  </div>
                </motion.div>
              </div>
            </div>
          </section>

          {/* Call to Action Section */}
          <section className="bg-white py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="mx-auto max-w-4xl space-y-6 text-center"
              >
                <h2 className="text-4xl font-bold" style={{ color: '#1D1D1F' }}>
                  Take Control of Your Health
                </h2>
                <p className="text-xl leading-relaxed text-slate-600">
                  Start your journey with NeuraLens today and gain insights into
                  your neurological well-being with cutting-edge AI technology.
                </p>
                <button
                  onClick={handleStartAssessment}
                  className="hover:scale-98 rounded-lg px-6 py-3 text-base font-semibold text-white transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  style={{
                    backgroundColor: '#007AFF',
                    backdropFilter: 'blur(20px)',
                  }}
                  onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                    e.currentTarget.style.backgroundColor = '#0056CC';
                  }}
                  onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                    e.currentTarget.style.backgroundColor = '#007AFF';
                  }}
                >
                  Start Health Check
                </button>
              </motion.div>
            </div>
          </section>

          {/* Apple-Style Footer */}
          <footer className="bg-white py-16">
            <div className="mx-auto max-w-7xl px-8">
              <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
                {/* Company Info */}
                <div className="space-y-4">
                  <h3
                    className="text-lg font-semibold"
                    style={{ color: '#1D1D1F' }}
                  >
                    NeuraLens
                  </h3>
                  <p className="text-sm text-gray-600">
                    Advanced neurological assessment through AI-powered
                    multi-modal analysis.
                  </p>
                </div>

                {/* Product */}
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-gray-900">
                    Product
                  </h4>
                  <ul className="space-y-2 text-sm">
                    <li>
                      <a
                        href="/assessment"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Health Assessment
                      </a>
                    </li>
                    <li>
                      <a
                        href="/dashboard"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Dashboard
                      </a>
                    </li>
                    <li>
                      <a
                        href="/results"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Results
                      </a>
                    </li>
                  </ul>
                </div>

                {/* Support */}
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-gray-900">
                    Support
                  </h4>
                  <ul className="space-y-2 text-sm">
                    <li>
                      <a
                        href="/help"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Help Center
                      </a>
                    </li>
                    <li>
                      <a
                        href="/contact"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Contact Us
                      </a>
                    </li>
                    <li>
                      <a
                        href="/faq"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        FAQ
                      </a>
                    </li>
                  </ul>
                </div>

                {/* Legal */}
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-gray-900">Legal</h4>
                  <ul className="space-y-2 text-sm">
                    <li>
                      <a
                        href="/privacy"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Privacy Policy
                      </a>
                    </li>
                    <li>
                      <a
                        href="/terms"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Terms of Service
                      </a>
                    </li>
                    <li>
                      <a
                        href="/security"
                        className="text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                      >
                        Security
                      </a>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Bottom Section */}
              <div className="mt-12 border-t border-gray-200 pt-8">
                <div className="flex flex-col items-center justify-between space-y-4 md:flex-row md:space-y-0">
                  <p className="text-sm text-gray-600">
                    © 2025 NeuraLens. All rights reserved.
                  </p>
                  <div className="flex space-x-6">
                    <a
                      href="/accessibility"
                      className="text-sm text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                    >
                      Accessibility
                    </a>
                    <a
                      href="/cookies"
                      className="text-sm text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                    >
                      Cookie Policy
                    </a>
                    <a
                      href="/compliance"
                      className="text-sm text-gray-600 transition-colors duration-200 hover:text-gray-900 hover:no-underline"
                    >
                      HIPAA Compliance
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </Layout>
    </ErrorBoundary>
  );
}
