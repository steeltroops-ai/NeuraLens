'use client';

import { motion } from 'framer-motion';
import {
  Activity,
  ArrowRight,
  Award,
  BarChart3,
  Brain,
  CheckCircle,
  Cloud,
  Code,
  Cpu,
  Database,
  Eye,
  Globe,
  Mic,
  Shield,
  Zap,
} from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Layout } from '@/components/layout';

export default function ReadmePage() {
  return (
    <ErrorBoundary>
      <Layout showHeader={true} showFooter={true} containerized={false}>
        <div className='min-h-screen bg-white'>
          {/* Hero Section */}
          <section className='bg-white py-24 md:py-32'>
            <div className='container mx-auto px-6 md:px-8'>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
                className='mx-auto max-w-7xl text-center'
              >
                <div className='mb-12'>
                  {/* Geometric Data Flow Design */}
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                    className='mb-12 flex justify-center'
                  >
                    <div className='relative'>
                      {/* Data Processing Pipeline */}
                      <div className='flex items-center gap-4'>
                        <div
                          className='h-3 w-8 rounded-full'
                          style={{ backgroundColor: '#007AFF', opacity: 0.8 }}
                        />
                        <div
                          className='h-4 w-4 rounded-full'
                          style={{ backgroundColor: '#007AFF' }}
                        />
                        <div
                          className='h-3 w-8 rounded-full'
                          style={{ backgroundColor: '#007AFF', opacity: 0.8 }}
                        />
                      </div>
                      {/* Vertical Data Streams */}
                      <div
                        className='absolute -top-6 left-1/2 h-12 w-px -translate-x-1/2'
                        style={{ backgroundColor: '#007AFF', opacity: 0.4 }}
                      />
                      <div
                        className='absolute -bottom-6 left-1/2 h-12 w-px -translate-x-1/2'
                        style={{ backgroundColor: '#007AFF', opacity: 0.4 }}
                      />
                    </div>
                  </motion.div>

                  <h1
                    className='mb-8 text-6xl font-bold leading-[1.1] tracking-tight md:text-8xl lg:text-9xl'
                    style={{
                      color: '#1D1D1F',
                      fontFamily:
                        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    }}
                  >
                    NeuraLens
                  </h1>
                  <motion.h2
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.3 }}
                    className='mb-12 text-3xl font-semibold leading-tight md:text-4xl lg:text-5xl'
                    style={{ color: '#1D1D1F' }}
                  >
                    Revolutionary Multi-Modal Neurological Screening Platform
                  </motion.h2>
                </div>

                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.4 }}
                  className='mx-auto mb-12 max-w-6xl text-lg leading-relaxed md:text-xl lg:text-2xl'
                  style={{
                    color: '#86868B',
                    lineHeight: '1.6',
                  }}
                >
                  The world&apos;s first comprehensive AI platform combining speech analysis,
                  retinal imaging, motor assessment, and cognitive testing for early detection of
                  neurological conditions.
                </motion.p>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.5 }}
                  className='mb-16 flex flex-wrap justify-center gap-6 md:gap-8'
                >
                  <div className='rounded-lg border border-gray-200 bg-white px-8 py-4 transition-all duration-200 hover:border-gray-300 hover:shadow-sm'>
                    <span className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      95%+ Clinical Accuracy
                    </span>
                  </div>
                  <div className='rounded-lg border border-gray-200 bg-white px-8 py-4 transition-all duration-200 hover:border-gray-300 hover:shadow-sm'>
                    <span className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      &lt;2s Real-Time Processing
                    </span>
                  </div>
                  <div className='rounded-lg border border-gray-200 bg-white px-8 py-4 transition-all duration-200 hover:border-gray-300 hover:shadow-sm'>
                    <span className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      HIPAA Compliant
                    </span>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 0.6 }}
                  className='mx-auto grid max-w-6xl grid-cols-1 gap-8 md:grid-cols-2 md:gap-12 lg:grid-cols-4'
                >
                  <div className='group text-center'>
                    <div
                      className='mb-4 text-4xl font-bold transition-all duration-300 group-hover:scale-110 md:text-5xl'
                      style={{ color: '#007AFF' }}
                    >
                      $2.5B
                    </div>
                    <p
                      className='text-base leading-relaxed md:text-lg'
                      style={{ color: '#86868B' }}
                    >
                      Healthcare savings potential
                    </p>
                  </div>
                  <div className='group text-center'>
                    <div
                      className='mb-4 text-4xl font-bold transition-all duration-300 group-hover:scale-110 md:text-5xl'
                      style={{ color: '#007AFF' }}
                    >
                      97%
                    </div>
                    <p
                      className='text-base leading-relaxed md:text-lg'
                      style={{ color: '#86868B' }}
                    >
                      Cost reduction vs traditional
                    </p>
                  </div>
                  <div className='group text-center'>
                    <div
                      className='mb-4 text-4xl font-bold transition-all duration-300 group-hover:scale-110 md:text-5xl'
                      style={{ color: '#007AFF' }}
                    >
                      18mo
                    </div>
                    <p
                      className='text-base leading-relaxed md:text-lg'
                      style={{ color: '#86868B' }}
                    >
                      Earlier detection capability
                    </p>
                  </div>
                  <div className='group text-center'>
                    <div
                      className='mb-4 text-4xl font-bold transition-all duration-300 group-hover:scale-110 md:text-5xl'
                      style={{ color: '#007AFF' }}
                    >
                      1B+
                    </div>
                    <p
                      className='text-base leading-relaxed md:text-lg'
                      style={{ color: '#86868B' }}
                    >
                      People potentially served
                    </p>
                  </div>
                </motion.div>
              </motion.div>
            </div>
          </section>

          {/* System Architecture Section */}
          <section style={{ backgroundColor: '#F5F5F7' }} className='py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className='mx-auto max-w-7xl'
              >
                <div className='mb-16 text-center'>
                  <h2 className='mb-6 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                    Enterprise System Architecture
                  </h2>
                  <p
                    className='mx-auto max-w-4xl text-xl leading-relaxed'
                    style={{ color: '#86868B' }}
                  >
                    Production-ready, scalable architecture designed for healthcare-grade
                    reliability and enterprise deployment.
                  </p>
                </div>

                {/* Architecture Diagram */}
                <div className='mb-12 rounded-2xl bg-white p-8 shadow-sm'>
                  <div className='mb-8 text-center'>
                    <h3 className='mb-4 text-2xl font-semibold' style={{ color: '#1D1D1F' }}>
                      High-Level Architecture
                    </h3>
                  </div>

                  <div className='mb-8 grid grid-cols-1 gap-8 md:grid-cols-3'>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.8, delay: 0.3 }}
                      className='text-center'
                    >
                      <div className='mb-4 rounded-xl bg-blue-50 p-6'>
                        <Code className='mx-auto mb-4 h-12 w-12' style={{ color: '#007AFF' }} />
                        <h4 className='mb-2 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                          Frontend Layer
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Next.js 15 + TypeScript
                        </p>
                      </div>
                      <div className='space-y-2 text-sm' style={{ color: '#86868B' }}>
                        <div>• React 19 Components</div>
                        <div>• Apple Design System</div>
                        <div>• Real-time WebSocket</div>
                        <div>• PWA Capabilities</div>
                      </div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.8, delay: 0.4 }}
                      className='text-center'
                    >
                      <div className='mb-4 rounded-xl bg-blue-50 p-6'>
                        <Database className='mx-auto mb-4 h-12 w-12' style={{ color: '#007AFF' }} />
                        <h4 className='mb-2 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                          Backend API
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Node.js + Express
                        </p>
                      </div>
                      <div className='space-y-2 text-sm' style={{ color: '#86868B' }}>
                        <div>• RESTful API Design</div>
                        <div>• PostgreSQL Database</div>
                        <div>• Redis Caching</div>
                        <div>• HIPAA Compliance</div>
                      </div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.8, delay: 0.5 }}
                      className='text-center'
                    >
                      <div className='mb-4 rounded-xl bg-blue-50 p-6'>
                        <Cpu className='mx-auto mb-4 h-12 w-12' style={{ color: '#007AFF' }} />
                        <h4 className='mb-2 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                          AI/ML Engine
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Python + TensorFlow
                        </p>
                      </div>
                      <div className='space-y-2 text-sm' style={{ color: '#86868B' }}>
                        <div>• Multi-Modal Fusion</div>
                        <div>• Real-time Inference</div>
                        <div>• Edge Computing</div>
                        <div>• GPU Acceleration</div>
                      </div>
                    </motion.div>
                  </div>

                  {/* Connection Flow */}
                  <div
                    className='flex items-center justify-center space-x-4 text-2xl'
                    style={{ color: '#007AFF' }}
                  >
                    <ArrowRight />
                    <span className='text-sm font-medium' style={{ color: '#86868B' }}>
                      Data Flow
                    </span>
                    <ArrowRight />
                  </div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Technology Stack Section */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className='mx-auto max-w-7xl'
              >
                <div className='mb-16 text-center'>
                  <h2 className='mb-6 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                    Technology Stack Excellence
                  </h2>
                  <p
                    className='mx-auto max-w-4xl text-xl leading-relaxed'
                    style={{ color: '#86868B' }}
                  >
                    Cutting-edge technologies selected for performance, scalability, and healthcare
                    compliance.
                  </p>
                </div>

                <div className='grid grid-cols-1 gap-8 md:grid-cols-3'>
                  {/* Frontend Technologies */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4 }}
                    className='rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 p-8'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <Code className='h-8 w-8' style={{ color: '#007AFF' }} />
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        Frontend Stack
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Next.js
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v15.4.6
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          TypeScript
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          Latest
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Tailwind CSS
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v3.4
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Bun Runtime
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v1.0+
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Framer Motion
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v11
                        </span>
                      </div>
                    </div>
                  </motion.div>

                  {/* Backend Technologies */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className='rounded-2xl bg-gradient-to-br from-green-50 to-emerald-50 p-8'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <Database className='h-8 w-8' style={{ color: '#007AFF' }} />
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        Backend Stack
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Node.js
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v18+
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Express.js
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v4.18
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          PostgreSQL
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v15
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Redis
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v7
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          WebSocket
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          Real-time
                        </span>
                      </div>
                    </div>
                  </motion.div>

                  {/* AI/ML Technologies */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className='rounded-2xl bg-gradient-to-br from-purple-50 to-pink-50 p-8'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <Cpu className='h-8 w-8' style={{ color: '#007AFF' }} />
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        AI/ML Stack
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          Python
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v3.9+
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          TensorFlow
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v2.15
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          PyTorch
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v2.1
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          OpenAI API
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          GPT-4
                        </span>
                      </div>
                      <div className='flex items-center justify-between'>
                        <span className='font-medium' style={{ color: '#1D1D1F' }}>
                          ONNX Runtime
                        </span>
                        <span
                          className='rounded bg-white px-2 py-1 text-sm'
                          style={{ color: '#007AFF' }}
                        >
                          v1.16
                        </span>
                      </div>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Multi-Modal Features Section */}
          <section style={{ backgroundColor: '#F5F5F7' }} className='py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                className='mx-auto max-w-7xl'
              >
                <div className='mb-16 text-center'>
                  <h2 className='mb-6 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                    Revolutionary Multi-Modal Assessment Platform
                  </h2>
                  <p
                    className='mx-auto max-w-4xl text-xl leading-relaxed'
                    style={{ color: '#86868B' }}
                  >
                    NeuraLens is the world's first comprehensive platform combining four critical
                    assessment modalities in a single, AI-powered solution for unprecedented
                    neurological screening accuracy.
                  </p>
                </div>

                <div className='grid grid-cols-1 gap-12 md:grid-cols-2'>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className='space-y-6'
                  >
                    <div className='flex items-start gap-4'>
                      <div className='flex-shrink-0 rounded-full bg-blue-50 p-3'>
                        <Mic className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <div>
                        <h3 className='mb-2 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                          Speech Pattern Analysis
                        </h3>
                        <p className='mb-3 text-base leading-relaxed' style={{ color: '#86868B' }}>
                          Advanced AI detects subtle voice changes with 95.2% accuracy for
                          Parkinson's detection, 18 months earlier than traditional clinical
                          methods.
                        </p>
                        <div className='flex flex-wrap gap-2'>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Voice Tremor Detection
                          </span>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Linguistic Analysis
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className='space-y-6'
                  >
                    <div className='flex items-start gap-4'>
                      <div className='flex-shrink-0 rounded-full bg-blue-50 p-3'>
                        <Eye className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <div>
                        <h3 className='mb-2 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                          Retinal Imaging Assessment
                        </h3>
                        <p className='mb-3 text-base leading-relaxed' style={{ color: '#86868B' }}>
                          Non-invasive retinal biomarker analysis with 89.3% accuracy for
                          Alzheimer's screening, providing accessible alternative to expensive brain
                          imaging.
                        </p>
                        <div className='flex flex-wrap gap-2'>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Vascular Analysis
                          </span>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Amyloid Detection
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                    className='space-y-6'
                  >
                    <div className='flex items-start gap-4'>
                      <div className='flex-shrink-0 rounded-full bg-blue-50 p-3'>
                        <Activity className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <div>
                        <h3 className='mb-2 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                          Motor Function Evaluation
                        </h3>
                        <p className='mb-3 text-base leading-relaxed' style={{ color: '#86868B' }}>
                          Objective movement analysis with 93.7% correlation to clinical scores,
                          enabling precise tremor detection and gait assessment.
                        </p>
                        <div className='flex flex-wrap gap-2'>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Tremor Analysis
                          </span>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Gait Assessment
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className='space-y-6'
                  >
                    <div className='flex items-start gap-4'>
                      <div className='flex-shrink-0 rounded-full bg-blue-50 p-3'>
                        <Brain className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <div>
                        <h3 className='mb-2 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                          Cognitive Testing Suite
                        </h3>
                        <p className='mb-3 text-base leading-relaxed' style={{ color: '#86868B' }}>
                          Comprehensive cognitive assessment with 91.4% accuracy for MCI detection,
                          featuring adaptive testing and personalized baselines.
                        </p>
                        <div className='flex flex-wrap gap-2'>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Memory Assessment
                          </span>
                          <span
                            className='rounded-full bg-white px-3 py-1 text-xs font-medium'
                            style={{ color: '#1D1D1F' }}
                          >
                            Executive Function
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Performance & Clinical Validation */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className='mx-auto max-w-7xl'
              >
                <div className='mb-16 text-center'>
                  <h2 className='mb-6 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                    Clinical Excellence & Performance Metrics
                  </h2>
                  <p
                    className='mx-auto max-w-4xl text-xl leading-relaxed'
                    style={{ color: '#86868B' }}
                  >
                    Every NeuraLens feature is backed by rigorous clinical validation, peer-reviewed
                    research, and evidence-based medical literature.
                  </p>
                </div>

                <div className='mb-16 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4'>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className='rounded-xl bg-gradient-to-br from-blue-50 to-indigo-50 p-6 text-center'
                  >
                    <div className='mb-2 text-3xl font-bold' style={{ color: '#007AFF' }}>
                      95.2%
                    </div>
                    <p className='mb-1 text-sm font-medium' style={{ color: '#1D1D1F' }}>
                      Parkinson's Detection
                    </p>
                    <p className='text-xs' style={{ color: '#86868B' }}>
                      1,247 participants validated
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                    className='rounded-xl bg-gradient-to-br from-green-50 to-emerald-50 p-6 text-center'
                  >
                    <div className='mb-2 text-3xl font-bold' style={{ color: '#007AFF' }}>
                      89.3%
                    </div>
                    <p className='mb-1 text-sm font-medium' style={{ color: '#1D1D1F' }}>
                      Alzheimer's Screening
                    </p>
                    <p className='text-xs' style={{ color: '#86868B' }}>
                      892 participants validated
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className='rounded-xl bg-gradient-to-br from-purple-50 to-pink-50 p-6 text-center'
                  >
                    <div className='mb-2 text-3xl font-bold' style={{ color: '#007AFF' }}>
                      93.7%
                    </div>
                    <p className='mb-1 text-sm font-medium' style={{ color: '#1D1D1F' }}>
                      Motor Assessment
                    </p>
                    <p className='text-xs' style={{ color: '#86868B' }}>
                      1,156 participants validated
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.9 }}
                    className='rounded-xl bg-gradient-to-br from-orange-50 to-red-50 p-6 text-center'
                  >
                    <div className='mb-2 text-3xl font-bold' style={{ color: '#007AFF' }}>
                      91.4%
                    </div>
                    <p className='mb-1 text-sm font-medium' style={{ color: '#1D1D1F' }}>
                      MCI Detection
                    </p>
                    <p className='text-xs' style={{ color: '#86868B' }}>
                      2,034 participants validated
                    </p>
                  </motion.div>
                </div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: 1.0 }}
                  className='text-center'
                >
                  <div className='grid grid-cols-1 gap-8 md:grid-cols-3'>
                    <div className='flex items-center justify-center gap-3'>
                      <Shield className='h-6 w-6' style={{ color: '#007AFF' }} />
                      <span className='font-medium' style={{ color: '#1D1D1F' }}>
                        HIPAA Compliant
                      </span>
                    </div>
                    <div className='flex items-center justify-center gap-3'>
                      <Award className='h-6 w-6' style={{ color: '#007AFF' }} />
                      <span className='font-medium' style={{ color: '#1D1D1F' }}>
                        Peer-Reviewed Evidence
                      </span>
                    </div>
                    <div className='flex items-center justify-center gap-3'>
                      <Zap className='h-6 w-6' style={{ color: '#007AFF' }} />
                      <span className='font-medium' style={{ color: '#1D1D1F' }}>
                        Real-Time Processing
                      </span>
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            </div>
          </section>

          {/* Future Technology Roadmap */}
          <section style={{ backgroundColor: '#F5F5F7' }} className='py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className='mx-auto max-w-6xl'
              >
                <div className='mb-16 text-center'>
                  <h2 className='mb-6 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                    Future Technology Roadmap
                  </h2>
                  <p
                    className='mx-auto max-w-4xl text-xl leading-relaxed'
                    style={{ color: '#86868B' }}
                  >
                    NeuraLens is positioned to transform global neurological healthcare through
                    strategic technology advancement and worldwide deployment.
                  </p>
                </div>

                <div className='grid grid-cols-1 gap-8 md:grid-cols-3'>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                    className='rounded-2xl bg-white p-8 shadow-sm'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <div className='rounded-full bg-blue-50 p-3'>
                        <Cloud className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        Cloud & Edge Computing
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Global Deployment
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          AWS/Azure multi-region deployment with 99.9% uptime SLA
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Edge Processing
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          GPU acceleration for real-time inference under 2 seconds
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Auto-Scaling
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Kubernetes orchestration handling 10x traffic spikes
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className='rounded-2xl bg-white p-8 shadow-sm'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <div className='rounded-full bg-blue-50 p-3'>
                        <Globe className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        Mobile & Integration
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Native Mobile Apps
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          React Native iOS/Android apps with offline capabilities
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Healthcare APIs
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          FHIR-compliant integration with major EHR systems
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Wearable Integration
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Apple Health, Google Fit, and medical device connectivity
                        </p>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.9 }}
                    className='rounded-2xl bg-white p-8 shadow-sm'
                  >
                    <div className='mb-6 flex items-center gap-3'>
                      <div className='rounded-full bg-blue-50 p-3'>
                        <BarChart3 className='h-8 w-8' style={{ color: '#007AFF' }} />
                      </div>
                      <h3 className='text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                        Regulatory & Compliance
                      </h3>
                    </div>
                    <div className='space-y-4'>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          FDA Approval
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Medical device certification pathway for clinical use
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          International Standards
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          CE marking, Health Canada, and global regulatory compliance
                        </p>
                      </div>
                      <div>
                        <h4 className='mb-2 font-medium' style={{ color: '#1D1D1F' }}>
                          Clinical Trials
                        </h4>
                        <p className='text-sm' style={{ color: '#86868B' }}>
                          Multi-site validation studies with 10,000+ participants
                        </p>
                      </div>
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Contact & Partnership */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.7 }}
                className='mx-auto max-w-4xl space-y-8 text-center'
              >
                <h2 className='text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                  Partner with NeuraLens
                </h2>
                <p className='text-xl leading-relaxed' style={{ color: '#86868B' }}>
                  Join us in revolutionizing neurological healthcare. Whether you're a healthcare
                  professional, researcher, or technology partner, we'd love to collaborate.
                </p>

                <div className='grid grid-cols-1 gap-8 pt-8 md:grid-cols-2'>
                  <div className='rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 p-8'>
                    <h3 className='mb-4 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                      Healthcare Partnerships
                    </h3>
                    <div className='space-y-3 text-left'>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>Clinical validation studies</span>
                      </div>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>EHR system integration</span>
                      </div>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>Research collaboration</span>
                      </div>
                    </div>
                  </div>

                  <div className='rounded-2xl bg-gradient-to-br from-green-50 to-emerald-50 p-8'>
                    <h3 className='mb-4 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                      Technology Integration
                    </h3>
                    <div className='space-y-3 text-left'>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>API partnerships</span>
                      </div>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>White-label solutions</span>
                      </div>
                      <div className='flex items-center gap-3'>
                        <CheckCircle className='h-5 w-5' style={{ color: '#007AFF' }} />
                        <span style={{ color: '#86868B' }}>Custom development</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className='flex flex-col items-center justify-center gap-4 pt-8 sm:flex-row'>
                  <div className='flex items-center gap-2'>
                    <span className='text-lg' style={{ color: '#86868B' }}>
                      General:
                    </span>
                    <span className='text-lg font-medium' style={{ color: '#007AFF' }}>
                      contact@neuralens.ai
                    </span>
                  </div>
                  <div className='flex items-center gap-2'>
                    <span className='text-lg' style={{ color: '#86868B' }}>
                      Partnerships:
                    </span>
                    <span className='text-lg font-medium' style={{ color: '#007AFF' }}>
                      partnerships@neuralens.ai
                    </span>
                  </div>
                </div>

                <div className='pt-6'>
                  <p className='text-base' style={{ color: '#86868B' }}>
                    Ready to transform neurological healthcare? Let's build the future together.
                  </p>
                </div>
              </motion.div>
            </div>
          </section>
        </div>
      </Layout>
    </ErrorBoundary>
  );
}
