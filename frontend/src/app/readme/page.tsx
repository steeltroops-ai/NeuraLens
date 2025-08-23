'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Code, Database, Cpu, Globe } from 'lucide-react';

export default function ReadmePage() {
  return (
    <ErrorBoundary>
      <Layout showHeader={true} showFooter={true} containerized={false}>
        <div className="min-h-screen bg-white">
          {/* Hero Section */}
          <section className="bg-white py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className="mx-auto max-w-4xl text-center space-y-6"
              >
                <h1 className="text-4xl font-bold" style={{ color: '#1D1D1F' }}>
                  NeuraLens Documentation
                </h1>
                <p className="text-xl leading-relaxed text-slate-600">
                  Technical overview and implementation details of the NeuraLens platform
                </p>
              </motion.div>
            </div>
          </section>

          {/* Architecture Section */}
          <section className="bg-gray-50 py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="mx-auto max-w-4xl space-y-6"
              >
                <h2 className="text-3xl font-bold text-center" style={{ color: '#1D1D1F' }}>
                  System Architecture
                </h2>
                <p className="text-xl leading-relaxed text-slate-600 text-center">
                  NeuraLens is built with modern web technologies and AI frameworks 
                  for optimal performance and scalability.
                </p>
              </motion.div>
            </div>
          </section>

          {/* Technology Stack */}
          <section className="bg-white py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className="mx-auto max-w-6xl"
              >
                <h2 className="text-3xl font-bold text-center mb-12" style={{ color: '#1D1D1F' }}>
                  Technology Stack
                </h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4 }}
                    className="text-center space-y-4"
                  >
                    <div className="flex justify-center">
                      <Globe className="h-12 w-12" style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className="text-lg font-semibold" style={{ color: '#1D1D1F' }}>
                      Frontend
                    </h3>
                    <p className="text-sm text-slate-600">
                      Next.js 15, React 19, TypeScript, Tailwind CSS
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className="text-center space-y-4"
                  >
                    <div className="flex justify-center">
                      <Cpu className="h-12 w-12" style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className="text-lg font-semibold" style={{ color: '#1D1D1F' }}>
                      AI Models
                    </h3>
                    <p className="text-sm text-slate-600">
                      TensorFlow.js, ONNX Runtime, MediaPipe
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className="text-center space-y-4"
                  >
                    <div className="flex justify-center">
                      <Database className="h-12 w-12" style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className="text-lg font-semibold" style={{ color: '#1D1D1F' }}>
                      Backend
                    </h3>
                    <p className="text-sm text-slate-600">
                      Supabase, PostgreSQL, Edge Functions
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                    className="text-center space-y-4"
                  >
                    <div className="flex justify-center">
                      <Code className="h-12 w-12" style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className="text-lg font-semibold" style={{ color: '#1D1D1F' }}>
                      Development
                    </h3>
                    <p className="text-sm text-slate-600">
                      Bun, ESLint, Prettier, Playwright
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Features Section */}
          <section className="bg-gray-50 py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className="mx-auto max-w-4xl space-y-6"
              >
                <h2 className="text-3xl font-bold text-center" style={{ color: '#1D1D1F' }}>
                  Key Features
                </h2>
                <div className="space-y-4 text-slate-600">
                  <p className="text-lg">
                    <strong style={{ color: '#1D1D1F' }}>Multi-Modal Assessment:</strong> 
                    Combines speech, retinal, motor, and cognitive analysis
                  </p>
                  <p className="text-lg">
                    <strong style={{ color: '#1D1D1F' }}>Real-Time Processing:</strong> 
                    Client-side inference with sub-100ms latency
                  </p>
                  <p className="text-lg">
                    <strong style={{ color: '#1D1D1F' }}>Privacy-First:</strong> 
                    HIPAA-compliant with local data processing
                  </p>
                  <p className="text-lg">
                    <strong style={{ color: '#1D1D1F' }}>Responsive Design:</strong> 
                    Optimized for desktop, tablet, and mobile devices
                  </p>
                  <p className="text-lg">
                    <strong style={{ color: '#1D1D1F' }}>Accessibility:</strong> 
                    WCAG 2.1 AA compliant with full keyboard navigation
                  </p>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Getting Started */}
          <section className="bg-white py-20">
            <div className="container mx-auto px-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className="mx-auto max-w-4xl space-y-6"
              >
                <h2 className="text-3xl font-bold text-center" style={{ color: '#1D1D1F' }}>
                  Getting Started
                </h2>
                <div className="space-y-4 text-slate-600">
                  <p className="text-lg">
                    1. Clone the repository and install dependencies with Bun
                  </p>
                  <p className="text-lg">
                    2. Configure environment variables for Supabase integration
                  </p>
                  <p className="text-lg">
                    3. Run the development server and navigate to localhost:3000
                  </p>
                  <p className="text-lg">
                    4. Complete the health assessment to experience the full platform
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
