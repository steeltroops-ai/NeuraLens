'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Layout } from '@/components/layout';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Brain, Shield, Zap, Users } from 'lucide-react';

export default function AboutPage() {
  return (
    <ErrorBoundary>
      <Layout showHeader={true} showFooter={true} containerized={false}>
        <div className='min-h-screen bg-white'>
          {/* Hero Section */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
                className='mx-auto max-w-4xl space-y-6 text-center'
              >
                <h1 className='text-4xl font-bold' style={{ color: '#1D1D1F' }}>
                  About MediLens
                </h1>
                <p className='text-xl leading-relaxed text-slate-600'>
                  Pioneering the future of neurological health assessment through advanced AI
                  technology
                </p>
              </motion.div>
            </div>
          </section>

          {/* Mission Section */}
          <section className='bg-gray-50 py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className='mx-auto max-w-4xl space-y-6 text-center'
              >
                <h2 className='text-3xl font-bold' style={{ color: '#1D1D1F' }}>
                  Our Mission
                </h2>
                <p className='text-xl leading-relaxed text-slate-600'>
                  To democratize access to advanced neurological health screening through
                  cutting-edge AI technology, enabling early detection and better health outcomes
                  for millions worldwide.
                </p>
              </motion.div>
            </div>
          </section>

          {/* Values Section */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className='mx-auto max-w-6xl'
              >
                <h2 className='mb-12 text-center text-3xl font-bold' style={{ color: '#1D1D1F' }}>
                  Our Values
                </h2>

                <div className='grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4'>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4 }}
                    className='space-y-4 text-center'
                  >
                    <div className='flex justify-center'>
                      <Brain className='h-12 w-12' style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      Innovation
                    </h3>
                    <p className='text-sm text-slate-600'>
                      Pushing the boundaries of AI-powered healthcare technology
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.5 }}
                    className='space-y-4 text-center'
                  >
                    <div className='flex justify-center'>
                      <Shield className='h-12 w-12' style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      Privacy
                    </h3>
                    <p className='text-sm text-slate-600'>
                      HIPAA-compliant security protecting your health data
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className='space-y-4 text-center'
                  >
                    <div className='flex justify-center'>
                      <Zap className='h-12 w-12' style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      Accuracy
                    </h3>
                    <p className='text-sm text-slate-600'>
                      Clinically validated AI models with 90%+ precision
                    </p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                    className='space-y-4 text-center'
                  >
                    <div className='flex justify-center'>
                      <Users className='h-12 w-12' style={{ color: '#007AFF' }} />
                    </div>
                    <h3 className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                      Accessibility
                    </h3>
                    <p className='text-sm text-slate-600'>
                      Making advanced healthcare accessible to everyone
                    </p>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Technology Section */}
          <section className='bg-gray-50 py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className='mx-auto max-w-4xl space-y-6 text-center'
              >
                <h2 className='text-3xl font-bold' style={{ color: '#1D1D1F' }}>
                  Advanced Technology
                </h2>
                <p className='text-xl leading-relaxed text-slate-600'>
                  Our platform combines speech analysis, retinal imaging, motor assessment, and
                  cognitive evaluation using state-of-the-art machine learning models to provide
                  comprehensive neurological health insights.
                </p>
              </motion.div>
            </div>
          </section>

          {/* Contact Section */}
          <section className='bg-white py-20'>
            <div className='container mx-auto px-4'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className='mx-auto max-w-4xl space-y-6 text-center'
              >
                <h2 className='text-3xl font-bold' style={{ color: '#1D1D1F' }}>
                  Get in Touch
                </h2>
                <p className='text-xl leading-relaxed text-slate-600'>
                  Have questions about MediLens? We'd love to hear from you.
                </p>
                <p className='text-lg text-slate-600'>
                  Email: <span style={{ color: '#007AFF' }}>contact@medilens.ai</span>
                </p>
              </motion.div>
            </div>
          </section>
        </div>
      </Layout>
    </ErrorBoundary>
  );
}
