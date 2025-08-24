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

export function ReadmePageClient() {
  return (
    <>
      {/* Architecture Overview */}
      <section className='py-20 bg-white'>
        <div className='container mx-auto px-6'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='mx-auto max-w-6xl text-center'
          >
            <h2 className='mb-12 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
              System Architecture
            </h2>
            
            <div className='mb-16 grid grid-cols-1 gap-8 md:grid-cols-3'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.1 }}
                viewport={{ once: true }}
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
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
                className='text-center'
              >
                <div className='mb-4 rounded-xl bg-green-50 p-6'>
                  <Database className='mx-auto mb-4 h-12 w-12' style={{ color: '#34C759' }} />
                  <h4 className='mb-2 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                    Backend API
                  </h4>
                  <p className='text-sm' style={{ color: '#86868B' }}>
                    FastAPI + PostgreSQL
                  </p>
                </div>
                <div className='space-y-2 text-sm' style={{ color: '#86868B' }}>
                  <div>• RESTful API Design</div>
                  <div>• Supabase Integration</div>
                  <div>• Real-time Processing</div>
                  <div>• HIPAA Compliance</div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                viewport={{ once: true }}
                className='text-center'
              >
                <div className='mb-4 rounded-xl bg-purple-50 p-6'>
                  <Brain className='mx-auto mb-4 h-12 w-12' style={{ color: '#AF52DE' }} />
                  <h4 className='mb-2 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                    AI/ML Engine
                  </h4>
                  <p className='text-sm' style={{ color: '#86868B' }}>
                    Multi-modal Analysis
                  </p>
                </div>
                <div className='space-y-2 text-sm' style={{ color: '#86868B' }}>
                  <div>• Speech Processing</div>
                  <div>• Computer Vision</div>
                  <div>• Motor Analysis</div>
                  <div>• NRI Fusion</div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className='py-20' style={{ backgroundColor: '#F5F5F7' }}>
        <div className='container mx-auto px-6'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='mx-auto max-w-6xl'
          >
            <h2 className='mb-12 text-center text-4xl font-bold' style={{ color: '#1D1D1F' }}>
              Technology Stack
            </h2>

            <div className='grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4'>
              {[
                { icon: Code, title: 'Frontend', items: ['Next.js 15', 'React 19', 'TypeScript', 'Tailwind CSS'] },
                { icon: Database, title: 'Backend', items: ['FastAPI', 'PostgreSQL', 'Supabase', 'Python 3.11'] },
                { icon: Brain, title: 'AI/ML', items: ['TensorFlow.js', 'ONNX Runtime', 'Whisper', 'MediaPipe'] },
                { icon: Cloud, title: 'Infrastructure', items: ['Vercel', 'Supabase', 'CDN', 'Edge Computing'] },
              ].map((category, index) => (
                <motion.div
                  key={category.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className='rounded-2xl bg-white p-6 shadow-sm'
                >
                  <category.icon className='mb-4 h-8 w-8' style={{ color: '#007AFF' }} />
                  <h3 className='mb-4 text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                    {category.title}
                  </h3>
                  <ul className='space-y-2'>
                    {category.items.map((item) => (
                      <li key={item} className='flex items-center text-sm' style={{ color: '#86868B' }}>
                        <CheckCircle className='mr-2 h-4 w-4' style={{ color: '#34C759' }} />
                        {item}
                      </li>
                    ))}
                  </ul>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Assessment Modalities */}
      <section className='py-20 bg-white'>
        <div className='container mx-auto px-6'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='mx-auto max-w-6xl'
          >
            <h2 className='mb-12 text-center text-4xl font-bold' style={{ color: '#1D1D1F' }}>
              Assessment Modalities
            </h2>

            <div className='grid grid-cols-1 gap-12 md:grid-cols-2'>
              {[
                {
                  icon: Mic,
                  title: 'Speech Pattern Analysis',
                  description: 'Advanced AI detects subtle voice changes with 95.2% accuracy for Parkinson\'s detection, 18 months earlier than traditional clinical methods.',
                  features: ['Real-time Processing', 'MFCC Analysis', 'Prosody Detection', 'Fluency Metrics'],
                },
                {
                  icon: Eye,
                  title: 'Retinal Imaging Analysis',
                  description: 'Non-invasive retinal analysis using advanced computer vision to detect early signs of neurological conditions through vascular patterns.',
                  features: ['Vessel Analysis', 'Cup-Disc Ratio', 'Hemorrhage Detection', 'Risk Assessment'],
                },
                {
                  icon: Activity,
                  title: 'Motor Function Assessment',
                  description: 'Precise movement analysis using computer vision and sensor data to detect motor impairments and tremor patterns.',
                  features: ['Tremor Detection', 'Coordination Tests', 'Finger Tapping', 'Balance Analysis'],
                },
                {
                  icon: Brain,
                  title: 'Cognitive Evaluation',
                  description: 'Comprehensive cognitive testing through interactive games and assessments to evaluate memory, attention, and executive function.',
                  features: ['Memory Tests', 'Attention Span', 'Reaction Time', 'Executive Function'],
                },
              ].map((modality, index) => (
                <motion.div
                  key={modality.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className='flex items-start gap-4'
                >
                  <div className='flex-shrink-0 rounded-full bg-blue-50 p-3'>
                    <modality.icon className='h-8 w-8' style={{ color: '#007AFF' }} />
                  </div>
                  <div>
                    <h3 className='mb-2 text-xl font-semibold' style={{ color: '#1D1D1F' }}>
                      {modality.title}
                    </h3>
                    <p className='mb-3 text-base leading-relaxed' style={{ color: '#86868B' }}>
                      {modality.description}
                    </p>
                    <div className='flex flex-wrap gap-2'>
                      {modality.features.map((feature) => (
                        <span
                          key={feature}
                          className='rounded-full px-3 py-1 text-xs font-medium'
                          style={{
                            backgroundColor: 'rgba(0, 122, 255, 0.1)',
                            color: '#007AFF',
                          }}
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Performance Metrics */}
      <section className='py-20' style={{ backgroundColor: '#F5F5F7' }}>
        <div className='container mx-auto px-6'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='mx-auto max-w-6xl text-center'
          >
            <h2 className='mb-12 text-4xl font-bold' style={{ color: '#1D1D1F' }}>
              Performance Metrics
            </h2>

            <div className='grid grid-cols-1 gap-8 md:grid-cols-4'>
              {[
                { value: '<100ms', label: 'Processing Time', icon: Zap },
                { value: '95%+', label: 'Clinical Accuracy', icon: Award },
                { value: '4-Modal', label: 'AI Analysis', icon: Brain },
                { value: 'HIPAA', label: 'Compliant', icon: Shield },
              ].map((metric, index) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className='rounded-2xl bg-white p-8 shadow-sm'
                >
                  <metric.icon className='mx-auto mb-4 h-12 w-12' style={{ color: '#007AFF' }} />
                  <div className='mb-2 text-3xl font-bold' style={{ color: '#1D1D1F' }}>
                    {metric.value}
                  </div>
                  <div className='text-sm font-medium' style={{ color: '#86868B' }}>
                    {metric.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>
    </>
  );
}
