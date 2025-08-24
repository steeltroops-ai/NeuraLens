'use client';

import { motion } from 'framer-motion';
import { Activity, Shield, Clock, Zap } from 'lucide-react';
import { lazy, Suspense } from 'react';
import { useSafeNavigation } from '@/components/SafeNavigation';
import { ErrorBoundary, AnatomicalLoadingSkeleton } from '@/components/ErrorBoundary';

// Lazy load heavy visual components for optimal performance
const SpeechWaveform = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.SpeechWaveform })),
);
const RetinalEye = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.RetinalEye })),
);
const HandKinematics = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.HandKinematics })),
);
const BrainNeural = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.BrainNeural })),
);
const NRIFusion = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.NRIFusion })),
);
const MultiModalNetwork = lazy(() =>
  import('@/components/visuals/SimpleGeometry').then(m => ({ default: m.MultiModalNetwork })),
);

export function HomePageClient() {
  const { navigate, preload } = useSafeNavigation();

  const handleStartAssessment = () => {
    navigate('/assessment');
  };

  const handleOpenDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <>
      {/* Interactive CTA Buttons */}
      <section className='py-12 bg-white'>
        <div className='container px-4 mx-auto'>
          <div className='flex flex-col items-center justify-center gap-4 sm:flex-row'>
            <motion.button
              onClick={handleStartAssessment}
              className='px-8 py-4 text-lg font-semibold text-white transition-all duration-150 hover:scale-98 rounded-xl focus:ring-2 focus:ring-blue-500 active:scale-95'
              style={{
                backgroundColor: '#007AFF',
                backdropFilter: 'blur(20px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              }}
              whileHover={{ scale: 0.98 }}
              whileTap={{ scale: 0.95 }}
              onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#0056CC';
                preload('/assessment'); // Prefetch on hover for instant navigation
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#007AFF';
              }}
              onFocus={() => preload('/assessment')}
            >
              Start Assessment
            </motion.button>

            <motion.button
              onClick={handleOpenDashboard}
              className='px-8 py-4 text-lg font-semibold transition-all duration-150 hover:scale-98 rounded-xl focus:ring-2 focus:ring-blue-500 active:scale-95'
              style={{
                backgroundColor: 'rgba(142, 142, 147, 0.2)',
                color: '#1D1D1F',
                backdropFilter: 'blur(20px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              }}
              whileHover={{ scale: 0.98 }}
              whileTap={{ scale: 0.95 }}
              onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = 'rgba(142, 142, 147, 0.3)';
                preload('/dashboard'); // Prefetch on hover for instant navigation
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = 'rgba(142, 142, 147, 0.2)';
              }}
              onFocus={() => preload('/dashboard')}
            >
              Dashboard
            </motion.button>
          </div>
        </div>
      </section>

      {/* Overview Section */}
      <section className='py-20 bg-white'>
        <div className='container px-4 mx-auto'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='max-w-6xl mx-auto'
          >
            <div className='grid grid-cols-1 gap-12 lg:grid-cols-3'>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.1 }}
                viewport={{ once: true }}
                className='space-y-4 text-center'
              >
                <h2 className='text-2xl font-bold' style={{ color: '#1D1D1F' }}>
                  What is NeuraLens?
                </h2>
                <p className='text-lg leading-relaxed text-slate-600'>
                  NeuraLens integrates advanced AI powered multi-modal analysis combining speech
                  patterns, retinal imaging, motor function assessment, and cognitive evaluation.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
                className='space-y-4 text-center'
              >
                <h2 className='text-2xl font-bold' style={{ color: '#1D1D1F' }}>
                  The Impact
                </h2>
                <p className='text-lg leading-relaxed text-slate-600'>
                  Neurological conditions affect 1 in 6 adults globally, representing over 1 billion
                  people worldwide. Early detection through advanced screening reduces healthcare
                  costs by 40% and significantly improves patient outcomes.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                viewport={{ once: true }}
                className='space-y-4 text-center'
              >
                <h2 className='text-2xl font-bold' style={{ color: '#1D1D1F' }}>
                  Advanced Features
                </h2>
                <p className='text-lg leading-relaxed text-slate-600'>
                  Our platform employs state of the art ML algorithms for comprehensive neurological
                  assessment. Real-time speech analysis, retinal imaging, precise motor function
                  evaluation, and cognitive testing.
                </p>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Assessment Modalities - Lazy Loaded */}
      <section className='relative py-24 bg-white'>
        <div className='px-8 mx-auto max-w-7xl'>
          <div className='grid items-center grid-cols-1 gap-16 lg:grid-cols-2'>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className='space-y-6'
            >
              <h2 className='mb-6 text-4xl font-bold text-slate-900'>Speech Analysis</h2>
              <div className='space-y-4 text-slate-600'>
                <p>
                  <span className='font-semibold text-slate-900'>Model:</span> Whisper-tiny (OpenAI)
                  optimized for neurological biomarker extraction
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Accuracy:</span> 90%+ validated on
                  DementiaBank dataset
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Latency:</span> &lt;100ms real-time
                  processing with client-side inference
                </p>
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className='relative h-64'
            >
              <ErrorBoundary fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                <Suspense fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                  <SpeechWaveform />
                </Suspense>
              </ErrorBoundary>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Retinal Analysis */}
      <section className='relative py-24' style={{ backgroundColor: '#F5F5F7' }}>
        <div className='px-8 mx-auto max-w-7xl'>
          <div className='grid items-center grid-cols-1 gap-16 lg:grid-cols-2'>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className='relative h-64'
            >
              <ErrorBoundary fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                <Suspense fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                  <RetinalEye />
                </Suspense>
              </ErrorBoundary>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className='space-y-6'
            >
              <h2 className='mb-6 text-4xl font-bold text-slate-900'>Retinal Analysis</h2>
              <div className='space-y-4 text-slate-600'>
                <p>
                  <span className='font-semibold text-slate-900'>Model:</span> EfficientNet-B0
                  fine-tuned on APTOS 2019 diabetic retinopathy dataset
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Accuracy:</span> 85%+ precision on
                  retinal biomarker detection
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Latency:</span> &lt;150ms with GPU
                  acceleration and intelligent caching
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Motor Assessment */}
      <section className='relative py-24 bg-white'>
        <div className='px-8 mx-auto max-w-7xl'>
          <div className='grid items-center grid-cols-1 gap-16 lg:grid-cols-2'>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className='space-y-6'
            >
              <h2 className='mb-6 text-4xl font-bold text-slate-900'>Motor Assessment</h2>
              <div className='space-y-4 text-slate-600'>
                <p>
                  <span className='font-semibold text-slate-900'>Model:</span> LSTM-based tremor
                  analysis with MediaPipe hand tracking
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Accuracy:</span> 88%+ on
                  Parkinson's disease detection benchmarks
                </p>
                <p>
                  <span className='font-semibold text-slate-900'>Latency:</span> &lt;50ms real-time
                  movement analysis with webcam input
                </p>
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className='relative h-64'
            >
              <ErrorBoundary fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                <Suspense fallback={<AnatomicalLoadingSkeleton className='h-64' />}>
                  <HandKinematics />
                </Suspense>
              </ErrorBoundary>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className='py-20 bg-white'>
        <div className='container px-4 mx-auto'>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className='max-w-4xl mx-auto space-y-6 text-center'
          >
            <h2 className='text-4xl font-bold' style={{ color: '#1D1D1F' }}>
              Take Control of Your Health
            </h2>
            <p className='text-xl leading-relaxed text-slate-600'>
              Start your journey with NeuraLens today and gain insights into your neurological
              well-being with cutting-edge AI technology.
            </p>
            <motion.button
              onClick={handleStartAssessment}
              className='px-8 py-4 text-lg font-semibold text-white transition-all duration-150 rounded-lg hover:scale-98 focus:outline-none focus:ring-2 focus:ring-blue-500'
              style={{
                backgroundColor: '#007AFF',
                backdropFilter: 'blur(20px)',
              }}
              whileHover={{ scale: 0.98 }}
              whileTap={{ scale: 0.95 }}
              onMouseEnter={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#0056CC';
                preload('/assessment'); // Prefetch on hover for instant navigation
              }}
              onMouseLeave={(e: React.MouseEvent<HTMLButtonElement>) => {
                e.currentTarget.style.backgroundColor = '#007AFF';
              }}
              onFocus={() => preload('/assessment')}
            >
              Start Health Check
            </motion.button>
          </motion.div>
        </div>
      </section>
    </>
  );
}
