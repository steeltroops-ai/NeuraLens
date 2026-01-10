import { Activity, Shield, Clock, Zap } from 'lucide-react';
import { Suspense } from 'react';
import { Metadata } from 'next';

import {
  ErrorBoundary,
  NetworkStatus,
  AnatomicalLoadingSkeleton,
} from '@/components/ErrorBoundary';
import { Layout } from '@/components/layout';
import { HomePageClient } from '@/components/pages/HomePageClient';

// Static metadata for SSG
export const metadata: Metadata = {
  title: 'NeuraLens - Early Detection, Better Outcomes',
  description:
    'Transforming neurological health with AI-powered insights for millions. Advanced multi-modal assessment platform for early detection of neurological conditions.',
  keywords: [
    'neurological assessment',
    'AI health',
    'early detection',
    'brain health',
    'medical AI',
  ],
  openGraph: {
    title: 'NeuraLens - Early Detection, Better Outcomes',
    description: 'Transforming neurological health with AI-powered insights for millions.',
    type: 'website',
  },
};

// Server component for static content
export default function HomePage() {
  return (
    <ErrorBoundary>
      <NetworkStatus />
      <Layout showHeader={true} showFooter={false} containerized={false}>
        <div className='min-h-screen bg-white'>
          {/* Static Hero Section - Server Rendered */}
          <section className='relative overflow-hidden bg-white'>
            {/* Subtle Neural Grid Background */}
            <div className='absolute inset-0 opacity-[0.02]'>
              <div
                className='w-full h-full'
                style={{
                  backgroundImage:
                    'radial-gradient(circle at 1px 1px, rgba(0,0,0,0.15) 1px, transparent 0)',
                  backgroundSize: '20px 20px',
                }}
              />
            </div>

            <div className='relative px-4 py-20 mx-auto max-w-7xl sm:px-6 lg:px-8 lg:py-32'>
              <div className='text-center'>
                <div className='mb-16 space-y-8'>
                  <h1 className='text-5xl font-bold leading-tight text-slate-900 sm:text-6xl lg:text-7xl'>
                    <span style={{ color: '#1D1D1F' }}>Neuralens</span>
                  </h1>
                  <p className='text-2xl font-medium text-slate-700 sm:text-3xl'>
                    Early Detection, Better Outcomes.
                  </p>
                  <p className='max-w-4xl mx-auto text-lg leading-relaxed text-slate-600 sm:text-xl'>
                    Transforming neurological health with AI-powered insights for millions.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Client-side Interactive Components */}
          <Suspense fallback={<AnatomicalLoadingSkeleton />}>
            <HomePageClient />
          </Suspense>
        </div>
      </Layout>
    </ErrorBoundary>
  );
}
