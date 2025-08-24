import { Metadata } from 'next';
import { Suspense } from 'react';
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

import { ErrorBoundary, AnatomicalLoadingSkeleton } from '@/components/ErrorBoundary';
import { Layout } from '@/components/layout';
import { ReadmePageClient } from '@/components/pages/ReadmePageClient';

// Static metadata for SSG optimization
export const metadata: Metadata = {
  title: 'NeuraLens Technical Documentation - Advanced AI Healthcare Platform',
  description:
    'Comprehensive technical documentation for NeuraLens multi-modal neurological assessment platform. Built with Next.js 15, TypeScript, and advanced AI/ML technologies.',
  keywords: [
    'technical documentation',
    'AI healthcare',
    'Next.js',
    'TypeScript',
    'machine learning',
    'neurological assessment',
  ],
  openGraph: {
    title: 'NeuraLens Technical Documentation',
    description: 'Advanced AI healthcare platform technical specifications and architecture.',
    type: 'website',
  },
};

// Server component for optimal SSG performance
export default function ReadmePage() {
  return (
    <ErrorBoundary>
      <Layout showHeader={true} showFooter={true} containerized={false}>
        <div className='min-h-screen bg-white'>
          {/* Static Hero Section - Server Rendered */}
          <section className='bg-white py-24 md:py-32'>
            <div className='container mx-auto px-6 md:px-8'>
              <div className='mx-auto max-w-7xl text-center'>
                <div className='mb-12'>
                  <h1 className='mb-8 text-5xl font-bold leading-tight text-slate-900 sm:text-6xl lg:text-7xl'>
                    <span style={{ color: '#1D1D1F' }}>Technical Documentation</span>
                  </h1>
                  <p className='text-2xl font-medium text-slate-700 sm:text-3xl'>
                    Advanced AI Healthcare Platform
                  </p>
                  <p className='mx-auto mt-6 max-w-4xl text-lg leading-relaxed text-slate-600 sm:text-xl'>
                    Comprehensive technical specifications for NeuraLens multi-modal neurological
                    assessment platform.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Client-side Interactive Components - Lazy loaded */}
          <Suspense fallback={<AnatomicalLoadingSkeleton />}>
            <ReadmePageClient />
          </Suspense>
        </div>
      </Layout>
    </ErrorBoundary>
  );
}
