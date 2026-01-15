import { Metadata } from 'next';
import { Suspense } from 'react';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { Layout } from '@/components/layout';
import { HomePageClient } from '@/components/pages/HomePageClient';

// Static metadata for optimal SEO - MediLens Branding
export const metadata: Metadata = {
  title: 'MediLens - AI-Powered Medical Diagnostics Platform',
  description:
    'MediLens is a centralized platform where doctors and patients can access multiple AI-powered diagnostic tools for various medical conditions. Specialized, validated, and easy to use.',
  keywords: [
    'medical diagnostics',
    'AI healthcare',
    'MediLens',
    'speech analysis',
    'retinal imaging',
    'motor assessment',
    'cognitive testing',
    'medical AI',
    'diagnostic platform',
    'healthcare technology',
    'clinical diagnostics',
    'AI-powered diagnosis',
  ],
  authors: [{ name: 'MediLens Team' }],
  creator: 'MediLens',
  publisher: 'MediLens',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://medilens.ai'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'MediLens - AI-Powered Medical Diagnostics Platform',
    description:
      'A centralized platform for AI-powered diagnostic tools. Access specialized, validated diagnostic modules for various medical conditions.',
    url: 'https://medilens.ai',
    siteName: 'MediLens',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'MediLens - AI-Powered Medical Diagnostics Platform',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'MediLens - AI-Powered Medical Diagnostics Platform',
    description:
      'A centralized platform for AI-powered diagnostic tools. Access specialized, validated diagnostic modules for various medical conditions.',
    images: ['/og-image.jpg'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
};

// Loading component for Suspense
function HomePageLoading() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-12 h-12 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
        <p className="text-[14px] text-slate-300 font-medium">Loading MediLens...</p>
      </div>
    </div>
  );
}

// Main page component - Server-side rendered
export default function HomePage() {
  return (
    <Layout containerized={false}>
      <ErrorBoundary>
        <Suspense fallback={<HomePageLoading />}>
          <HomePageClient />
        </Suspense>
      </ErrorBoundary>
    </Layout>
  );
}

// Enable static generation for optimal performance
export const dynamic = 'force-static';
export const revalidate = 3600; // Revalidate every hour
