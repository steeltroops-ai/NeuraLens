import { Metadata } from 'next';
import { Suspense } from 'react';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { Layout } from '@/components/layout';
import { HomePageClient } from '@/components/pages/HomePageClient';

// Static metadata for optimal SEO
export const metadata: Metadata = {
  title: 'NeuraLens - Advanced Neurological Assessment Platform',
  description:
    'Revolutionary AI-powered neurological assessment platform. Early detection through multi-modal analysis of speech, retinal imaging, motor function, and cognitive patterns.',
  keywords: [
    'neurological assessment',
    'AI healthcare',
    'early detection',
    'speech analysis',
    'retinal imaging',
    'motor function',
    'cognitive assessment',
    'Parkinson\'s detection',
    'neurodegenerative diseases',
    'medical AI',
  ],
  authors: [{ name: 'NeuraLens Team' }],
  creator: 'NeuraLens',
  publisher: 'NeuraLens',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://neuralens.ai'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'NeuraLens - Advanced Neurological Assessment Platform',
    description:
      'Revolutionary AI-powered neurological assessment platform for early detection and better outcomes.',
    url: 'https://neuralens.ai',
    siteName: 'NeuraLens',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'NeuraLens - Advanced Neurological Assessment Platform',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuraLens - Advanced Neurological Assessment Platform',
    description:
      'Revolutionary AI-powered neurological assessment platform for early detection and better outcomes.',
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
        <p className="text-slate-600 font-medium">Loading NeuraLens...</p>
      </div>
    </div>
  );
}

// Main page component - Server-side rendered
export default function HomePage() {
  return (
    <Layout>
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
