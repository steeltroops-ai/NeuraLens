import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: {
    default: 'NeuroLens-X | Neurological Risk Assessment',
    template: '%s | NeuroLens-X',
  },
  description:
    'Multi-modal neurological risk assessment platform for early detection of neurological disorders through AI-powered speech, retinal, and risk analysis.',
  keywords: [
    'neurology',
    'neurological assessment',
    'early detection',
    'AI healthcare',
    'speech analysis',
    'retinal imaging',
    'risk assessment',
    'dementia screening',
    "Parkinson's detection",
    'cognitive health',
  ],
  authors: [{ name: 'NeuroLens-X Team' }],
  creator: 'NeuroLens-X',
  publisher: 'NeuroLens-X',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://neurolens-x.com'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://neurolens-x.com',
    siteName: 'NeuroLens-X',
    title: 'NeuroLens-X | Neurological Risk Assessment',
    description:
      'Multi-modal neurological risk assessment platform for early detection of neurological disorders.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'NeuroLens-X - Neurological Risk Assessment Platform',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuroLens-X | Neurological Risk Assessment',
    description:
      'Multi-modal neurological risk assessment platform for early detection of neurological disorders.',
    images: ['/twitter-image.png'],
    creator: '@neurolens_x',
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
    yandex: 'your-yandex-verification-code',
    yahoo: 'your-yahoo-verification-code',
  },
  category: 'healthcare',
  classification: 'Medical Technology',
  referrer: 'origin-when-cross-origin',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#3B82F6' },
    { media: '(prefers-color-scheme: dark)', color: '#3B82F6' },
  ],
  colorScheme: 'dark light',
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        {/* PWA Meta Tags */}
        <link rel="manifest" href="/manifest.json" />
        <meta name="application-name" content="NeuroLens-X" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="NeuroLens-X" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="msapplication-config" content="/browserconfig.xml" />
        <meta name="msapplication-TileColor" content="#3B82F6" />
        <meta name="msapplication-tap-highlight" content="no" />

        {/* Apple Touch Icons */}
        <link rel="apple-touch-icon" href="/icons/icon-152x152.png" />
        <link
          rel="apple-touch-icon"
          sizes="152x152"
          href="/icons/icon-152x152.png"
        />
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/icons/icon-180x180.png"
        />

        {/* Favicons */}
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href="/favicon-32x32.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href="/favicon-16x16.png"
        />
        <link rel="shortcut icon" href="/favicon.ico" />

        {/* Preconnect to external domains */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />

        {/* DNS Prefetch */}
        <link rel="dns-prefetch" href="https://fonts.googleapis.com" />
        <link rel="dns-prefetch" href="https://fonts.gstatic.com" />

        {/* Preload critical resources */}
        <link
          rel="preload"
          href="/fonts/inter-var.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />

        {/* Security Headers */}
        <meta httpEquiv="X-Content-Type-Options" content="nosniff" />
        <meta httpEquiv="X-XSS-Protection" content="1; mode=block" />
        <meta
          httpEquiv="Referrer-Policy"
          content="strict-origin-when-cross-origin"
        />

        {/* Performance Hints */}
        <meta httpEquiv="Accept-CH" content="DPR, Viewport-Width, Width" />

        {/* Structured Data */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'MedicalWebPage',
              name: 'NeuroLens-X',
              description: 'Multi-modal neurological risk assessment platform',
              url: 'https://neurolens-x.com',
              medicalAudience: {
                '@type': 'MedicalAudience',
                audienceType: 'Patient',
              },
              about: {
                '@type': 'MedicalCondition',
                name: 'Neurological Disorders',
              },
              provider: {
                '@type': 'Organization',
                name: 'NeuroLens-X',
                url: 'https://neurolens-x.com',
              },
            }),
          }}
        />
      </head>
      <body className="min-h-screen bg-surface-background text-text-primary antialiased">
        {/* Skip Links for Accessibility */}
        <a
          href="#main-content"
          className="skip-link sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-primary-500 focus:px-4 focus:py-2 focus:text-white focus:shadow-lg"
        >
          Skip to main content
        </a>
        <a
          href="#main-navigation"
          className="skip-link sr-only focus:not-sr-only focus:absolute focus:left-32 focus:top-4 focus:z-50 focus:rounded-md focus:bg-primary-500 focus:px-4 focus:py-2 focus:text-white focus:shadow-lg"
        >
          Skip to navigation
        </a>

        {/* Main Application */}
        <div id="root" className="relative">
          {children}
        </div>

        {/* Live Regions for Screen Readers */}
        <div
          id="live-region-announcements"
          aria-live="polite"
          aria-atomic="true"
          className="sr-only"
        />
        <div
          id="live-region-alerts"
          aria-live="assertive"
          aria-atomic="true"
          className="sr-only"
        />

        {/* PWA Install Prompt */}
        <div id="pwa-install-prompt" className="pwa-install-prompt" />

        {/* Service Worker Registration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js')
                    .then(function(registration) {
                      console.log('SW registered: ', registration);
                    })
                    .catch(function(registrationError) {
                      console.log('SW registration failed: ', registrationError);
                    });
                });
              }
            `,
          }}
        />

        {/* Accessibility Initialization */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Initialize accessibility features
              (function() {
                // Detect reduced motion preference
                if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                  document.documentElement.classList.add('reduce-motion');
                }
                
                // Detect high contrast preference
                if (window.matchMedia('(prefers-contrast: high)').matches) {
                  document.documentElement.classList.add('high-contrast');
                }
                
                // Focus management for keyboard users
                let hadKeyboardEvent = true;
                const keyboardThrottleTimeout = 100;
                
                function handleKeyboardEvent(e) {
                  if (e.metaKey || e.altKey || e.ctrlKey) {
                    return;
                  }
                  hadKeyboardEvent = true;
                }
                
                function handlePointerEvent() {
                  hadKeyboardEvent = false;
                  setTimeout(() => {
                    if (!hadKeyboardEvent) {
                      document.body.classList.remove('keyboard-user');
                    }
                  }, keyboardThrottleTimeout);
                }
                
                document.addEventListener('keydown', handleKeyboardEvent, true);
                document.addEventListener('mousedown', handlePointerEvent, true);
                document.addEventListener('pointerdown', handlePointerEvent, true);
                document.addEventListener('touchstart', handlePointerEvent, true);
                
                // Add keyboard user class when tab is pressed
                document.addEventListener('keydown', function(e) {
                  if (e.key === 'Tab') {
                    document.body.classList.add('keyboard-user');
                  }
                });
              })();
            `,
          }}
        />
      </body>
    </html>
  );
}
