/**
 * NeuroLens-X Root Layout
 * Global layout with Neuro-Minimalist styling and PWA support
 */

import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

// Optimized font loading
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: {
    default: "NeuroLens-X | Neurological Risk Assessment Platform",
    template: "%s | NeuroLens-X",
  },
  description:
    "Advanced multi-modal neurological risk assessment platform using AI-powered analysis of speech, retinal, motor, and cognitive biomarkers.",
  keywords: [
    "neurological assessment",
    "AI healthcare",
    "medical screening",
    "biomarker analysis",
    "speech analysis",
    "retinal screening",
    "motor assessment",
    "cognitive evaluation",
  ],
  authors: [{ name: "NeuroLens Team" }],
  creator: "NeuroLens-X",
  publisher: "NeuroLens-X",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL("https://neurolens-x.vercel.app"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://neurolens-x.vercel.app",
    title: "NeuroLens-X | Neurological Risk Assessment Platform",
    description:
      "Advanced multi-modal neurological risk assessment platform using AI-powered analysis.",
    siteName: "NeuroLens-X",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NeuroLens-X Platform",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroLens-X | Neurological Risk Assessment Platform",
    description:
      "Advanced multi-modal neurological risk assessment platform using AI-powered analysis.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  verification: {
    google: "your-google-verification-code",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#3b82f6" },
    { media: "(prefers-color-scheme: dark)", color: "#1e3a8a" },
  ],
  colorScheme: "light",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={cn(inter.variable, "scroll-smooth")}>
      <head>
        {/* Preload critical resources */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin=""
        />

        {/* PWA manifest */}
        <link rel="manifest" href="/manifest.json" />

        {/* Apple PWA meta tags */}
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="NeuroLens-X" />
        <link rel="apple-touch-icon" href="/icon-192x192.png" />

        {/* Microsoft PWA meta tags */}
        <meta name="msapplication-TileColor" content="#1e3a8a" />
        <meta name="msapplication-config" content="/browserconfig.xml" />

        {/* Favicon */}
        <link rel="icon" type="image/x-icon" href="/favicon.ico" />
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

        {/* Performance hints */}
        <link rel="dns-prefetch" href="//api.neurolens-x.com" />
        <link rel="preconnect" href="https://api.neurolens-x.com" />
      </head>
      <body
        className={cn(
          "min-h-screen bg-gradient-to-br from-neutral-50/80 via-primary-50/60 to-secondary-50/80",
          "font-sans antialiased",
          "neural-grid-primary",
          // Accessibility improvements
          "focus-within:outline-none",
          // Performance optimizations
          "will-change-scroll"
        )}
      >
        {/* Premium Skip to main content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50 glass-medium text-primary-900 px-6 py-3 rounded-xl font-medium shadow-neural-md transition-all duration-300"
        >
          Skip to main content
        </a>

        {/* Main application content */}
        <div id="main-content" className="relative min-h-screen">
          {children}
        </div>

        {/* Service Worker Registration - Disabled for development */}
        {process.env.NODE_ENV === "production" && (
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
        )}

        {/* Performance monitoring */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Core Web Vitals monitoring
              function sendToAnalytics(metric) {
                // Replace with your analytics endpoint
                console.log('Performance metric:', metric);
              }

              // Monitor LCP, FID, CLS (disabled for development)
              if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
                try {
                  import('web-vitals').then(({ getCLS, getFID, getLCP }) => {
                    getCLS(sendToAnalytics);
                    getFID(sendToAnalytics);
                    getLCP(sendToAnalytics);
                  }).catch(err => console.log('Web vitals not available:', err));
                } catch (e) {
                  console.log('Web vitals import failed:', e);
                }
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
