import { Inter } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import { PatientProvider } from "@/context/PatientContext";

import type { Metadata, Viewport } from "next";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: {
    default: "MediLens | AI-Powered Medical Diagnostics",
    template: "%s | MediLens",
  },
  description:
    "AI-powered medical diagnostics platform providing multiple diagnostic tools for various medical conditions - all in one place.",
  keywords: [
    "medical diagnostics",
    "AI healthcare",
    "early detection",
    "speech analysis",
    "retinal imaging",
    "motor assessment",
    "cognitive testing",
    "Parkinson's detection",
    "diabetic retinopathy",
    "neurological screening",
  ],
  authors: [{ name: "MediLens Team" }],
  creator: "MediLens",
  publisher: "MediLens",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL("https://medilens.com"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://medilens.com",
    siteName: "MediLens",
    title: "MediLens | AI-Powered Medical Diagnostics",
    description:
      "Centralized web platform providing multiple AI-powered diagnostic tools for various medical conditions.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "MediLens - AI-Powered Medical Diagnostics Platform",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "MediLens | AI-Powered Medical Diagnostics",
    description:
      "Centralized web platform providing multiple AI-powered diagnostic tools for various medical conditions.",
    images: ["/twitter-image.png"],
    creator: "@medilens_ai",
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
    yandex: "your-yandex-verification-code",
    yahoo: "your-yahoo-verification-code",
  },
  category: "healthcare",
  classification: "Medical Technology",
  referrer: "origin-when-cross-origin",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#3B82F6" },
    { media: "(prefers-color-scheme: dark)", color: "#3B82F6" },
  ],
  colorScheme: "dark light",
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <ClerkProvider>
      <PatientProvider>
        <html
          lang="en"
          className={inter.variable}
          data-scroll-behavior="smooth"
          suppressHydrationWarning
        >
          <head>
            {/* PWA Meta Tags */}
            <link rel="manifest" href="/manifest.json" />
            <meta name="application-name" content="MediLens" />
            <meta name="apple-mobile-web-app-capable" content="yes" />
            <meta
              name="apple-mobile-web-app-status-bar-style"
              content="default"
            />
            <meta name="apple-mobile-web-app-title" content="MediLens" />
            <meta name="mobile-web-app-capable" content="yes" />
            <meta name="msapplication-config" content="/browserconfig.xml" />
            <meta name="msapplication-TileColor" content="#3B82F6" />
            <meta name="msapplication-tap-highlight" content="no" />

            {/* Apple Touch Icons */}
            <link
              rel="apple-touch-icon"
              sizes="180x180"
              href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2'/></svg>"
            />

            {/* Favicons */}
            <link
              rel="icon"
              type="image/svg+xml"
              href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ef4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2'/></svg>"
            />

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
            {/* Font preloading removed - using Google Fonts instead */}

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
                  "@context": "https://schema.org",
                  "@type": "MedicalWebPage",
                  name: "MediLens",
                  description:
                    "AI-powered medical diagnostics platform providing multiple diagnostic tools for various medical conditions",
                  url: "https://medilens.com",
                  medicalAudience: {
                    "@type": "MedicalAudience",
                    audienceType: "Patient",
                  },
                  about: {
                    "@type": "MedicalCondition",
                    name: "Medical Diagnostics",
                  },
                  provider: {
                    "@type": "Organization",
                    name: "MediLens",
                    url: "https://medilens.com",
                  },
                }),
              }}
            />
          </head>
          <body
            className="bg-surface-background min-h-screen text-text-primary antialiased"
            suppressHydrationWarning
          >
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

            {/* Service Worker Registration - Production Only */}
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
      </PatientProvider>
    </ClerkProvider>
  );
}
