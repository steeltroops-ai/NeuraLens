'use client';

import React, { useEffect } from 'react';
import { Header } from './Header';
import { Footer } from './Footer';
import { cn } from '@/utils/cn';
import { initializeAccessibility } from '@/utils/accessibility';

interface LayoutProps {
  children: React.ReactNode;
  className?: string;
  showHeader?: boolean;
  showFooter?: boolean;
  fullHeight?: boolean;
  containerized?: boolean;
}

export const Layout: React.FC<LayoutProps> = ({
  children,
  className,
  showHeader = true,
  showFooter = true,
  fullHeight = false,
  containerized = true,
}) => {
  // Initialize accessibility features
  useEffect(() => {
    initializeAccessibility();
  }, []);

  return (
    <div
      className={cn(
        'flex min-h-screen flex-col',
        fullHeight && 'h-screen',
        className
      )}
    >
      {/* Header */}
      {showHeader && <Header />}

      {/* Main Content */}
      <main
        className={cn(
          'flex-1',
          showHeader && 'pt-16 lg:pt-20',
          containerized && 'container mx-auto px-4'
        )}
        id="main-content"
        role="main"
      >
        {children}
      </main>

      {/* Footer */}
      {showFooter && <Footer />}

      {/* PWA Install Prompt */}
      <PWAInstallPrompt />

      {/* Accessibility Announcements */}
      <div
        id="accessibility-announcements"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />
    </div>
  );
};

/**
 * PWA Install Prompt Component
 */
const PWAInstallPrompt: React.FC = () => {
  const [showPrompt, setShowPrompt] = React.useState(false);
  const [deferredPrompt, setDeferredPrompt] = React.useState<any>(null);

  useEffect(() => {
    const handleBeforeInstallPrompt = (e: Event) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault();
      // Stash the event so it can be triggered later
      setDeferredPrompt(e);
      // Show the install prompt
      setShowPrompt(true);
    };

    const handleAppInstalled = () => {
      // Hide the install prompt
      setShowPrompt(false);
      setDeferredPrompt(null);

      // Log install to analytics
      console.log('PWA was installed');
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener(
        'beforeinstallprompt',
        handleBeforeInstallPrompt
      );
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    // Show the install prompt
    deferredPrompt.prompt();

    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;

    if (outcome === 'accepted') {
      console.log('User accepted the install prompt');
    } else {
      console.log('User dismissed the install prompt');
    }

    // Clear the deferredPrompt
    setDeferredPrompt(null);
    setShowPrompt(false);
  };

  const handleDismiss = () => {
    setShowPrompt(false);
    // Don't show again for this session
    if (
      typeof window !== 'undefined' &&
      typeof sessionStorage !== 'undefined'
    ) {
      sessionStorage.setItem('pwa-prompt-dismissed', 'true');
    }
  };

  // Don't show if already dismissed this session or if we're on server
  if (typeof window === 'undefined' || typeof sessionStorage === 'undefined') {
    return null;
  }

  if (sessionStorage.getItem('pwa-prompt-dismissed')) {
    return null;
  }

  if (!showPrompt || !deferredPrompt) {
    return null;
  }

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 mx-auto max-w-md">
      <div className="rounded-xl border border-neutral-700 bg-surface-primary p-4 shadow-xl">
        <div className="flex items-start space-x-3">
          <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-600">
            <svg
              className="h-6 w-6 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
          </div>

          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-text-primary">
              Install NeuroLens-X
            </h3>
            <p className="mt-1 text-xs text-text-secondary">
              Install our app for faster access and offline capability.
            </p>

            <div className="mt-3 flex space-x-2">
              <button
                onClick={handleInstallClick}
                className="rounded-lg bg-primary-500 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-surface-primary"
              >
                Install
              </button>
              <button
                onClick={handleDismiss}
                className="rounded-lg px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-neutral-500 focus:ring-offset-2 focus:ring-offset-surface-primary"
              >
                Not now
              </button>
            </div>
          </div>

          <button
            onClick={handleDismiss}
            className="rounded-lg p-1 text-text-muted transition-colors hover:text-text-secondary focus:outline-none focus:ring-2 focus:ring-neutral-500 focus:ring-offset-2 focus:ring-offset-surface-primary"
            aria-label="Dismiss install prompt"
          >
            <svg
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Assessment Layout - Specialized layout for assessment flow
 */
interface AssessmentLayoutProps {
  children: React.ReactNode;
  currentStep?: number;
  totalSteps?: number;
  stepTitle?: string;
  onBack?: () => void;
  onExit?: () => void;
  showProgress?: boolean;
}

export const AssessmentLayout: React.FC<AssessmentLayoutProps> = ({
  children,
  currentStep = 1,
  totalSteps = 4,
  stepTitle,
  onBack,
  onExit,
  showProgress = true,
}) => {
  const progressPercentage = ((currentStep - 1) / (totalSteps - 1)) * 100;

  return (
    <div className="min-h-screen bg-surface-background">
      {/* Assessment Header */}
      <header className="border-b border-neutral-800 bg-surface-primary">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Back Button */}
            <div className="flex items-center space-x-4">
              {onBack && (
                <button
                  onClick={onBack}
                  className="rounded-lg p-2 text-text-secondary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-surface-primary"
                  aria-label="Go back"
                >
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 19l-7-7 7-7"
                    />
                  </svg>
                </button>
              )}

              <div>
                <h1 className="text-lg font-semibold text-text-primary">
                  {stepTitle || `Step ${currentStep} of ${totalSteps}`}
                </h1>
                {showProgress && (
                  <p className="text-sm text-text-secondary">
                    Assessment Progress
                  </p>
                )}
              </div>
            </div>

            {/* Exit Button */}
            {onExit && (
              <button
                onClick={onExit}
                className="rounded-lg p-2 text-text-secondary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-surface-primary"
                aria-label="Exit assessment"
              >
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>

          {/* Progress Bar */}
          {showProgress && (
            <div className="mt-4">
              <div className="h-2 w-full rounded-full bg-neutral-800">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500 ease-out"
                  style={{ width: `${progressPercentage}%` }}
                />
              </div>
              <div className="mt-2 flex justify-between text-xs text-text-muted">
                <span>Step {currentStep}</span>
                <span>{Math.round(progressPercentage)}% Complete</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Assessment Content */}
      <main className="flex-1 py-8" role="main">
        {children}
      </main>
    </div>
  );
};

export default Layout;
