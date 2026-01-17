'use client';

import React, { useEffect, useRef } from 'react';

import { cn } from '@/utils/cn';

import { Footer } from './Footer';
import { Header } from './Header';

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
  // Initialize accessibility features only once globally
  const accessibilityInitialized = useRef(false);

  useEffect(() => {
    if (!accessibilityInitialized.current) {
      // Lazy load accessibility initialization to avoid blocking render
      import('@/utils/accessibility').then(({ initializeAccessibility }) => {
        initializeAccessibility();
        accessibilityInitialized.current = true;
      });
    }
  }, []);

  return (
    <div className={cn('flex min-h-screen flex-col', fullHeight && 'h-screen', className)}>
      {/* Header */}
      {showHeader && <Header />}

      {/* Main Content */}
      <main
        className={cn(
          'flex-1',
          showHeader && 'pt-14',
          containerized && 'container mx-auto px-4',
        )}
        id='main-content'
        role='main'
      >
        {children}
      </main>

      {/* Footer */}
      {showFooter && <Footer />}

      {/* Accessibility Announcements */}
      <div
        id='accessibility-announcements'
        aria-live='polite'
        aria-atomic='true'
        className='sr-only'
      />
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
    <div className='bg-surface-background min-h-screen'>
      {/* Assessment Header */}
      <header className='bg-surface-primary border-b border-neutral-800'>
        <div className='container mx-auto px-4 py-4'>
          <div className='flex items-center justify-between'>
            {/* Back Button */}
            <div className='flex items-center space-x-4'>
              {onBack && (
                <button
                  onClick={onBack}
                  className='focus:ring-offset-surface-primary rounded-lg p-2 text-text-secondary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2'
                  aria-label='Go back'
                >
                  <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                    <path
                      strokeLinecap='round'
                      strokeLinejoin='round'
                      strokeWidth={2}
                      d='M15 19l-7-7 7-7'
                    />
                  </svg>
                </button>
              )}

              <div>
                <h1 className='text-lg font-semibold text-text-primary'>
                  {stepTitle || `Step ${currentStep} of ${totalSteps}`}
                </h1>
                {showProgress && <p className='text-sm text-text-secondary'>Assessment Progress</p>}
              </div>
            </div>

            {/* Exit Button */}
            {onExit && (
              <button
                onClick={onExit}
                className='focus:ring-offset-surface-primary rounded-lg p-2 text-text-secondary transition-colors hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2'
                aria-label='Exit assessment'
              >
                <svg className='h-5 w-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                  <path
                    strokeLinecap='round'
                    strokeLinejoin='round'
                    strokeWidth={2}
                    d='M6 18L18 6M6 6l12 12'
                  />
                </svg>
              </button>
            )}
          </div>

          {/* Progress Bar */}
          {showProgress && (
            <div className='mt-4'>
              <div className='h-2 w-full rounded-full bg-neutral-800'>
                <div
                  className='h-2 rounded-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500 ease-out'
                  style={{ width: `${progressPercentage}%` }}
                />
              </div>
              <div className='text-text-muted mt-2 flex justify-between text-xs'>
                <span>Step {currentStep}</span>
                <span>{Math.round(progressPercentage)}% Complete</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Assessment Content */}
      <main className='flex-1 py-8' role='main'>
        {children}
      </main>
    </div>
  );
};

export default Layout;
