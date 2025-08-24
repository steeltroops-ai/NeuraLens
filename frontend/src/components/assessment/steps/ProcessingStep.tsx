'use client';

import React, { useEffect, useState } from 'react';

import { Button, Loading } from '@/components/ui';
import { mlModelIntegrator, type AssessmentProgress } from '@/lib/ml';

interface ProcessingStepProps {
  sessionId: string;
  isProcessing: boolean;
  error: string | null;
  onRetry: () => void;
}

export const ProcessingStep: React.FC<ProcessingStepProps> = ({
  sessionId,
  isProcessing,
  error,
  onRetry,
}) => {
  const [progress, setProgress] = useState<AssessmentProgress | null>(null);

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        const currentProgress = mlModelIntegrator.getAssessmentProgress(sessionId);
        if (currentProgress) {
          setProgress(currentProgress);
        }
      }, 500);

      return () => clearInterval(interval);
    }

    return () => {}; // Return empty cleanup function when not processing
  }, [sessionId, isProcessing]);

  const processingSteps = [
    { id: 'speech', label: 'Analyzing Speech Patterns', icon: '🎤' },
    { id: 'retinal', label: 'Processing Retinal Image', icon: '👁️' },
    { id: 'risk', label: 'Calculating Risk Factors', icon: '📊' },
    { id: 'fusion', label: 'Computing NRI Score', icon: '🧠' },
  ];

  return (
    <div className='min-h-screen bg-gray-50 py-12'>
      <div className='container mx-auto px-6'>
        <div className='mx-auto max-w-4xl space-y-12'>
          {/* Apple-Style Header */}
          <div className='animate-fade-in space-y-6 text-center'>
            <div className='shadow-success mx-auto flex h-24 w-24 items-center justify-center rounded-apple-xl bg-gradient-to-br from-success-500 to-success-600'>
              {isProcessing ? (
                <div className='relative'>
                  <div className='h-12 w-12 animate-spin rounded-full border-4 border-white border-t-transparent' />
                  <div className='absolute inset-0 flex items-center justify-center'>
                    <div className='h-6 w-6 animate-pulse rounded-full bg-white' />
                  </div>
                </div>
              ) : error ? (
                <div className='flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-error-400 to-error-500'>
                  <svg className='h-8 w-8 text-white' fill='currentColor' viewBox='0 0 24 24'>
                    <path
                      fillRule='evenodd'
                      d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z'
                      clipRule='evenodd'
                    />
                  </svg>
                </div>
              ) : (
                <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                  <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                </svg>
              )}
            </div>
            <h1 className='text-4xl font-bold tracking-tight text-text-primary'>
              {isProcessing
                ? 'Processing Health Check'
                : error
                  ? 'Processing Error'
                  : 'Health Check Complete'}
            </h1>
            <p className='mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary'>
              {isProcessing
                ? 'Our AI models are analyzing your health check data'
                : error
                  ? 'An error occurred during processing'
                  : 'Your brain health evaluation is ready'}
            </p>
          </div>

          {/* Apple-Style Processing Status */}
          {isProcessing && (
            <div className='card-apple animate-slide-up p-12'>
              <div className='space-y-8'>
                {/* Overall Progress */}
                <div className='space-y-6 text-center'>
                  <div className='animate-pulse text-6xl font-bold text-success-500'>
                    {progress?.progress || 0}%
                  </div>
                  <div className='h-4 w-full overflow-hidden rounded-full bg-gray-200'>
                    <div
                      className='h-4 rounded-full bg-gradient-to-r from-success-400 to-success-500 shadow-lg transition-all duration-700 ease-out'
                      style={{ width: `${progress?.progress || 0}%` }}
                    />
                  </div>
                  <p className='text-xl font-medium text-text-primary'>
                    {progress?.currentStep || 'Initializing...'}
                  </p>
                  {progress?.estimatedTimeRemaining && progress.estimatedTimeRemaining > 0 && (
                    <p className='inline-block rounded-full bg-gray-100 px-4 py-2 text-sm text-text-secondary'>
                      Estimated time remaining: {Math.round(progress.estimatedTimeRemaining)}s
                    </p>
                  )}
                </div>

                {/* Apple-Style Step Progress */}
                <div className='space-y-4'>
                  {processingSteps.map((step, index) => {
                    const isCompleted = progress?.completedModalities.includes(step.id) || false;
                    const isCurrent =
                      progress?.currentStep.toLowerCase().includes(step.id) || false;

                    return (
                      <div
                        key={step.id}
                        className={`flex items-center space-x-6 rounded-apple-lg p-6 transition-all duration-300 ${
                          isCurrent
                            ? 'scale-105 border-2 border-success-200 bg-success-50'
                            : isCompleted
                              ? 'border border-gray-200 bg-gray-50'
                              : 'border border-gray-100 bg-white'
                        }`}
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <div
                          className={`text-3xl transition-transform duration-300 ${isCurrent ? 'animate-bounce' : ''}`}
                        >
                          {step.icon}
                        </div>
                        <div className='flex-1'>
                          <div
                            className={`text-lg font-semibold ${
                              isCurrent ? 'text-success-700' : 'text-text-primary'
                            }`}
                          >
                            {step.label}
                          </div>
                        </div>
                        <div className='flex-shrink-0'>
                          {isCompleted ? (
                            <div className='flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg'>
                              <svg
                                className='h-5 w-5 text-white'
                                fill='currentColor'
                                viewBox='0 0 24 24'
                              >
                                <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                              </svg>
                            </div>
                          ) : isCurrent ? (
                            <div className='flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500'>
                              <div className='h-4 w-4 animate-pulse rounded-full bg-white' />
                            </div>
                          ) : (
                            <div className='h-8 w-8 rounded-full border-2 border-gray-300 bg-white' />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Apple-Style Error State */}
          {error && (
            <div className='animate-scale-in rounded-apple-lg border border-error-200 bg-error-50 p-10'>
              <div className='space-y-6 text-center'>
                <div className='mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-error-400 to-error-500 shadow-lg'>
                  <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                    <path
                      fillRule='evenodd'
                      d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z'
                      clipRule='evenodd'
                    />
                  </svg>
                </div>
                <div className='space-y-3'>
                  <h2 className='text-2xl font-semibold text-error-700'>Processing Failed</h2>
                  <p className='text-lg text-error-600'>{error}</p>
                </div>

                <div className='space-y-4'>
                  <Button
                    onClick={onRetry}
                    size='xl'
                    className='hover:shadow-error-hover shadow-error'
                  >
                    <svg
                      className='mr-3 h-6 w-6'
                      fill='none'
                      stroke='currentColor'
                      viewBox='0 0 24 24'
                    >
                      <path
                        strokeLinecap='round'
                        strokeLinejoin='round'
                        strokeWidth={2}
                        d='M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15'
                      />
                    </svg>
                    Retry Processing
                  </Button>
                  <p className='inline-block rounded-full bg-error-100 px-4 py-2 text-sm text-error-600'>
                    If the problem persists, please check your internet connection and try again.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Apple-Style Success State */}
          {!isProcessing && !error && (
            <div className='animate-scale-in rounded-apple-lg border border-success-200 bg-success-50 p-10'>
              <div className='space-y-6 text-center'>
                <div className='mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg'>
                  <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                    <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                  </svg>
                </div>
                <div className='space-y-3'>
                  <h2 className='text-2xl font-semibold text-success-700'>Assessment Complete</h2>
                  <p className='text-lg text-success-600'>
                    Your neurological risk assessment has been successfully processed.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Apple-Style Processing Info */}
          <div className='animate-scale-in rounded-apple-lg border border-medical-100 bg-medical-50 p-8'>
            <h3 className='mb-6 text-center text-xl font-semibold text-text-primary'>
              What's Happening?
            </h3>
            <div className='grid grid-cols-1 gap-8 md:grid-cols-2'>
              <div className='space-y-4'>
                <h4 className='flex items-center text-lg font-semibold text-text-primary'>
                  <div className='mr-3 flex h-8 w-8 items-center justify-center rounded-apple bg-medical-100'>
                    <svg
                      className='h-4 w-4 text-medical-600'
                      fill='currentColor'
                      viewBox='0 0 24 24'
                    >
                      <path d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' />
                    </svg>
                  </div>
                  AI Analysis
                </h4>
                <ul className='space-y-2 text-text-secondary'>
                  <li className='flex items-center'>
                    <div className='mr-3 h-2 w-2 rounded-full bg-medical-400' />
                    Voice biomarker extraction
                  </li>
                  <li className='flex items-center'>
                    <div className='mr-3 h-2 w-2 rounded-full bg-medical-400' />
                    Retinal vascular analysis
                  </li>
                  <li className='flex items-center'>
                    <div className='mr-3 h-2 w-2 rounded-full bg-medical-400' />
                    Risk factor computation
                  </li>
                  <li className='flex items-center'>
                    <div className='mr-3 h-2 w-2 rounded-full bg-medical-400' />
                    Multi-modal data fusion
                  </li>
                </ul>
              </div>
              <div className='space-y-4'>
                <h4 className='flex items-center text-lg font-semibold text-text-primary'>
                  <div className='mr-3 flex h-8 w-8 items-center justify-center rounded-apple bg-success-100'>
                    <svg
                      className='h-4 w-4 text-success-600'
                      fill='currentColor'
                      viewBox='0 0 24 24'
                    >
                      <path
                        fillRule='evenodd'
                        d='M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z'
                        clipRule='evenodd'
                      />
                    </svg>
                  </div>
                  Quality Assurance
                </h4>
                <ul className='space-y-1'>
                  <li>• Data quality validation</li>
                  <li>• Confidence calculation</li>
                  <li>• Uncertainty quantification</li>
                  <li>• Clinical correlation</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
