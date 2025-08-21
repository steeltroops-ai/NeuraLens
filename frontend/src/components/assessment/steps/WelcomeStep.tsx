'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui';

interface WelcomeStepProps {
  onNext: () => void;
  onExit: () => void;
}

export const WelcomeStep: React.FC<WelcomeStepProps> = ({ onNext, onExit }) => {
  const [hasConsented, setHasConsented] = useState(false);
  const [hasReadDisclaimer, setHasReadDisclaimer] = useState(false);

  const canProceed = hasConsented && hasReadDisclaimer;

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-4xl space-y-12">
          {/* Apple-Style Header */}
          <div className="animate-fade-in space-y-6 text-center">
            <div className="from-medical-500 to-medical-600 rounded-apple-xl shadow-medical mx-auto flex h-20 w-20 items-center justify-center bg-gradient-to-br">
              <svg
                className="h-10 w-10 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold tracking-tight text-text-primary">
              Welcome to NeuroLens-X
            </h1>
            <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary">
              90-second brain health check powered by AI
            </p>
          </div>

          {/* Apple-Style Assessment Overview */}
          <div className="card-apple animate-slide-up p-10">
            <h2 className="mb-8 text-center text-3xl font-semibold text-text-primary">
              Four Quick Health Checks
            </h2>

            <div className="mb-10 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
              {/* Speech Assessment */}
              <div className="group space-y-4 text-center">
                <div className="rounded-apple-lg bg-medical-50 group-hover:bg-medical-100 mx-auto flex h-16 w-16 items-center justify-center transition-all duration-300 group-hover:scale-110">
                  <svg
                    className="text-medical-500 h-8 w-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">
                  Voice Evaluation
                </h3>
                <p className="text-sm leading-relaxed text-text-secondary">
                  Simple voice recording
                </p>
                <div className="text-medical-500 bg-medical-50 inline-block rounded-full px-3 py-1 text-sm font-medium">
                  30 seconds
                </div>
              </div>

              {/* Retinal Imaging */}
              <div className="group space-y-4 text-center">
                <div className="rounded-apple-lg bg-neural-50 group-hover:bg-neural-100 mx-auto flex h-16 w-16 items-center justify-center transition-all duration-300 group-hover:scale-110">
                  <svg
                    className="text-neural-500 h-8 w-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">
                  Eye Health Scan
                </h3>
                <p className="text-sm leading-relaxed text-text-secondary">
                  Quick eye photo analysis
                </p>
                <div className="text-neural-500 bg-neural-50 inline-block rounded-full px-3 py-1 text-sm font-medium">
                  15 seconds
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="group space-y-4 text-center">
                <div className="rounded-apple-lg bg-warning-50 group-hover:bg-warning-100 mx-auto flex h-16 w-16 items-center justify-center transition-all duration-300 group-hover:scale-110">
                  <svg
                    className="text-warning-500 h-8 w-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">
                  Health Questionnaire
                </h3>
                <p className="text-sm leading-relaxed text-text-secondary">
                  Simple health questions
                </p>
                <div className="text-warning-500 bg-warning-50 inline-block rounded-full px-3 py-1 text-sm font-medium">
                  30 seconds
                </div>
              </div>

              {/* AI Analysis */}
              <div className="group space-y-4 text-center">
                <div className="rounded-apple-lg bg-success-50 group-hover:bg-success-100 mx-auto flex h-16 w-16 items-center justify-center transition-all duration-300 group-hover:scale-110">
                  <svg
                    className="text-success-500 h-8 w-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-text-primary">
                  Health Score
                </h3>
                <p className="text-sm leading-relaxed text-text-secondary">
                  Personalized brain health score
                </p>
                <div className="text-success-500 bg-success-50 inline-block rounded-full px-3 py-1 text-sm font-medium">
                  Instant
                </div>
              </div>
            </div>

            {/* Apple-Style Total Time Badge */}
            <div className="mt-8 text-center">
              <div className="bg-medical-50 rounded-apple-xl border-medical-100 inline-flex items-center gap-4 border px-8 py-6">
                <div className="rounded-apple bg-medical-100 flex h-12 w-12 items-center justify-center">
                  <svg
                    className="text-medical-500 h-6 w-6"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="text-left">
                  <h4 className="text-lg font-semibold text-text-primary">
                    Total Screening Time
                  </h4>
                  <p className="text-sm text-text-secondary">
                    Under 90 seconds for complete health check
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Apple-Style Privacy & Security */}
          <div className="card-apple animate-scale-in p-10">
            <h2 className="mb-8 text-center text-3xl font-semibold text-text-primary">
              Privacy & Security
            </h2>

            <div className="mb-8 grid grid-cols-1 gap-8 md:grid-cols-3">
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="bg-success-50 rounded-apple-lg flex h-16 w-16 items-center justify-center">
                  <svg
                    className="text-success-500 h-8 w-8"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-text-primary">
                    Local Processing
                  </h4>
                  <p className="text-sm leading-relaxed text-text-secondary">
                    All analysis performed in your browser for maximum privacy
                  </p>
                </div>
              </div>

              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="bg-medical-50 rounded-apple-lg flex h-16 w-16 items-center justify-center">
                  <svg
                    className="text-medical-500 h-8 w-8"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-text-primary">
                    HIPAA Compliant
                  </h4>
                  <p className="text-sm leading-relaxed text-text-secondary">
                    Healthcare data protection standards maintained
                  </p>
                </div>
              </div>

              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="bg-success-50 rounded-apple-lg flex h-16 w-16 items-center justify-center">
                  <svg
                    className="text-success-500 h-8 w-8"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-text-primary">
                    No Data Storage
                  </h4>
                  <p className="text-sm leading-relaxed text-text-secondary">
                    Your data is not stored on our servers
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Apple-Style Medical Disclaimer */}
          <div className="bg-warning-50 border-warning-200 rounded-apple-lg animate-scale-in border p-8">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="bg-warning-100 rounded-apple flex h-12 w-12 items-center justify-center">
                  <svg
                    className="text-warning-600 h-6 w-6"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
              </div>
              <div className="flex-1">
                <h3 className="text-warning-800 mb-3 text-lg font-semibold">
                  Important Medical Disclaimer
                </h3>
                <p className="text-warning-700 text-sm leading-relaxed">
                  This assessment is a screening tool and not a diagnostic
                  device. Results should not replace professional medical
                  advice, diagnosis, or treatment. Always consult with qualified
                  healthcare professionals for medical decisions. If you have
                  immediate health concerns, contact your healthcare provider or
                  emergency services.
                </p>
              </div>
            </div>
          </div>

          {/* Apple-Style Consent Section */}
          <div className="card-apple animate-slide-up p-10">
            <h2 className="mb-8 text-center text-3xl font-semibold text-text-primary">
              Consent & Agreement
            </h2>

            <div className="space-y-6">
              <label className="rounded-apple-lg flex cursor-pointer items-start space-x-4 p-4 transition-colors duration-200 hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={hasReadDisclaimer}
                  onChange={(e) => setHasReadDisclaimer(e.target.checked)}
                  className="rounded-apple text-medical-500 focus:ring-medical-500 mt-1 h-5 w-5 border-2 border-gray-300 focus:ring-2 focus:ring-offset-2"
                />
                <span className="text-base leading-relaxed text-text-primary">
                  I understand this is a health screening tool and not a medical
                  diagnosis. I have read the medical disclaimer above.
                </span>
              </label>

              <label className="rounded-apple-lg flex cursor-pointer items-start space-x-4 p-4 transition-colors duration-200 hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={hasConsented}
                  onChange={(e) => setHasConsented(e.target.checked)}
                  className="rounded-apple text-medical-500 focus:ring-medical-500 mt-1 h-5 w-5 border-2 border-gray-300 focus:ring-2 focus:ring-offset-2"
                />
                <span className="text-base leading-relaxed text-text-primary">
                  I consent to participate in this brain health screening. I
                  understand my data will be processed locally in my browser and
                  not stored on external servers.
                </span>
              </label>
            </div>
          </div>

          {/* Apple-Style Action Buttons */}
          <div className="flex animate-fade-in flex-col justify-center gap-6 sm:flex-row">
            <Button
              variant="secondary"
              size="xl"
              onClick={onExit}
              className="min-w-[200px]"
            >
              Exit Health Check
            </Button>

            <Button
              variant="primary"
              size="xl"
              onClick={onNext}
              disabled={!canProceed}
              className="shadow-medical hover:shadow-medical-hover min-w-[200px]"
            >
              Begin Health Check
              <svg
                className="ml-3 h-6 w-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
