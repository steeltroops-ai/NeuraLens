'use client';

import React, { useState } from 'react';
import { Button, Card } from '@/components/ui';

interface WelcomeStepProps {
  onNext: () => void;
  onExit: () => void;
}

export const WelcomeStep: React.FC<WelcomeStepProps> = ({ onNext, onExit }) => {
  const [hasConsented, setHasConsented] = useState(false);
  const [hasReadDisclaimer, setHasReadDisclaimer] = useState(false);

  const canProceed = hasConsented && hasReadDisclaimer;

  return (
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl flex items-center justify-center mx-auto">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </div>
            <h1 className="text-4xl font-bold text-text-primary">
              Welcome to NeuroLens-X
            </h1>
            <p className="text-xl text-text-secondary max-w-2xl mx-auto">
              Your comprehensive neurological risk assessment platform
            </p>
          </div>

          {/* Assessment Overview */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              What to Expect
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {/* Speech Assessment */}
              <div className="text-center space-y-3">
                <div className="w-12 h-12 bg-primary-500/10 rounded-lg flex items-center justify-center mx-auto">
                  <svg className="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-text-primary">Speech Analysis</h3>
                <p className="text-sm text-text-secondary">Voice biomarker detection through speech pattern analysis</p>
                <div className="text-xs text-primary-400">~3 minutes</div>
              </div>

              {/* Retinal Imaging */}
              <div className="text-center space-y-3">
                <div className="w-12 h-12 bg-primary-500/10 rounded-lg flex items-center justify-center mx-auto">
                  <svg className="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-text-primary">Retinal Imaging</h3>
                <p className="text-sm text-text-secondary">Vascular pattern analysis for early pathological changes</p>
                <div className="text-xs text-primary-400">~2 minutes</div>
              </div>

              {/* Risk Assessment */}
              <div className="text-center space-y-3">
                <div className="w-12 h-12 bg-primary-500/10 rounded-lg flex items-center justify-center mx-auto">
                  <svg className="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-text-primary">Risk Assessment</h3>
                <p className="text-sm text-text-secondary">Comprehensive health and lifestyle questionnaire</p>
                <div className="text-xs text-primary-400">~5 minutes</div>
              </div>

              {/* AI Analysis */}
              <div className="text-center space-y-3">
                <div className="w-12 h-12 bg-primary-500/10 rounded-lg flex items-center justify-center mx-auto">
                  <svg className="w-6 h-6 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="font-semibold text-text-primary">AI Analysis</h3>
                <p className="text-sm text-text-secondary">Unified Neuro-Risk Index calculation</p>
                <div className="text-xs text-primary-400">~1 minute</div>
              </div>
            </div>

            <div className="bg-primary-500/5 border border-primary-500/20 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-primary-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div>
                  <h4 className="font-medium text-primary-400 mb-1">Total Assessment Time</h4>
                  <p className="text-sm text-text-secondary">
                    Approximately 11 minutes for complete multi-modal assessment
                  </p>
                </div>
              </div>
            </div>
          </Card>

          {/* Privacy & Security */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              Privacy & Security
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-success mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                </svg>
                <div>
                  <h4 className="font-medium text-text-primary">Local Processing</h4>
                  <p className="text-sm text-text-secondary">All analysis performed in your browser for maximum privacy</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-success mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <h4 className="font-medium text-text-primary">HIPAA Compliant</h4>
                  <p className="text-sm text-text-secondary">Healthcare data protection standards maintained</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <svg className="w-5 h-5 text-success mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <h4 className="font-medium text-text-primary">No Data Storage</h4>
                  <p className="text-sm text-text-secondary">Your data is not stored on our servers</p>
                </div>
              </div>
            </div>
          </Card>

          {/* Medical Disclaimer */}
          <Card className="p-8 border-amber-500/20 bg-amber-500/5">
            <div className="flex items-start space-x-3">
              <svg className="w-6 h-6 text-amber-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="font-semibold text-amber-400 mb-2">Important Medical Disclaimer</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  This assessment is a screening tool and not a diagnostic device. Results should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions. If you have immediate health concerns, contact your healthcare provider or emergency services.
                </p>
              </div>
            </div>
          </Card>

          {/* Consent Checkboxes */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              Consent & Agreement
            </h2>
            
            <div className="space-y-4">
              <label className="flex items-start space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasReadDisclaimer}
                  onChange={(e) => setHasReadDisclaimer(e.target.checked)}
                  className="mt-1 w-4 h-4 text-primary-500 bg-surface-secondary border-neutral-600 rounded focus:ring-primary-500 focus:ring-2"
                />
                <span className="text-sm text-text-secondary">
                  I have read and understand the medical disclaimer above. I understand this is a screening tool and not a diagnostic device.
                </span>
              </label>
              
              <label className="flex items-start space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasConsented}
                  onChange={(e) => setHasConsented(e.target.checked)}
                  className="mt-1 w-4 h-4 text-primary-500 bg-surface-secondary border-neutral-600 rounded focus:ring-primary-500 focus:ring-2"
                />
                <span className="text-sm text-text-secondary">
                  I consent to participate in this neurological risk assessment. I understand my data will be processed locally in my browser and not stored on external servers.
                </span>
              </label>
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              variant="secondary"
              size="lg"
              onClick={onExit}
              className="px-8"
            >
              Exit Assessment
            </Button>
            
            <Button
              variant="primary"
              size="lg"
              onClick={onNext}
              disabled={!canProceed}
              className="px-8"
            >
              Begin Assessment
              <svg className="w-5 h-5 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
