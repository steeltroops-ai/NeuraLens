'use client';

import React, { useEffect, useState } from 'react';
import { Button, Card, Loading } from '@/components/ui';
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
        const currentProgress =
          mlModelIntegrator.getAssessmentProgress(sessionId);
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
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-4xl space-y-8">
          {/* Header */}
          <div className="space-y-4 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600">
              {isProcessing ? (
                <Loading size="md" className="text-white" />
              ) : error ? (
                <svg
                  className="h-8 w-8 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <svg
                  className="h-8 w-8 text-white"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                </svg>
              )}
            </div>
            <h1 className="text-3xl font-bold text-text-primary">
              {isProcessing
                ? 'Processing Assessment'
                : error
                  ? 'Processing Error'
                  : 'Assessment Complete'}
            </h1>
            <p className="mx-auto max-w-2xl text-lg text-text-secondary">
              {isProcessing
                ? 'Our AI models are analyzing your assessment data'
                : error
                  ? 'An error occurred during processing'
                  : 'Your neurological risk assessment is ready'}
            </p>
          </div>

          {/* Processing Status */}
          {isProcessing && (
            <Card className="p-8">
              <div className="space-y-6">
                {/* Overall Progress */}
                <div className="space-y-4 text-center">
                  <div className="text-4xl font-bold text-primary-400">
                    {progress?.progress || 0}%
                  </div>
                  <div className="h-3 w-full rounded-full bg-neutral-800">
                    <div
                      className="h-3 rounded-full bg-primary-500 transition-all duration-500"
                      style={{ width: `${progress?.progress || 0}%` }}
                    />
                  </div>
                  <p className="text-text-secondary">
                    {progress?.currentStep || 'Initializing...'}
                  </p>
                  {progress?.estimatedTimeRemaining &&
                    progress.estimatedTimeRemaining > 0 && (
                      <p className="text-sm text-text-muted">
                        Estimated time remaining:{' '}
                        {Math.round(progress.estimatedTimeRemaining)}s
                      </p>
                    )}
                </div>

                {/* Step Progress */}
                <div className="space-y-4">
                  {processingSteps.map((step) => {
                    const isCompleted =
                      progress?.completedModalities.includes(step.id) || false;
                    const isCurrent =
                      progress?.currentStep.toLowerCase().includes(step.id) ||
                      false;

                    return (
                      <div
                        key={step.id}
                        className="flex items-center space-x-4 rounded-lg bg-surface-secondary p-4"
                      >
                        <div className="text-2xl">{step.icon}</div>
                        <div className="flex-1">
                          <div className="font-medium text-text-primary">
                            {step.label}
                          </div>
                        </div>
                        <div className="flex-shrink-0">
                          {isCompleted ? (
                            <svg
                              className="h-6 w-6 text-success"
                              fill="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                            </svg>
                          ) : isCurrent ? (
                            <Loading size="sm" />
                          ) : (
                            <div className="h-6 w-6 rounded-full border-2 border-neutral-600" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </Card>
          )}

          {/* Error State */}
          {error && (
            <Card className="border-red-500/20 bg-red-500/5 p-8">
              <div className="space-y-6 text-center">
                <div className="text-6xl">❌</div>
                <div>
                  <h2 className="mb-2 text-2xl font-semibold text-red-400">
                    Processing Failed
                  </h2>
                  <p className="mb-4 text-text-secondary">{error}</p>
                </div>

                <div className="space-y-3">
                  <Button onClick={onRetry} size="lg" className="px-8">
                    Retry Processing
                  </Button>
                  <p className="text-sm text-text-muted">
                    If the problem persists, please check your internet
                    connection and try again.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {/* Success State */}
          {!isProcessing && !error && (
            <Card className="border-success/20 bg-success/5 p-8">
              <div className="space-y-6 text-center">
                <div className="text-6xl">✅</div>
                <div>
                  <h2 className="mb-2 text-2xl font-semibold text-success">
                    Assessment Complete
                  </h2>
                  <p className="text-text-secondary">
                    Your neurological risk assessment has been successfully
                    processed.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {/* Processing Info */}
          <Card className="p-6">
            <h3 className="mb-4 text-lg font-semibold text-text-primary">
              What's Happening?
            </h3>
            <div className="grid grid-cols-1 gap-4 text-sm text-text-secondary md:grid-cols-2">
              <div>
                <h4 className="mb-2 font-medium text-text-primary">
                  AI Analysis
                </h4>
                <ul className="space-y-1">
                  <li>• Voice biomarker extraction</li>
                  <li>• Retinal vascular analysis</li>
                  <li>• Risk factor computation</li>
                  <li>• Multi-modal data fusion</li>
                </ul>
              </div>
              <div>
                <h4 className="mb-2 font-medium text-text-primary">
                  Quality Assurance
                </h4>
                <ul className="space-y-1">
                  <li>• Data quality validation</li>
                  <li>• Confidence calculation</li>
                  <li>• Uncertainty quantification</li>
                  <li>• Clinical correlation</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
