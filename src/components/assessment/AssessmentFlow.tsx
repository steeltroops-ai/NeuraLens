'use client';

import React, { useState, useCallback } from 'react';
import { AssessmentLayout } from '@/components/layout';
import { Button, Card, Progress } from '@/components/ui';
import { WelcomeStep } from './steps/WelcomeStep';
import { SpeechAssessmentStep } from './steps/SpeechAssessmentStep';
import { RetinalAssessmentStep } from './steps/RetinalAssessmentStep';
import { RiskAssessmentStep } from './steps/RiskAssessmentStep';
import { ProcessingStep } from './steps/ProcessingStep';
import { ResultsStep } from './steps/ResultsStep';
import {
  mlModelIntegrator,
  generateSessionId,
  type AssessmentRequest,
  type CompleteAssessmentResult,
  type RiskAssessmentData,
} from '@/lib/ml';

export type AssessmentStep =
  | 'welcome'
  | 'speech'
  | 'retinal'
  | 'risk'
  | 'processing'
  | 'results';

export interface AssessmentData {
  sessionId: string;
  audioFile?: File;
  retinalImage?: File;
  riskData?: RiskAssessmentData;
  results?: CompleteAssessmentResult;
}

const ASSESSMENT_STEPS: Array<{
  id: AssessmentStep;
  title: string;
  description: string;
  estimatedTime: number; // minutes
}> = [
  {
    id: 'welcome',
    title: 'Welcome',
    description: 'Introduction and consent',
    estimatedTime: 1,
  },
  {
    id: 'speech',
    title: 'Speech Analysis',
    description: 'Voice biomarker assessment',
    estimatedTime: 3,
  },
  {
    id: 'retinal',
    title: 'Retinal Imaging',
    description: 'Eye health evaluation',
    estimatedTime: 2,
  },
  {
    id: 'risk',
    title: 'Risk Assessment',
    description: 'Health and lifestyle questionnaire',
    estimatedTime: 5,
  },
  {
    id: 'processing',
    title: 'Processing',
    description: 'AI analysis in progress',
    estimatedTime: 1,
  },
  {
    id: 'results',
    title: 'Results',
    description: 'Your neurological risk assessment',
    estimatedTime: 0,
  },
];

export const AssessmentFlow: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<AssessmentStep>('welcome');
  const [assessmentData, setAssessmentData] = useState<AssessmentData>({
    sessionId: generateSessionId(),
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Get current step info
  const currentStepIndex = ASSESSMENT_STEPS.findIndex(
    (step) => step.id === currentStep
  );
  const currentStepInfo = ASSESSMENT_STEPS[currentStepIndex];
  const totalSteps = ASSESSMENT_STEPS.length - 1; // Exclude results step from count

  // Navigation handlers
  const handleNext = useCallback(() => {
    if (currentStepIndex === -1) return;
    const nextIndex = currentStepIndex + 1;
    if (nextIndex < ASSESSMENT_STEPS.length) {
      const nextStep = ASSESSMENT_STEPS[nextIndex];
      if (nextStep) {
        setCurrentStep(nextStep.id);
      }
    }
  }, [currentStepIndex]);

  const handleBack = useCallback(() => {
    if (currentStepIndex === -1) return;
    const prevIndex = currentStepIndex - 1;
    if (prevIndex >= 0) {
      const prevStep = ASSESSMENT_STEPS[prevIndex];
      if (prevStep) {
        setCurrentStep(prevStep.id);
      }
    }
  }, [currentStepIndex]);

  const handleExit = useCallback(() => {
    if (
      typeof window !== 'undefined' &&
      confirm(
        'Are you sure you want to exit the assessment? Your progress will be lost.'
      )
    ) {
      window.location.href = '/';
    }
  }, []);

  // Data update handlers
  const updateAssessmentData = useCallback(
    (updates: Partial<AssessmentData>) => {
      setAssessmentData((prev) => ({ ...prev, ...updates }));
    },
    []
  );

  // Process assessment
  const processAssessment = useCallback(async () => {
    setIsProcessing(true);
    setError(null);

    try {
      const request: AssessmentRequest = {
        sessionId: assessmentData.sessionId,
        ...(assessmentData.audioFile && {
          audioFile: assessmentData.audioFile,
        }),
        ...(assessmentData.retinalImage && {
          retinalImage: assessmentData.retinalImage,
        }),
        ...(assessmentData.riskData && { riskData: assessmentData.riskData }),
      };

      const results = await mlModelIntegrator.processAssessment(request);

      updateAssessmentData({ results });
      setCurrentStep('results');
    } catch (err) {
      console.error('Assessment processing failed:', err);
      setError((err as Error).message);
    } finally {
      setIsProcessing(false);
    }
  }, [assessmentData, updateAssessmentData]);

  // Step-specific handlers
  const handleSpeechComplete = useCallback(
    (audioFile: File) => {
      updateAssessmentData({ audioFile });
      handleNext();
    },
    [updateAssessmentData, handleNext]
  );

  const handleRetinalComplete = useCallback(
    (retinalImage: File) => {
      updateAssessmentData({ retinalImage });
      handleNext();
    },
    [updateAssessmentData, handleNext]
  );

  const handleRiskComplete = useCallback(
    (riskData: RiskAssessmentData) => {
      updateAssessmentData({ riskData });
      setCurrentStep('processing');
      // Start processing automatically
      setTimeout(() => {
        processAssessment();
      }, 1000);
    },
    [updateAssessmentData, processAssessment]
  );

  // Render current step
  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'welcome':
        return <WelcomeStep onNext={handleNext} onExit={handleExit} />;

      case 'speech':
        return (
          <SpeechAssessmentStep
            onComplete={handleSpeechComplete}
            onBack={handleBack}
            onSkip={handleNext}
          />
        );

      case 'retinal':
        return (
          <RetinalAssessmentStep
            onComplete={handleRetinalComplete}
            onBack={handleBack}
            onSkip={handleNext}
          />
        );

      case 'risk':
        return (
          <RiskAssessmentStep
            onComplete={handleRiskComplete}
            onBack={handleBack}
            {...(assessmentData.riskData && {
              initialData: assessmentData.riskData,
            })}
          />
        );

      case 'processing':
        return (
          <ProcessingStep
            sessionId={assessmentData.sessionId}
            isProcessing={isProcessing}
            error={error}
            onRetry={processAssessment}
          />
        );

      case 'results':
        return assessmentData.results ? (
          <ResultsStep
            results={assessmentData.results}
            onRestart={() => {
              setAssessmentData({ sessionId: generateSessionId() });
              setCurrentStep('welcome');
            }}
            onExit={() => {
              if (typeof window !== 'undefined') {
                window.location.href = '/';
              }
            }}
          />
        ) : (
          <div className="p-8 text-center">
            <h2 className="mb-4 text-2xl font-bold text-text-primary">
              No Results Available
            </h2>
            <p className="mb-8 text-text-secondary">
              Please complete the assessment to view results.
            </p>
            <Button onClick={() => setCurrentStep('welcome')}>
              Start Over
            </Button>
          </div>
        );

      default:
        return null;
    }
  };

  // Calculate progress percentage
  const progressPercentage =
    currentStep === 'results'
      ? 100
      : (currentStepIndex / (totalSteps - 1)) * 100;

  const layoutProps: any = {
    currentStep: currentStepIndex + 1,
    totalSteps: totalSteps,
    stepTitle: currentStepInfo?.title || 'Assessment',
    showProgress: currentStep !== 'results',
  };

  if (
    currentStep !== 'welcome' &&
    currentStep !== 'processing' &&
    currentStep !== 'results'
  ) {
    layoutProps.onBack = handleBack;
  }

  if (currentStep !== 'processing' && currentStep !== 'results') {
    layoutProps.onExit = handleExit;
  }

  return (
    <AssessmentLayout {...layoutProps}>
      <div className="min-h-screen bg-surface-background">
        {/* Progress Indicator */}
        {currentStep !== 'results' && (
          <div className="border-b border-neutral-800 bg-surface-primary py-4">
            <div className="container mx-auto px-4">
              <div className="mx-auto max-w-4xl">
                <div className="mb-4 flex items-center justify-between">
                  <h1 className="text-2xl font-bold text-text-primary">
                    {currentStepInfo?.title}
                  </h1>
                  <div className="text-sm text-text-secondary">
                    Step {currentStepIndex + 1} of {totalSteps}
                  </div>
                </div>

                <Progress
                  value={progressPercentage}
                  className="h-2"
                  showLabel={false}
                />

                <div className="mt-2 flex justify-between text-xs text-text-muted">
                  <span>{currentStepInfo?.description}</span>
                  <span>
                    {currentStepInfo?.estimatedTime &&
                      currentStepInfo.estimatedTime > 0 &&
                      `~${currentStepInfo.estimatedTime} min`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step Content */}
        <div className="flex-1">{renderCurrentStep()}</div>

        {/* Error Display */}
        {error && (
          <div className="fixed bottom-4 left-4 right-4 z-50">
            <Card className="border-red-500/20 bg-red-500/10 p-4">
              <div className="flex items-center space-x-3">
                <svg
                  className="h-5 w-5 flex-shrink-0 text-red-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-red-400">
                    Assessment Error
                  </h3>
                  <p className="mt-1 text-sm text-red-300">{error}</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setError(null)}
                  className="text-red-400 hover:text-red-300"
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
                </Button>
              </div>
            </Card>
          </div>
        )}
      </div>
    </AssessmentLayout>
  );
};
