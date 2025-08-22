'use client';

import React, { useState, useCallback } from 'react';
import { AssessmentLayout } from '@/components/layout';
import { Button, Card, Progress } from '@/components/ui';
import { WelcomeStep } from './steps/WelcomeStep';
import { SpeechAssessmentStep } from './steps/SpeechAssessmentStep';
import { RetinalAssessmentStep } from './steps/RetinalAssessmentStep';
import { CognitiveAssessmentStep } from './steps/CognitiveAssessmentStep';
import { RiskAssessmentStep } from './steps/RiskAssessmentStep';
import { MotorAssessmentStep } from './steps/MotorAssessmentStep';
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
  | 'cognitive'
  | 'risk'
  | 'motor'
  | 'processing'
  | 'results';

export interface AssessmentData {
  sessionId: string;
  audioFile?: File;
  retinalImage?: File;
  cognitiveData?: any; // Cognitive assessment data
  riskData?: RiskAssessmentData;
  motorData?: any; // Motor assessment data
  results?: CompleteAssessmentResult;
}

const ASSESSMENT_STEPS: Array<{
  id: AssessmentStep;
  title: string;
  description: string;
  estimatedTime: number; // seconds
}> = [
  {
    id: 'welcome',
    title: 'Welcome',
    description: 'Introduction and consent',
    estimatedTime: 15,
  },
  {
    id: 'speech',
    title: 'Voice Evaluation',
    description: 'Simple voice recording',
    estimatedTime: 30,
  },
  {
    id: 'retinal',
    title: 'Eye Health Scan',
    description: 'Quick eye photo analysis',
    estimatedTime: 15,
  },
  {
    id: 'cognitive',
    title: 'Cognitive Assessment',
    description: 'Memory, attention, and thinking tests',
    estimatedTime: 180,
  },
  {
    id: 'risk',
    title: 'Health Questionnaire',
    description: 'Brief health and lifestyle questions',
    estimatedTime: 30,
  },
  {
    id: 'motor',
    title: 'Movement Check',
    description: 'Simple finger tapping test',
    estimatedTime: 15,
  },
  {
    id: 'processing',
    title: 'Processing',
    description: 'AI analysis in progress',
    estimatedTime: 15,
  },
  {
    id: 'results',
    title: 'Your Results',
    description: 'Your personalized brain health report',
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

  const handleCognitiveComplete = useCallback(
    (cognitiveData: any) => {
      updateAssessmentData({ cognitiveData });
      handleNext();
    },
    [updateAssessmentData, handleNext]
  );

  const handleRiskComplete = useCallback(
    (riskData: RiskAssessmentData) => {
      updateAssessmentData({ riskData });
      handleNext();
    },
    [updateAssessmentData, handleNext]
  );

  const handleMotorComplete = useCallback(
    (motorData: any) => {
      updateAssessmentData({ motorData });
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

      case 'cognitive':
        return (
          <CognitiveAssessmentStep
            onComplete={handleCognitiveComplete}
            onBack={handleBack}
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

      case 'motor':
        return (
          <MotorAssessmentStep
            onComplete={handleMotorComplete}
            onBack={handleBack}
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
    <div className="min-h-screen bg-gray-50">
      {/* Apple-Style Progress Header */}
      {currentStep !== 'results' && (
        <div className="sticky top-0 z-40 border-b border-gray-200 bg-white bg-white/95 backdrop-blur-sm">
          <div className="container mx-auto px-6 py-4">
            <div className="mx-auto max-w-4xl">
              {/* Header with Back Button */}
              <div className="mb-6 flex items-center justify-between">
                {currentStep !== 'welcome' && currentStep !== 'processing' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleBack}
                    className="flex items-center gap-2 text-medical-500 hover:text-medical-600"
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
                        d="M15 19l-7-7 7-7"
                      />
                    </svg>
                    Back
                  </Button>
                )}

                <div className="flex-1 text-center">
                  <h1 className="text-2xl font-semibold tracking-tight text-text-primary">
                    {currentStepInfo?.title}
                  </h1>
                  <p className="mt-1 text-sm text-text-secondary">
                    {currentStepInfo?.description}
                  </p>
                </div>

                {(currentStep as AssessmentStep) !== 'processing' &&
                  (currentStep as AssessmentStep) !== 'results' && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleExit}
                      className="text-text-secondary hover:text-text-primary"
                    >
                      Exit
                    </Button>
                  )}
              </div>

              {/* Apple-Style Progress Dots */}
              <div className="mb-4 flex items-center justify-center space-x-3">
                {ASSESSMENT_STEPS.slice(0, -2).map((step, index) => (
                  <div key={step.id} className="flex items-center">
                    <div
                      className={`h-3 w-3 rounded-full transition-all duration-300 ${
                        index < currentStepIndex
                          ? 'scale-110 bg-success-500'
                          : index === currentStepIndex
                            ? 'scale-125 bg-medical-500 shadow-lg'
                            : 'bg-gray-300'
                      }`}
                    />
                    {index < ASSESSMENT_STEPS.length - 3 && (
                      <div
                        className={`mx-2 h-0.5 w-8 transition-colors duration-300 ${
                          index < currentStepIndex
                            ? 'bg-success-500'
                            : 'bg-gray-300'
                        }`}
                      />
                    )}
                  </div>
                ))}
              </div>

              {/* Step Counter */}
              <div className="text-center">
                <span className="text-sm font-medium text-text-secondary">
                  Step {currentStepIndex + 1} of {totalSteps - 2}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Apple-Style Step Content */}
      <div className="flex-1 py-8">
        <div className="container mx-auto px-6">
          <div className="mx-auto max-w-4xl">{renderCurrentStep()}</div>
        </div>
      </div>

      {/* Apple-Style Error Toast */}
      {error && (
        <div className="fixed bottom-6 left-6 right-6 z-50 animate-slide-up">
          <div className="mx-auto max-w-md">
            <div className="rounded-apple-lg border border-error-200 bg-white p-4 shadow-apple">
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <div className="flex h-8 w-8 items-center justify-center rounded-apple bg-error-50">
                    <svg
                      className="h-4 w-4 text-error-500"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-text-primary">
                    Assessment Error
                  </h3>
                  <p className="mt-1 text-sm leading-relaxed text-text-secondary">
                    {error}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setError(null)}
                  className="flex-shrink-0 p-1 text-text-secondary hover:text-text-primary"
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
