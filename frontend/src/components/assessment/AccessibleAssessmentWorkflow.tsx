/**
 * Accessible Assessment Workflow Component
 * WCAG 2.1 AA compliant assessment workflow with comprehensive accessibility features
 */

import React, { useState, useCallback, useRef } from 'react';
import { useAssessmentWorkflow } from '@/hooks/useAssessmentWorkflow';
import {
  useAssessmentAccessibility,
  useFocusManagement,
  useSkipLink,
  useLiveRegion,
} from '@/hooks/useAccessibility';
import {
  AssessmentInput,
  AssessmentResults as AssessmentResultsType,
} from '@/lib/assessment/workflow';
import {
  validateAudioFile,
  validateImageFile,
  ValidationResult,
} from '@/lib/assessment/validation';
import { LoadingButton, AssessmentLoading, LoadingTimeout } from '@/components/ui/LoadingStates';
import { AssessmentResults } from '@/components/assessment/AssessmentResults';
import { ErrorBoundary, AssessmentErrorFallback } from '@/components/ui/ErrorDisplay';
import { Upload, Play, RotateCcw, CheckCircle, AlertTriangle } from 'lucide-react';

interface AccessibleAssessmentWorkflowProps {
  sessionId: string;
  onComplete?: (results: any) => void;
  className?: string;
}

export function AccessibleAssessmentWorkflow({
  sessionId,
  onComplete,
  className = '',
}: AccessibleAssessmentWorkflowProps) {
  const [files, setFiles] = useState<{
    speech?: File;
    retinal?: File;
  }>({});
  const [validationResults, setValidationResults] = useState<{
    speech?: ValidationResult;
    retinal?: ValidationResult;
  }>({});
  const [currentStep, setCurrentStep] = useState(0);

  // Accessibility hooks
  const {
    announceStepChange,
    announceAssessmentComplete,
    announceAssessmentError,
    announceFileUpload,
    announceValidationError,
    prefersReducedMotion,
  } = useAssessmentAccessibility();

  const { saveFocus, restoreFocus } = useFocusManagement();
  const { SkipLink, targetRef } = useSkipLink();
  const { announce, LiveRegion } = useLiveRegion('polite');

  // Assessment workflow
  const workflow = useAssessmentWorkflow(sessionId, {
    enableProgressTracking: true,
    enablePersistence: true,
    onStepCompleted: step => {
      const stepNumber = getStepNumber(step);
      announceStepChange(step, stepNumber, 8);
      setCurrentStep(stepNumber);
    },
    onError: error => {
      announceAssessmentError(error);
    },
    onCompleted: results => {
      announceAssessmentComplete(results.overallRiskCategory, results.nriResult?.nri_score || 0);
      onComplete?.(results);
    },
  });

  // Refs for focus management
  const fileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({});
  const startButtonRef = useRef<HTMLButtonElement>(null);

  // Helper function to get step number
  const getStepNumber = useCallback((step: string): number => {
    const steps = [
      'upload',
      'validation',
      'speech_processing',
      'retinal_processing',
      'motor_processing',
      'cognitive_processing',
      'nri_fusion',
      'results',
    ];
    return steps.indexOf(step) + 1;
  }, []);

  // Handle file upload with accessibility announcements
  const handleFileUpload = useCallback(
    async (type: 'speech' | 'retinal', file: File) => {
      setFiles(prev => ({ ...prev, [type]: file }));
      announceFileUpload(file.name, type);

      // Validate file
      let validation;
      if (type === 'speech') {
        validation = await validateAudioFile(file);
      } else {
        validation = await validateImageFile(file);
      }

      setValidationResults(prev => ({
        ...prev,
        [type]: validation,
      }));

      // Announce validation results
      if (!validation.isValid) {
        announceValidationError(validation.errors);
      } else {
        announce(`${type} file validated successfully`);
      }
    },
    [announceFileUpload, announceValidationError, announce],
  );

  // Generate mock data
  const generateMockMotorData = useCallback(() => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      data.push({
        x: Math.sin(i * 0.1) * 0.1 + Math.random() * 0.05,
        y: Math.cos(i * 0.1) * 0.1 + Math.random() * 0.05,
        z: 9.8 + Math.random() * 0.1,
      });
    }
    return data;
  }, []);

  const generateMockCognitiveData = useCallback(() => {
    const responseTimes = [];
    const accuracy = [];

    for (let i = 0; i < 20; i++) {
      responseTimes.push(1000 + Math.random() * 500);
      accuracy.push(Math.random() > 0.2 ? 1 : 0);
    }

    return {
      response_times: responseTimes,
      accuracy,
      memory: {
        immediate_recall: 0.8 + Math.random() * 0.2,
        delayed_recall: 0.7 + Math.random() * 0.2,
      },
      attention: {
        sustained_attention: 0.85 + Math.random() * 0.15,
      },
      executive: {
        planning: 0.75 + Math.random() * 0.2,
        flexibility: 0.8 + Math.random() * 0.15,
      },
    };
  }, []);

  // Start assessment
  const startAssessment = useCallback(async () => {
    const input: AssessmentInput = {
      sessionId,
      speechFile: files.speech,
      retinalImage: files.retinal,
      motorData: {
        accelerometer: generateMockMotorData(),
      },
      cognitiveData: generateMockCognitiveData(),
    };

    // Save focus before starting
    saveFocus();
    announce('Starting assessment workflow');

    try {
      await workflow.executeAssessment(input);
    } catch (error) {
      console.error('Assessment failed:', error);
    }
  }, [
    sessionId,
    files,
    workflow,
    generateMockMotorData,
    generateMockCognitiveData,
    saveFocus,
    announce,
  ]);

  // Reset workflow
  const resetWorkflow = useCallback(() => {
    workflow.resetWorkflow();
    setFiles({});
    setValidationResults({});
    setCurrentStep(0);
    restoreFocus();
    announce('Assessment workflow reset');
  }, [workflow, restoreFocus, announce]);

  // Keyboard event handlers
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === 'Escape' && workflow.isRunning) {
        workflow.cancelAssessment();
        announce('Assessment cancelled');
      }
    },
    [workflow, announce],
  );

  return (
    <ErrorBoundary
      fallback={AssessmentErrorFallback}
      onError={error => announceAssessmentError(error.message)}
      level='component'
      name='AccessibleAssessmentWorkflow'
    >
      <div
        className={`rounded-lg bg-white shadow-lg ${className}`}
        onKeyDown={handleKeyDown}
        role='main'
        aria-label='Assessment workflow'
      >
        {/* Skip link */}
        <SkipLink href='#assessment-content'>Skip to assessment content</SkipLink>

        {/* Live region for announcements */}
        <LiveRegion />

        {/* Header */}
        <div className='border-b border-gray-200 p-6'>
          <h1 className='text-3xl font-bold text-gray-900' id='assessment-title'>
            MediLens Assessment Workflow
          </h1>
          <p className='mt-2 text-gray-600'>
            Complete neurological assessment with real-time ML processing
          </p>
          <p className='mt-1 text-sm text-gray-500'>
            Session ID: <span className='font-mono'>{sessionId}</span>
          </p>
        </div>

        {/* Main content */}
        <div
          id='assessment-content'
          ref={targetRef as React.RefObject<HTMLDivElement>}
          tabIndex={-1}
        >
          {/* File Upload Section */}
          {!workflow.isRunning && !workflow.isCompleted && (
            <section className='border-b border-gray-200 p-6' aria-labelledby='upload-heading'>
              <h2 id='upload-heading' className='mb-4 text-xl font-semibold text-gray-900'>
                Upload Assessment Files
              </h2>

              <div
                className='grid grid-cols-1 gap-6 md:grid-cols-2'
                role='group'
                aria-label='File upload options'
              >
                {/* Speech File Upload */}
                <div className='rounded-lg border-2 border-dashed border-gray-300 p-6 text-center focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2'>
                  <Upload className='mx-auto mb-3 h-8 w-8 text-gray-400' aria-hidden='true' />
                  <h3 className='mb-2 text-lg font-medium text-gray-900'>Speech Audio</h3>
                  <p className='mb-4 text-sm text-gray-600'>
                    Upload an audio file for speech analysis (WAV, MP3, M4A)
                  </p>

                  <input
                    ref={el => {
                      fileInputRefs.current.speech = el;
                    }}
                    type='file'
                    accept='audio/*'
                    onChange={e => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload('speech', file);
                    }}
                    className='sr-only'
                    id='speech-upload'
                    aria-describedby='speech-description'
                  />
                  <label
                    htmlFor='speech-upload'
                    className='inline-flex cursor-pointer items-center rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                  >
                    Choose Audio File
                  </label>
                  <div id='speech-description' className='sr-only'>
                    Upload an audio file for speech pattern analysis. Supported formats: WAV, MP3,
                    M4A. Maximum file size: 50MB.
                  </div>

                  {files.speech && (
                    <div className='mt-3 text-sm text-gray-600' role='status' aria-live='polite'>
                      <p>
                        <strong>Selected:</strong> {files.speech.name}
                      </p>
                      <p>
                        <strong>Size:</strong> {(files.speech.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  )}

                  {validationResults.speech && (
                    <div className='mt-3' role='alert'>
                      {validationResults.speech.isValid ? (
                        <div className='flex items-center gap-2 text-sm text-green-600'>
                          <CheckCircle className='h-4 w-4' aria-hidden='true' />
                          <span>File validated successfully</span>
                        </div>
                      ) : (
                        <div className='text-sm text-red-600'>
                          <div className='mb-2 flex items-center gap-2'>
                            <AlertTriangle className='h-4 w-4' aria-hidden='true' />
                            <span>Validation failed:</span>
                          </div>
                          <ul className='list-inside list-disc space-y-1'>
                            {validationResults.speech.errors.map((error: string, index: number) => (
                              <li key={index}>{error}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Retinal Image Upload */}
                <div className='rounded-lg border-2 border-dashed border-gray-300 p-6 text-center focus-within:border-green-500 focus-within:ring-2 focus-within:ring-green-500 focus-within:ring-offset-2'>
                  <Upload className='mx-auto mb-3 h-8 w-8 text-gray-400' aria-hidden='true' />
                  <h3 className='mb-2 text-lg font-medium text-gray-900'>Retinal Image</h3>
                  <p className='mb-4 text-sm text-gray-600'>
                    Upload a retinal fundus image for analysis (JPEG, PNG, TIFF)
                  </p>

                  <input
                    ref={el => {
                      fileInputRefs.current.retinal = el;
                    }}
                    type='file'
                    accept='image/*'
                    onChange={e => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload('retinal', file);
                    }}
                    className='sr-only'
                    id='retinal-upload'
                    aria-describedby='retinal-description'
                  />
                  <label
                    htmlFor='retinal-upload'
                    className='inline-flex cursor-pointer items-center rounded-lg bg-green-600 px-4 py-2 text-white transition-colors hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2'
                  >
                    Choose Image File
                  </label>
                  <div id='retinal-description' className='sr-only'>
                    Upload a retinal fundus image for vessel and optic disc analysis. Supported
                    formats: JPEG, PNG, TIFF. Maximum file size: 20MB.
                  </div>

                  {files.retinal && (
                    <div className='mt-3 text-sm text-gray-600' role='status' aria-live='polite'>
                      <p>
                        <strong>Selected:</strong> {files.retinal.name}
                      </p>
                      <p>
                        <strong>Size:</strong> {(files.retinal.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  )}

                  {validationResults.retinal && (
                    <div className='mt-3' role='alert'>
                      {validationResults.retinal.isValid ? (
                        <div className='flex items-center gap-2 text-sm text-green-600'>
                          <CheckCircle className='h-4 w-4' aria-hidden='true' />
                          <span>File validated successfully</span>
                        </div>
                      ) : (
                        <div className='text-sm text-red-600'>
                          <div className='mb-2 flex items-center gap-2'>
                            <AlertTriangle className='h-4 w-4' aria-hidden='true' />
                            <span>Validation failed:</span>
                          </div>
                          <ul className='list-inside list-disc space-y-1'>
                            {validationResults.retinal.errors.map(
                              (error: string, index: number) => (
                                <li key={index}>{error}</li>
                              ),
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Start Assessment Button */}
              <div className='mt-6 text-center'>
                <LoadingButton
                  loading={false}
                  onClick={startAssessment}
                  disabled={!files.speech && !files.retinal}
                  loadingText='Starting assessment...'
                  className='inline-flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50'
                  aria-describedby='start-description'
                >
                  <Play className='h-5 w-5' aria-hidden='true' />
                  Start Complete Assessment
                </LoadingButton>
                <div id='start-description' className='mt-2 text-sm text-gray-500'>
                  Assessment will include speech, retinal, motor, and cognitive analysis
                </div>
              </div>
            </section>
          )}

          {/* Progress Section */}
          {workflow.isRunning && (
            <LoadingTimeout
              isLoading={workflow.isRunning}
              timeout={120000} // 2 minutes
              onTimeout={() => announceAssessmentError('Assessment timeout')}
              timeoutMessage='The assessment is taking longer than expected. This may be due to network issues or high server load.'
            >
              <section className='border-b border-gray-200 p-6' aria-labelledby='progress-heading'>
                <h2 id='progress-heading' className='mb-6 text-xl font-semibold text-gray-900'>
                  Assessment Progress
                </h2>

                <div
                  role='progressbar'
                  aria-valuenow={Math.round(workflow.progress.progressPercentage)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`Assessment progress: ${Math.round(workflow.progress.progressPercentage)}% complete`}
                  className='mb-6'
                >
                  <AssessmentLoading
                    type={workflow.progress.currentStep as any}
                    progress={workflow.progress.progressPercentage}
                    message={`Processing ${workflow.progress.currentStep}...`}
                  />
                </div>

                <div className='rounded-lg bg-gray-50 p-4'>
                  <div className='mb-2 flex items-center justify-between'>
                    <span className='text-sm font-medium text-gray-700'>
                      Current Step: {workflow.progress.currentStep}
                    </span>
                    <span className='text-sm text-gray-600'>
                      {Math.round(workflow.progress.progressPercentage)}% Complete
                    </span>
                  </div>

                  <div className='h-2 w-full rounded-full bg-gray-200' role='presentation'>
                    <div
                      className={`h-2 rounded-full transition-all ${prefersReducedMotion ? '' : 'duration-500'} bg-blue-600`}
                      style={{ width: `${workflow.progress.progressPercentage}%` }}
                    />
                  </div>

                  {workflow.progress.estimatedTimeRemaining && (
                    <p className='mt-2 text-xs text-gray-500'>
                      Estimated time remaining: {workflow.progress.estimatedTimeRemaining}s
                    </p>
                  )}
                </div>

                <button
                  onClick={() => {
                    workflow.cancelAssessment();
                    announce('Assessment cancelled');
                  }}
                  className='mt-4 rounded text-sm text-gray-600 hover:text-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2'
                  aria-label='Cancel assessment'
                >
                  Press Escape or click here to cancel
                </button>
              </section>
            </LoadingTimeout>
          )}

          {/* Error Section */}
          {workflow.hasError && (
            <section
              className='border-b border-gray-200 p-6'
              role='alert'
              aria-labelledby='error-heading'
            >
              <h2 id='error-heading' className='sr-only'>
                Assessment Error
              </h2>
              <AssessmentErrorFallback
                error={new Error(workflow.error || 'Unknown error')}
                resetError={resetWorkflow}
                onRetryAssessment={startAssessment}
                assessmentStep={workflow.progress.currentStep}
              />
            </section>
          )}

          {/* Results Section */}
          {workflow.isCompleted && workflow.results && (
            <section className='p-6' aria-labelledby='results-heading'>
              <h2 id='results-heading' className='sr-only'>
                Assessment Results
              </h2>
              <AssessmentResults
                results={workflow.results as AssessmentResultsType}
                onExport={format => announce(`Exporting results as ${format}`)}
                onShare={() => announce('Sharing results')}
                onCompare={() => announce('Opening comparison view')}
              />
            </section>
          )}

          {/* Reset Button */}
          {(workflow.isCompleted || workflow.hasError) && (
            <div className='border-t border-gray-200 p-6 text-center'>
              <LoadingButton
                loading={false}
                onClick={resetWorkflow}
                className='inline-flex items-center gap-2 rounded-lg border border-blue-600 px-6 py-3 text-blue-600 transition-colors hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                aria-label='Start new assessment'
              >
                <RotateCcw className='h-5 w-5' aria-hidden='true' />
                Start New Assessment
              </LoadingButton>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
}
