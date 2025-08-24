/**
 * Complete Assessment Workflow Test Page
 * End-to-end testing of the assessment workflow with real data processing
 */

import React, { useState, useCallback } from 'react';
import { useAssessmentWorkflow } from '@/hooks/useAssessmentWorkflow';
import {
  AssessmentInput,
  AssessmentResults as AssessmentResultsType,
} from '@/lib/assessment/workflow';
import { validateAudioFile, validateImageFile } from '@/lib/assessment/validation';
import { ProgressSteps } from '@/components/ui/LoadingStates';
import { AssessmentResults } from '@/components/assessment/AssessmentResults';
import { ErrorDisplay } from '@/components/ui/ErrorDisplay';
import { Upload, Play, RotateCcw, Download } from 'lucide-react';

export default function AssessmentWorkflowTestPage() {
  const [sessionId] = useState(() => `test-session-${Date.now()}`);
  const [files, setFiles] = useState<{
    speech?: File;
    retinal?: File;
  }>({});
  const [validationResults, setValidationResults] = useState<any>({});

  const workflow = useAssessmentWorkflow(sessionId, {
    enableProgressTracking: true,
    enablePersistence: true,
    onStepCompleted: step => {
      console.log(`Step completed: ${step}`);
    },
    onError: error => {
      console.error('Workflow error:', error);
    },
    onCompleted: results => {
      console.log('Assessment completed:', results);
    },
  });

  // Handle file upload
  const handleFileUpload = useCallback(async (type: 'speech' | 'retinal', file: File) => {
    setFiles(prev => ({ ...prev, [type]: file }));

    // Validate file
    let validation;
    if (type === 'speech') {
      validation = await validateAudioFile(file);
    } else {
      validation = await validateImageFile(file);
    }

    setValidationResults((prev: any) => ({
      ...prev,
      [type]: validation,
    }));
  }, []);

  // Generate mock motor data
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

  // Generate mock cognitive data
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

    try {
      await workflow.executeAssessment(input);
    } catch (error) {
      console.error('Assessment failed:', error);
    }
  }, [sessionId, files, workflow, generateMockMotorData, generateMockCognitiveData]);

  // Export results
  const handleExport = useCallback(
    async (format: 'pdf' | 'json') => {
      if (!workflow.results) return;

      if (format === 'json') {
        const dataStr = JSON.stringify(workflow.results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `assessment-${sessionId}.json`;
        link.click();
        URL.revokeObjectURL(url);
      } else {
        // PDF export would be implemented here
        alert('PDF export not implemented in demo');
      }
    },
    [workflow.results, sessionId],
  );

  return (
    <div className='min-h-screen bg-gray-50 py-8'>
      <div className='mx-auto max-w-6xl px-4'>
        <div className='rounded-lg bg-white shadow-lg'>
          {/* Header */}
          <div className='border-b border-gray-200 p-6'>
            <h1 className='text-3xl font-bold text-gray-900'>Complete Assessment Workflow Test</h1>
            <p className='mt-2 text-gray-600'>
              End-to-end testing of the NeuraLens assessment workflow with real ML processing
            </p>
            <p className='mt-1 text-sm text-gray-500'>Session ID: {sessionId}</p>
          </div>

          {/* File Upload Section */}
          {!workflow.isRunning && !workflow.isCompleted && (
            <div className='border-b border-gray-200 p-6'>
              <h2 className='mb-4 text-xl font-semibold text-gray-900'>Upload Assessment Files</h2>

              <div className='grid grid-cols-1 gap-6 md:grid-cols-2'>
                {/* Speech File Upload */}
                <div className='rounded-lg border-2 border-dashed border-gray-300 p-6 text-center'>
                  <Upload className='mx-auto mb-3 h-8 w-8 text-gray-400' />
                  <h3 className='mb-2 text-lg font-medium text-gray-900'>Speech Audio</h3>
                  <input
                    type='file'
                    accept='audio/*'
                    onChange={e => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload('speech', file);
                    }}
                    className='hidden'
                    id='speech-upload'
                  />
                  <label
                    htmlFor='speech-upload'
                    className='inline-flex cursor-pointer items-center rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700'
                  >
                    Choose Audio File
                  </label>

                  {files.speech && (
                    <div className='mt-3 text-sm text-gray-600'>
                      <p>Selected: {files.speech.name}</p>
                      <p>Size: {(files.speech.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  )}

                  {validationResults.speech && (
                    <div className='mt-3'>
                      {validationResults.speech.isValid ? (
                        <p className='text-sm text-green-600'>✓ File validated successfully</p>
                      ) : (
                        <div className='text-sm text-red-600'>
                          <p>✗ Validation failed:</p>
                          <ul className='mt-1 list-inside list-disc'>
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
                <div className='rounded-lg border-2 border-dashed border-gray-300 p-6 text-center'>
                  <Upload className='mx-auto mb-3 h-8 w-8 text-gray-400' />
                  <h3 className='mb-2 text-lg font-medium text-gray-900'>Retinal Image</h3>
                  <input
                    type='file'
                    accept='image/*'
                    onChange={e => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload('retinal', file);
                    }}
                    className='hidden'
                    id='retinal-upload'
                  />
                  <label
                    htmlFor='retinal-upload'
                    className='inline-flex cursor-pointer items-center rounded-lg bg-green-600 px-4 py-2 text-white transition-colors hover:bg-green-700'
                  >
                    Choose Image File
                  </label>

                  {files.retinal && (
                    <div className='mt-3 text-sm text-gray-600'>
                      <p>Selected: {files.retinal.name}</p>
                      <p>Size: {(files.retinal.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  )}

                  {validationResults.retinal && (
                    <div className='mt-3'>
                      {validationResults.retinal.isValid ? (
                        <p className='text-sm text-green-600'>✓ File validated successfully</p>
                      ) : (
                        <div className='text-sm text-red-600'>
                          <p>✗ Validation failed:</p>
                          <ul className='mt-1 list-inside list-disc'>
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
                <button
                  onClick={startAssessment}
                  disabled={!files.speech && !files.retinal}
                  className='inline-flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50'
                >
                  <Play className='h-5 w-5' />
                  Start Complete Assessment
                </button>
                <p className='mt-2 text-sm text-gray-500'>
                  Assessment will include speech, retinal, motor, and cognitive analysis
                </p>
              </div>
            </div>
          )}

          {/* Progress Section */}
          {workflow.isRunning && (
            <div className='border-b border-gray-200 p-6'>
              <h2 className='mb-6 text-xl font-semibold text-gray-900'>Assessment Progress</h2>

              <ProgressSteps
                steps={[
                  'Upload',
                  'Validation',
                  'Speech',
                  'Retinal',
                  'Motor',
                  'Cognitive',
                  'NRI Fusion',
                  'Results',
                ]}
                currentStep={workflow.progress.completedSteps.length}
                completedSteps={workflow.progress.completedSteps.map((_, index) => index)}
                className='mb-6'
              />

              <div className='rounded-lg bg-gray-50 p-4'>
                <div className='mb-2 flex items-center justify-between'>
                  <span className='text-sm font-medium text-gray-700'>
                    Current Step: {workflow.progress.currentStep}
                  </span>
                  <span className='text-sm text-gray-600'>
                    {Math.round(workflow.progress.progressPercentage)}% Complete
                  </span>
                </div>

                <div className='h-2 w-full rounded-full bg-gray-200'>
                  <div
                    className='h-2 rounded-full bg-blue-600 transition-all duration-500'
                    style={{ width: `${workflow.progress.progressPercentage}%` }}
                  />
                </div>

                {workflow.progress.estimatedTimeRemaining && (
                  <p className='mt-2 text-xs text-gray-500'>
                    Estimated time remaining: {workflow.progress.estimatedTimeRemaining}s
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Error Section */}
          {workflow.hasError && (
            <div className='border-b border-gray-200 p-6'>
              <ErrorDisplay
                error={workflow.error || 'Unknown error occurred'}
                onRetry={() => workflow.resetWorkflow()}
              />
            </div>
          )}

          {/* Results Section */}
          {workflow.isCompleted && workflow.results && (
            <div className='p-6'>
              <AssessmentResults
                results={workflow.results as AssessmentResultsType}
                onExport={handleExport}
                onShare={() => alert('Share functionality not implemented in demo')}
                onCompare={() => alert('Compare functionality not implemented in demo')}
              />
            </div>
          )}

          {/* Reset Button */}
          {(workflow.isCompleted || workflow.hasError) && (
            <div className='border-t border-gray-200 p-6 text-center'>
              <button
                onClick={() => {
                  workflow.resetWorkflow();
                  setFiles({});
                  setValidationResults({});
                }}
                className='inline-flex items-center gap-2 rounded-lg border border-blue-600 px-6 py-3 text-blue-600 transition-colors hover:bg-blue-50'
              >
                <RotateCcw className='h-5 w-5' />
                Start New Assessment
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
