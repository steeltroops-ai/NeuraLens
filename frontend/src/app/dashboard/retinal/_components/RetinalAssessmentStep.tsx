/**
 * RetinalAssessmentStep Component
 * 
 * Step-by-step workflow component for retinal analysis:
 * 1. Image upload with drag-and-drop
 * 2. Validation and quality check
 * 3. Processing with progress
 * 4. Results display
 * 
 * Features:
 * - Wizard-style progression
 * - Keyboard navigation
 * - ARIA accessibility labels
 * - Error recovery with retry
 * 
 * Requirements: All (workflow)
 * 
 * @module app/dashboard/retinal/_components/RetinalAssessmentStep
 */

'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  CheckCircle2, 
  AlertCircle,
  Loader2,
  ArrowLeft,
  RefreshCw,
  HelpCircle,
  Eye,
  Activity
} from 'lucide-react';
import { useRetinalAnalysis, AnalysisStage } from '@/hooks/useRetinalAnalysis';
import { ImageUploadCard } from '@/components/retinal/ImageUploadCard';
import { RetinalResultsCard } from '@/components/retinal/RetinalResultsCard';

// ============================================================================
// Types
// ============================================================================

type WorkflowStep = 'upload' | 'validating' | 'processing' | 'results' | 'error';

interface StepIndicatorProps {
  currentStep: WorkflowStep;
  isProcessing: boolean;
}

interface RetinalAssessmentStepProps {
  /** Patient ID for the assessment */
  patientId?: string;
  /** Called when step changes */
  onStepChange?: (step: WorkflowStep) => void;
  /** Called when result is available */
  onResultAvailable?: (assessmentId: string) => void;
  /** Called when processing state changes */
  onProcessingChange?: (isProcessing: boolean) => void;
}

// ============================================================================
// Step Indicator Component
// ============================================================================

function StepIndicator({ currentStep, isProcessing }: StepIndicatorProps) {
  const steps = [
    { id: 'upload', label: 'Upload Image', icon: Upload },
    { id: 'validating', label: 'Validating', icon: CheckCircle2 },
    { id: 'processing', label: 'Processing', icon: Activity },
    { id: 'results', label: 'Results', icon: Eye },
  ];

  const getStepStatus = (stepId: string) => {
    const stepOrder = ['upload', 'validating', 'processing', 'results'];
    const currentIndex = stepOrder.indexOf(currentStep === 'error' ? 'upload' : currentStep);
    const stepIndex = stepOrder.indexOf(stepId);
    
    if (stepIndex < currentIndex) return 'complete';
    if (stepIndex === currentIndex) return 'current';
    return 'upcoming';
  };

  return (
    <nav 
      aria-label="Assessment progress"
      className="flex items-center justify-between mb-8"
    >
      <ol className="flex items-center w-full">
        {steps.map((step, index) => {
          const status = getStepStatus(step.id);
          const Icon = step.icon;
          const isLast = index === steps.length - 1;

          return (
            <li 
              key={step.id} 
              className={`flex items-center ${!isLast ? 'flex-1' : ''}`}
              aria-current={status === 'current' ? 'step' : undefined}
            >
              {/* Step Circle */}
              <div className="flex flex-col items-center">
                <div
                  className={`
                    flex h-10 w-10 items-center justify-center rounded-full border-2 transition-all
                    ${status === 'complete' 
                      ? 'border-cyan-500 bg-cyan-500 text-white' 
                      : status === 'current'
                        ? 'border-cyan-500 bg-white text-cyan-600'
                        : 'border-zinc-300 bg-white text-zinc-400'
                    }
                  `}
                  role="img"
                  aria-label={`${step.label}: ${status}`}
                >
                  {status === 'complete' ? (
                    <CheckCircle2 className="h-5 w-5" />
                  ) : status === 'current' && isProcessing ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </div>
                <span 
                  className={`
                    mt-2 text-xs font-medium
                    ${status === 'current' ? 'text-cyan-600' : 'text-zinc-500'}
                  `}
                >
                  {step.label}
                </span>
              </div>

              {/* Connector Line */}
              {!isLast && (
                <div 
                  className={`
                    flex-1 h-0.5 mx-4
                    ${status === 'complete' ? 'bg-cyan-500' : 'bg-zinc-200'}
                  `}
                  aria-hidden="true"
                />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

// ============================================================================
// Error Display Component
// ============================================================================

interface ErrorDisplayProps {
  message: string;
  onRetry: () => void;
  onBack: () => void;
  retryCount: number;
  maxRetries: number;
}

function ErrorDisplay({ message, onRetry, onBack, retryCount, maxRetries }: ErrorDisplayProps) {
  const canRetry = retryCount < maxRetries;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl border border-red-200 bg-red-50 p-8 text-center"
      role="alert"
      aria-live="assertive"
    >
      <div className="flex justify-center mb-4">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
          <AlertCircle className="h-8 w-8 text-red-600" />
        </div>
      </div>

      <h3 className="text-lg font-semibold text-red-900 mb-2">
        Analysis Failed
      </h3>
      
      <p className="text-red-700 mb-4 max-w-md mx-auto">
        {message}
      </p>

      {/* Recovery Steps */}
      <div className="text-left max-w-sm mx-auto mb-6">
        <p className="text-sm font-medium text-red-800 mb-2">Try these steps:</p>
        <ul className="space-y-1 text-sm text-red-700">
          <li className="flex items-start gap-2">
            <span className="mt-1 h-1.5 w-1.5 rounded-full bg-red-500 flex-shrink-0" />
            Check your internet connection
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-1 h-1.5 w-1.5 rounded-full bg-red-500 flex-shrink-0" />
            Ensure the image is a valid fundus photograph
          </li>
          <li className="flex items-start gap-2">
            <span className="mt-1 h-1.5 w-1.5 rounded-full bg-red-500 flex-shrink-0" />
            Try a different image file
          </li>
        </ul>
      </div>

      {/* Retry Info */}
      {retryCount > 0 && (
        <p className="text-xs text-red-600 mb-4">
          Retry attempt {retryCount} of {maxRetries}
        </p>
      )}

      {/* Action Buttons */}
      <div className="flex justify-center gap-3">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-4 py-2.5 border border-zinc-300 text-zinc-700 rounded-lg font-medium hover:bg-zinc-50 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2"
          aria-label="Go back to upload"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Upload
        </button>
        
        {canRetry && (
          <button
            onClick={onRetry}
            className="flex items-center gap-2 px-4 py-2.5 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
            aria-label={`Retry analysis, attempt ${retryCount + 1} of ${maxRetries}`}
          >
            <RefreshCw className="h-4 w-4" />
            Retry Analysis
          </button>
        )}
      </div>

      {/* Help Link */}
      <div className="mt-6 pt-4 border-t border-red-200">
        <a
          href="/support"
          className="inline-flex items-center gap-1 text-sm text-red-700 hover:text-red-900"
        >
          <HelpCircle className="h-4 w-4" />
          Contact Support
        </a>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Processing Display Component
// ============================================================================

interface ProcessingDisplayProps {
  progress: number;
  stage: AnalysisStage;
}

function ProcessingDisplay({ progress, stage }: ProcessingDisplayProps) {
  const stageMessages = {
    validating: 'Validating image quality...',
    uploading: 'Uploading image...',
    processing: 'Running AI analysis...',
    idle: 'Preparing...',
    complete: 'Analysis complete!',
    error: 'Analysis failed',
  };

  const stageDetails = {
    validating: 'Checking resolution, focus, and anatomical features',
    uploading: 'Securely transferring image to analysis server',
    processing: 'Extracting biomarkers and calculating risk score',
    idle: '',
    complete: '',
    error: '',
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50 to-cyan-50 p-8"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="flex flex-col items-center text-center">
        {/* Animated Icon */}
        <div className="relative mb-6">
          <div className="h-20 w-20 rounded-full border-4 border-blue-200 flex items-center justify-center">
            <Loader2 className="h-10 w-10 text-blue-600 animate-spin" />
          </div>
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-transparent border-t-cyan-500"
            animate={{ rotate: 360 }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
          />
        </div>

        {/* Stage Message */}
        <h3 className="text-lg font-semibold text-blue-900 mb-2">
          {stageMessages[stage] || 'Processing...'}
        </h3>
        <p className="text-sm text-blue-700 mb-6 max-w-md">
          {stageDetails[stage]}
        </p>

        {/* Progress Bar */}
        <div className="w-full max-w-md">
          <div className="flex justify-between text-sm text-blue-700 mb-2">
            <span>Progress</span>
            <span>{progress}%</span>
          </div>
          <div className="h-3 rounded-full bg-blue-200 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </div>

        {/* Accessibility Message */}
        <p className="sr-only">
          {stage === 'processing' 
            ? `Analysis in progress, ${progress}% complete` 
            : stageMessages[stage]
          }
        </p>
      </div>
    </motion.div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function RetinalAssessmentStep({
  patientId = 'DEMO-PATIENT',
  onStepChange,
  onResultAvailable,
  onProcessingChange,
}: RetinalAssessmentStepProps) {
  const {
    result,
    validation,
    loading,
    stage,
    uploadProgress,
    error,
    analyze,
    validate,
    downloadReport,
    reset,
    clearError,
  } = useRetinalAnalysis();

  const [currentStep, setCurrentStep] = useState<WorkflowStep>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 3;

  // For keyboard navigation
  const containerRef = useRef<HTMLDivElement>(null);

  // ============================================================================
  // Step Transitions
  // ============================================================================

  useEffect(() => {
    let newStep: WorkflowStep = 'upload';

    if (stage === 'validating') {
      newStep = 'validating';
    } else if (stage === 'uploading' || stage === 'processing') {
      newStep = 'processing';
    } else if (stage === 'complete' && result) {
      newStep = 'results';
    } else if (stage === 'error' || error) {
      newStep = 'error';
    }

    setCurrentStep(newStep);
    onStepChange?.(newStep);
  }, [stage, result, error, onStepChange]);

  // Notify parent of processing state
  useEffect(() => {
    onProcessingChange?.(loading);
  }, [loading, onProcessingChange]);

  // Notify parent when result is available
  useEffect(() => {
    if (result?.assessment_id) {
      onResultAvailable?.(result.assessment_id);
    }
  }, [result, onResultAvailable]);

  // ============================================================================
  // Handlers
  // ============================================================================

  const handleFileSelect = useCallback(async (file: File) => {
    setSelectedFile(file);
    setRetryCount(0);
    
    // Validate first
    const validationResult = await validate(file);
    
    if (validationResult?.is_valid) {
      // Proceed to analysis
      await analyze(file, patientId);
    }
  }, [validate, analyze, patientId]);

  const handleRetry = useCallback(async () => {
    if (!selectedFile || retryCount >= maxRetries) return;
    
    setRetryCount(prev => prev + 1);
    clearError();
    
    // Re-run analysis
    await analyze(selectedFile, patientId);
  }, [selectedFile, retryCount, maxRetries, clearError, analyze, patientId]);

  const handleBack = useCallback(() => {
    reset();
    setSelectedFile(null);
    setRetryCount(0);
    setCurrentStep('upload');
  }, [reset]);

  const handleNewAnalysis = useCallback(() => {
    handleBack();
  }, [handleBack]);

  const handleReportDownload = useCallback(async (assessmentId: string) => {
    await downloadReport(assessmentId);
  }, [downloadReport]);

  // ============================================================================
  // Keyboard Navigation
  // ============================================================================

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to reset on error
      if (e.key === 'Escape' && currentStep === 'error') {
        handleBack();
      }
      // Enter to retry on error
      if (e.key === 'Enter' && currentStep === 'error' && retryCount < maxRetries) {
        handleRetry();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentStep, retryCount, handleBack, handleRetry]);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div 
      ref={containerRef}
      className="space-y-6"
      role="region"
      aria-label="Retinal Analysis Assessment"
    >
      {/* Step Indicator */}
      <StepIndicator 
        currentStep={currentStep} 
        isProcessing={loading} 
      />

      {/* Step Content */}
      <AnimatePresence mode="wait">
        {/* Upload Step */}
        {currentStep === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <ImageUploadCard
              onFileSelect={handleFileSelect}
              disabled={loading}
              title="Upload Retinal Image"
              description="Upload a high-quality fundus photograph for AI-powered neurological risk analysis"
            />
          </motion.div>
        )}

        {/* Validating Step */}
        {currentStep === 'validating' && (
          <motion.div
            key="validating"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <ProcessingDisplay 
              progress={uploadProgress} 
              stage={stage} 
            />
          </motion.div>
        )}

        {/* Processing Step */}
        {currentStep === 'processing' && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <ProcessingDisplay 
              progress={uploadProgress} 
              stage={stage} 
            />
          </motion.div>
        )}

        {/* Results Step */}
        {currentStep === 'results' && result && (
          <motion.div
            key="results"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <RetinalResultsCard
              result={result}
              showVisualizations={true}
              enableReport={true}
              onReportDownload={handleReportDownload}
              onNewAnalysis={handleNewAnalysis}
            />
          </motion.div>
        )}

        {/* Error Step */}
        {currentStep === 'error' && error && (
          <motion.div
            key="error"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <ErrorDisplay
              message={error}
              onRetry={handleRetry}
              onBack={handleBack}
              retryCount={retryCount}
              maxRetries={maxRetries}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Screen Reader Announcements */}
      <div 
        className="sr-only" 
        role="status" 
        aria-live="polite"
        aria-atomic="true"
      >
        {currentStep === 'upload' && 'Ready to upload retinal image'}
        {currentStep === 'validating' && 'Validating image quality'}
        {currentStep === 'processing' && `Processing image, ${uploadProgress}% complete`}
        {currentStep === 'results' && 'Analysis complete, viewing results'}
        {currentStep === 'error' && `Error: ${error}`}
      </div>
    </div>
  );
}
