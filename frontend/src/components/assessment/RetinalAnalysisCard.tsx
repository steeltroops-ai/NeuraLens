/**
 * Retinal Analysis Card Component for Neuralens Dashboard
 *
 * This component provides the user interface for retinal analysis testing,
 * integrating with the Retinal Processor and Image Capture System for real-time
 * neurological assessment through fundus image analysis.
 *
 * Key Features:
 * - Neuro-Minimalist UI design with glassmorphism effects
 * - Image upload and camera capture functionality
 * - Real-time processing feedback with progress indicators
 * - Comprehensive error handling and user feedback
 * - Integration with NRI (Neuro-Risk Index) calculation
 *
 * Design System Compliance:
 * - Colors: Deep Blue #1e3a8a, Teal #0d9488, Electric Blue #3b82f6
 * - Typography: Inter font, 16px base, 8px grid system
 * - Glassmorphism: backdrop-filter blur(10px), opacity 0.85
 * - Responsive: 320px-1024px breakpoints
 * - Accessibility: WCAG 2.1 AA compliance
 */

'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  Upload,
  Eye,
  AlertCircle,
  CheckCircle,
  Loader,
  X,
  RotateCcw,
  Zap,
} from 'lucide-react';
import {
  RetinalResult,
  RetinalAnalysisError,
  RetinalProcessingState,
  RETINAL_ANALYSIS_CONSTANTS,
} from '../../types/retinal-analysis';
import { RetinalProcessor } from '../../lib/ml/retinal/retinal-processor';
import { ImageCaptureSystem } from '../../lib/ml/retinal/image-capture';

interface RetinalAnalysisCardProps {
  onResult?: (result: RetinalResult) => void;
  onError?: (error: RetinalAnalysisError) => void;
  className?: string;
  allowCamera?: boolean;
  allowUpload?: boolean;
}

export const RetinalAnalysisCard: React.FC<RetinalAnalysisCardProps> = ({
  onResult,
  onError,
  className = '',
  allowCamera = true,
  allowUpload = true,
}) => {
  // Component state management
  const [processingState, setProcessingState] =
    useState<RetinalProcessingState>({
      status: 'idle',
      progress: 0,
      currentStep: '',
    });

  const [analysisResult, setAnalysisResult] = useState<RetinalResult | null>(
    null
  );
  const [isInitialized, setIsInitialized] = useState(false);
  const [initializationError, setInitializationError] = useState<string | null>(
    null
  );
  const [showCamera, setShowCamera] = useState(false);

  // Refs for ML components
  const retinalProcessorRef = useRef<RetinalProcessor | null>(null);
  const imageCaptureRef = useRef<ImageCaptureSystem | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);

  /**
   * Initialize ML components on component mount
   * Sets up Retinal Processor and Image Capture System with error handling
   */
  useEffect(() => {
    const initializeComponents = async () => {
      try {
        console.log('[RetinalAnalysisCard] Initializing ML components...');

        // Initialize Retinal Processor with ONNX model
        retinalProcessorRef.current = new RetinalProcessor({
          debug: process.env.NODE_ENV === 'development',
        });

        await retinalProcessorRef.current.initialize();

        // Initialize Image Capture System
        imageCaptureRef.current = new ImageCaptureSystem(
          setProcessingState,
          process.env.NODE_ENV === 'development'
        );

        setIsInitialized(true);
        console.log('[RetinalAnalysisCard] Initialization complete');
      } catch (error) {
        console.error('[RetinalAnalysisCard] Initialization failed:', error);
        const errorMessage =
          error instanceof RetinalAnalysisError
            ? error.message
            : 'Failed to initialize retinal analysis components';

        setInitializationError(errorMessage);

        if (onError) {
          onError(
            error instanceof RetinalAnalysisError
              ? error
              : new RetinalAnalysisError(
                  errorMessage,
                  'INITIALIZATION_ERROR',
                  error
                )
          );
        }
      }
    };

    initializeComponents();

    // Cleanup on unmount
    return () => {
      retinalProcessorRef.current?.dispose();
      imageCaptureRef.current?.dispose();
    };
  }, [onError]);

  /**
   * Handle successful retinal analysis results
   * Updates state and notifies parent component
   */
  const handleAnalysisResult = useCallback(
    (result: RetinalResult) => {
      console.log('[RetinalAnalysisCard] Analysis result received:', result);

      setAnalysisResult(result);
      setProcessingState((prev) => ({ ...prev, status: 'complete' }));

      if (onResult) {
        onResult(result);
      }

      // Send results to backend API for NRI integration
      sendResultsToAPI(result);
    },
    [onResult]
  );

  /**
   * Handle retinal analysis errors
   * Updates error state and notifies parent component
   */
  const handleAnalysisError = useCallback(
    (error: RetinalAnalysisError) => {
      console.error('[RetinalAnalysisCard] Analysis error:', error);

      setProcessingState((prev) => ({
        ...prev,
        status: 'error',
        error: error.message,
      }));

      if (onError) {
        onError(error);
      }
    },
    [onError]
  );

  /**
   * Process uploaded or captured image
   */
  const processImage = useCallback(
    async (imageFile: File) => {
      try {
        if (!retinalProcessorRef.current || !isInitialized) {
          throw new RetinalAnalysisError(
            'Components not initialized',
            'NOT_INITIALIZED'
          );
        }

        console.log('[RetinalAnalysisCard] Processing retinal image...');

        setProcessingState((prev) => ({
          ...prev,
          status: 'analyzing',
          progress: 0.5,
          currentStep: 'Analyzing retinal features...',
        }));

        // Process image with ML model
        const result =
          await retinalProcessorRef.current.processImage(imageFile);

        handleAnalysisResult(result);
      } catch (error) {
        console.error('[RetinalAnalysisCard] Processing failed:', error);

        const analysisError =
          error instanceof RetinalAnalysisError
            ? error
            : new RetinalAnalysisError(
                'Failed to process retinal image',
                'PROCESSING_ERROR',
                error
              );

        handleAnalysisError(analysisError);
      }
    },
    [isInitialized, handleAnalysisResult, handleAnalysisError]
  );

  /**
   * Handle file upload
   */
  const handleFileUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file || !imageCaptureRef.current) return;

      try {
        const validatedFile =
          await imageCaptureRef.current.handleFileUpload(file);
        await processImage(validatedFile);
      } catch (error) {
        const analysisError =
          error instanceof RetinalAnalysisError
            ? error
            : new RetinalAnalysisError(
                'File upload failed',
                'UPLOAD_ERROR',
                error
              );

        handleAnalysisError(analysisError);
      }
    },
    [processImage, handleAnalysisError]
  );

  /**
   * Handle camera capture
   */
  const handleCameraCapture = useCallback(async () => {
    try {
      if (!imageCaptureRef.current) {
        throw new RetinalAnalysisError(
          'Image capture not initialized',
          'NOT_INITIALIZED'
        );
      }

      // Initialize camera if not already done
      if (!showCamera) {
        await imageCaptureRef.current.initializeCamera();
        setShowCamera(true);

        // Add video element to container
        const videoElement = imageCaptureRef.current.getVideoElement();
        if (videoElement && videoContainerRef.current) {
          videoContainerRef.current.appendChild(videoElement);
        }
        return;
      }

      // Capture image from camera
      const capturedFile = await imageCaptureRef.current.captureImage();
      await processImage(capturedFile);

      // Hide camera after capture
      setShowCamera(false);
    } catch (error) {
      const analysisError =
        error instanceof RetinalAnalysisError
          ? error
          : new RetinalAnalysisError(
              'Camera capture failed',
              'CAMERA_ERROR',
              error
            );

      handleAnalysisError(analysisError);
    }
  }, [showCamera, processImage, handleAnalysisError]);

  /**
   * Send analysis results to backend API
   */
  const sendResultsToAPI = async (result: RetinalResult) => {
    try {
      const response = await fetch('/api/retinal', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          result,
          sessionId: `session_${Date.now()}`,
          timestamp: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const apiResult = await response.json();
      console.log('[RetinalAnalysisCard] API response:', apiResult);
    } catch (error) {
      console.warn('[RetinalAnalysisCard] API request failed:', error);
      // Don't throw error - API failure shouldn't break the UI
    }
  };

  /**
   * Reset component to initial state
   */
  const handleReset = useCallback(() => {
    setAnalysisResult(null);
    setProcessingState({
      status: 'idle',
      progress: 0,
      currentStep: '',
    });
    setShowCamera(false);

    // Clear file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  /**
   * Render upload interface
   */
  const renderUploadInterface = () => {
    if (processingState.status !== 'idle') return null;

    return (
      <div className="space-y-4">
        {/* Upload Options */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {allowUpload && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => fileInputRef.current?.click()}
              className="flex flex-col items-center justify-center space-y-2 rounded-xl border-2 border-dashed border-blue-300 bg-blue-50/50 p-6 text-blue-600 transition-all duration-200 hover:border-blue-400 hover:bg-blue-50/70"
            >
              <Upload className="h-8 w-8" />
              <span className="font-medium">Upload Image</span>
              <span className="text-sm text-blue-500">
                JPEG, PNG up to 10MB
              </span>
            </motion.button>
          )}

          {allowCamera && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleCameraCapture}
              className="flex flex-col items-center justify-center space-y-2 rounded-xl border-2 border-dashed border-teal-300 bg-teal-50/50 p-6 text-teal-600 transition-all duration-200 hover:border-teal-400 hover:bg-teal-50/70"
            >
              <Camera className="h-8 w-8" />
              <span className="font-medium">Use Camera</span>
              <span className="text-sm text-teal-500">
                Capture retinal image
              </span>
            </motion.button>
          )}
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept={RETINAL_ANALYSIS_CONSTANTS.SUPPORTED_FORMATS.join(',')}
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>
    );
  };

  /**
   * Render camera preview
   */
  const renderCameraPreview = () => {
    if (!showCamera) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className="space-y-4"
      >
        <div className="relative overflow-hidden rounded-xl bg-gray-900">
          <div ref={videoContainerRef} className="aspect-video w-full" />

          {/* Camera controls */}
          <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 space-x-4">
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={handleCameraCapture}
              className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-gray-900 shadow-lg"
            >
              <Camera className="h-6 w-6" />
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setShowCamera(false)}
              className="flex h-12 w-12 items-center justify-center rounded-full bg-red-500 text-white shadow-lg"
            >
              <X className="h-6 w-6" />
            </motion.button>
          </div>
        </div>
      </motion.div>
    );
  };

  /**
   * Render processing status
   */
  const renderProcessingStatus = () => {
    if (processingState.status === 'idle') return null;

    const statusConfig = {
      uploading: { color: 'blue', icon: Upload },
      preprocessing: { color: 'yellow', icon: Loader },
      analyzing: { color: 'purple', icon: Eye },
      complete: { color: 'green', icon: CheckCircle },
      error: { color: 'red', icon: AlertCircle },
    };

    const config = statusConfig[processingState.status];
    const Icon = config.icon;

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-3"
      >
        <div className="flex items-center space-x-3">
          <Icon
            className={`h-5 w-5 text-${config.color}-600 ${processingState.status === 'analyzing' ? 'animate-pulse' : ''}`}
          />
          <span className="font-medium text-gray-900">
            {processingState.currentStep}
          </span>
        </div>

        {processingState.progress > 0 && (
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div
              className={`bg-${config.color}-600 h-2 rounded-full transition-all duration-300`}
              style={{ width: `${processingState.progress * 100}%` }}
            />
          </div>
        )}

        {processingState.error && (
          <div className="flex items-center space-x-2 text-red-600">
            <AlertCircle className="h-4 w-4" />
            <span className="text-sm">{processingState.error}</span>
          </div>
        )}
      </motion.div>
    );
  };

  // Show initialization error if components failed to load
  if (initializationError) {
    return (
      <div
        className={`rounded-2xl border border-red-200 bg-red-50/80 p-6 backdrop-blur-sm ${className}`}
      >
        <div className="flex items-center space-x-2 text-red-600">
          <AlertCircle className="h-5 w-5" />
          <h3 className="font-semibold">Retinal Analysis Unavailable</h3>
        </div>
        <p className="mt-2 text-sm text-red-600">{initializationError}</p>
      </div>
    );
  }

  return (
    <div
      className={`rounded-2xl border border-gray-200 bg-white/85 p-6 shadow-lg backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className="mb-6">
        <h3 className="mb-2 text-xl font-semibold text-gray-900">
          Retinal Analysis
        </h3>
        <p className="text-sm text-gray-600">
          Upload or capture a retinal fundus image for neurological assessment
        </p>
      </div>

      {/* Main Content */}
      <div className="space-y-6">
        {/* Upload Interface */}
        {renderUploadInterface()}

        {/* Camera Preview */}
        <AnimatePresence>{renderCameraPreview()}</AnimatePresence>

        {/* Processing Status */}
        {renderProcessingStatus()}

        {/* Analysis Results */}
        <AnimatePresence>
          {analysisResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6 rounded-xl border border-gray-200 bg-white/80 p-4 backdrop-blur-sm"
            >
              <div className="mb-4 flex items-center space-x-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                <h4 className="font-semibold text-gray-900">
                  Analysis Complete
                </h4>
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {/* Primary Metrics */}
                <div className="space-y-3">
                  <div>
                    <span className="text-sm text-gray-600">
                      Vascular Score:
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="text-lg font-bold text-blue-600">
                        {(analysisResult.vascularScore * 100).toFixed(1)}%
                      </div>
                      <div
                        className={`rounded-full px-2 py-1 text-xs font-medium ${
                          analysisResult.vascularScore < 0.3
                            ? 'bg-green-100 text-green-700'
                            : analysisResult.vascularScore < 0.7
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-red-100 text-red-700'
                        }`}
                      >
                        {analysisResult.vascularScore < 0.3
                          ? 'Normal'
                          : analysisResult.vascularScore < 0.7
                            ? 'Moderate'
                            : 'High Risk'}
                      </div>
                    </div>
                  </div>

                  <div>
                    <span className="text-sm text-gray-600">
                      Cup-Disc Ratio:
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="text-lg font-bold text-teal-600">
                        {analysisResult.cupDiscRatio.toFixed(2)}
                      </div>
                      <div
                        className={`rounded-full px-2 py-1 text-xs font-medium ${
                          analysisResult.cupDiscRatio < 0.4
                            ? 'bg-green-100 text-green-700'
                            : analysisResult.cupDiscRatio < 0.6
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-red-100 text-red-700'
                        }`}
                      >
                        {analysisResult.cupDiscRatio < 0.4
                          ? 'Normal'
                          : analysisResult.cupDiscRatio < 0.6
                            ? 'Elevated'
                            : 'High Risk'}
                      </div>
                    </div>
                  </div>

                  <div>
                    <span className="text-sm text-gray-600">Confidence:</span>
                    <div className="text-lg font-bold text-purple-600">
                      {(analysisResult.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Risk Features */}
                <div className="space-y-2">
                  <h5 className="text-sm font-medium text-gray-700">
                    Risk Features
                  </h5>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Vessel Density:</span>
                      <span className="font-medium">
                        {(
                          analysisResult.riskFeatures.vesselDensity * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Tortuosity Index:</span>
                      <span className="font-medium">
                        {analysisResult.riskFeatures.tortuosityIndex.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">AV Ratio:</span>
                      <span className="font-medium">
                        {analysisResult.riskFeatures.arteriovenousRatio.toFixed(
                          2
                        )}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Image Quality:</span>
                      <span className="font-medium">
                        {(
                          analysisResult.riskFeatures.imageQuality * 100
                        ).toFixed(0)}
                        %
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="mt-4 border-t border-gray-200 pt-4">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>
                    Processing:{' '}
                    {analysisResult.metadata.processingTime.toFixed(0)}ms
                  </span>
                  <span>Model: {analysisResult.metadata.modelVersion}</span>
                  <span>
                    Quality:{' '}
                    {(analysisResult.riskFeatures.imageQuality * 100).toFixed(
                      0
                    )}
                    %
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Reset Button */}
        {(processingState.status === 'complete' ||
          processingState.status === 'error') && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleReset}
            className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-blue-600 transition-colors hover:text-blue-700"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Analyze Another Image</span>
          </motion.button>
        )}
      </div>
    </div>
  );
};
