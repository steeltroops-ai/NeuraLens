/**
 * Speech Analysis Card Component for Neuralens Dashboard
 *
 * This component provides the user interface for speech analysis testing,
 * integrating with the Speech Processor and Audio Recorder for real-time
 * neurological assessment through voice pattern analysis.
 *
 * Key Features:
 * - Neuro-Minimalist UI design with glassmorphism effects
 * - Real-time audio level visualization during recording
 * - Progress tracking with 30-second recording duration
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

import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Play, Square, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import React, { useState, useEffect, useCallback, useRef } from 'react';

import { AudioRecorder } from '../../lib/ml/audio-recorder';
import { SpeechProcessor } from '../../lib/ml/speech-processor';
import { SpeechAnalysisError, SPEECH_ANALYSIS_CONSTANTS } from '../../types/speech-analysis';

import type { SpeechResult, RecordingState } from '../../types/speech-analysis';

interface SpeechAnalysisCardProps {
  onResult?: (result: SpeechResult) => void;
  onError?: (error: SpeechAnalysisError) => void;
  className?: string;
}

export const SpeechAnalysisCard: React.FC<SpeechAnalysisCardProps> = ({
  onResult,
  onError,
  className = '',
}) => {
  // Component state management
  const [recordingState, setRecordingState] = useState<RecordingState>({
    status: 'idle',
    progress: 0,
    audioLevel: 0,
    recordedDuration: 0,
  });

  const [analysisResult, setAnalysisResult] = useState<SpeechResult | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [initializationError, setInitializationError] = useState<string | null>(null);

  // Refs for ML components
  const speechProcessorRef = useRef<SpeechProcessor | null>(null);
  const audioRecorderRef = useRef<AudioRecorder | null>(null);

  /**
   * Initialize ML components on component mount
   * Sets up Speech Processor and Audio Recorder with error handling
   */
  useEffect(() => {
    const initializeComponents = async () => {
      try {
        console.log('[SpeechAnalysisCard] Initializing ML components...');

        // Initialize Speech Processor with ONNX model
        speechProcessorRef.current = new SpeechProcessor({
          debug: process.env.NODE_ENV === 'development',
        });

        await speechProcessorRef.current.initialize();

        // Initialize Audio Recorder with WebRTC
        audioRecorderRef.current = new AudioRecorder(
          {
            sampleRate: SPEECH_ANALYSIS_CONSTANTS.TARGET_SAMPLE_RATE,
            duration: SPEECH_ANALYSIS_CONSTANTS.DEFAULT_DURATION,
            minAudioLevel: SPEECH_ANALYSIS_CONSTANTS.MIN_AUDIO_LEVEL,
            noiseReduction: true,
          },
          setRecordingState,
          process.env.NODE_ENV === 'development',
        );

        await audioRecorderRef.current.initialize();

        setIsInitialized(true);
        console.log('[SpeechAnalysisCard] Initialization complete');
      } catch (error) {
        console.error('[SpeechAnalysisCard] Initialization failed:', error);
        const errorMessage =
          error instanceof SpeechAnalysisError
            ? error.message
            : 'Failed to initialize speech analysis components';

        setInitializationError(errorMessage);

        if (onError) {
          onError(
            error instanceof SpeechAnalysisError
              ? error
              : new SpeechAnalysisError(errorMessage, 'INITIALIZATION_ERROR', error),
          );
        }
      }
    };

    initializeComponents();

    // Cleanup on unmount
    return () => {
      speechProcessorRef.current?.dispose();
      audioRecorderRef.current?.dispose();
    };
  }, [onError]);

  /**
   * Start speech recording and analysis
   * Initiates 30-second audio capture with real-time feedback
   */
  const handleStartRecording = useCallback(async () => {
    try {
      if (!audioRecorderRef.current || !isInitialized) {
        throw new SpeechAnalysisError('Components not initialized', 'NOT_INITIALIZED');
      }

      console.log('[SpeechAnalysisCard] Starting speech recording...');

      // Clear previous results
      setAnalysisResult(null);

      // Start recording
      await audioRecorderRef.current.startRecording();
    } catch (error) {
      console.error('[SpeechAnalysisCard] Failed to start recording:', error);

      const analysisError =
        error instanceof SpeechAnalysisError
          ? error
          : new SpeechAnalysisError('Failed to start recording', 'RECORDING_ERROR', error);

      if (onError) {
        onError(analysisError);
      }
    }
  }, [isInitialized, onError]);

  /**
   * Stop recording and process audio
   * Ends recording and runs ML analysis on captured audio
   */
  const handleStopRecording = useCallback(async () => {
    try {
      if (!audioRecorderRef.current || !speechProcessorRef.current) {
        throw new SpeechAnalysisError('Components not available', 'COMPONENTS_NOT_AVAILABLE');
      }

      console.log('[SpeechAnalysisCard] Stopping recording and processing...');

      // Stop recording
      audioRecorderRef.current.stopRecording();

      // Get recorded audio buffer
      const audioBuffer = await audioRecorderRef.current.getAudioBuffer();

      // Process audio with ML model
      const result = await speechProcessorRef.current.process(audioBuffer);

      // Update state with results
      setAnalysisResult(result);
      setRecordingState(prev => ({ ...prev, status: 'complete' }));

      console.log('[SpeechAnalysisCard] Analysis complete:', result);

      // Send results to parent component
      if (onResult) {
        onResult(result);
      }

      // Send results to backend API for NRI integration
      await sendResultsToAPI(result);
    } catch (error) {
      console.error('[SpeechAnalysisCard] Processing failed:', error);

      const analysisError =
        error instanceof SpeechAnalysisError
          ? error
          : new SpeechAnalysisError('Failed to process audio', 'PROCESSING_ERROR', error);

      setRecordingState(prev => ({
        ...prev,
        status: 'error',
        error: analysisError.message,
      }));

      if (onError) {
        onError(analysisError);
      }
    }
  }, [onResult, onError]);

  /**
   * Send analysis results to backend API
   * Integrates with FastAPI endpoint for NRI calculation
   */
  const sendResultsToAPI = async (result: SpeechResult) => {
    try {
      const response = await fetch('/api/speech', {
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
      console.log('[SpeechAnalysisCard] API response:', apiResult);
    } catch (error) {
      console.warn('[SpeechAnalysisCard] API request failed:', error);
      // Don't throw error - API failure shouldn't break the UI
    }
  };

  /**
   * Reset component to initial state
   */
  const handleReset = useCallback(() => {
    setAnalysisResult(null);
    setRecordingState({
      status: 'idle',
      progress: 0,
      audioLevel: 0,
      recordedDuration: 0,
    });
  }, []);

  /**
   * Render recording button with appropriate state
   */
  const renderRecordingButton = () => {
    const { status } = recordingState;

    if (status === 'recording') {
      return (
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleStopRecording}
          className='flex h-16 w-16 items-center justify-center rounded-full bg-red-500 text-white shadow-lg transition-all duration-200 hover:bg-red-600 hover:shadow-xl'
          aria-label='Stop recording'
        >
          <Square className='h-6 w-6' />
        </motion.button>
      );
    }

    if (status === 'processing') {
      return (
        <div className='flex h-16 w-16 items-center justify-center rounded-full bg-blue-500 text-white shadow-lg'>
          <Loader className='h-6 w-6 animate-spin' />
        </div>
      );
    }

    return (
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleStartRecording}
        disabled={!isInitialized}
        className='flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-r from-blue-600 to-teal-600 text-white shadow-lg transition-all duration-200 hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-50'
        aria-label='Start recording'
      >
        <Mic className='h-6 w-6' />
      </motion.button>
    );
  };

  /**
   * Render audio level visualization
   */
  const renderAudioLevel = () => {
    if (recordingState.status !== 'recording') return null;

    const levelBars = Array.from({ length: 5 }, (_, i) => {
      const threshold = (i + 1) * 0.2;
      const isActive = recordingState.audioLevel > threshold;

      return (
        <div
          key={i}
          className={`h-8 w-2 rounded-full transition-all duration-100 ${
            isActive ? 'bg-green-500' : 'bg-gray-300'
          }`}
          style={{
            height: `${Math.max(8, recordingState.audioLevel * 32)}px`,
          }}
        />
      );
    });

    return <div className='flex h-8 items-end space-x-1'>{levelBars}</div>;
  };

  /**
   * Render progress indicator
   */
  const renderProgress = () => {
    if (recordingState.status === 'idle') return null;

    return (
      <div className='w-full'>
        <div className='mb-2 flex justify-between text-sm text-gray-600'>
          <span>
            {recordingState.status === 'recording'
              ? 'Recording...'
              : recordingState.status === 'processing'
                ? 'Processing...'
                : 'Complete'}
          </span>
          <span>
            {Math.round(recordingState.recordedDuration)}s /{' '}
            {SPEECH_ANALYSIS_CONSTANTS.DEFAULT_DURATION}s
          </span>
        </div>
        <div className='h-2 w-full rounded-full bg-gray-200'>
          <div
            className='h-2 rounded-full bg-gradient-to-r from-blue-600 to-teal-600 transition-all duration-200'
            style={{ width: `${recordingState.progress * 100}%` }}
          />
        </div>
      </div>
    );
  };

  /**
   * Render analysis results
   */
  const renderResults = () => {
    if (!analysisResult) return null;

    const { fluencyScore, confidence, biomarkers } = analysisResult;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className='mt-6 rounded-xl border border-gray-200 bg-white/80 p-4 backdrop-blur-sm'
      >
        <div className='mb-4 flex items-center space-x-2'>
          <CheckCircle className='h-5 w-5 text-green-500' />
          <h4 className='font-semibold text-gray-900'>Analysis Complete</h4>
        </div>

        <div className='grid grid-cols-2 gap-4 text-sm'>
          <div>
            <span className='text-gray-600'>Fluency Score:</span>
            <div className='font-semibold text-blue-600'>{(fluencyScore * 100).toFixed(1)}%</div>
          </div>
          <div>
            <span className='text-gray-600'>Confidence:</span>
            <div className='font-semibold text-teal-600'>{(confidence * 100).toFixed(1)}%</div>
          </div>
          <div>
            <span className='text-gray-600'>Speech Rate:</span>
            <div className='font-semibold'>{biomarkers.speechRate} WPM</div>
          </div>
          <div>
            <span className='text-gray-600'>Pause Frequency:</span>
            <div className='font-semibold'>{biomarkers.pauseFrequency}/min</div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Show initialization error if components failed to load
  if (initializationError) {
    return (
      <div
        className={`rounded-2xl border border-red-200 bg-red-50/80 p-6 backdrop-blur-sm ${className}`}
      >
        <div className='flex items-center space-x-2 text-red-600'>
          <AlertCircle className='h-5 w-5' />
          <h3 className='font-semibold'>Speech Analysis Unavailable</h3>
        </div>
        <p className='mt-2 text-sm text-red-600'>{initializationError}</p>
      </div>
    );
  }

  return (
    <div
      className={`rounded-2xl border border-gray-200 bg-white/85 p-6 shadow-lg backdrop-blur-md ${className}`}
    >
      {/* Header */}
      <div className='mb-6'>
        <h3 className='mb-2 text-xl font-semibold text-gray-900'>Speech Analysis</h3>
        <p className='text-sm text-gray-600'>
          Record a 30-second speech sample for neurological assessment
        </p>
      </div>

      {/* Recording Interface */}
      <div className='flex flex-col items-center space-y-6'>
        {/* Recording Button */}
        <div className='flex items-center space-x-4'>
          {renderRecordingButton()}
          {renderAudioLevel()}
        </div>

        {/* Progress Indicator */}
        {renderProgress()}

        {/* Error Display */}
        <AnimatePresence>
          {recordingState.error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className='flex items-center space-x-2 text-red-600'
            >
              <AlertCircle className='h-4 w-4' />
              <span className='text-sm'>{recordingState.error}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Reset Button */}
        {(recordingState.status === 'complete' || recordingState.status === 'error') && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleReset}
            className='px-4 py-2 text-sm font-medium text-blue-600 transition-colors hover:text-blue-700'
          >
            Record Again
          </motion.button>
        )}
      </div>

      {/* Results Display */}
      {renderResults()}
    </div>
  );
};
