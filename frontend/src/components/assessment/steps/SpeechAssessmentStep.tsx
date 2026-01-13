'use client';

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';

import { Button } from '@/components/ui';
import { AudioVisualizer } from '@/components/assessment/AudioVisualizer';
import {
  RecordingErrorHandler,
  RecordingError,
} from '@/lib/recording/error-handler';
import {
  BUTTON_ARIA_LABELS,
  STATE_ARIA_DESCRIPTIONS,
  getStateAnnouncement,
  getAudioLevelAriaLabel,
  getRecordingTimeAriaLabel,
  handleRecordingKeyboard,
  FOCUS_STYLES,
  getKeyboardShortcutsHelp,
} from '@/lib/recording/accessibility';
import {
  RecordingState,
  RecordingStateManager,
  createRecordingStateManager,
  RecordingManagerState,
} from '@/lib/recording/state-manager';

interface SpeechAssessmentStepProps {
  onComplete: (audioFile: File) => void;
  onBack: () => void;
  onSkip: () => void;
}

/**
 * SpeechAssessmentStep Component
 * 
 * Implements voice recording for neurological assessment with:
 * - State machine-based recording management (Requirements 5.3)
 * - Real-time audio visualization (Requirements 6.1, 6.2, 6.3)
 * - Categorized error handling (Requirements 7.1, 7.2, 7.3, 7.4)
 * - Full accessibility support (Requirements 8.1, 8.2, 8.3, 8.4, 8.5)
 */
export const SpeechAssessmentStep: React.FC<SpeechAssessmentStepProps> = ({
  onComplete,
  onBack,
  onSkip,
}) => {
  // Create state manager instance
  const stateManagerRef = useRef<RecordingStateManager>(createRecordingStateManager());

  // State from the state manager
  const [managerState, setManagerState] = useState<RecordingManagerState>(
    stateManagerRef.current.fullState
  );

  // State for screen reader announcements
  const [announcement, setAnnouncement] = useState<string>('');
  const previousStateRef = useRef<RecordingState>('idle');

  // Audio resources refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);
  const recordButtonRef = useRef<HTMLButtonElement>(null);

  // Low audio warning state
  const [showLowAudioWarning, setShowLowAudioWarning] = useState(false);

  // Constants
  const MAX_RECORDING_TIME = 120; // 2 minutes
  const MIN_RECORDING_TIME = 5; // 5 seconds
  const TARGET_SAMPLE_RATE = 16000;

  // ARIA IDs
  const AUDIO_LEVEL_ID = 'audio-level-description';
  const RECORDING_STATUS_ID = 'recording-status';
  const ERROR_MESSAGE_ID = 'error-message';
  const LIVE_REGION_ID = 'recording-announcements';

  // Subscribe to state manager changes
  useEffect(() => {
    const unsubscribe = stateManagerRef.current.subscribe((newState) => {
      setManagerState(newState);
    });
    return unsubscribe;
  }, []);

  // Get current state for convenience
  const currentState = managerState.state;
  const recordingTime = managerState.recordingTime;
  const audioLevel = managerState.audioLevel;
  const error = managerState.error as RecordingError | null;

  // Derived state flags
  const isRecording = currentState === 'recording';
  const isInitializing = currentState === 'initializing';
  const hasRecording = currentState === 'completed';
  const hasError = currentState === 'error';
  const isIdle = currentState === 'idle';

  // Announce state changes to screen readers
  useEffect(() => {
    if (previousStateRef.current !== currentState) {
      const announcementText = getStateAnnouncement(currentState, {
        recordingTime: recordingTime,
        errorMessage: error?.message,
      });
      setAnnouncement(announcementText);
      previousStateRef.current = currentState;
    }
  }, [currentState, recordingTime, error?.message]);

  /**
   * Cleanup all audio resources
   */
  const cleanupResources = useCallback(() => {
    // Stop all media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Clear timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Clear analyser reference
    analyserRef.current = null;

    // Clear media recorder
    mediaRecorderRef.current = null;
  }, []);

  /**
   * Initialize audio recording
   */
  const initializeRecording = useCallback(async () => {
    const manager = stateManagerRef.current;

    // Transition to initializing state
    if (!manager.dispatch('START_INIT')) {
      console.warn('Cannot start initialization from current state');
      return;
    }

    try {
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: TARGET_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // Create audio context for level monitoring
      audioContextRef.current = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      // Create MediaRecorder
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
      };

      // Fallback for Safari
      if (!MediaRecorder.isTypeSupported(options.mimeType || '')) {
        options.mimeType = 'audio/mp4';
      }

      mediaRecorderRef.current = new MediaRecorder(stream, options);

      mediaRecorderRef.current.ondataavailable = event => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        recordedBlobRef.current = blob;
        manager.setAudioBlob(blob);
      };

      // Start recording immediately after successful initialization
      audioChunksRef.current = [];
      recordedBlobRef.current = null;
      mediaRecorderRef.current.start(100); // Collect data every 100ms

      // Transition to recording state
      manager.dispatch('INIT_SUCCESS');

      // Start timer
      timerRef.current = setInterval(() => {
        const currentTime = stateManagerRef.current.recordingTime + 1;
        if (currentTime >= MAX_RECORDING_TIME) {
          handleStopRecording();
        } else {
          stateManagerRef.current.updateRecordingTime(currentTime);
        }
      }, 1000);

    } catch (err) {
      console.error('Failed to initialize recording:', err);

      // Use RecordingErrorHandler to categorize the error
      const categorizedError = RecordingErrorHandler.categorizeError(
        err instanceof Error ? err : new Error(String(err))
      );

      manager.setError(categorizedError);
      cleanupResources();
    }
  }, [cleanupResources]);

  /**
   * Stop recording
   */
  const handleStopRecording = useCallback(() => {
    const manager = stateManagerRef.current;

    if (mediaRecorderRef.current && manager.state === 'recording') {
      mediaRecorderRef.current.stop();

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      // Transition to completed state
      manager.dispatch('STOP');
    }
  }, []);

  /**
   * Handle audio level updates from AudioVisualizer
   */
  const handleAudioLevelChange = useCallback((level: number) => {
    stateManagerRef.current.updateAudioLevel(level / 100); // Normalize to 0-1
  }, []);

  /**
   * Handle low audio warning from AudioVisualizer
   */
  const handleLowAudioWarning = useCallback((isLow: boolean) => {
    setShowLowAudioWarning(isLow);
  }, []);

  /**
   * Handle keyboard events for recording controls
   */
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    handleRecordingKeyboard(event, {
      onStartStop: () => {
        if (isRecording) {
          handleStopRecording();
        } else if (isIdle && !hasError) {
          initializeRecording();
        }
      },
    });
  }, [isRecording, isIdle, hasError, handleStopRecording, initializeRecording]);

  /**
   * Convert blob to File and complete
   */
  const handleComplete = useCallback(async () => {
    if (!recordedBlobRef.current) {
      const noRecordingError = RecordingErrorHandler.createError('processing_failed');
      noRecordingError.message = 'No recording available. Please record audio first.';
      stateManagerRef.current.setError(noRecordingError);
      return;
    }

    if (recordingTime < MIN_RECORDING_TIME) {
      const tooShortError = RecordingErrorHandler.createError('processing_failed');
      tooShortError.message = `Recording too short. Please record for at least ${MIN_RECORDING_TIME} seconds.`;
      stateManagerRef.current.setError(tooShortError);
      return;
    }

    try {
      // Convert blob to WAV format file
      const audioFile = new File([recordedBlobRef.current], `speech-sample-${Date.now()}.wav`, {
        type: 'audio/wav',
      });

      onComplete(audioFile);
    } catch (err) {
      console.error('Failed to process recording:', err);

      // Use RecordingErrorHandler to categorize the error
      const categorizedError = RecordingErrorHandler.categorizeError(
        err instanceof Error ? err : new Error(String(err))
      );

      stateManagerRef.current.setError(categorizedError);
    }
  }, [recordingTime, onComplete]);

  /**
   * Reset recording state
   */
  const handleReset = useCallback(() => {
    cleanupResources();
    recordedBlobRef.current = null;
    audioChunksRef.current = [];
    setShowLowAudioWarning(false);

    // Reset state manager
    if (stateManagerRef.current.state === 'completed') {
      stateManagerRef.current.dispatch('RESET');
    } else if (stateManagerRef.current.state === 'error') {
      stateManagerRef.current.dispatch('RETRY');
    }
  }, [cleanupResources]);

  /**
   * Retry after error
   */
  const handleRetry = useCallback(() => {
    handleReset();
    // Small delay to ensure state is reset before reinitializing
    setTimeout(() => {
      initializeRecording();
    }, 100);
  }, [handleReset, initializeRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupResources();
    };
  }, [cleanupResources]);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Memoize the status text
  const statusText = useMemo(() => {
    if (isInitializing) return 'Initializing Microphone...';
    if (isRecording) return 'Recording in Progress';
    if (hasRecording) return 'Recording Complete';
    if (hasError) return 'Recording Error';
    return 'Ready to Record';
  }, [isInitializing, isRecording, hasRecording, hasError]);

  // Memoize the description text
  const descriptionText = useMemo(() => {
    if (isInitializing) {
      return 'Setting up your microphone and audio processing...';
    }
    if (isRecording) {
      return `Please speak clearly into your microphone. ${formatTime(MAX_RECORDING_TIME - recordingTime)} remaining`;
    }
    if (hasRecording) {
      return `Your ${formatTime(recordingTime)} speech sample has been captured successfully`;
    }
    if (hasError && error) {
      return error.message;
    }
    return 'Tap the record button or press Space to begin your voice assessment. Minimum 5 seconds required.';
  }, [isInitializing, isRecording, hasRecording, hasError, error, recordingTime]);

  return (
    <div
      className='min-h-screen bg-gray-50 py-12'
      onKeyDown={handleKeyDown}
      role="region"
      aria-label="Voice evaluation assessment"
    >
      {/* Screen Reader Live Region for Announcements */}
      <div
        id={LIVE_REGION_ID}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcement}
      </div>

      <div className='container mx-auto px-6'>
        <div className='mx-auto max-w-4xl space-y-12'>
          {/* Apple-Style Header */}
          <div className='animate-fade-in space-y-6 text-center'>
            <div
              className='mx-auto flex h-20 w-20 items-center justify-center rounded-apple-xl bg-gradient-to-br from-medical-500 to-medical-600 shadow-medical'
              aria-hidden="true"
            >
              <svg
                className='h-10 w-10 text-white'
                fill='none'
                stroke='currentColor'
                viewBox='0 0 24 24'
              >
                <path
                  strokeLinecap='round'
                  strokeLinejoin='round'
                  strokeWidth={2}
                  d='M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z'
                />
              </svg>
            </div>
            <h1 className='text-4xl font-bold tracking-tight text-text-primary'>
              Voice Evaluation
            </h1>
            <p className='mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary'>
              Record a 30-second voice sample to check for early signs of neurological changes
            </p>
          </div>

          {/* Apple-Style Recording Interface */}
          <div
            className='card-apple animate-slide-up p-12'
            role="application"
            aria-label="Voice recording interface"
            aria-describedby={RECORDING_STATUS_ID}
          >
            <div className='space-y-8 text-center'>
              {/* Recording Visual Indicator */}
              <div
                className='mx-auto flex h-40 w-40 items-center justify-center rounded-full bg-gradient-to-br from-medical-50 to-medical-100 shadow-inner'
                role="img"
                aria-label={STATE_ARIA_DESCRIPTIONS[currentState]}
              >
                {isRecording ? (
                  <div className='relative h-20 w-20'>
                    <div
                      className='h-20 w-20 animate-pulse rounded-full bg-gradient-to-br from-error-400 to-error-500 shadow-lg'
                      style={{
                        transform: `scale(${1 + audioLevel * 0.3})`,
                        transition: 'transform 0.1s ease-out',
                      }}
                      aria-hidden="true"
                    />
                    <div
                      className='absolute inset-0 flex items-center justify-center'
                      aria-hidden="true"
                    >
                      <div className='text-sm font-bold text-white'>
                        {formatTime(recordingTime)}
                      </div>
                    </div>
                  </div>
                ) : hasRecording ? (
                  <div
                    className='flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg'
                    aria-hidden="true"
                  >
                    <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                      <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                    </svg>
                  </div>
                ) : isInitializing ? (
                  <div
                    className='h-20 w-20 animate-spin rounded-full border-4 border-medical-200 border-t-medical-500'
                    role="progressbar"
                    aria-label="Initializing microphone"
                  />
                ) : (
                  <div
                    className='flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-medical-400 to-medical-500 shadow-lg'
                    aria-hidden="true"
                  >
                    <svg
                      className='h-12 w-12 text-white'
                      fill='none'
                      stroke='currentColor'
                      viewBox='0 0 24 24'
                    >
                      <path
                        strokeLinecap='round'
                        strokeLinejoin='round'
                        strokeWidth={2}
                        d='M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z'
                      />
                    </svg>
                  </div>
                )}
              </div>

              {/* Status Text */}
              <div className='space-y-4'>
                <h3
                  id={RECORDING_STATUS_ID}
                  className='text-2xl font-semibold text-text-primary'
                >
                  {statusText}
                </h3>
                <p className='text-lg leading-relaxed text-text-secondary'>
                  {descriptionText}
                </p>

                {/* Error Guidance Display */}
                {hasError && error && (
                  <div
                    id={ERROR_MESSAGE_ID}
                    className='mx-auto max-w-md rounded-apple-lg border border-error-200 bg-error-50 p-4'
                    role="alert"
                    aria-live="assertive"
                  >
                    <p className='text-sm text-error-700'>
                      {error.guidance}
                    </p>
                  </div>
                )}

                {/* Audio Visualizer - Integrated Component */}
                {isRecording && (
                  <div
                    className='mx-auto w-80'
                    role="region"
                    aria-label="Audio level monitor"
                  >
                    <AudioVisualizer
                      analyser={analyserRef.current}
                      isActive={isRecording}
                      onAudioLevelChange={handleAudioLevelChange}
                      onLowAudioWarning={handleLowAudioWarning}
                      lowAudioThreshold={10}
                      lowAudioDuration={2000}
                      showWaveform={true}
                      showLevelBar={true}
                      ariaLabel="Real-time audio level visualization"
                    />
                    {/* Screen reader only: detailed audio level */}
                    <span className="sr-only">
                      {getRecordingTimeAriaLabel(recordingTime, MAX_RECORDING_TIME)}
                    </span>
                  </div>
                )}

                {/* Low Audio Warning (shown by AudioVisualizer, but we can add extra context) */}
                {showLowAudioWarning && isRecording && (
                  <span className="sr-only" role="alert">
                    {getAudioLevelAriaLabel(audioLevel * 100)}
                  </span>
                )}
              </div>

              {/* Recording Controls */}
              <div className='flex justify-center space-x-4'>
                {isIdle && !hasError && (
                  <Button
                    ref={recordButtonRef}
                    onClick={initializeRecording}
                    size='xl'
                    className={`shadow-medical hover:shadow-medical-hover ${FOCUS_STYLES.recordButton}`}
                    aria-label={BUTTON_ARIA_LABELS.startRecording}
                    aria-describedby={RECORDING_STATUS_ID}
                  >
                    <svg
                      className='mr-3 h-6 w-6'
                      fill='none'
                      stroke='currentColor'
                      viewBox='0 0 24 24'
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap='round'
                        strokeLinejoin='round'
                        strokeWidth={2}
                        d='M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z'
                      />
                    </svg>
                    Start Recording
                  </Button>
                )}

                {isRecording && (
                  <Button
                    onClick={handleStopRecording}
                    variant='secondary'
                    size='xl'
                    className={`shadow-medical hover:shadow-medical-hover ${FOCUS_STYLES.button}`}
                    aria-label={BUTTON_ARIA_LABELS.stopRecording}
                    aria-describedby={AUDIO_LEVEL_ID}
                  >
                    <svg
                      className='mr-3 h-6 w-6'
                      fill='currentColor'
                      viewBox='0 0 24 24'
                      aria-hidden="true"
                    >
                      <path d='M6 6h12v12H6z' />
                    </svg>
                    Stop Recording
                  </Button>
                )}

                {hasRecording && (
                  <div className='space-y-6'>
                    <Button
                      onClick={handleComplete}
                      disabled={recordingTime < MIN_RECORDING_TIME}
                      size='xl'
                      className={`shadow-medical hover:shadow-medical-hover disabled:cursor-not-allowed disabled:opacity-50 ${FOCUS_STYLES.button}`}
                      aria-label={BUTTON_ARIA_LABELS.continueAnalysis}
                      aria-disabled={recordingTime < MIN_RECORDING_TIME}
                    >
                      Continue with Analysis
                      <svg
                        className='ml-3 h-6 w-6'
                        fill='none'
                        stroke='currentColor'
                        viewBox='0 0 24 24'
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap='round'
                          strokeLinejoin='round'
                          strokeWidth={2}
                          d='M13 7l5 5m0 0l-5 5m5-5H6'
                        />
                      </svg>
                    </Button>
                    <Button
                      variant='secondary'
                      onClick={handleReset}
                      size='lg'
                      className={`px-8 ${FOCUS_STYLES.button}`}
                      aria-label={BUTTON_ARIA_LABELS.recordAgain}
                    >
                      Record Again
                    </Button>
                  </div>
                )}

                {hasError && (
                  <Button
                    onClick={handleRetry}
                    size='xl'
                    className={`shadow-medical hover:shadow-medical-hover ${FOCUS_STYLES.button}`}
                    aria-label={BUTTON_ARIA_LABELS.tryAgain}
                    aria-describedby={ERROR_MESSAGE_ID}
                  >
                    <svg
                      className='mr-3 h-6 w-6'
                      fill='none'
                      stroke='currentColor'
                      viewBox='0 0 24 24'
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap='round'
                        strokeLinejoin='round'
                        strokeWidth={2}
                        d='M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15'
                      />
                    </svg>
                    Try Again
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Apple-Style Instructions */}
          <div
            className='animate-scale-in rounded-apple-lg border border-medical-100 bg-medical-50 p-8'
            role="region"
            aria-label="Recording instructions"
          >
            <h4 className='mb-6 text-center text-xl font-semibold text-text-primary'>
              Recording Instructions
            </h4>
            <div className='grid grid-cols-1 gap-6 md:grid-cols-2'>
              <div className='flex items-start space-x-3'>
                <div
                  className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'
                  aria-hidden="true"
                >
                  <span className='text-sm font-semibold text-medical-600'>1</span>
                </div>
                <div>
                  <h5 className='font-medium text-text-primary'>Quiet Environment</h5>
                  <p className='text-sm text-text-secondary'>
                    Find a quiet space with minimal background noise
                  </p>
                </div>
              </div>

              <div className='flex items-start space-x-3'>
                <div
                  className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'
                  aria-hidden="true"
                >
                  <span className='text-sm font-semibold text-medical-600'>2</span>
                </div>
                <div>
                  <h5 className='font-medium text-text-primary'>Clear Speech</h5>
                  <p className='text-sm text-text-secondary'>
                    Speak clearly at your normal pace and volume
                  </p>
                </div>
              </div>

              <div className='flex items-start space-x-3'>
                <div
                  className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'
                  aria-hidden="true"
                >
                  <span className='text-sm font-semibold text-medical-600'>3</span>
                </div>
                <div>
                  <h5 className='font-medium text-text-primary'>30 Seconds</h5>
                  <p className='text-sm text-text-secondary'>
                    Recording will automatically stop after 30 seconds
                  </p>
                </div>
              </div>

              <div className='flex items-start space-x-3'>
                <div
                  className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'
                  aria-hidden="true"
                >
                  <span className='text-sm font-semibold text-medical-600'>4</span>
                </div>
                <div>
                  <h5 className='font-medium text-text-primary'>Re-record Option</h5>
                  <p className='text-sm text-text-secondary'>
                    You can record again if you're not satisfied
                  </p>
                </div>
              </div>
            </div>

            {/* Keyboard shortcuts info for screen readers */}
            <div className="mt-6 text-center text-sm text-text-secondary">
              <p>
                <strong>Keyboard shortcuts:</strong> {getKeyboardShortcutsHelp()}
              </p>
            </div>
          </div>

          {/* Apple-Style Navigation */}
          <div className='flex animate-fade-in justify-center gap-6'>
            <Button
              variant='secondary'
              onClick={onBack}
              size='lg'
              className={`px-8 ${FOCUS_STYLES.button}`}
              aria-label={BUTTON_ARIA_LABELS.goBack}
            >
              Go Back
            </Button>
            <Button
              variant='secondary'
              onClick={onSkip}
              size='lg'
              className={`px-8 ${FOCUS_STYLES.button}`}
              aria-label={BUTTON_ARIA_LABELS.skipStep}
            >
              Skip This Step
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
