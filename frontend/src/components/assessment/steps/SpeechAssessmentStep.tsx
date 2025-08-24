'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';

import { Button } from '@/components/ui';

interface SpeechAssessmentStepProps {
  onComplete: (audioFile: File) => void;
  onBack: () => void;
  onSkip: () => void;
}

interface RecordingState {
  isRecording: boolean;
  hasRecording: boolean;
  isPaused: boolean;
  recordingTime: number;
  audioLevel: number;
  error: string | null;
  isInitializing: boolean;
}

export const SpeechAssessmentStep: React.FC<SpeechAssessmentStepProps> = ({
  onComplete,
  onBack,
  onSkip,
}) => {
  const [recordingState, setRecordingState] = useState<RecordingState>({
    isRecording: false,
    hasRecording: false,
    isPaused: false,
    recordingTime: 0,
    audioLevel: 0,
    error: null,
    isInitializing: false,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);

  // Constants
  const MAX_RECORDING_TIME = 120; // 2 minutes
  const MIN_RECORDING_TIME = 5; // 5 seconds
  const TARGET_SAMPLE_RATE = 16000;

  // Initialize audio recording
  const initializeRecording = useCallback(async () => {
    try {
      setRecordingState(prev => ({ ...prev, isInitializing: true, error: null }));

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
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
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
        setRecordingState(prev => ({ ...prev, hasRecording: true }));
      };

      setRecordingState(prev => ({ ...prev, isInitializing: false }));
    } catch (error) {
      console.error('Failed to initialize recording:', error);
      let errorMessage = 'Failed to access microphone. ';

      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          errorMessage += 'Please allow microphone access and try again.';
        } else if (error.name === 'NotFoundError') {
          errorMessage += 'No microphone found. Please connect a microphone.';
        } else {
          errorMessage += error.message;
        }
      }

      setRecordingState(prev => ({
        ...prev,
        error: errorMessage,
        isInitializing: false,
      }));
    }
  }, []);

  // Monitor audio levels
  const monitorAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
    const normalizedLevel = Math.min(average / 128, 1);

    setRecordingState(prev => ({ ...prev, audioLevel: normalizedLevel }));

    if (recordingState.isRecording) {
      animationFrameRef.current = requestAnimationFrame(monitorAudioLevel);
    }
  }, [recordingState.isRecording]);

  // Start recording
  const handleStartRecording = useCallback(async () => {
    try {
      if (!mediaRecorderRef.current) {
        await initializeRecording();
        return;
      }

      audioChunksRef.current = [];
      recordedBlobRef.current = null;

      setRecordingState(prev => ({
        ...prev,
        isRecording: true,
        hasRecording: false,
        recordingTime: 0,
        error: null,
      }));

      mediaRecorderRef.current.start(100); // Collect data every 100ms

      // Start audio level monitoring
      monitorAudioLevel();

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingState(prev => {
          const newTime = prev.recordingTime + 1;
          if (newTime >= MAX_RECORDING_TIME) {
            handleStopRecording();
            return prev;
          }
          return { ...prev, recordingTime: newTime };
        });
      }, 1000);
    } catch (error) {
      console.error('Failed to start recording:', error);
      setRecordingState(prev => ({
        ...prev,
        error: 'Failed to start recording. Please try again.',
      }));
    }
  }, [initializeRecording, monitorAudioLevel]);

  // Stop recording
  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current && recordingState.isRecording) {
      mediaRecorderRef.current.stop();

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      setRecordingState(prev => ({
        ...prev,
        isRecording: false,
        audioLevel: 0,
      }));
    }
  }, [recordingState.isRecording]);

  // Convert blob to File and complete
  const handleComplete = useCallback(async () => {
    if (!recordedBlobRef.current) {
      setRecordingState(prev => ({
        ...prev,
        error: 'No recording available. Please record audio first.',
      }));
      return;
    }

    if (recordingState.recordingTime < MIN_RECORDING_TIME) {
      setRecordingState(prev => ({
        ...prev,
        error: `Recording too short. Please record for at least ${MIN_RECORDING_TIME} seconds.`,
      }));
      return;
    }

    try {
      // Convert blob to WAV format file
      const audioFile = new File([recordedBlobRef.current], `speech-sample-${Date.now()}.wav`, {
        type: 'audio/wav',
      });

      onComplete(audioFile);
    } catch (error) {
      console.error('Failed to process recording:', error);
      setRecordingState(prev => ({
        ...prev,
        error: 'Failed to process recording. Please try again.',
      }));
    }
  }, [recordingState.recordingTime, onComplete]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className='min-h-screen bg-gray-50 py-12'>
      <div className='container mx-auto px-6'>
        <div className='mx-auto max-w-4xl space-y-12'>
          {/* Apple-Style Header */}
          <div className='animate-fade-in space-y-6 text-center'>
            <div className='mx-auto flex h-20 w-20 items-center justify-center rounded-apple-xl bg-gradient-to-br from-medical-500 to-medical-600 shadow-medical'>
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
          <div className='card-apple animate-slide-up p-12'>
            <div className='space-y-8 text-center'>
              <div className='mx-auto flex h-40 w-40 items-center justify-center rounded-full bg-gradient-to-br from-medical-50 to-medical-100 shadow-inner'>
                {recordingState.isRecording ? (
                  <div className='relative h-20 w-20'>
                    <div
                      className='h-20 w-20 animate-pulse rounded-full bg-gradient-to-br from-error-400 to-error-500 shadow-lg'
                      style={{
                        transform: `scale(${1 + recordingState.audioLevel * 0.3})`,
                        transition: 'transform 0.1s ease-out',
                      }}
                    />
                    <div className='absolute inset-0 flex items-center justify-center'>
                      <div className='text-sm font-bold text-white'>
                        {formatTime(recordingState.recordingTime)}
                      </div>
                    </div>
                  </div>
                ) : recordingState.hasRecording ? (
                  <div className='flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg'>
                    <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                      <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                    </svg>
                  </div>
                ) : recordingState.isInitializing ? (
                  <div className='h-20 w-20 animate-spin rounded-full border-4 border-medical-200 border-t-medical-500' />
                ) : (
                  <div className='flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-medical-400 to-medical-500 shadow-lg'>
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

              <div className='space-y-4'>
                <h3 className='text-2xl font-semibold text-text-primary'>
                  {recordingState.isInitializing
                    ? 'Initializing Microphone...'
                    : recordingState.isRecording
                      ? 'Recording in Progress'
                      : recordingState.hasRecording
                        ? 'Recording Complete'
                        : recordingState.error
                          ? 'Recording Error'
                          : 'Ready to Record'}
                </h3>
                <p className='text-lg leading-relaxed text-text-secondary'>
                  {recordingState.isInitializing
                    ? 'Setting up your microphone and audio processing...'
                    : recordingState.isRecording
                      ? `Please speak clearly into your microphone. ${formatTime(MAX_RECORDING_TIME - recordingState.recordingTime)} remaining`
                      : recordingState.hasRecording
                        ? `Your ${formatTime(recordingState.recordingTime)} speech sample has been captured successfully`
                        : recordingState.error
                          ? recordingState.error
                          : 'Tap the record button to begin your voice assessment. Minimum 5 seconds required.'}
                </p>

                {/* Audio Level Indicator */}
                {recordingState.isRecording && (
                  <div className='mx-auto w-64'>
                    <div className='mb-2 text-sm text-text-secondary'>Audio Level</div>
                    <div className='h-2 overflow-hidden rounded-full bg-gray-200'>
                      <div
                        className='h-full bg-gradient-to-r from-green-400 to-green-600 transition-all duration-100'
                        style={{ width: `${recordingState.audioLevel * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Recording Controls */}
              <div className='flex justify-center space-x-4'>
                {!recordingState.isRecording &&
                  !recordingState.hasRecording &&
                  !recordingState.isInitializing && (
                    <Button
                      onClick={handleStartRecording}
                      disabled={!!recordingState.error}
                      size='xl'
                      className='shadow-medical hover:shadow-medical-hover disabled:cursor-not-allowed disabled:opacity-50'
                    >
                      <svg
                        className='mr-3 h-6 w-6'
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
                      {recordingState.error ? 'Microphone Error' : 'Start Recording'}
                    </Button>
                  )}

                {recordingState.isRecording && (
                  <Button
                    onClick={handleStopRecording}
                    variant='secondary'
                    size='xl'
                    className='shadow-medical hover:shadow-medical-hover'
                  >
                    <svg className='mr-3 h-6 w-6' fill='currentColor' viewBox='0 0 24 24'>
                      <path d='M6 6h12v12H6z' />
                    </svg>
                    Stop Recording
                  </Button>
                )}

                {recordingState.hasRecording && (
                  <div className='space-y-6'>
                    <Button
                      onClick={handleComplete}
                      disabled={recordingState.recordingTime < MIN_RECORDING_TIME}
                      size='xl'
                      className='shadow-medical hover:shadow-medical-hover disabled:cursor-not-allowed disabled:opacity-50'
                    >
                      Continue with Analysis
                      <svg
                        className='ml-3 h-6 w-6'
                        fill='none'
                        stroke='currentColor'
                        viewBox='0 0 24 24'
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
                      onClick={() => {
                        setRecordingState(prev => ({
                          ...prev,
                          isRecording: false,
                          hasRecording: false,
                          recordingTime: 0,
                          error: null,
                        }));
                        recordedBlobRef.current = null;
                      }}
                      size='lg'
                      className='px-8'
                    >
                      Record Again
                    </Button>
                  </div>
                )}

                {recordingState.error && (
                  <Button
                    onClick={() => {
                      setRecordingState(prev => ({ ...prev, error: null }));
                      initializeRecording();
                    }}
                    size='xl'
                    className='shadow-medical hover:shadow-medical-hover'
                  >
                    Try Again
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Apple-Style Instructions */}
          <div className='animate-scale-in rounded-apple-lg border border-medical-100 bg-medical-50 p-8'>
            <h4 className='mb-6 text-center text-xl font-semibold text-text-primary'>
              Recording Instructions
            </h4>
            <div className='grid grid-cols-1 gap-6 md:grid-cols-2'>
              <div className='flex items-start space-x-3'>
                <div className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'>
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
                <div className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'>
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
                <div className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'>
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
                <div className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-medical-100'>
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
          </div>

          {/* Apple-Style Navigation */}
          <div className='flex animate-fade-in justify-center gap-6'>
            <Button variant='secondary' onClick={onSkip} size='lg' className='px-8'>
              Skip This Step
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
