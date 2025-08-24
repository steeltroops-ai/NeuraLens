'use client';

import React, { useState } from 'react';

import { Button } from '@/components/ui';

interface SpeechAssessmentStepProps {
  onComplete: (audioFile: File) => void;
  onBack: () => void;
  onSkip: () => void;
}

export const SpeechAssessmentStep: React.FC<SpeechAssessmentStepProps> = ({
  onComplete,
  onBack,
  onSkip,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecording, setHasRecording] = useState(false);

  const handleStartRecording = () => {
    setIsRecording(true);
    // Simulate recording
    setTimeout(() => {
      setIsRecording(false);
      setHasRecording(true);
    }, 3000);
  };

  const handleComplete = () => {
    // Create a mock audio file for demo purposes
    const mockAudioFile = new File(['mock audio data'], 'speech-sample.wav', {
      type: 'audio/wav',
    });
    onComplete(mockAudioFile);
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
                {isRecording ? (
                  <div className='h-20 w-20 animate-pulse rounded-full bg-gradient-to-br from-error-400 to-error-500 shadow-lg' />
                ) : hasRecording ? (
                  <div className='flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg'>
                    <svg className='h-12 w-12 text-white' fill='currentColor' viewBox='0 0 24 24'>
                      <path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' />
                    </svg>
                  </div>
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
                  {isRecording
                    ? 'Recording in Progress'
                    : hasRecording
                      ? 'Recording Complete'
                      : 'Ready to Record'}
                </h3>
                <p className='text-lg leading-relaxed text-text-secondary'>
                  {isRecording
                    ? 'Please speak clearly into your microphone for 30 seconds'
                    : hasRecording
                      ? 'Your speech sample has been captured successfully'
                      : 'Tap the record button to begin your voice assessment'}
                </p>
              </div>

              {!isRecording && !hasRecording && (
                <Button
                  onClick={handleStartRecording}
                  size='xl'
                  className='shadow-medical hover:shadow-medical-hover'
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
                  Start Recording
                </Button>
              )}

              {hasRecording && (
                <div className='space-y-6'>
                  <Button
                    onClick={handleComplete}
                    size='xl'
                    className='shadow-medical hover:shadow-medical-hover'
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
                      setHasRecording(false);
                      setIsRecording(false);
                    }}
                    size='lg'
                    className='px-8'
                  >
                    Record Again
                  </Button>
                </div>
              )}
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
