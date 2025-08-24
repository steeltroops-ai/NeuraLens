'use client';

import React, { useState, useEffect, useRef } from 'react';

import { Button, Card, Progress } from '@/components/ui';

interface MotorAssessmentStepProps {
  onComplete: (motorData: any) => void;
  onBack: () => void;
}

export const MotorAssessmentStep: React.FC<MotorAssessmentStepProps> = ({ onComplete, onBack }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [tapCount, setTapCount] = useState(0);
  const [tapTimes, setTapTimes] = useState<number[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const startTimeRef = useRef<number>(0);

  const RECORDING_DURATION = 15; // 15 seconds

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording && countdown > 0) {
      interval = setInterval(() => {
        setCountdown(prev => prev - 1);
      }, 1000);
    } else if (isRecording && countdown === 0) {
      handleStopRecording();
    }
    return () => clearInterval(interval);
  }, [isRecording, countdown]);

  const handleStartRecording = () => {
    setIsRecording(true);
    setCountdown(RECORDING_DURATION);
    setTapCount(0);
    setTapTimes([]);
    setIsComplete(false);
    startTimeRef.current = Date.now();
  };

  const handleTap = () => {
    if (!isRecording) return;

    const currentTime = Date.now();
    const relativeTime = currentTime - startTimeRef.current;

    setTapCount(prev => prev + 1);
    setTapTimes(prev => [...prev, relativeTime]);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    setCountdown(0);
    setIsComplete(true);
  };

  const handleComplete = () => {
    const motorData = {
      tapCount,
      tapTimes,
      duration: RECORDING_DURATION * 1000,
      averageInterval:
        tapTimes.length > 1
          ? tapTimes.reduce((acc, time, index) => {
              if (index === 0) return acc;
              const prevTime = tapTimes[index - 1];
              return acc + (time - (prevTime ?? 0));
            }, 0) /
            (tapTimes.length - 1)
          : 0,
      timestamp: new Date().toISOString(),
    };

    onComplete(motorData);
  };

  const progress = isRecording
    ? ((RECORDING_DURATION - countdown) / RECORDING_DURATION) * 100
    : isComplete
      ? 100
      : 0;

  return (
    <div className='mx-auto max-w-2xl space-y-8'>
      {/* Header */}
      <div className='space-y-4 text-center'>
        <h2 className='text-3xl font-bold text-text-primary'>Movement Check</h2>
        <p className='text-lg text-text-secondary'>
          Simple finger tapping test to assess motor function
        </p>
      </div>

      {/* Instructions Card */}
      <Card className='p-8'>
        <div className='space-y-6'>
          <h3 className='text-xl font-semibold text-text-primary'>Instructions</h3>
          <div className='space-y-4'>
            <div className='flex items-start space-x-3'>
              <div className='flex h-6 w-6 items-center justify-center rounded-full bg-primary-500 text-sm font-semibold text-white'>
                1
              </div>
              <p className='text-text-secondary'>Place your device on a flat surface</p>
            </div>
            <div className='flex items-start space-x-3'>
              <div className='flex h-6 w-6 items-center justify-center rounded-full bg-primary-500 text-sm font-semibold text-white'>
                2
              </div>
              <p className='text-text-secondary'>
                Use your index finger to tap the button below as quickly and consistently as
                possible
              </p>
            </div>
            <div className='flex items-start space-x-3'>
              <div className='flex h-6 w-6 items-center justify-center rounded-full bg-primary-500 text-sm font-semibold text-white'>
                3
              </div>
              <p className='text-text-secondary'>
                Continue tapping for 15 seconds until the timer stops
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Recording Interface */}
      <Card className='p-8'>
        <div className='space-y-6 text-center'>
          {!isRecording && !isComplete && (
            <div className='space-y-6'>
              <div className='text-6xl'>✋</div>
              <p className='text-text-secondary'>Ready to start the finger tapping test?</p>
              <Button
                variant='primary'
                size='lg'
                onClick={handleStartRecording}
                className='w-full max-w-xs'
              >
                Start Test
              </Button>
            </div>
          )}

          {isRecording && (
            <div className='space-y-6'>
              <div className='text-4xl font-bold text-primary-500'>{countdown}s</div>
              <Progress value={progress} className='w-full' />
              <div className='space-y-4'>
                <p className='text-text-secondary'>Tap the button below as quickly as possible</p>
                <button
                  onClick={handleTap}
                  className='h-32 w-32 rounded-full bg-primary-500 text-2xl font-bold text-white shadow-lg transition-all duration-150 hover:bg-primary-600 hover:shadow-xl active:scale-95'
                >
                  TAP
                </button>
                <div className='text-lg font-semibold text-text-primary'>Taps: {tapCount}</div>
              </div>
            </div>
          )}

          {isComplete && (
            <div className='space-y-6'>
              <div className='text-6xl text-success-500'>✓</div>
              <div className='space-y-2'>
                <h3 className='text-xl font-semibold text-text-primary'>Test Complete!</h3>
                <p className='text-text-secondary'>You completed {tapCount} taps in 15 seconds</p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Navigation */}
      <div className='flex justify-between'>
        <Button variant='secondary' onClick={onBack} disabled={isRecording}>
          Back
        </Button>
        <Button variant='primary' onClick={handleComplete} disabled={!isComplete}>
          Continue
        </Button>
      </div>
    </div>
  );
};
