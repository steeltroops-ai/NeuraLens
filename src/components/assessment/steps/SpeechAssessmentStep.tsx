'use client';

import React, { useState } from 'react';
import { Button, Card } from '@/components/ui';

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
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-text-primary">
              Speech Analysis
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Record a short speech sample for voice biomarker analysis
            </p>
          </div>

          {/* Recording Interface */}
          <Card className="p-8">
            <div className="text-center space-y-6">
              <div className="w-32 h-32 mx-auto bg-primary-500/10 rounded-full flex items-center justify-center">
                {isRecording ? (
                  <div className="w-16 h-16 bg-red-500 rounded-full animate-pulse" />
                ) : hasRecording ? (
                  <svg className="w-16 h-16 text-success" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                  </svg>
                ) : (
                  <svg className="w-16 h-16 text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                )}
              </div>

              <div>
                <h3 className="text-xl font-semibold text-text-primary mb-2">
                  {isRecording ? 'Recording...' : hasRecording ? 'Recording Complete' : 'Ready to Record'}
                </h3>
                <p className="text-text-secondary">
                  {isRecording 
                    ? 'Please speak clearly into your microphone'
                    : hasRecording 
                    ? 'Your speech sample has been recorded successfully'
                    : 'Click the button below to start recording'
                  }
                </p>
              </div>

              {!isRecording && !hasRecording && (
                <Button
                  onClick={handleStartRecording}
                  size="lg"
                  className="px-8"
                >
                  Start Recording
                </Button>
              )}

              {hasRecording && (
                <div className="space-y-4">
                  <Button
                    onClick={handleComplete}
                    size="lg"
                    className="px-8"
                  >
                    Continue with Analysis
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => {
                      setHasRecording(false);
                      setIsRecording(false);
                    }}
                    size="sm"
                  >
                    Record Again
                  </Button>
                </div>
              )}
            </div>
          </Card>

          {/* Instructions */}
          <Card className="p-6">
            <h4 className="font-semibold text-text-primary mb-3">Recording Instructions</h4>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li>• Ensure you're in a quiet environment</li>
              <li>• Speak clearly and at a normal pace</li>
              <li>• The recording will last approximately 30 seconds</li>
              <li>• You can re-record if needed</li>
            </ul>
          </Card>

          {/* Navigation */}
          <div className="flex justify-between">
            <Button variant="secondary" onClick={onBack}>
              Back
            </Button>
            <Button variant="ghost" onClick={onSkip}>
              Skip This Step
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
