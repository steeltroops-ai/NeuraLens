/**
 * Speech Assessment Component for Neuralens Dashboard
 *
 * This component integrates the ML-powered Speech Analysis Card into the
 * Dashboard interface, providing comprehensive speech analysis capabilities
 * with real-time neurological assessment through voice pattern analysis.
 *
 * Key Features:
 * - Integration with SpeechAnalysisCard component
 * - ML-powered speech processing with Whisper-tiny model
 * - Real-time audio recording and analysis
 * - Comprehensive biomarker extraction and display
 * - NRI (Neuro-Risk Index) integration
 * - Performance metrics and error handling
 *
 * Technical Implementation:
 * - Uses ONNX Runtime for client-side ML inference
 * - WebRTC audio recording with real-time feedback
 * - MFCC feature extraction for speech analysis
 * - Integration with FastAPI backend for result processing
 * - Comprehensive error handling and user feedback
 */

'use client';

import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic,
  Activity,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Info,
  Brain,
  Zap,
} from 'lucide-react';
import React, { useState, useCallback } from 'react';

import { SPEECH_ANALYSIS_CONSTANTS } from '../../types/speech-analysis';
import { SpeechAnalysisCard } from '../assessment/SpeechAnalysisCard';

import type { SpeechResult, SpeechAnalysisError } from '../../types/speech-analysis';

interface SpeechAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

export default function SpeechAssessment({ onProcessingChange }: SpeechAssessmentProps) {
  // Component state for ML-powered speech analysis
  const [analysisResult, setAnalysisResult] = useState<SpeechResult | null>(null);
  const [error, setError] = useState<SpeechAnalysisError | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState({
    totalAnalyses: 0,
    averageLatency: 0,
    averageAccuracy: 0,
    lastAnalysisTime: null as Date | null,
  });

  /**
   * Handle successful speech analysis results
   * Updates performance metrics and notifies parent component
   */
  const handleAnalysisResult = useCallback(
    (result: SpeechResult) => {
      console.log('[SpeechAssessment] Analysis result received:', result);

      setAnalysisResult(result);
      setError(null);
      setIsProcessing(false);

      // Update performance metrics
      setPerformanceMetrics(prev => ({
        totalAnalyses: prev.totalAnalyses + 1,
        averageLatency:
          (prev.averageLatency * prev.totalAnalyses + result.metadata.processingTime) /
          (prev.totalAnalyses + 1),
        averageAccuracy:
          (prev.averageAccuracy * prev.totalAnalyses + result.confidence) /
          (prev.totalAnalyses + 1),
        lastAnalysisTime: result.metadata.timestamp,
      }));

      // Notify parent component that processing is complete
      onProcessingChange(false);
    },
    [onProcessingChange],
  );

  /**
   * Handle speech analysis errors
   * Updates error state and notifies parent component
   */
  const handleAnalysisError = useCallback(
    (error: SpeechAnalysisError) => {
      console.error('[SpeechAssessment] Analysis error:', error);

      setError(error);
      setIsProcessing(false);

      // Notify parent component that processing failed
      onProcessingChange(false);
    },
    [onProcessingChange],
  );

  /**
   * Handle recording state changes
   * Updates processing state based on recording status
   */
  const handleStateChange = useCallback(
    (state: any) => {
      const isCurrentlyProcessing = state.status === 'recording' || state.status === 'processing';

      if (isCurrentlyProcessing !== isProcessing) {
        setIsProcessing(isCurrentlyProcessing);
        onProcessingChange(isCurrentlyProcessing);
      }
    },
    [isProcessing, onProcessingChange],
  );

  /**
   * Calculate NRI contribution from speech analysis
   * Converts fluency score to NRI scale (0-100)
   */
  const calculateNRIContribution = (result: SpeechResult): number => {
    // Convert fluency score (0-1) to risk score (0-100)
    // Lower fluency = higher risk
    const riskScore = (1 - result.fluencyScore) * 100;

    // Apply confidence weighting
    const weightedScore = riskScore * result.confidence;

    return Math.round(Math.max(0, Math.min(100, weightedScore)));
  };

  /**
   * Get risk level classification based on NRI score
   */
  const getRiskLevel = (nriScore: number) => {
    if (nriScore < 25) return { level: 'Low', color: 'text-green-600', bg: 'bg-green-50' };
    if (nriScore < 50)
      return {
        level: 'Moderate',
        color: 'text-yellow-600',
        bg: 'bg-yellow-50',
      };
    if (nriScore < 75) return { level: 'High', color: 'text-orange-600', bg: 'bg-orange-50' };
    return { level: 'Critical', color: 'text-red-600', bg: 'bg-red-50' };
  };

  /**
   * Render performance metrics display
   */
  const renderPerformanceMetrics = () => (
    <div className='grid grid-cols-1 gap-3 text-xs sm:grid-cols-2 sm:gap-4 sm:text-sm lg:grid-cols-4'>
      <div className='flex items-center space-x-2 rounded-lg bg-slate-50 p-2 text-slate-600 sm:p-3'>
        <Clock className='h-4 w-4 flex-shrink-0' />
        <span>Target: &lt;{SPEECH_ANALYSIS_CONSTANTS.MAX_LATENCY}ms</span>
      </div>
      <div className='flex items-center space-x-2 rounded-lg bg-slate-50 p-2 text-slate-600 sm:p-3'>
        <Activity className='h-4 w-4 flex-shrink-0' />
        <span>Target: {SPEECH_ANALYSIS_CONSTANTS.MIN_ACCURACY * 100}%+ Accuracy</span>
      </div>
      <div className='flex items-center space-x-2 rounded-lg bg-slate-50 p-2 text-slate-600 sm:p-3'>
        <TrendingUp className='h-4 w-4 flex-shrink-0' />
        <span>Analyses: {performanceMetrics.totalAnalyses}</span>
      </div>
      <div className='flex items-center space-x-2 rounded-lg bg-slate-50 p-2 text-slate-600 sm:p-3'>
        <Brain className='h-4 w-4 flex-shrink-0' />
        <span>Model: Whisper-tiny</span>
      </div>
    </div>
  );

  return (
    <div className='space-y-6'>
      {/* Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6'>
        <div className='mb-4 flex flex-col space-y-3 sm:flex-row sm:items-center sm:space-x-3 sm:space-y-0'>
          <div className='w-fit rounded-lg bg-gradient-to-r from-blue-500 to-blue-600 p-3'>
            <Mic className='h-5 w-5 text-white sm:h-6 sm:w-6' />
          </div>
          <div className='flex-1'>
            <h1 className='text-xl font-bold text-slate-900 sm:text-2xl'>Speech Analysis</h1>
            <p className='text-sm text-slate-600 sm:text-base'>
              ML-powered neurological assessment through voice pattern analysis
            </p>
          </div>
        </div>

        {renderPerformanceMetrics()}
      </div>

      {/* ML-Powered Speech Analysis Interface */}
      <SpeechAnalysisCard
        onResult={handleAnalysisResult}
        onError={handleAnalysisError}
        className='w-full'
      />

      {/* Analysis Results Display */}
      <AnimatePresence>
        {analysisResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'
          >
            <div className='mb-4 flex items-center space-x-2'>
              <CheckCircle className='h-5 w-5 text-green-600' />
              <h2 className='text-lg font-semibold text-slate-900'>Analysis Results</h2>
            </div>

            <div className='grid grid-cols-1 gap-6 md:grid-cols-2'>
              {/* Primary Metrics */}
              <div className='space-y-4'>
                <div>
                  <h3 className='mb-2 text-sm font-medium text-slate-700'>Fluency Assessment</h3>
                  <div className='flex items-center space-x-3'>
                    <div className='text-2xl font-bold text-blue-600'>
                      {(analysisResult.fluencyScore * 100).toFixed(1)}%
                    </div>
                    <div className='text-sm text-slate-600'>
                      Confidence: {(analysisResult.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className='mb-2 text-sm font-medium text-slate-700'>NRI Contribution</h3>
                  <div className='flex items-center space-x-3'>
                    <div className='text-2xl font-bold text-purple-600'>
                      {calculateNRIContribution(analysisResult)}
                    </div>
                    <div
                      className={`rounded-full px-2 py-1 text-xs font-medium ${getRiskLevel(calculateNRIContribution(analysisResult)).bg} ${getRiskLevel(calculateNRIContribution(analysisResult)).color}`}
                    >
                      {getRiskLevel(calculateNRIContribution(analysisResult)).level} Risk
                    </div>
                  </div>
                </div>
              </div>

              {/* Biomarkers */}
              <div className='space-y-3'>
                <h3 className='text-sm font-medium text-slate-700'>Speech Biomarkers</h3>
                <div className='space-y-2 text-sm'>
                  <div className='flex justify-between'>
                    <span className='text-slate-600'>Speech Rate:</span>
                    <span className='font-medium'>{analysisResult.biomarkers.speechRate} WPM</span>
                  </div>
                  <div className='flex justify-between'>
                    <span className='text-slate-600'>Pause Frequency:</span>
                    <span className='font-medium'>
                      {analysisResult.biomarkers.pauseFrequency}/min
                    </span>
                  </div>
                  <div className='flex justify-between'>
                    <span className='text-slate-600'>Avg Pause Duration:</span>
                    <span className='font-medium'>{analysisResult.biomarkers.pauseDuration}ms</span>
                  </div>
                  <div className='flex justify-between'>
                    <span className='text-slate-600'>Pitch Variation:</span>
                    <span className='font-medium'>
                      {(analysisResult.biomarkers.pitchVariation * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className='mt-6 border-t pt-4'>
              <div className='flex items-center justify-between text-sm text-slate-600'>
                <span>Processing Time: {analysisResult.metadata.processingTime.toFixed(1)}ms</span>
                <span>Model: {analysisResult.metadata.modelVersion}</span>
                <span>Sample Rate: {analysisResult.metadata.sampleRate}Hz</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className='rounded-xl border border-red-200 bg-red-50 p-6'
          >
            <div className='flex items-center space-x-2 text-red-600'>
              <AlertCircle className='h-5 w-5' />
              <h3 className='font-semibold'>Analysis Error</h3>
            </div>
            <p className='mt-2 text-sm text-red-600'>{error.message}</p>
            {error.code && <p className='mt-1 text-xs text-red-500'>Error Code: {error.code}</p>}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
