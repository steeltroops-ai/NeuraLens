/**
 * Assessment Results Display Component
 * Comprehensive results visualization with export and comparison features
 */

import { useState, useCallback } from 'react';
import {
  Download,
  Share2,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Brain,
  Eye,
  Activity,
} from 'lucide-react';
import type { AssessmentResults } from '@/lib/assessment/workflow';
import { LoadingButton } from '@/components/ui/LoadingStates';

// Props interface
interface AssessmentResultsProps {
  results: AssessmentResults;
  onExport?: (format: 'pdf' | 'json') => void;
  onShare?: () => void;
  onCompare?: () => void;
  className?: string;
}

// Risk category colors
const getRiskCategoryColor = (category: string) => {
  switch (category) {
    case 'low':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'moderate':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    case 'high':
      return 'text-red-600 bg-red-50 border-red-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
};

// Risk category icons
const getRiskCategoryIcon = (category: string) => {
  switch (category) {
    case 'low':
      return <CheckCircle className='h-5 w-5' />;
    case 'moderate':
      return <AlertTriangle className='h-5 w-5' />;
    case 'high':
      return <AlertTriangle className='h-5 w-5' />;
    default:
      return <AlertTriangle className='h-5 w-5' />;
  }
};

// Main component
export function AssessmentResults({
  results,
  onExport,
  onShare,
  onCompare,
  className = '',
}: AssessmentResultsProps) {
  const [exportLoading, setExportLoading] = useState<string | null>(null);

  // Handle export
  const handleExport = useCallback(
    async (format: 'pdf' | 'json') => {
      if (!onExport) return;

      setExportLoading(format);
      try {
        await onExport(format);
      } finally {
        setExportLoading(null);
      }
    },
    [onExport],
  );

  return (
    <div className={`rounded-lg bg-white shadow-lg ${className}`}>
      {/* Header */}
      <div className='border-b border-gray-200 p-6'>
        <div className='flex items-center justify-between'>
          <div>
            <h2 className='text-2xl font-bold text-gray-900'>Assessment Results</h2>
            <p className='mt-1 text-gray-600'>Session: {results.sessionId}</p>
            <p className='text-sm text-gray-500'>
              Completed: {new Date(results.completionTime).toLocaleString()}
            </p>
          </div>

          <div className='flex items-center gap-3'>
            {onCompare && (
              <button
                onClick={onCompare}
                className='flex items-center gap-2 rounded-lg border border-blue-600 px-4 py-2 text-blue-600 transition-colors hover:bg-blue-50'
              >
                <TrendingUp className='h-4 w-4' />
                Compare
              </button>
            )}

            {onShare && (
              <button
                onClick={onShare}
                className='flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-gray-600 transition-colors hover:bg-gray-50'
              >
                <Share2 className='h-4 w-4' />
                Share
              </button>
            )}

            {onExport && (
              <div className='flex items-center gap-2'>
                <LoadingButton
                  loading={exportLoading === 'pdf'}
                  onClick={() => handleExport('pdf')}
                  className='flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700'
                >
                  <Download className='h-4 w-4' />
                  PDF
                </LoadingButton>

                <LoadingButton
                  loading={exportLoading === 'json'}
                  onClick={() => handleExport('json')}
                  className='flex items-center gap-2 rounded-lg border border-blue-600 px-4 py-2 text-blue-600 transition-colors hover:bg-blue-50'
                >
                  <Download className='h-4 w-4' />
                  JSON
                </LoadingButton>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Overall Risk Assessment */}
      <div className='border-b border-gray-200 p-6'>
        <div className='mb-6 flex items-center justify-between'>
          <h3 className='text-lg font-semibold text-gray-900'>Overall Risk Assessment</h3>
          <div className='flex items-center gap-2'>
            <Clock className='h-4 w-4 text-gray-500' />
            <span className='text-sm text-gray-500'>
              {(results.totalProcessingTime / 1000).toFixed(1)}s processing time
            </span>
          </div>
        </div>

        <div className='grid grid-cols-1 gap-6 md:grid-cols-3'>
          {/* NRI Score */}
          <div className='text-center'>
            <div className='relative mx-auto mb-3 h-24 w-24'>
              <svg className='h-24 w-24 -rotate-90 transform' viewBox='0 0 100 100'>
                <circle
                  cx='50'
                  cy='50'
                  r='40'
                  stroke='currentColor'
                  strokeWidth='8'
                  fill='none'
                  className='text-gray-200'
                />
                <circle
                  cx='50'
                  cy='50'
                  r='40'
                  stroke='currentColor'
                  strokeWidth='8'
                  fill='none'
                  strokeDasharray={`${(results.nriResult?.nri_score || 0) * 2.51} 251`}
                  className='text-blue-600'
                />
              </svg>
              <div className='absolute inset-0 flex items-center justify-center'>
                <span className='text-xl font-bold text-gray-900'>
                  {Math.round((results.nriResult?.nri_score || 0) * 100)}
                </span>
              </div>
            </div>
            <p className='text-sm font-medium text-gray-900'>NRI Score</p>
            <p className='text-xs text-gray-500'>Neurological Risk Index</p>
          </div>

          {/* Risk Category */}
          <div className='text-center'>
            <div
              className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 ${getRiskCategoryColor(results.overallRiskCategory)} `}
            >
              {getRiskCategoryIcon(results.overallRiskCategory)}
              <span className='font-medium capitalize'>{results.overallRiskCategory} Risk</span>
            </div>
            <p className='mt-2 text-sm text-gray-600'>Risk Category</p>
          </div>

          {/* Confidence */}
          <div className='text-center'>
            <div className='mb-1 text-3xl font-bold text-gray-900'>
              {Math.round((results.nriResult?.confidence || 0) * 100)}%
            </div>
            <p className='text-sm font-medium text-gray-900'>Confidence</p>
            <p className='text-xs text-gray-500'>Analysis Certainty</p>
          </div>
        </div>
      </div>

      {/* Modality Results */}
      <div className='p-6'>
        <h3 className='mb-6 text-lg font-semibold text-gray-900'>Detailed Analysis Results</h3>

        <div className='grid grid-cols-1 gap-6 lg:grid-cols-2'>
          {/* Speech Analysis */}
          {results.speechResult && (
            <div className='rounded-lg border border-gray-200 p-4'>
              <div className='mb-4 flex items-center gap-3'>
                <div className='rounded-lg bg-blue-100 p-2'>
                  <Activity className='h-5 w-5 text-blue-600' />
                </div>
                <div>
                  <h4 className='font-semibold text-gray-900'>Speech Analysis</h4>
                  <p className='text-sm text-gray-600'>
                    Risk Score: {results.speechResult.risk_score.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className='space-y-2'>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Fluency</span>
                  <span className='text-sm font-medium'>
                    {(results.speechResult.biomarkers.fluency_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Voice Tremor</span>
                  <span className='text-sm font-medium'>
                    {(results.speechResult.biomarkers.voice_tremor * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Clarity</span>
                  <span className='text-sm font-medium'>
                    {(results.speechResult.biomarkers.articulation_clarity * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Retinal Analysis */}
          {results.retinalResult && (
            <div className='rounded-lg border border-gray-200 p-4'>
              <div className='mb-4 flex items-center gap-3'>
                <div className='rounded-lg bg-green-100 p-2'>
                  <Eye className='h-5 w-5 text-green-600' />
                </div>
                <div>
                  <h4 className='font-semibold text-gray-900'>Retinal Analysis</h4>
                  <p className='text-sm text-gray-600'>
                    Risk Score: {results.retinalResult.risk_score.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className='space-y-2'>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Vessel Density</span>
                  <span className='text-sm font-medium'>
                    {(results.retinalResult.biomarkers.vessel_density * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Cup-Disc Ratio</span>
                  <span className='text-sm font-medium'>
                    {results.retinalResult.biomarkers.cup_disc_ratio.toFixed(2)}
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>AV Ratio</span>
                  <span className='text-sm font-medium'>
                    {results.retinalResult.biomarkers.av_ratio.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Motor Assessment */}
          {results.motorResult && (
            <div className='rounded-lg border border-gray-200 p-4'>
              <div className='mb-4 flex items-center gap-3'>
                <div className='rounded-lg bg-orange-100 p-2'>
                  <Activity className='h-5 w-5 text-orange-600' />
                </div>
                <div>
                  <h4 className='font-semibold text-gray-900'>Motor Assessment</h4>
                  <p className='text-sm text-gray-600'>
                    Risk Score: {results.motorResult.risk_score.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className='space-y-2'>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Coordination</span>
                  <span className='text-sm font-medium'>
                    {(results.motorResult.biomarkers.coordination_index * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Tremor Severity</span>
                  <span className='text-sm font-medium'>
                    {(results.motorResult.biomarkers.tremor_severity * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Movement Quality</span>
                  <span className='text-sm font-medium capitalize'>
                    {results.motorResult.movement_quality}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Cognitive Assessment */}
          {results.cognitiveResult && (
            <div className='rounded-lg border border-gray-200 p-4'>
              <div className='mb-4 flex items-center gap-3'>
                <div className='rounded-lg bg-purple-100 p-2'>
                  <Brain className='h-5 w-5 text-purple-600' />
                </div>
                <div>
                  <h4 className='font-semibold text-gray-900'>Cognitive Assessment</h4>
                  <p className='text-sm text-gray-600'>
                    Risk Score: {results.cognitiveResult.risk_score.toFixed(3)}
                  </p>
                </div>
              </div>

              <div className='space-y-2'>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Memory</span>
                  <span className='text-sm font-medium'>
                    {(results.cognitiveResult.biomarkers.memory_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Attention</span>
                  <span className='text-sm font-medium'>
                    {(results.cognitiveResult.biomarkers.attention_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='flex justify-between'>
                  <span className='text-sm text-gray-600'>Processing Speed</span>
                  <span className='text-sm font-medium'>
                    {(results.cognitiveResult.biomarkers.processing_speed * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recommendations */}
      {results.nriResult?.recommendations && results.nriResult.recommendations.length > 0 && (
        <div className='border-t border-gray-200 bg-gray-50 p-6'>
          <h3 className='mb-4 text-lg font-semibold text-gray-900'>Recommendations</h3>
          <ul className='space-y-2'>
            {results.nriResult.recommendations.map((recommendation, index) => (
              <li key={index} className='flex items-start gap-2'>
                <CheckCircle className='mt-0.5 h-4 w-4 flex-shrink-0 text-green-600' />
                <span className='text-sm text-gray-700'>{recommendation}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
