/**
 * Comprehensive Results Dashboard
 * Real-time assessment results display with clinical insights and accessibility
 */

import React, { useState, useEffect, useCallback } from 'react';
import { AssessmentResults } from '@/lib/assessment/workflow';
import { useScreenReader, useReducedMotion } from '@/hooks/useAccessibility';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  Brain,
  Eye,
  Activity,
  Zap,
  Calendar,
  FileText,
  Share2,
} from 'lucide-react';

// Dashboard props interface
interface ResultsDashboardProps {
  results: AssessmentResults;
  isRealTime?: boolean;
  onExport?: (format: string) => void;
  onShare?: () => void;
  onScheduleFollowUp?: () => void;
  className?: string;
}

// Risk trend data interface
interface RiskTrendData {
  date: string;
  nriScore: number;
  riskCategory: string;
}

// Real-time update interface
interface RealTimeUpdate {
  type: 'progress' | 'result' | 'recommendation';
  data: any;
  timestamp: string;
}

export function ResultsDashboard({
  results,
  isRealTime = false,
  onExport,
  onShare,
  onScheduleFollowUp,
  className = '',
}: ResultsDashboardProps) {
  const [realtimeUpdates, setRealtimeUpdates] = useState<RealTimeUpdate[]>([]);
  const [selectedModality, setSelectedModality] = useState<string>('overview');
  const [trendData, setTrendData] = useState<RiskTrendData[]>([]);

  const { announce } = useScreenReader();
  const { prefersReducedMotion } = useReducedMotion();

  // Simulate real-time updates (in production, this would use WebSocket)
  useEffect(() => {
    if (!isRealTime) return;

    const interval = setInterval(() => {
      const mockUpdate: RealTimeUpdate = {
        type: 'progress',
        data: { progress: Math.random() * 100 },
        timestamp: new Date().toISOString(),
      };

      setRealtimeUpdates(prev => [...prev.slice(-9), mockUpdate]);
    }, 2000);

    return () => clearInterval(interval);
  }, [isRealTime]);

  // Generate mock trend data
  useEffect(() => {
    const mockTrend: RiskTrendData[] = Array.from({ length: 7 }, (_, i) => ({
      date: new Date(Date.now() - (6 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0] || '',
      nriScore: 0.3 + Math.random() * 0.4,
      riskCategory: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'moderate' : 'low',
    }));
    setTrendData(mockTrend);
  }, []);

  // Get risk category styling
  const getRiskCategoryStyle = useCallback((category: string) => {
    switch (category) {
      case 'low':
        return 'text-green-700 bg-green-100 border-green-200';
      case 'moderate':
        return 'text-yellow-700 bg-yellow-100 border-yellow-200';
      case 'high':
        return 'text-red-700 bg-red-100 border-red-200';
      default:
        return 'text-gray-700 bg-gray-100 border-gray-200';
    }
  }, []);

  // Get risk category icon
  const getRiskCategoryIcon = useCallback((category: string) => {
    switch (category) {
      case 'low':
        return <CheckCircle className='h-5 w-5' aria-hidden='true' />;
      case 'moderate':
        return <AlertTriangle className='h-5 w-5' aria-hidden='true' />;
      case 'high':
        return <AlertTriangle className='h-5 w-5' aria-hidden='true' />;
      default:
        return <AlertTriangle className='h-5 w-5' aria-hidden='true' />;
    }
  }, []);

  // Handle modality selection
  const handleModalitySelect = useCallback(
    (modality: string) => {
      setSelectedModality(modality);
      announce(`Selected ${modality} view`);
    },
    [announce],
  );

  return (
    <div
      className={`rounded-lg bg-white shadow-lg ${className}`}
      role='main'
      aria-label='Assessment results dashboard'
    >
      {/* Dashboard Header */}
      <header className='border-b border-gray-200 p-6'>
        <div className='flex items-center justify-between'>
          <div>
            <h1 className='text-2xl font-bold text-gray-900'>Assessment Results Dashboard</h1>
            <p className='mt-1 text-gray-600'>
              Session: {results.sessionId} • Completed:{' '}
              {new Date(results.completionTime).toLocaleString()}
            </p>
          </div>

          <div className='flex items-center gap-3'>
            {isRealTime && (
              <div
                className='flex items-center gap-2 text-green-600'
                role='status'
                aria-live='polite'
              >
                <div
                  className={`h-2 w-2 rounded-full bg-green-600 ${prefersReducedMotion ? '' : 'animate-pulse'}`}
                />
                <span className='text-sm font-medium'>Live Updates</span>
              </div>
            )}

            {onScheduleFollowUp && (
              <button
                onClick={onScheduleFollowUp}
                className='flex items-center gap-2 rounded-lg border border-blue-600 px-4 py-2 text-blue-600 transition-colors hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                aria-label='Schedule follow-up appointment'
              >
                <Calendar className='h-4 w-4' />
                Schedule Follow-up
              </button>
            )}

            {onShare && (
              <button
                onClick={onShare}
                className='flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-gray-600 transition-colors hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2'
                aria-label='Share results'
              >
                <Share2 className='h-4 w-4' />
                Share
              </button>
            )}

            {onExport && (
              <button
                onClick={() => onExport('pdf')}
                className='flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                aria-label='Export results as PDF'
              >
                <FileText className='h-4 w-4' />
                Export PDF
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className='border-b border-gray-200' role='tablist' aria-label='Assessment result views'>
        <div className='flex space-x-8 px-6'>
          {[
            { id: 'overview', label: 'Overview', icon: TrendingUp },
            { id: 'speech', label: 'Speech Analysis', icon: Activity },
            { id: 'retinal', label: 'Retinal Analysis', icon: Eye },
            { id: 'motor', label: 'Motor Assessment', icon: Activity },
            { id: 'cognitive', label: 'Cognitive Assessment', icon: Brain },
            { id: 'nri', label: 'NRI Analysis', icon: Zap },
          ].map(tab => {
            const Icon = tab.icon;
            const isSelected = selectedModality === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => handleModalitySelect(tab.id)}
                role='tab'
                aria-selected={isSelected}
                aria-controls={`${tab.id}-panel`}
                className={`flex items-center gap-2 border-b-2 px-1 py-4 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  isSelected
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                } `}
              >
                <Icon className='h-4 w-4' aria-hidden='true' />
                {tab.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main Content */}
      <main className='p-6'>
        {/* Overview Panel */}
        {selectedModality === 'overview' && (
          <div id='overview-panel' role='tabpanel' aria-labelledby='overview-tab'>
            {/* Overall Risk Assessment */}
            <section className='mb-8' aria-labelledby='risk-assessment-heading'>
              <h2 id='risk-assessment-heading' className='mb-4 text-lg font-semibold text-gray-900'>
                Overall Risk Assessment
              </h2>

              <div className='grid grid-cols-1 gap-6 md:grid-cols-4'>
                {/* NRI Score Circle */}
                <div className='text-center'>
                  <div className='relative mx-auto mb-3 h-24 w-24'>
                    <svg
                      className='h-24 w-24 -rotate-90 transform'
                      viewBox='0 0 100 100'
                      role='img'
                      aria-label={`NRI Score: ${Math.round((results.nriResult?.nri_score || 0) * 100)} percent`}
                    >
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
                        style={{
                          transition: prefersReducedMotion
                            ? 'none'
                            : 'stroke-dasharray 1s ease-in-out',
                        }}
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
                    className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 ${getRiskCategoryStyle(results.overallRiskCategory)} `}
                  >
                    {getRiskCategoryIcon(results.overallRiskCategory)}
                    <span className='font-medium capitalize'>
                      {results.overallRiskCategory} Risk
                    </span>
                  </div>
                  <p className='mt-2 text-sm text-gray-600'>Risk Category</p>
                </div>

                {/* Confidence Score */}
                <div className='text-center'>
                  <div className='mb-1 text-3xl font-bold text-gray-900'>
                    {Math.round((results.nriResult?.confidence || 0) * 100)}%
                  </div>
                  <p className='text-sm font-medium text-gray-900'>Confidence</p>
                  <p className='text-xs text-gray-500'>Analysis Certainty</p>
                </div>

                {/* Processing Time */}
                <div className='text-center'>
                  <div className='mb-2 flex items-center justify-center gap-2'>
                    <Clock className='h-5 w-5 text-gray-500' aria-hidden='true' />
                    <span className='text-2xl font-bold text-gray-900'>
                      {(results.totalProcessingTime / 1000).toFixed(1)}s
                    </span>
                  </div>
                  <p className='text-sm font-medium text-gray-900'>Processing Time</p>
                  <p className='text-xs text-gray-500'>Total Analysis Duration</p>
                </div>
              </div>
            </section>

            {/* Risk Trend Chart */}
            <section className='mb-8' aria-labelledby='trend-heading'>
              <h2 id='trend-heading' className='mb-4 text-lg font-semibold text-gray-900'>
                Risk Trend Analysis
              </h2>

              <div className='rounded-lg bg-gray-50 p-4'>
                <div className='mb-4 flex items-center justify-between'>
                  <span className='text-sm font-medium text-gray-700'>7-Day NRI Trend</span>
                  <div className='flex items-center gap-2'>
                    <TrendingUp className='h-4 w-4 text-green-600' aria-hidden='true' />
                    <span className='text-sm text-green-600'>Improving</span>
                  </div>
                </div>

                {/* Simple trend visualization */}
                <div
                  className='flex h-20 items-end justify-between gap-1'
                  role='img'
                  aria-label='7-day risk trend chart'
                >
                  {trendData.map((point, index) => (
                    <div
                      key={index}
                      className='rounded-t bg-blue-600'
                      style={{
                        height: `${point.nriScore * 100}%`,
                        width: '12%',
                      }}
                      title={`${point.date}: ${(point.nriScore * 100).toFixed(1)}%`}
                    />
                  ))}
                </div>

                <div className='mt-2 flex justify-between text-xs text-gray-500'>
                  <span>7 days ago</span>
                  <span>Today</span>
                </div>
              </div>
            </section>

            {/* Modality Summary */}
            <section aria-labelledby='modality-summary-heading'>
              <h2
                id='modality-summary-heading'
                className='mb-4 text-lg font-semibold text-gray-900'
              >
                Assessment Modality Summary
              </h2>

              <div className='grid grid-cols-1 gap-4 lg:grid-cols-2'>
                {/* Speech Analysis Summary */}
                {results.speechResult && (
                  <div className='rounded-lg border border-gray-200 p-4'>
                    <div className='mb-3 flex items-center gap-3'>
                      <div className='rounded-lg bg-blue-100 p-2'>
                        <Activity className='h-5 w-5 text-blue-600' aria-hidden='true' />
                      </div>
                      <div>
                        <h3 className='font-semibold text-gray-900'>Speech Analysis</h3>
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
                    </div>
                  </div>
                )}

                {/* Retinal Analysis Summary */}
                {results.retinalResult && (
                  <div className='rounded-lg border border-gray-200 p-4'>
                    <div className='mb-3 flex items-center gap-3'>
                      <div className='rounded-lg bg-green-100 p-2'>
                        <Eye className='h-5 w-5 text-green-600' aria-hidden='true' />
                      </div>
                      <div>
                        <h3 className='font-semibold text-gray-900'>Retinal Analysis</h3>
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
                    </div>
                  </div>
                )}

                {/* Motor Assessment Summary */}
                {results.motorResult && (
                  <div className='rounded-lg border border-gray-200 p-4'>
                    <div className='mb-3 flex items-center gap-3'>
                      <div className='rounded-lg bg-orange-100 p-2'>
                        <Activity className='h-5 w-5 text-orange-600' aria-hidden='true' />
                      </div>
                      <div>
                        <h3 className='font-semibold text-gray-900'>Motor Assessment</h3>
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
                    </div>
                  </div>
                )}

                {/* Cognitive Assessment Summary */}
                {results.cognitiveResult && (
                  <div className='rounded-lg border border-gray-200 p-4'>
                    <div className='mb-3 flex items-center gap-3'>
                      <div className='rounded-lg bg-purple-100 p-2'>
                        <Brain className='h-5 w-5 text-purple-600' aria-hidden='true' />
                      </div>
                      <div>
                        <h3 className='font-semibold text-gray-900'>Cognitive Assessment</h3>
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
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {/* Individual Modality Panels */}
        {selectedModality !== 'overview' && (
          <div
            id={`${selectedModality}-panel`}
            role='tabpanel'
            aria-labelledby={`${selectedModality}-tab`}
          >
            <div className='py-8 text-center'>
              <p className='text-gray-600'>
                Detailed {selectedModality} analysis view would be implemented here
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
