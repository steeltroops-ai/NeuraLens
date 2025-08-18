'use client';

import React, { useState, useEffect } from 'react';
import { Button, Card, Progress } from '@/components/ui';
import { NRIScoreDisplay } from './NRIScoreDisplay';
import { ModalityBreakdown } from './ModalityBreakdown';
import { ClinicalRecommendations } from './ClinicalRecommendations';
import { ResultsExport } from './ResultsExport';
import type { CompleteAssessmentResult } from '@/lib/ml';

// Mock data for demonstration
const mockResults: CompleteAssessmentResult = {
  sessionId: 'demo_session_123',
  nriResult: {
    nriScore: 42,
    confidence: 87,
    riskCategory: 'moderate',
    modalityContributions: {
      speech: 0.3,
      retinal: 0.25,
      risk: 0.35,
      motor: 0.1,
    },
    uncertaintyFactors: ['Limited motor assessment data'],
    recommendations: [
      'Regular monitoring with healthcare provider',
      'Focus on modifiable risk factors',
      'Consider lifestyle interventions',
      'Annual cognitive screening',
    ],
    processingTime: 15420,
    dataCompleteness: 75,
    clinicalNotes: [
      'Unified NRI Score: 42/100',
      'Speech Analysis: 38/100 (Quality: 92%)',
      'Retinal Analysis: 45/100 (Quality: 88%)',
      'Risk Assessment: 48/100 (Quality: 95%)',
      'Total Processing Time: 15420ms',
    ],
  },
  modalityResults: {
    speech: {
      riskScore: 38,
      confidence: 92,
      findings: ['Slight reduction in speech rate', 'Normal voice quality'],
      processingTime: 8200,
      qualityScore: 0.92,
      features: {
        speechRate: 145,
        pauseDuration: 850,
        pauseFrequency: 4.2,
        articulationRate: 9.7,
        fundamentalFreq: 185,
        f0Variability: 12.5,
        jitter: 0.008,
        shimmer: 0.045,
        harmonicNoiseRatio: 22.3,
        spectralCentroid: 1200,
        spectralBandwidth: 2100,
        tremorFrequency: 5.2,
        tremorAmplitude: 0.08,
        voiceTremor: 0.15,
        stressPattern: [0.6, 0.4, 0.8, 0.5],
        intonationRange: 8.5,
        rhythmVariability: 0.25,
      },
    },
    retinal: {
      riskScore: 45,
      confidence: 88,
      findings: [
        'Normal vessel density',
        'Slight increase in vessel tortuosity',
      ],
      processingTime: 6100,
      imageQuality: 0.88,
      recommendations: ['Annual eye examination recommended'],
      features: {
        vesselDensity: 18.5,
        vesselTortuosity: 1.3,
        vesselWidth: 8.2,
        branchingAngle: 72,
        opticDiscArea: 2850,
        cupDiscRatio: 0.28,
        rimArea: 2052,
        maculaArea: 14500,
        fovealThickness: 185,
        maculaPigmentation: 0.65,
        microaneurysms: 0,
        hemorrhages: 0,
        exudates: 0,
        cottonWoolSpots: 0,
        retinalNerveLayer: 92,
        ganglionCellLayer: 88,
        vascularComplexity: 1.68,
        imageSharpness: 0.85,
        illumination: 0.78,
        contrast: 0.72,
      },
    },
    risk: {
      overallRisk: 48,
      confidence: 95,
      categoryRisks: {
        demographic: 35,
        medical: 25,
        family: 60,
        lifestyle: 30,
        cognitive: 20,
      },
      modifiableFactors: [
        'Increase physical activity',
        'Improve sleep quality',
      ],
      nonModifiableFactors: ['Advanced age', 'Family history of dementia'],
      recommendations: ['Regular monitoring with healthcare provider'],
      processingTime: 1120,
    },
  },
  metadata: {
    totalProcessingTime: 15420,
    timestamp: new Date(),
    version: '1.0.0',
    dataQuality: 0.88,
  },
};

export const ResultsDashboard: React.FC = () => {
  const [results, setResults] = useState<CompleteAssessmentResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<
    'overview' | 'details' | 'recommendations'
  >('overview');

  useEffect(() => {
    // In a real app, this would fetch results from storage or API
    // For demo, we'll use mock data
    setTimeout(() => {
      setResults(mockResults);
      setIsLoading(false);
    }, 1000);
  }, []);

  if (isLoading) {
    return (
      <div className="mx-auto max-w-6xl">
        <div className="py-12 text-center">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-2 border-primary-500 border-t-transparent"></div>
          <h2 className="mb-2 text-xl font-semibold text-text-primary">
            Loading Results
          </h2>
          <p className="text-text-secondary">
            Retrieving your assessment results...
          </p>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="mx-auto max-w-6xl">
        <Card className="p-8 text-center">
          <svg
            className="mx-auto mb-4 h-16 w-16 text-text-muted"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <h2 className="mb-2 text-2xl font-bold text-text-primary">
            No Results Found
          </h2>
          <p className="mb-6 text-text-secondary">
            We couldn't find any assessment results. Please complete an
            assessment first.
          </p>
          <Button
            onClick={() => {
              if (typeof window !== 'undefined') {
                window.location.href = '/assessment';
              }
            }}
          >
            Start Assessment
          </Button>
        </Card>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'details', label: 'Detailed Analysis', icon: '🔍' },
    { id: 'recommendations', label: 'Recommendations', icon: '💡' },
  ] as const;

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      {/* Header */}
      <div className="space-y-4 text-center">
        <h1 className="text-4xl font-bold text-text-primary">
          Assessment Results
        </h1>
        <p className="text-lg text-text-secondary">
          Your comprehensive neurological risk assessment
        </p>
        <div className="text-sm text-text-muted">
          Completed on {results.metadata.timestamp.toLocaleDateString()} at{' '}
          {results.metadata.timestamp.toLocaleTimeString()}
        </div>
      </div>

      {/* NRI Score Display */}
      <NRIScoreDisplay nriResult={results.nriResult} />

      {/* Tab Navigation */}
      <div className="border-b border-neutral-800">
        <nav className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`border-b-2 px-2 py-4 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-400'
                  : 'border-transparent text-text-secondary hover:border-neutral-600 hover:text-text-primary'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Quick Stats */}
            <div className="grid grid-cols-1 gap-6 md:grid-cols-4">
              <Card className="p-6 text-center">
                <div className="mb-2 text-3xl font-bold text-primary-400">
                  {results.nriResult.nriScore}
                </div>
                <div className="text-sm text-text-secondary">NRI Score</div>
              </Card>

              <Card className="p-6 text-center">
                <div className="mb-2 text-3xl font-bold text-text-primary">
                  {results.nriResult.confidence}%
                </div>
                <div className="text-sm text-text-secondary">Confidence</div>
              </Card>

              <Card className="p-6 text-center">
                <div className="mb-2 text-3xl font-bold text-text-primary">
                  {results.nriResult.dataCompleteness}%
                </div>
                <div className="text-sm text-text-secondary">Data Complete</div>
              </Card>

              <Card className="p-6 text-center">
                <div className="mb-2 text-3xl font-bold text-text-primary">
                  {Math.round(results.metadata.totalProcessingTime / 1000)}s
                </div>
                <div className="text-sm text-text-secondary">
                  Processing Time
                </div>
              </Card>
            </div>

            {/* Modality Breakdown */}
            <ModalityBreakdown
              modalityResults={results.modalityResults}
              contributions={results.nriResult.modalityContributions}
            />
          </div>
        )}

        {activeTab === 'details' && (
          <div className="space-y-8">
            {/* Detailed Analysis */}
            <Card className="p-8">
              <h3 className="mb-6 text-2xl font-semibold text-text-primary">
                Detailed Analysis
              </h3>

              {/* Clinical Notes */}
              <div className="space-y-4">
                <h4 className="text-lg font-medium text-text-primary">
                  Clinical Notes
                </h4>
                <div className="rounded-lg bg-surface-secondary p-4">
                  {results.nriResult.clinicalNotes.map((note, index) => (
                    <div
                      key={index}
                      className="py-1 text-sm text-text-secondary"
                    >
                      {note}
                    </div>
                  ))}
                </div>
              </div>

              {/* Uncertainty Factors */}
              {results.nriResult.uncertaintyFactors.length > 0 && (
                <div className="mt-6 space-y-4">
                  <h4 className="text-lg font-medium text-text-primary">
                    Uncertainty Factors
                  </h4>
                  <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-4">
                    {results.nriResult.uncertaintyFactors.map(
                      (factor, index) => (
                        <div
                          key={index}
                          className="flex items-center py-1 text-sm text-amber-300"
                        >
                          <svg
                            className="mr-2 h-4 w-4 flex-shrink-0"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                          >
                            <path
                              fillRule="evenodd"
                              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                              clipRule="evenodd"
                            />
                          </svg>
                          {factor}
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </Card>
          </div>
        )}

        {activeTab === 'recommendations' && (
          <ClinicalRecommendations
            recommendations={results.nriResult.recommendations}
            riskCategory={results.nriResult.riskCategory}
            modalityResults={results.modalityResults}
          />
        )}
      </div>

      {/* Export Options */}
      <ResultsExport results={results} />

      {/* Action Buttons */}
      <div className="flex flex-col justify-center gap-4 sm:flex-row">
        <Button
          variant="secondary"
          size="lg"
          onClick={() => {
            if (typeof window !== 'undefined') {
              window.location.href = '/assessment';
            }
          }}
          className="px-8"
        >
          Take New Assessment
        </Button>

        <Button
          variant="primary"
          size="lg"
          onClick={() => {
            if (typeof window !== 'undefined') {
              window.print();
            }
          }}
          className="px-8"
        >
          Print Results
        </Button>
      </div>
    </div>
  );
};
