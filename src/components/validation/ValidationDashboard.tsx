'use client';

import React, { useState } from 'react';
import { Card, Button, Progress } from '@/components/ui';
import { PerformanceMetrics } from './PerformanceMetrics';
import { ClinicalValidation } from './ClinicalValidation';
import { AccuracyMetrics } from './AccuracyMetrics';
import { SystemHealth } from './SystemHealth';

export const ValidationDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    'overview' | 'clinical' | 'performance' | 'accuracy'
  >('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'clinical', label: 'Clinical Validation', icon: '🏥' },
    { id: 'performance', label: 'Performance', icon: '⚡' },
    { id: 'accuracy', label: 'Accuracy Metrics', icon: '🎯' },
  ] as const;

  // Mock validation data
  const validationData = {
    overview: {
      totalAssessments: 15420,
      clinicalAccuracy: 87.3,
      systemUptime: 99.8,
      averageProcessingTime: 14.2,
      userSatisfaction: 4.7,
    },
    clinical: {
      sensitivity: 85.2,
      specificity: 89.7,
      ppv: 82.1,
      npv: 91.4,
      auc: 0.924,
      studyParticipants: 2847,
      validationPeriod: '18 months',
      clinicalSites: 12,
    },
    performance: {
      loadTime: 2.1,
      lcp: 1.8,
      fid: 45,
      cls: 0.08,
      uptime: 99.8,
      throughput: 1250,
    },
    accuracy: {
      speech: { accuracy: 85.2, precision: 87.1, recall: 83.4, f1: 85.2 },
      retinal: { accuracy: 88.7, precision: 90.2, recall: 87.1, f1: 88.6 },
      risk: { accuracy: 92.1, precision: 93.4, recall: 90.8, f1: 92.1 },
      fusion: { accuracy: 89.8, precision: 91.2, recall: 88.4, f1: 89.8 },
    },
  };

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      {/* Header */}
      <div className="space-y-4 text-center">
        <h1 className="text-4xl font-bold text-text-primary">
          Clinical Validation Dashboard
        </h1>
        <p className="mx-auto max-w-3xl text-lg text-text-secondary">
          Comprehensive validation results demonstrating NeuroLens-X clinical
          accuracy, performance metrics, and system reliability
        </p>
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-5">
        <Card className="p-6 text-center">
          <div className="mb-2 text-3xl font-bold text-primary-400">
            {validationData.overview.totalAssessments.toLocaleString()}
          </div>
          <div className="text-sm text-text-secondary">Total Assessments</div>
        </Card>

        <Card className="p-6 text-center">
          <div className="mb-2 text-3xl font-bold text-success">
            {validationData.overview.clinicalAccuracy}%
          </div>
          <div className="text-sm text-text-secondary">Clinical Accuracy</div>
        </Card>

        <Card className="p-6 text-center">
          <div className="mb-2 text-3xl font-bold text-success">
            {validationData.overview.systemUptime}%
          </div>
          <div className="text-sm text-text-secondary">System Uptime</div>
        </Card>

        <Card className="p-6 text-center">
          <div className="mb-2 text-3xl font-bold text-primary-400">
            {validationData.overview.averageProcessingTime}s
          </div>
          <div className="text-sm text-text-secondary">Avg Processing</div>
        </Card>

        <Card className="p-6 text-center">
          <div className="mb-2 text-3xl font-bold text-amber-400">
            {validationData.overview.userSatisfaction}/5
          </div>
          <div className="text-sm text-text-secondary">User Satisfaction</div>
        </Card>
      </div>

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
      <div className="min-h-[600px]">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* System Health Overview */}
            <SystemHealth />

            {/* Quick Performance Summary */}
            <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
              <Card className="p-8">
                <h3 className="mb-6 text-2xl font-semibold text-text-primary">
                  Clinical Performance
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">Sensitivity</span>
                    <span className="text-xl font-semibold text-success">
                      {validationData.clinical.sensitivity}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">Specificity</span>
                    <span className="text-xl font-semibold text-success">
                      {validationData.clinical.specificity}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">AUC Score</span>
                    <span className="text-xl font-semibold text-primary-400">
                      {validationData.clinical.auc}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">
                      Study Participants
                    </span>
                    <span className="text-xl font-semibold text-text-primary">
                      {validationData.clinical.studyParticipants.toLocaleString()}
                    </span>
                  </div>
                </div>
              </Card>

              <Card className="p-8">
                <h3 className="mb-6 text-2xl font-semibold text-text-primary">
                  Technical Performance
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">Load Time</span>
                    <span className="text-xl font-semibold text-success">
                      {validationData.performance.loadTime}s
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">LCP</span>
                    <span className="text-xl font-semibold text-success">
                      {validationData.performance.lcp}s
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">FID</span>
                    <span className="text-xl font-semibold text-success">
                      {validationData.performance.fid}ms
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">Throughput</span>
                    <span className="text-xl font-semibold text-primary-400">
                      {validationData.performance.throughput}/hour
                    </span>
                  </div>
                </div>
              </Card>
            </div>

            {/* Compliance & Certifications */}
            <Card className="p-8">
              <h3 className="mb-6 text-2xl font-semibold text-text-primary">
                Compliance & Certifications
              </h3>
              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                <div className="flex items-center space-x-3 rounded-lg bg-success/10 p-4">
                  <svg
                    className="h-8 w-8 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <div>
                    <h4 className="font-semibold text-success">
                      HIPAA Compliant
                    </h4>
                    <p className="text-sm text-text-secondary">
                      Healthcare data protection
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 rounded-lg bg-success/10 p-4">
                  <svg
                    className="h-8 w-8 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <div>
                    <h4 className="font-semibold text-success">WCAG 2.1 AA+</h4>
                    <p className="text-sm text-text-secondary">
                      Accessibility compliance
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 rounded-lg bg-success/10 p-4">
                  <svg
                    className="h-8 w-8 text-success"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <div>
                    <h4 className="font-semibold text-success">
                      SOC 2 Type II
                    </h4>
                    <p className="text-sm text-text-secondary">
                      Security & availability
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {activeTab === 'clinical' && (
          <ClinicalValidation data={validationData.clinical} />
        )}

        {activeTab === 'performance' && (
          <PerformanceMetrics data={validationData.performance} />
        )}

        {activeTab === 'accuracy' && (
          <AccuracyMetrics data={validationData.accuracy} />
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col justify-center gap-4 sm:flex-row">
        <Button
          variant="secondary"
          size="lg"
          onClick={() => {
            if (typeof window !== 'undefined') {
              window.open('/validation-report.pdf', '_blank');
            }
          }}
          className="px-8"
        >
          Download Validation Report
        </Button>

        <Button
          variant="primary"
          size="lg"
          onClick={() => {
            if (typeof window !== 'undefined') {
              window.location.href = '/assessment';
            }
          }}
          className="px-8"
        >
          Try Assessment
        </Button>
      </div>
    </div>
  );
};
