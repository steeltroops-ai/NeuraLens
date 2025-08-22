'use client';

import React from 'react';
import { Button, Card } from '@/components/ui';
// Removed dependencies on deleted results components
import type { CompleteAssessmentResult } from '@/lib/ml';

interface ResultsStepProps {
  results: CompleteAssessmentResult;
  onRestart: () => void;
  onExit: () => void;
}

export const ResultsStep: React.FC<ResultsStepProps> = ({
  results,
  onRestart,
  onExit,
}) => {
  const handleDownloadReport = () => {
    // In a real implementation, this would generate and download a PDF report
    if (typeof window !== 'undefined') {
      window.print();
    }
  };

  const handleShareResults = async () => {
    if (typeof window === 'undefined') return;

    if (navigator.share) {
      try {
        await navigator.share({
          title: 'NeuroLens-X Health Check Results',
          text: `My brain health check shows a health score of ${results.nriResult.nriScore} (${results.nriResult.riskCategory} risk)`,
          url: window.location.href,
        });
      } catch (error) {
        console.error('Sharing failed:', error);
      }
    } else {
      // Fallback: copy to clipboard
      const shareText = `My NeuroLens-X health check: Health Score ${results.nriResult.nriScore} (${results.nriResult.riskCategory} risk)`;
      try {
        await navigator.clipboard.writeText(shareText);
        alert('Results copied to clipboard!');
      } catch (error) {
        console.error('Copy failed:', error);
      }
    }
  };

  return (
    <div className="min-h-screen py-12 bg-gray-50">
      <div className="container px-6 mx-auto">
        <div className="max-w-6xl mx-auto space-y-12">
          {/* Apple-Style Header */}
          <div className="space-y-6 text-center animate-fade-in">
            <div className="flex items-center justify-center w-24 h-24 mx-auto shadow-success rounded-apple-xl bg-gradient-to-br from-success-500 to-success-600">
              <svg
                className="w-12 h-12 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
              </svg>
            </div>
            <h1 className="text-5xl font-bold tracking-tight text-text-primary">
              Health Check Complete
            </h1>
            <p className="max-w-2xl mx-auto text-xl leading-relaxed text-text-secondary">
              Your comprehensive brain health evaluation results
            </p>
            <div className="inline-block px-6 py-3 bg-white border border-gray-200 shadow-sm rounded-apple-lg">
              <div className="text-sm text-text-secondary">
                Completed on {results.metadata.timestamp.toLocaleDateString()}{' '}
                at {results.metadata.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* NRI Score Display */}
          <Card className="p-6">
            <div className="text-center">
              <div className="mb-4">
                <div className="text-6xl font-bold text-blue-600">
                  {results.nriResult.nriScore}
                </div>
                <div className="text-lg text-slate-600">
                  Neurological Risk Index
                </div>
              </div>
              <div className="text-sm text-slate-500">
                Risk Category:{' '}
                <span className="font-medium">
                  {results.nriResult.riskCategory}
                </span>
              </div>
            </div>
          </Card>

          {/* Quick Summary */}
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
              <div className="text-sm text-text-secondary">Processing Time</div>
            </Card>
          </div>

          {/* Clinical Recommendations */}
          <Card className="p-6">
            <h3 className="mb-4 text-lg font-semibold text-slate-900">
              Clinical Recommendations
            </h3>
            <div className="space-y-3">
              {results.nriResult.recommendations.map(
                (recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 mt-1 bg-blue-600 rounded-full"></div>
                    <p className="text-sm text-slate-700">{recommendation}</p>
                  </div>
                )
              )}
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            <Button
              onClick={handleDownloadReport}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Download Report
            </Button>

            <Button
              onClick={handleShareResults}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"
                />
              </svg>
              Share Results
            </Button>

            <Button
              onClick={onRestart}
              variant="primary"
              size="lg"
              className="w-full"
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              New Health Check
            </Button>

            <Button
              onClick={onExit}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                />
              </svg>
              Go Home
            </Button>
          </div>

          {/* Important Disclaimer */}
          <Card className="p-6 border-amber-500/20 bg-amber-500/5">
            <div className="flex items-start space-x-3">
              <svg
                className="flex-shrink-0 w-6 h-6 mt-1 text-amber-400"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <h3 className="mb-2 font-semibold text-amber-400">
                  Important Medical Disclaimer
                </h3>
                <p className="text-sm leading-relaxed text-text-secondary">
                  This health check is a screening tool and not a medical
                  diagnosis. Results should not replace professional medical
                  advice, diagnosis, or treatment. Always consult with qualified
                  healthcare professionals for medical decisions. If you have
                  immediate health concerns, contact your healthcare provider or
                  emergency services.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
