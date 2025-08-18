'use client';

import React from 'react';
import { Button, Card } from '@/components/ui';
import { NRIScoreDisplay } from '@/components/results/NRIScoreDisplay';
import { ClinicalRecommendations } from '@/components/results/ClinicalRecommendations';
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
          title: 'NeuroLens-X Assessment Results',
          text: `My neurological risk assessment shows an NRI score of ${results.nriResult.nriScore} (${results.nriResult.riskCategory} risk)`,
          url: window.location.href,
        });
      } catch (error) {
        console.error('Sharing failed:', error);
      }
    } else {
      // Fallback: copy to clipboard
      const shareText = `My NeuroLens-X assessment: NRI Score ${results.nriResult.nriScore} (${results.nriResult.riskCategory} risk)`;
      try {
        await navigator.clipboard.writeText(shareText);
        alert('Results copied to clipboard!');
      } catch (error) {
        console.error('Copy failed:', error);
      }
    }
  };

  return (
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-success to-green-600 rounded-2xl flex items-center justify-center mx-auto">
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
              </svg>
            </div>
            <h1 className="text-4xl font-bold text-text-primary">
              Assessment Complete
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Your comprehensive neurological risk assessment results
            </p>
            <div className="text-sm text-text-muted">
              Completed on {results.metadata.timestamp.toLocaleDateString()} at{' '}
              {results.metadata.timestamp.toLocaleTimeString()}
            </div>
          </div>

          {/* NRI Score Display */}
          <NRIScoreDisplay nriResult={results.nriResult} />

          {/* Quick Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="p-6 text-center">
              <div className="text-3xl font-bold text-primary-400 mb-2">
                {results.nriResult.nriScore}
              </div>
              <div className="text-sm text-text-secondary">NRI Score</div>
            </Card>
            
            <Card className="p-6 text-center">
              <div className="text-3xl font-bold text-text-primary mb-2">
                {results.nriResult.confidence}%
              </div>
              <div className="text-sm text-text-secondary">Confidence</div>
            </Card>
            
            <Card className="p-6 text-center">
              <div className="text-3xl font-bold text-text-primary mb-2">
                {results.nriResult.dataCompleteness}%
              </div>
              <div className="text-sm text-text-secondary">Data Complete</div>
            </Card>
            
            <Card className="p-6 text-center">
              <div className="text-3xl font-bold text-text-primary mb-2">
                {Math.round(results.metadata.totalProcessingTime / 1000)}s
              </div>
              <div className="text-sm text-text-secondary">Processing Time</div>
            </Card>
          </div>

          {/* Clinical Recommendations */}
          <ClinicalRecommendations 
            recommendations={results.nriResult.recommendations}
            riskCategory={results.nriResult.riskCategory}
            modalityResults={results.modalityResults}
          />

          {/* Action Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Button
              onClick={handleDownloadReport}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Report
            </Button>
            
            <Button
              onClick={handleShareResults}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
              </svg>
              Share Results
            </Button>
            
            <Button
              onClick={onRestart}
              variant="primary"
              size="lg"
              className="w-full"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              New Assessment
            </Button>
            
            <Button
              onClick={onExit}
              variant="secondary"
              size="lg"
              className="w-full"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Go Home
            </Button>
          </div>

          {/* Important Disclaimer */}
          <Card className="p-6 border-amber-500/20 bg-amber-500/5">
            <div className="flex items-start space-x-3">
              <svg className="w-6 h-6 text-amber-400 mt-1 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="font-semibold text-amber-400 mb-2">Important Medical Disclaimer</h3>
                <p className="text-sm text-text-secondary leading-relaxed">
                  This assessment is a screening tool and not a diagnostic device. Results should not replace 
                  professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
                  professionals for medical decisions. If you have immediate health concerns, contact your 
                  healthcare provider or emergency services.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
