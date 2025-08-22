'use client';

import React, { useState } from 'react';
import { Button, Card } from '@/components/ui';
import type { CompleteAssessmentResult } from '@/lib/ml';

interface ResultsExportProps {
  results: CompleteAssessmentResult;
}

export const ResultsExport: React.FC<ResultsExportProps> = ({ results }) => {
  const [isExporting, setIsExporting] = useState(false);

  const handleExportPDF = async () => {
    setIsExporting(true);
    try {
      // In a real implementation, this would generate a PDF
      // For now, we'll simulate the process
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Create a simple text export for demonstration
      const exportData = {
        sessionId: results.sessionId,
        timestamp: results.metadata.timestamp.toISOString(),
        nriScore: results.nriResult.nriScore,
        riskCategory: results.nriResult.riskCategory,
        confidence: results.nriResult.confidence,
        recommendations: results.nriResult.recommendations,
        clinicalNotes: results.nriResult.clinicalNotes,
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `neurolens-x-results-${results.sessionId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
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
      const shareText = `My NeuroLens-X assessment: NRI Score ${results.nriResult.nriScore} (${results.nriResult.riskCategory} risk). View at ${window.location.href}`;
      await navigator.clipboard.writeText(shareText);
      alert('Results copied to clipboard!');
    }
  };

  return (
    <Card className="p-6">
      <h3 className="mb-6 text-center text-xl font-semibold text-text-primary">
        Export & Share Results
      </h3>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* PDF Export */}
        <div className="space-y-3 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-red-500/10">
            <svg
              className="h-6 w-6 text-red-400"
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
          </div>
          <h4 className="font-medium text-text-primary">PDF Report</h4>
          <p className="text-sm text-text-secondary">
            Download a comprehensive clinical report
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleExportPDF}
            disabled={isExporting}
            className="w-full"
          >
            {isExporting ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"></div>
                Generating...
              </>
            ) : (
              'Download PDF'
            )}
          </Button>
        </div>

        {/* Share Results */}
        <div className="space-y-3 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-blue-500/10">
            <svg
              className="h-6 w-6 text-blue-400"
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
          </div>
          <h4 className="font-medium text-text-primary">Share Results</h4>
          <p className="text-sm text-text-secondary">
            Share with healthcare providers
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleShareResults}
            className="w-full"
          >
            Share
          </Button>
        </div>

        {/* Print Results */}
        <div className="space-y-3 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-lg bg-green-500/10">
            <svg
              className="h-6 w-6 text-green-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"
              />
            </svg>
          </div>
          <h4 className="font-medium text-text-primary">Print Results</h4>
          <p className="text-sm text-text-secondary">
            Print a hard copy for your records
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              if (typeof window !== 'undefined') {
                window.print();
              }
            }}
            className="w-full"
          >
            Print
          </Button>
        </div>
      </div>

      {/* Privacy Notice */}
      <div className="mt-6 rounded-lg bg-surface-secondary p-4">
        <div className="flex items-start space-x-3">
          <svg
            className="mt-0.5 h-5 w-5 flex-shrink-0 text-primary-400"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
              clipRule="evenodd"
            />
          </svg>
          <div>
            <h5 className="mb-1 font-medium text-primary-400">
              Privacy Protected
            </h5>
            <p className="text-sm text-text-secondary">
              Your assessment data is processed locally and not stored on our
              servers. Exported files contain only the information you choose to
              share.
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
};
