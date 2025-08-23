'use client';

import React from 'react';
import { Card, Button } from '@/components/ui';
import { Download, FileText, Share2 } from 'lucide-react';

interface ResultsExportProps {
  nriScore: number;
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
  sessionId?: string;
  className?: string;
}

export const ResultsExport: React.FC<ResultsExportProps> = ({
  nriScore,
  riskLevel,
  sessionId,
  className = '',
}) => {
  const handleExportPDF = () => {
    // Placeholder for PDF export functionality
    console.log('Exporting PDF report...');
    // In a real implementation, this would generate and download a PDF
  };

  const handleExportJSON = () => {
    // Placeholder for JSON export functionality
    const data = {
      sessionId,
      nriScore,
      riskLevel,
      timestamp: new Date().toISOString(),
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuralens-results-${sessionId || Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleShare = () => {
    // Placeholder for sharing functionality
    if (navigator.share) {
      navigator.share({
        title: 'NeuraLens Assessment Results',
        text: `NRI Score: ${nriScore.toFixed(1)} (${riskLevel} risk)`,
        url: window.location.href,
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(
        `NeuraLens Assessment Results\nNRI Score: ${nriScore.toFixed(1)} (${riskLevel} risk)\n${window.location.href}`
      );
    }
  };

  return (
    <Card className={`p-6 ${className}`}>
      <h3 className="mb-4 text-lg font-semibold text-slate-900">
        Export Results
      </h3>

      <div className="space-y-3">
        <Button
          onClick={handleExportPDF}
          variant="secondary"
          className="w-full justify-start"
        >
          <FileText className="mr-2 h-4 w-4" />
          Export PDF Report
        </Button>

        <Button
          onClick={handleExportJSON}
          variant="secondary"
          className="w-full justify-start"
        >
          <Download className="mr-2 h-4 w-4" />
          Download Data (JSON)
        </Button>

        <Button
          onClick={handleShare}
          variant="secondary"
          className="w-full justify-start"
        >
          <Share2 className="mr-2 h-4 w-4" />
          Share Results
        </Button>
      </div>

      <div className="mt-4 rounded-lg bg-gray-50 p-3">
        <div className="text-sm text-gray-600">
          <strong>Session ID:</strong> {sessionId || 'N/A'}
        </div>
        <div className="mt-1 text-xs text-gray-500">
          Generated on {new Date().toLocaleDateString()}
        </div>
      </div>
    </Card>
  );
};
