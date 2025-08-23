'use client';

import React from 'react';
import { Card } from '@/components/ui';

interface NRIScoreDisplayProps {
  score: number;
  confidence: number;
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
  className?: string;
}

export const NRIScoreDisplay: React.FC<NRIScoreDisplayProps> = ({
  score,
  confidence,
  riskLevel,
  className = ''
}) => {
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-50';
      case 'moderate': return 'text-yellow-600 bg-yellow-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'critical': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <Card className={`p-6 ${className}`}>
      <div className="text-center">
        <h3 className="text-lg font-semibold text-slate-900 mb-4">
          Neurological Risk Index (NRI)
        </h3>
        
        <div className="mb-4">
          <div className="text-4xl font-bold text-slate-900 mb-2">
            {score.toFixed(1)}
          </div>
          <div className="text-sm text-slate-500">
            out of 100
          </div>
        </div>

        <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(riskLevel)}`}>
          {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk
        </div>

        <div className="mt-4 text-sm text-slate-600">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      </div>
    </Card>
  );
};
