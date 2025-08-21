'use client';

import React from 'react';
import { Card } from '@/components/ui';

interface ModalityBreakdownProps {
  modalityResults: any;
  contributions: Record<string, number>;
}

export const ModalityBreakdown: React.FC<ModalityBreakdownProps> = ({ 
  modalityResults, 
  contributions 
}) => {
  const modalities = [
    {
      id: 'speech',
      name: 'Speech Analysis',
      icon: '🎤',
      description: 'Voice biomarker detection',
      result: modalityResults.speech,
    },
    {
      id: 'retinal',
      name: 'Retinal Imaging',
      icon: '👁️',
      description: 'Vascular pattern analysis',
      result: modalityResults.retinal,
    },
    {
      id: 'risk',
      name: 'Risk Assessment',
      icon: '📊',
      description: 'Health and lifestyle factors',
      result: modalityResults.risk,
    },
  ];

  return (
    <div className="space-y-6">
      <h3 className="text-2xl font-semibold text-text-primary text-center">
        Modality Breakdown
      </h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {modalities.map((modality) => {
          const contribution = Math.round((contributions[modality.id] || 0) * 100);
          const result = modality.result;
          
          if (!result) {
            return (
              <Card key={modality.id} className="p-6 opacity-50">
                <div className="text-center">
                  <div className="text-4xl mb-4">{modality.icon}</div>
                  <h4 className="text-lg font-semibold text-text-primary mb-2">
                    {modality.name}
                  </h4>
                  <p className="text-sm text-text-muted">Not available</p>
                </div>
              </Card>
            );
          }

          return (
            <Card key={modality.id} className="p-6">
              <div className="text-center mb-4">
                <div className="text-4xl mb-2">{modality.icon}</div>
                <h4 className="text-lg font-semibold text-text-primary mb-1">
                  {modality.name}
                </h4>
                <p className="text-sm text-text-secondary mb-4">
                  {modality.description}
                </p>
              </div>

              <div className="space-y-4">
                {/* Score */}
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary-400 mb-1">
                    {result.riskScore || result.overallRisk}
                  </div>
                  <div className="text-sm text-text-muted">Risk Score</div>
                </div>

                {/* Contribution */}
                <div className="bg-surface-secondary rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-text-secondary">Contribution</span>
                    <span className="text-sm font-semibold text-text-primary">
                      {contribution}%
                    </span>
                  </div>
                  <div className="w-full bg-neutral-800 rounded-full h-2">
                    <div 
                      className="bg-primary-500 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${contribution}%` }}
                    />
                  </div>
                </div>

                {/* Key Findings */}
                <div>
                  <h5 className="text-sm font-medium text-text-primary mb-2">
                    Key Findings
                  </h5>
                  <div className="space-y-1">
                    {(result.findings || []).slice(0, 2).map((finding: string, index: number) => (
                      <div key={index} className="text-xs text-text-secondary flex items-start">
                        <span className="text-primary-400 mr-2">•</span>
                        {finding}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};
