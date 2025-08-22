'use client';

import React from 'react';
import { Card } from '@/components/ui';

interface ClinicalRecommendationsProps {
  recommendations: string[];
  riskCategory: string;
  modalityResults: any;
}

export const ClinicalRecommendations: React.FC<ClinicalRecommendationsProps> = ({
  recommendations,
  riskCategory,
  modalityResults,
}) => {
  const getRiskCategoryInfo = (category: string) => {
    switch (category) {
      case 'low':
        return { color: 'text-green-400', bgColor: 'bg-green-500/10', icon: '✅' };
      case 'moderate':
        return { color: 'text-amber-400', bgColor: 'bg-amber-500/10', icon: '⚠️' };
      case 'high':
        return { color: 'text-orange-400', bgColor: 'bg-orange-500/10', icon: '🔶' };
      case 'critical':
        return { color: 'text-red-400', bgColor: 'bg-red-500/10', icon: '🚨' };
      default:
        return { color: 'text-neutral-400', bgColor: 'bg-neutral-500/10', icon: '❓' };
    }
  };

  const riskInfo = getRiskCategoryInfo(riskCategory);

  return (
    <div className="space-y-8">
      {/* Primary Recommendations */}
      <Card className="p-8">
        <div className="flex items-center mb-6">
          <span className="text-2xl mr-3">{riskInfo.icon}</span>
          <h3 className="text-2xl font-semibold text-text-primary">
            Clinical Recommendations
          </h3>
        </div>

        <div className="space-y-4">
          {recommendations.map((recommendation, index) => (
            <div key={index} className="flex items-start space-x-3 p-4 bg-surface-secondary rounded-lg">
              <div className="w-6 h-6 bg-primary-500 text-white rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 mt-0.5">
                {index + 1}
              </div>
              <div className="flex-1">
                <p className="text-text-primary">{recommendation}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Modality-Specific Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Lifestyle Recommendations */}
        <Card className="p-6">
          <h4 className="text-lg font-semibold text-text-primary mb-4 flex items-center">
            <span className="mr-2">🏃‍♂️</span>
            Lifestyle Modifications
          </h4>
          <div className="space-y-3">
            {modalityResults.risk?.modifiableFactors?.map((factor: string, index: number) => (
              <div key={index} className="flex items-start space-x-2">
                <span className="text-success text-sm mt-1">✓</span>
                <span className="text-sm text-text-secondary">{factor}</span>
              </div>
            )) || (
              <p className="text-sm text-text-muted">No specific lifestyle modifications identified</p>
            )}
          </div>
        </Card>

        {/* Follow-up Care */}
        <Card className="p-6">
          <h4 className="text-lg font-semibold text-text-primary mb-4 flex items-center">
            <span className="mr-2">🏥</span>
            Follow-up Care
          </h4>
          <div className="space-y-3">
            <div className="flex items-start space-x-2">
              <span className="text-primary-400 text-sm mt-1">•</span>
              <span className="text-sm text-text-secondary">
                Schedule follow-up assessment in 6-12 months
              </span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-primary-400 text-sm mt-1">•</span>
              <span className="text-sm text-text-secondary">
                Discuss results with your healthcare provider
              </span>
            </div>
            <div className="flex items-start space-x-2">
              <span className="text-primary-400 text-sm mt-1">•</span>
              <span className="text-sm text-text-secondary">
                Monitor for any new symptoms or changes
              </span>
            </div>
          </div>
        </Card>
      </div>

      {/* Emergency Information */}
      {riskCategory === 'critical' && (
        <Card className="p-6 border-red-500/20 bg-red-500/5">
          <div className="flex items-start space-x-3">
            <span className="text-2xl">🚨</span>
            <div>
              <h4 className="text-lg font-semibold text-red-400 mb-2">
                Urgent Medical Attention Required
              </h4>
              <p className="text-sm text-text-secondary mb-4">
                Your assessment indicates a critical risk level. Please contact your healthcare provider immediately or seek emergency medical attention if you experience any concerning symptoms.
              </p>
              <div className="space-y-2">
                <div className="text-sm text-text-secondary">
                  <strong>Emergency Signs:</strong> Sudden confusion, severe headache, difficulty speaking, vision changes, weakness, or loss of coordination
                </div>
                <div className="text-sm text-text-secondary">
                  <strong>Emergency Number:</strong> Call 911 or your local emergency services
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
