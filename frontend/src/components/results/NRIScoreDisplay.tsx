'use client';

import React from 'react';
import { Card } from '@/components/ui';
import type { NRIFusionResult } from '@/lib/ml';

interface NRIScoreDisplayProps {
  nriResult: NRIFusionResult;
}

const getRiskCategoryInfo = (category: string) => {
  switch (category) {
    case 'low':
      return {
        color: '#10B981',
        bgColor: 'bg-green-500/10',
        borderColor: 'border-green-500/20',
        textColor: 'text-green-400',
        label: 'Low Risk',
        description: 'Low neurological risk detected',
        icon: '✅',
      };
    case 'moderate':
      return {
        color: '#F59E0B',
        bgColor: 'bg-amber-500/10',
        borderColor: 'border-amber-500/20',
        textColor: 'text-amber-400',
        label: 'Moderate Risk',
        description: 'Moderate neurological risk detected',
        icon: '⚠️',
      };
    case 'high':
      return {
        color: '#F97316',
        bgColor: 'bg-orange-500/10',
        borderColor: 'border-orange-500/20',
        textColor: 'text-orange-400',
        label: 'High Risk',
        description: 'High neurological risk detected',
        icon: '🔶',
      };
    case 'critical':
      return {
        color: '#EF4444',
        bgColor: 'bg-red-500/10',
        borderColor: 'border-red-500/20',
        textColor: 'text-red-400',
        label: 'Critical Risk',
        description: 'Critical neurological risk detected',
        icon: '🚨',
      };
    default:
      return {
        color: '#6B7280',
        bgColor: 'bg-neutral-500/10',
        borderColor: 'border-neutral-500/20',
        textColor: 'text-neutral-400',
        label: 'Unknown',
        description: 'Risk level unknown',
        icon: '❓',
      };
  }
};

export const NRIScoreDisplay: React.FC<NRIScoreDisplayProps> = ({ nriResult }) => {
  const riskInfo = getRiskCategoryInfo(nriResult.riskCategory);
  
  // Calculate score position for visual indicator
  const scorePosition = (nriResult.nriScore / 100) * 100;

  return (
    <Card className="p-8 relative overflow-hidden">
      {/* Background gradient effect */}
      <div 
        className="absolute inset-0 opacity-5"
        style={{
          background: `radial-gradient(circle at 50% 50%, ${riskInfo.color} 0%, transparent 70%)`
        }}
      />
      
      <div className="relative z-10">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-text-primary mb-2">
            Neuro-Risk Index (NRI)
          </h2>
          <p className="text-text-secondary">
            Your unified neurological risk assessment score
          </p>
        </div>

        {/* Main Score Display */}
        <div className="flex flex-col lg:flex-row items-center gap-8 mb-8">
          {/* Score Circle */}
          <div className="relative">
            <div className="w-48 h-48 rounded-full border-8 border-neutral-800 flex items-center justify-center relative">
              {/* Progress ring */}
              <svg className="absolute inset-0 w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle
                  cx="50"
                  cy="50"
                  r="42"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="4"
                  className="text-neutral-800"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="42"
                  fill="none"
                  stroke={riskInfo.color}
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={`${scorePosition * 2.64} 264`}
                  className="transition-all duration-1000 ease-out"
                />
              </svg>
              
              {/* Score text */}
              <div className="text-center">
                <div className="text-5xl font-black text-gradient-primary mb-1">
                  {nriResult.nriScore}
                </div>
                <div className="text-sm text-text-muted uppercase tracking-wider">
                  out of 100
                </div>
              </div>
            </div>
          </div>

          {/* Risk Category & Details */}
          <div className="flex-1 space-y-6">
            {/* Risk Category Badge */}
            <div className={`inline-flex items-center px-6 py-3 rounded-full ${riskInfo.bgColor} ${riskInfo.borderColor} border`}>
              <span className="text-2xl mr-3">{riskInfo.icon}</span>
              <div>
                <div className={`text-lg font-semibold ${riskInfo.textColor}`}>
                  {riskInfo.label}
                </div>
                <div className="text-sm text-text-secondary">
                  {riskInfo.description}
                </div>
              </div>
            </div>

            {/* Confidence & Data Quality */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-surface-secondary rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-secondary">Confidence</span>
                  <span className="text-lg font-semibold text-text-primary">
                    {nriResult.confidence}%
                  </span>
                </div>
                <div className="w-full bg-neutral-800 rounded-full h-2">
                  <div 
                    className="bg-primary-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${nriResult.confidence}%` }}
                  />
                </div>
              </div>

              <div className="bg-surface-secondary rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-secondary">Data Complete</span>
                  <span className="text-lg font-semibold text-text-primary">
                    {nriResult.dataCompleteness}%
                  </span>
                </div>
                <div className="w-full bg-neutral-800 rounded-full h-2">
                  <div 
                    className="bg-success h-2 rounded-full transition-all duration-500"
                    style={{ width: `${nriResult.dataCompleteness}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Processing Info */}
            <div className="text-sm text-text-muted">
              <div className="flex items-center space-x-4">
                <span>
                  ⏱️ Processed in {Math.round(nriResult.processingTime)}ms
                </span>
                <span>
                  🧠 AI Model v1.0.0
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Scale Visualization */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-text-primary text-center">
            Risk Scale
          </h3>
          
          <div className="relative">
            {/* Scale bar */}
            <div className="h-4 rounded-full overflow-hidden bg-gradient-to-r from-green-500 via-amber-500 via-orange-500 to-red-500">
              {/* Score indicator */}
              <div 
                className="absolute top-0 w-1 h-4 bg-white shadow-lg transform -translate-x-0.5"
                style={{ left: `${scorePosition}%` }}
              />
            </div>
            
            {/* Scale labels */}
            <div className="flex justify-between mt-2 text-xs text-text-muted">
              <span>0 - Low</span>
              <span>25</span>
              <span>50 - Moderate</span>
              <span>75</span>
              <span>100 - Critical</span>
            </div>
          </div>
        </div>

        {/* Modality Contributions */}
        <div className="mt-8 space-y-4">
          <h3 className="text-lg font-semibold text-text-primary text-center">
            Assessment Contributions
          </h3>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(nriResult.modalityContributions).map(([modality, contribution]) => {
              const percentage = Math.round(contribution * 100);
              const modalityInfo = {
                speech: { icon: '🎤', label: 'Speech', color: 'bg-blue-500' },
                retinal: { icon: '👁️', label: 'Retinal', color: 'bg-purple-500' },
                risk: { icon: '📊', label: 'Risk Factors', color: 'bg-green-500' },
                motor: { icon: '🤲', label: 'Motor', color: 'bg-orange-500' },
              }[modality] || { icon: '❓', label: modality, color: 'bg-gray-500' };

              return (
                <div key={modality} className="bg-surface-secondary rounded-lg p-4 text-center">
                  <div className="text-2xl mb-2">{modalityInfo.icon}</div>
                  <div className="text-sm font-medium text-text-primary mb-1">
                    {modalityInfo.label}
                  </div>
                  <div className="text-lg font-bold text-primary-400">
                    {percentage}%
                  </div>
                  <div className="w-full bg-neutral-800 rounded-full h-1 mt-2">
                    <div 
                      className={`${modalityInfo.color} h-1 rounded-full transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Important Note */}
        <div className="mt-8 bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <svg className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="font-medium text-amber-400 mb-1">Important</h4>
              <p className="text-sm text-text-secondary">
                This assessment is a screening tool and not a diagnostic device. 
                Always consult with qualified healthcare professionals for medical decisions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
