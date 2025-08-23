'use client';

import React from 'react';
import { Card } from '@/components/ui';
import { Mic, Eye, Hand, Brain } from 'lucide-react';

interface ModalityScore {
  name: string;
  score: number;
  confidence: number;
  icon: React.ReactNode;
  color: string;
}

interface ModalityBreakdownProps {
  speechScore?: number;
  retinalScore?: number;
  motorScore?: number;
  cognitiveScore?: number;
  className?: string;
}

export const ModalityBreakdown: React.FC<ModalityBreakdownProps> = ({
  speechScore = 0,
  retinalScore = 0,
  motorScore = 0,
  cognitiveScore = 0,
  className = ''
}) => {
  const modalities: ModalityScore[] = [
    {
      name: 'Speech Analysis',
      score: speechScore,
      confidence: 0.9,
      icon: <Mic className="w-5 h-5" />,
      color: 'text-blue-600'
    },
    {
      name: 'Retinal Analysis',
      score: retinalScore,
      confidence: 0.85,
      icon: <Eye className="w-5 h-5" />,
      color: 'text-teal-600'
    },
    {
      name: 'Motor Assessment',
      score: motorScore,
      confidence: 0.88,
      icon: <Hand className="w-5 h-5" />,
      color: 'text-purple-600'
    },
    {
      name: 'Cognitive Evaluation',
      score: cognitiveScore,
      confidence: 0.92,
      icon: <Brain className="w-5 h-5" />,
      color: 'text-green-600'
    }
  ];

  return (
    <Card className={`p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-slate-900 mb-4">
        Modality Breakdown
      </h3>
      
      <div className="space-y-4">
        {modalities.map((modality, index) => (
          <div key={index} className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`${modality.color}`}>
                {modality.icon}
              </div>
              <div>
                <div className="font-medium text-slate-900">
                  {modality.name}
                </div>
                <div className="text-sm text-slate-500">
                  Confidence: {(modality.confidence * 100).toFixed(0)}%
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className="font-semibold text-slate-900">
                {modality.score.toFixed(1)}
              </div>
              <div className="text-sm text-slate-500">
                /100
              </div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};
