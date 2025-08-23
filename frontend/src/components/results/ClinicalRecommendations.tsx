'use client';

import React from 'react';
import { Card } from '@/components/ui';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';

interface Recommendation {
  type: 'warning' | 'info' | 'success';
  title: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
}

interface ClinicalRecommendationsProps {
  nriScore: number;
  riskLevel: 'low' | 'moderate' | 'high' | 'critical';
  className?: string;
}

export const ClinicalRecommendations: React.FC<ClinicalRecommendationsProps> = ({
  nriScore,
  riskLevel,
  className = ''
}) => {
  const getRecommendations = (): Recommendation[] => {
    const recommendations: Recommendation[] = [];

    if (riskLevel === 'critical' || nriScore > 80) {
      recommendations.push({
        type: 'warning',
        title: 'Immediate Clinical Evaluation Recommended',
        description: 'High neurological risk detected. Consider immediate consultation with a neurologist.',
        priority: 'high'
      });
    } else if (riskLevel === 'high' || nriScore > 60) {
      recommendations.push({
        type: 'warning',
        title: 'Follow-up Assessment Recommended',
        description: 'Elevated risk indicators suggest follow-up within 3-6 months.',
        priority: 'medium'
      });
    } else if (riskLevel === 'moderate' || nriScore > 30) {
      recommendations.push({
        type: 'info',
        title: 'Routine Monitoring Suggested',
        description: 'Consider annual neurological screening and lifestyle modifications.',
        priority: 'medium'
      });
    } else {
      recommendations.push({
        type: 'success',
        title: 'Low Risk Profile',
        description: 'Continue regular health maintenance and periodic screening.',
        priority: 'low'
      });
    }

    // Add general recommendations
    recommendations.push({
      type: 'info',
      title: 'Lifestyle Recommendations',
      description: 'Maintain regular exercise, healthy diet, and cognitive engagement activities.',
      priority: 'low'
    });

    return recommendations;
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'warning': return <AlertTriangle className="w-5 h-5 text-orange-600" />;
      case 'success': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'info': return <Info className="w-5 h-5 text-blue-600" />;
      default: return <Info className="w-5 h-5 text-gray-600" />;
    }
  };

  const recommendations = getRecommendations();

  return (
    <Card className={`p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-slate-900 mb-4">
        Clinical Recommendations
      </h3>
      
      <div className="space-y-4">
        {recommendations.map((rec, index) => (
          <div key={index} className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-0.5">
              {getIcon(rec.type)}
            </div>
            <div className="flex-1">
              <div className="font-medium text-slate-900 mb-1">
                {rec.title}
              </div>
              <div className="text-sm text-slate-600">
                {rec.description}
              </div>
              <div className="mt-2">
                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                  rec.priority === 'high' ? 'bg-red-100 text-red-800' :
                  rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {rec.priority.charAt(0).toUpperCase() + rec.priority.slice(1)} Priority
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <div className="text-sm text-blue-800">
          <strong>Disclaimer:</strong> These recommendations are for informational purposes only and should not replace professional medical advice. Always consult with qualified healthcare providers for proper diagnosis and treatment.
        </div>
      </div>
    </Card>
  );
};
