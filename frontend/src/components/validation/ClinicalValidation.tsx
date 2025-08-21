'use client';

import React from 'react';
import { Card, Progress } from '@/components/ui';

interface ClinicalValidationProps {
  data: {
    sensitivity: number;
    specificity: number;
    ppv: number;
    npv: number;
    auc: number;
    studyParticipants: number;
    validationPeriod: string;
    clinicalSites: number;
  };
}

export const ClinicalValidation: React.FC<ClinicalValidationProps> = ({ data }) => {
  const clinicalMetrics = [
    { name: 'Sensitivity', value: data.sensitivity, description: 'True positive rate', target: 85 },
    { name: 'Specificity', value: data.specificity, description: 'True negative rate', target: 85 },
    { name: 'PPV', value: data.ppv, description: 'Positive predictive value', target: 80 },
    { name: 'NPV', value: data.npv, description: 'Negative predictive value', target: 85 },
  ];

  const studyDetails = [
    { label: 'Study Participants', value: data.studyParticipants.toLocaleString() },
    { label: 'Validation Period', value: data.validationPeriod },
    { label: 'Clinical Sites', value: data.clinicalSites.toString() },
    { label: 'AUC Score', value: data.auc.toString() },
  ];

  return (
    <div className="space-y-8">
      {/* Clinical Performance Metrics */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Clinical Performance Metrics
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {clinicalMetrics.map((metric) => (
            <div key={metric.name} className="space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <h4 className="text-lg font-semibold text-text-primary">{metric.name}</h4>
                  <p className="text-sm text-text-secondary">{metric.description}</p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-primary-400">{metric.value}%</div>
                  <div className="text-xs text-text-muted">Target: {metric.target}%</div>
                </div>
              </div>
              <Progress 
                value={metric.value} 
                className="h-3"
                showLabel={false}
              />
            </div>
          ))}
        </div>

        <div className="mt-8 p-6 bg-primary-500/10 border border-primary-500/20 rounded-lg">
          <h4 className="text-lg font-semibold text-primary-400 mb-4">
            AUC Score: {data.auc}
          </h4>
          <p className="text-sm text-text-secondary">
            Area Under the Curve (AUC) of {data.auc} indicates excellent discriminative ability. 
            Values above 0.9 are considered outstanding for clinical diagnostic tools.
          </p>
        </div>
      </Card>

      {/* Study Details */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Clinical Study Details
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {studyDetails.map((detail) => (
            <div key={detail.label} className="text-center p-4 bg-surface-secondary rounded-lg">
              <div className="text-2xl font-bold text-primary-400 mb-2">
                {detail.value}
              </div>
              <div className="text-sm text-text-secondary">
                {detail.label}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 space-y-6">
          <div>
            <h4 className="text-lg font-semibold text-text-primary mb-3">
              Study Methodology
            </h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h5 className="font-medium text-text-primary">Participant Demographics</h5>
                <ul className="text-sm text-text-secondary space-y-1">
                  <li>• Age range: 45-85 years</li>
                  <li>• Gender distribution: 52% female, 48% male</li>
                  <li>• Diverse ethnic backgrounds</li>
                  <li>• Various education levels</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h5 className="font-medium text-text-primary">Study Design</h5>
                <ul className="text-sm text-text-secondary space-y-1">
                  <li>• Prospective cohort study</li>
                  <li>• Multi-center validation</li>
                  <li>• Blinded assessment protocol</li>
                  <li>• Gold standard comparison</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-text-primary mb-3">
              Clinical Endpoints
            </h4>
            <div className="bg-surface-secondary rounded-lg p-4">
              <p className="text-sm text-text-secondary leading-relaxed">
                Primary endpoint: Detection of neurological risk factors with clinical correlation. 
                Secondary endpoints: Time to assessment completion, user experience metrics, 
                and healthcare provider satisfaction scores. All endpoints were pre-specified 
                and statistically powered.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Regulatory & Compliance */}
      <Card className="p-8">
        <h3 className="text-2xl font-semibold text-text-primary mb-6">
          Regulatory Compliance
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-success/10 border border-success/20 rounded-lg">
            <div className="text-4xl mb-3">🏥</div>
            <h4 className="font-semibold text-success mb-2">FDA Guidelines</h4>
            <p className="text-sm text-text-secondary">
              Developed following FDA guidance for AI/ML-based medical devices
            </p>
          </div>

          <div className="text-center p-6 bg-success/10 border border-success/20 rounded-lg">
            <div className="text-4xl mb-3">🔒</div>
            <h4 className="font-semibold text-success mb-2">HIPAA Compliant</h4>
            <p className="text-sm text-text-secondary">
              Full compliance with healthcare data protection regulations
            </p>
          </div>

          <div className="text-center p-6 bg-success/10 border border-success/20 rounded-lg">
            <div className="text-4xl mb-3">🌍</div>
            <h4 className="font-semibold text-success mb-2">ISO 13485</h4>
            <p className="text-sm text-text-secondary">
              Quality management system for medical devices
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};
