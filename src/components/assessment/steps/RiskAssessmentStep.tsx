'use client';

import React, { useState } from 'react';
import { Button, Card, Input } from '@/components/ui';
import type { RiskAssessmentData } from '@/lib/ml';

interface RiskAssessmentStepProps {
  onComplete: (riskData: RiskAssessmentData) => void;
  onBack: () => void;
  initialData?: RiskAssessmentData;
}

export const RiskAssessmentStep: React.FC<RiskAssessmentStepProps> = ({
  onComplete,
  onBack,
  initialData,
}) => {
  const [formData, setFormData] = useState<Partial<RiskAssessmentData>>(
    initialData || {
      demographics: {
        age: 0,
        sex: 'male',
        ethnicity: '',
        education: 0,
        occupation: '',
        handedness: 'right',
      },
      medicalHistory: {
        hypertension: false,
        diabetes: false,
        heartDisease: false,
        stroke: false,
        cholesterol: 0,
        headInjury: false,
        seizures: false,
        migraines: false,
        sleepDisorders: false,
        depression: false,
        anxiety: false,
        cognitiveComplaints: false,
        thyroidDisease: false,
        kidneyDisease: false,
        liverDisease: false,
        autoimmune: false,
        medications: [],
        supplements: [],
      },
      familyHistory: {
        alzheimers: false,
        parkinsons: false,
        huntingtons: false,
        stroke: false,
        dementia: false,
        depression: false,
        diabetes: false,
        heartDisease: false,
        apoeE4: null,
        familyHistoryAge: 0,
      },
      lifestyle: {
        exerciseFrequency: 0,
        exerciseIntensity: 'moderate',
        dietQuality: 5,
        mediterraneanDiet: false,
        alcohol: 0,
        smoking: 'never',
        cognitiveActivity: 5,
        socialEngagement: 5,
        sleepQuality: 5,
        sleepDuration: 8,
        stressLevel: 5,
        stressManagement: false,
        airPollution: 5,
        occupationalExposure: false,
      },
      cognitive: {
        memoryComplaints: false,
        concentrationIssues: false,
        languageProblems: false,
        executiveIssues: false,
        spatialIssues: false,
        dailyActivities: 10,
        instrumentalActivities: 10,
        moodChanges: false,
        apathy: false,
        irritability: false,
      },
    }
  );

  const [currentSection, setCurrentSection] = useState(0);

  const sections = [
    { title: 'Demographics', key: 'demographics' },
    { title: 'Medical History', key: 'medicalHistory' },
    { title: 'Family History', key: 'familyHistory' },
    { title: 'Lifestyle', key: 'lifestyle' },
    { title: 'Cognitive Assessment', key: 'cognitive' },
  ];

  const handleComplete = () => {
    onComplete(formData as RiskAssessmentData);
  };

  const renderDemographics = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Age
          </label>
          <Input
            type="number"
            value={formData.demographics?.age || ''}
            onChange={(e) =>
              setFormData({
                ...formData,
                demographics: {
                  ...formData.demographics!,
                  age: parseInt(e.target.value) || 0,
                },
              })
            }
            placeholder="Enter your age"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-primary mb-2">
            Sex
          </label>
          <select
            value={formData.demographics?.sex || 'male'}
            onChange={(e) =>
              setFormData({
                ...formData,
                demographics: {
                  ...formData.demographics!,
                  sex: e.target.value as 'male' | 'female' | 'other',
                },
              })
            }
            className="w-full px-4 py-3 bg-surface-secondary border border-neutral-700 rounded-lg text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Ethnicity
        </label>
        <Input
          value={formData.demographics?.ethnicity || ''}
          onChange={(e) =>
            setFormData({
              ...formData,
              demographics: {
                ...formData.demographics!,
                ethnicity: e.target.value,
              },
            })
          }
          placeholder="Enter your ethnicity"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Years of Education
        </label>
        <Input
          type="number"
          value={formData.demographics?.education || ''}
          onChange={(e) =>
            setFormData({
              ...formData,
              demographics: {
                ...formData.demographics!,
                education: parseInt(e.target.value) || 0,
              },
            })
          }
          placeholder="Years of formal education"
        />
      </div>
    </div>
  );

  const renderMedicalHistory = () => (
    <div className="space-y-6">
      <h4 className="text-lg font-semibold text-text-primary">
        Medical Conditions
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { key: 'hypertension', label: 'High Blood Pressure' },
          { key: 'diabetes', label: 'Diabetes' },
          { key: 'heartDisease', label: 'Heart Disease' },
          { key: 'stroke', label: 'Previous Stroke' },
          { key: 'headInjury', label: 'Head Injury' },
          { key: 'depression', label: 'Depression' },
        ].map(({ key, label }) => (
          <label key={key} className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={formData.medicalHistory?.[key as keyof typeof formData.medicalHistory] as boolean || false}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  medicalHistory: {
                    ...formData.medicalHistory!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="w-4 h-4 text-primary-500 bg-surface-secondary border-neutral-600 rounded focus:ring-primary-500"
            />
            <span className="text-text-secondary">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderFamilyHistory = () => (
    <div className="space-y-6">
      <h4 className="text-lg font-semibold text-text-primary">
        Family History of Neurological Conditions
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { key: 'alzheimers', label: "Alzheimer's Disease" },
          { key: 'parkinsons', label: "Parkinson's Disease" },
          { key: 'dementia', label: 'Dementia' },
          { key: 'stroke', label: 'Stroke' },
        ].map(({ key, label }) => (
          <label key={key} className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={formData.familyHistory?.[key as keyof typeof formData.familyHistory] as boolean || false}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  familyHistory: {
                    ...formData.familyHistory!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="w-4 h-4 text-primary-500 bg-surface-secondary border-neutral-600 rounded focus:ring-primary-500"
            />
            <span className="text-text-secondary">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderLifestyle = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Exercise Frequency (days per week)
        </label>
        <Input
          type="number"
          min="0"
          max="7"
          value={formData.lifestyle?.exerciseFrequency || ''}
          onChange={(e) =>
            setFormData({
              ...formData,
              lifestyle: {
                ...formData.lifestyle!,
                exerciseFrequency: parseInt(e.target.value) || 0,
              },
            })
          }
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Diet Quality (1-10 scale)
        </label>
        <Input
          type="number"
          min="1"
          max="10"
          value={formData.lifestyle?.dietQuality || ''}
          onChange={(e) =>
            setFormData({
              ...formData,
              lifestyle: {
                ...formData.lifestyle!,
                dietQuality: parseInt(e.target.value) || 5,
              },
            })
          }
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Smoking Status
        </label>
        <select
          value={formData.lifestyle?.smoking || 'never'}
          onChange={(e) =>
            setFormData({
              ...formData,
              lifestyle: {
                ...formData.lifestyle!,
                smoking: e.target.value as 'never' | 'former' | 'current',
              },
            })
          }
          className="w-full px-4 py-3 bg-surface-secondary border border-neutral-700 rounded-lg text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          <option value="never">Never</option>
          <option value="former">Former</option>
          <option value="current">Current</option>
        </select>
      </div>
    </div>
  );

  const renderCognitive = () => (
    <div className="space-y-6">
      <h4 className="text-lg font-semibold text-text-primary">
        Cognitive Concerns
      </h4>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { key: 'memoryComplaints', label: 'Memory Problems' },
          { key: 'concentrationIssues', label: 'Concentration Issues' },
          { key: 'languageProblems', label: 'Language Difficulties' },
          { key: 'moodChanges', label: 'Mood Changes' },
        ].map(({ key, label }) => (
          <label key={key} className="flex items-center space-x-3 cursor-pointer">
            <input
              type="checkbox"
              checked={formData.cognitive?.[key as keyof typeof formData.cognitive] as boolean || false}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  cognitive: {
                    ...formData.cognitive!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="w-4 h-4 text-primary-500 bg-surface-secondary border-neutral-600 rounded focus:ring-primary-500"
            />
            <span className="text-text-secondary">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderCurrentSection = () => {
    switch (currentSection) {
      case 0: return renderDemographics();
      case 1: return renderMedicalHistory();
      case 2: return renderFamilyHistory();
      case 3: return renderLifestyle();
      case 4: return renderCognitive();
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 rounded-2xl flex items-center justify-center mx-auto">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-text-primary">
              Risk Assessment
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Complete health and lifestyle questionnaire
            </p>
          </div>

          {/* Progress */}
          <div className="flex justify-center">
            <div className="flex space-x-2">
              {sections.map((_, index) => (
                <div
                  key={index}
                  className={`w-3 h-3 rounded-full ${
                    index === currentSection
                      ? 'bg-primary-500'
                      : index < currentSection
                      ? 'bg-success'
                      : 'bg-neutral-600'
                  }`}
                />
              ))}
            </div>
          </div>

          {/* Form */}
          <Card className="p-8">
            <h2 className="text-2xl font-semibold text-text-primary mb-6">
              {sections[currentSection].title}
            </h2>
            {renderCurrentSection()}
          </Card>

          {/* Navigation */}
          <div className="flex justify-between">
            <Button
              variant="secondary"
              onClick={currentSection === 0 ? onBack : () => setCurrentSection(currentSection - 1)}
            >
              {currentSection === 0 ? 'Back' : 'Previous'}
            </Button>
            
            <div className="space-x-4">
              {currentSection < sections.length - 1 ? (
                <Button onClick={() => setCurrentSection(currentSection + 1)}>
                  Next Section
                </Button>
              ) : (
                <Button onClick={handleComplete}>
                  Complete Assessment
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
