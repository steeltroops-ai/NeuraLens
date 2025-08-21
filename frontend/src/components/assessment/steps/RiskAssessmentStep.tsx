'use client';

import React, { useState } from 'react';
import { Button, Input } from '@/components/ui';
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
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="mb-2 block text-sm font-medium text-text-primary">
            Age
          </label>
          <Input
            type="number"
            value={formData.demographics?.age?.toString() || ''}
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
          <label className="mb-2 block text-sm font-medium text-text-primary">
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
            className="w-full rounded-lg border border-neutral-700 bg-surface-secondary px-4 py-3 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
      </div>

      <div>
        <label className="mb-2 block text-sm font-medium text-text-primary">
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
        <label className="mb-2 block text-sm font-medium text-text-primary">
          Years of Education
        </label>
        <Input
          type="number"
          value={formData.demographics?.education?.toString() || ''}
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
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[
          { key: 'hypertension', label: 'High Blood Pressure' },
          { key: 'diabetes', label: 'Diabetes' },
          { key: 'heartDisease', label: 'Heart Disease' },
          { key: 'stroke', label: 'Previous Stroke' },
          { key: 'headInjury', label: 'Head Injury' },
          { key: 'depression', label: 'Depression' },
        ].map(({ key, label }) => (
          <label
            key={key}
            className="flex cursor-pointer items-center space-x-3"
          >
            <input
              type="checkbox"
              checked={
                (formData.medicalHistory?.[
                  key as keyof typeof formData.medicalHistory
                ] as boolean) || false
              }
              onChange={(e) =>
                setFormData({
                  ...formData,
                  medicalHistory: {
                    ...formData.medicalHistory!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-neutral-600 bg-surface-secondary text-primary-500 focus:ring-primary-500"
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
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[
          { key: 'alzheimers', label: "Alzheimer's Disease" },
          { key: 'parkinsons', label: "Parkinson's Disease" },
          { key: 'dementia', label: 'Dementia' },
          { key: 'stroke', label: 'Stroke' },
        ].map(({ key, label }) => (
          <label
            key={key}
            className="flex cursor-pointer items-center space-x-3"
          >
            <input
              type="checkbox"
              checked={
                (formData.familyHistory?.[
                  key as keyof typeof formData.familyHistory
                ] as boolean) || false
              }
              onChange={(e) =>
                setFormData({
                  ...formData,
                  familyHistory: {
                    ...formData.familyHistory!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-neutral-600 bg-surface-secondary text-primary-500 focus:ring-primary-500"
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
        <label className="mb-2 block text-sm font-medium text-text-primary">
          Exercise Frequency (days per week)
        </label>
        <Input
          type="number"
          min="0"
          max="7"
          value={formData.lifestyle?.exerciseFrequency?.toString() || ''}
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
        <label className="mb-2 block text-sm font-medium text-text-primary">
          Diet Quality (1-10 scale)
        </label>
        <Input
          type="number"
          min="1"
          max="10"
          value={formData.lifestyle?.dietQuality?.toString() || ''}
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
        <label className="mb-2 block text-sm font-medium text-text-primary">
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
          className="w-full rounded-lg border border-neutral-700 bg-surface-secondary px-4 py-3 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500"
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
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {[
          { key: 'memoryComplaints', label: 'Memory Problems' },
          { key: 'concentrationIssues', label: 'Concentration Issues' },
          { key: 'languageProblems', label: 'Language Difficulties' },
          { key: 'moodChanges', label: 'Mood Changes' },
        ].map(({ key, label }) => (
          <label
            key={key}
            className="flex cursor-pointer items-center space-x-3"
          >
            <input
              type="checkbox"
              checked={
                (formData.cognitive?.[
                  key as keyof typeof formData.cognitive
                ] as boolean) || false
              }
              onChange={(e) =>
                setFormData({
                  ...formData,
                  cognitive: {
                    ...formData.cognitive!,
                    [key]: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-neutral-600 bg-surface-secondary text-primary-500 focus:ring-primary-500"
            />
            <span className="text-text-secondary">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderCurrentSection = () => {
    switch (currentSection) {
      case 0:
        return renderDemographics();
      case 1:
        return renderMedicalHistory();
      case 2:
        return renderFamilyHistory();
      case 3:
        return renderLifestyle();
      case 4:
        return renderCognitive();
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-4xl space-y-12">
          {/* Apple-Style Header */}
          <div className="animate-fade-in space-y-6 text-center">
            <div className="rounded-apple-xl from-warning-500 to-warning-600 mx-auto flex h-20 w-20 items-center justify-center bg-gradient-to-br shadow-warning">
              <svg
                className="h-10 w-10 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <h1 className="text-4xl font-bold tracking-tight text-text-primary">
              Health Questionnaire
            </h1>
            <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary">
              Answer simple questions about your health and lifestyle
            </p>
          </div>

          {/* Apple-Style Progress */}
          <div className="flex justify-center animate-scale-in">
            <div className="flex items-center space-x-3">
              {sections.map((section, index) => (
                <div key={index} className="flex items-center">
                  <div
                    className={`h-4 w-4 rounded-full transition-all duration-300 ${
                      index === currentSection
                        ? 'bg-warning-500 scale-125 shadow-lg'
                        : index < currentSection
                          ? 'bg-success-500 scale-110'
                          : 'bg-gray-300'
                    }`}
                  />
                  {index < sections.length - 1 && (
                    <div
                      className={`w-8 h-0.5 mx-2 transition-colors duration-300 ${
                        index < currentSection ? 'bg-success-500' : 'bg-gray-300'
                      }`}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Apple-Style Form */}
          <div className="card-apple p-10 animate-slide-up">
            <h2 className="mb-8 text-3xl font-semibold text-text-primary text-center">
              {sections[currentSection]?.title || 'Assessment'}
            </h2>
            {renderCurrentSection()}
          </div>

          {/* Apple-Style Navigation */}
          <div className="flex justify-between items-center animate-fade-in">
            <Button
              variant="secondary"
              size="lg"
              onClick={
                currentSection === 0
                  ? onBack
                  : () => setCurrentSection(currentSection - 1)
              }
              className="px-8"
            >
              {currentSection === 0 ? 'Back' : 'Previous'}
            </Button>

            <div className="text-center">
              <span className="text-sm text-text-secondary">
                {currentSection + 1} of {sections.length}
              </span>
            </div>

            <div>
              {currentSection < sections.length - 1 ? (
                <Button
                  onClick={() => setCurrentSection(currentSection + 1)}
                  size="lg"
                  className="px-8 shadow-warning hover:shadow-warning-hover"
                >
                  Next Section
                  <svg className="ml-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Button>
              ) : (
                <Button
                  onClick={handleComplete}
                  size="lg"
                  className="px-8 shadow-warning hover:shadow-warning-hover"
                >
                  Complete Assessment
                  <svg className="ml-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
