/**
 * Comprehensive Results Dashboard Page
 * Complete dashboard with real-time updates, clinical recommendations, and export functionality
 */

import React, { useState, useEffect, useCallback } from 'react';
import { ResultsDashboard } from '@/components/dashboard/ResultsDashboard';
import { ClinicalRecommendations } from '@/components/dashboard/ClinicalRecommendations';
import { ExportInterface } from '@/components/dashboard/ExportInterface';
import { ErrorBoundary, DefaultErrorFallback } from '@/components/ui/ErrorDisplay';
import { useScreenReader, useSkipLink } from '@/hooks/useAccessibility';
import { ClinicalRecommendationEngine } from '@/lib/clinical/recommendations';
import { AssessmentResults } from '@/lib/assessment/workflow';
import {
  BarChart3,
  FileText,
  Stethoscope,
  Calendar,
  Bell,
  Settings,
  RefreshCw,
} from 'lucide-react';

// Mock assessment results for demonstration
const MOCK_RESULTS: AssessmentResults = {
  sessionId: 'demo-session-2024',
  completionTime: new Date().toISOString(),
  totalProcessingTime: 45000,
  overallRiskCategory: 'moderate',
  metadata: {
    startTime: new Date(Date.now() - 45000).toISOString(),
    endTime: new Date().toISOString(),
    stepsCompleted: [
      'upload',
      'validation',
      'speech_processing',
      'retinal_processing',
      'motor_processing',
      'cognitive_processing',
      'nri_fusion',
      'results',
    ],
    errors: [],
  },
  nriResult: {
    session_id: 'demo_comprehensive',
    nri_score: 0.45,
    confidence: 0.82,
    risk_category: 'moderate',
    uncertainty: 0.18,
    consistency_score: 0.85,
    modality_contributions: [
      { modality: 'speech', weight: 0.3, confidence: 0.8, risk_score: 0.4 },
      { modality: 'retinal', weight: 0.25, confidence: 0.9, risk_score: 0.3 },
      { modality: 'motor', weight: 0.25, confidence: 0.85, risk_score: 0.5 },
      { modality: 'cognitive', weight: 0.2, confidence: 0.75, risk_score: 0.35 },
    ],
    processing_time: 5000,
    timestamp: new Date().toISOString(),
    recommendations: [
      'Consider follow-up assessment in 6 months',
      'Implement regular exercise routine',
      'Monitor symptoms closely',
    ],
    follow_up_actions: ['Schedule follow-up appointment'],
  },
  speechResult: {
    session_id: 'demo_comprehensive',
    risk_score: 0.38,
    confidence: 0.85,
    processing_time: 12000,
    timestamp: new Date().toISOString(),
    recommendations: ['Monitor speech patterns', 'Consider speech therapy if needed'],
    quality_score: 0.92,
    biomarkers: {
      fluency_score: 0.78,
      voice_tremor: 0.25,
      articulation_clarity: 0.88,
      speaking_rate: 0.82,
      pause_frequency: 0.65,
      pause_pattern: 0.75,
      prosody_variation: 0.68,
    },
    file_info: {
      duration: 30000,
      sample_rate: 44100,
      channels: 1,
    },
  },
  retinalResult: {
    session_id: 'demo_comprehensive',
    risk_score: 0.42,
    confidence: 0.79,
    processing_time: 18000,
    timestamp: new Date().toISOString(),
    recommendations: ['Monitor retinal health', 'Regular eye examinations recommended'],
    quality_score: 0.89,
    detected_conditions: ['Normal findings', 'No significant abnormalities'],
    biomarkers: {
      vessel_density: 0.72,
      cup_disc_ratio: 0.35,
      av_ratio: 0.68,
      vessel_tortuosity: 0.45,
      exudate_area: 0.02,
      microaneurysm_count: 1,
      hemorrhage_count: 0,
    },
    image_info: {
      width: 1024,
      height: 1024,
      format: 'JPEG',
      file_size: 2048000,
    },
  },
  motorResult: {
    session_id: 'demo_comprehensive',
    risk_score: 0.51,
    confidence: 0.87,
    processing_time: 8000,
    timestamp: new Date().toISOString(),
    recommendations: ['Monitor motor function', 'Regular physical therapy recommended'],
    assessment_type: 'tremor',
    movement_quality: 'fair',
    biomarkers: {
      coordination_index: 0.65,
      tremor_severity: 0.38,
      fatigue_index: 0.42,
      movement_frequency: 0.58,
      amplitude_variation: 0.35,
      asymmetry_score: 0.28,
    },
  },
  cognitiveResult: {
    session_id: 'demo_comprehensive',
    risk_score: 0.35,
    confidence: 0.81,
    processing_time: 7000,
    timestamp: new Date().toISOString(),
    recommendations: ['Monitor cognitive function', 'Consider cognitive training exercises'],
    overall_score: 0.72,
    test_battery: ['memory', 'attention', 'executive'],
    domain_scores: {
      memory: 0.68,
      attention: 0.75,
      executive: 0.71,
      processing: 0.78,
    },
    biomarkers: {
      memory_score: 0.68,
      attention_score: 0.75,
      processing_speed: 0.78,
      executive_score: 0.71,
      language_score: 0.73,
      cognitive_flexibility: 0.69,
    },
  },
};

export default function ComprehensiveDashboardPage() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'recommendations' | 'export'>(
    'dashboard',
  );
  const [isRealTime, setIsRealTime] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [notifications, setNotifications] = useState<string[]>([]);

  const { announce } = useScreenReader();
  const { SkipLink, targetRef } = useSkipLink();

  // Generate clinical recommendations
  const recommendations = ClinicalRecommendationEngine.generateRecommendations(MOCK_RESULTS);

  // Simulate real-time updates
  useEffect(() => {
    if (!isRealTime) return;

    const interval = setInterval(() => {
      setLastUpdated(new Date());

      // Simulate occasional notifications
      if (Math.random() > 0.8) {
        const newNotification = `System update: ${new Date().toLocaleTimeString()}`;
        setNotifications(prev => [newNotification, ...prev.slice(0, 4)]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isRealTime]);

  // Handle tab change
  const handleTabChange = useCallback(
    (tab: typeof activeTab) => {
      setActiveTab(tab);
      announce(`Switched to ${tab} view`);
    },
    [announce],
  );

  // Handle export completion
  const handleExportComplete = useCallback(
    (result: any) => {
      announce('Export completed successfully');
      setNotifications(prev => [`Export completed: ${result.fileName}`, ...prev.slice(0, 4)]);
    },
    [announce],
  );

  // Handle follow-up scheduling
  const handleScheduleFollowUp = useCallback(() => {
    announce('Opening follow-up scheduling');
    // In a real app, this would open a scheduling interface
    alert('Follow-up scheduling would open here');
  }, [announce]);

  // Handle recommendation completion
  const handleMarkRecommendationComplete = useCallback(
    (recommendationId: string) => {
      announce('Recommendation marked as complete');
      setNotifications(prev => [
        `Recommendation completed: ${recommendationId}`,
        ...prev.slice(0, 4),
      ]);
    },
    [announce],
  );

  // Handle sharing
  const handleShare = useCallback(() => {
    announce('Opening sharing options');
    // In a real app, this would open sharing interface
    alert('Sharing options would open here');
  }, [announce]);

  return (
    <ErrorBoundary fallback={DefaultErrorFallback} level='page' name='ComprehensiveDashboard'>
      <div className='min-h-screen bg-gray-50'>
        {/* Skip link */}
        <SkipLink href='#main-content'>Skip to main content</SkipLink>

        {/* Header */}
        <header className='border-b border-gray-200 bg-white shadow-sm'>
          <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
            <div className='flex h-16 items-center justify-between'>
              <div className='flex items-center gap-4'>
                <h1 className='text-xl font-semibold text-gray-900'>NeuraLens Dashboard</h1>

                <div className='flex items-center gap-2 text-sm text-gray-600'>
                  <div
                    className={`h-2 w-2 rounded-full ${isRealTime ? 'bg-green-500' : 'bg-gray-400'}`}
                  />
                  <span>{isRealTime ? 'Live' : 'Static'}</span>
                  <span>•</span>
                  <span>Updated: {lastUpdated.toLocaleTimeString()}</span>
                </div>
              </div>

              <div className='flex items-center gap-3'>
                {/* Notifications */}
                {notifications.length > 0 && (
                  <div className='relative'>
                    <button
                      className='rounded-lg p-2 text-gray-600 hover:text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                      aria-label={`${notifications.length} notifications`}
                    >
                      <Bell className='h-5 w-5' />
                      <span className='absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs text-white'>
                        {notifications.length}
                      </span>
                    </button>
                  </div>
                )}

                {/* Real-time toggle */}
                <button
                  onClick={() => {
                    setIsRealTime(!isRealTime);
                    announce(
                      isRealTime ? 'Real-time updates disabled' : 'Real-time updates enabled',
                    );
                  }}
                  className='flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                  aria-pressed={isRealTime}
                >
                  <RefreshCw className={`h-4 w-4 ${isRealTime ? 'animate-spin' : ''}`} />
                  {isRealTime ? 'Live' : 'Static'}
                </button>

                {/* Settings */}
                <button
                  className='rounded-lg p-2 text-gray-600 hover:text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                  aria-label='Dashboard settings'
                >
                  <Settings className='h-5 w-5' />
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav
          className='border-b border-gray-200 bg-white'
          role='tablist'
          aria-label='Dashboard sections'
        >
          <div className='mx-auto max-w-7xl px-4 sm:px-6 lg:px-8'>
            <div className='flex space-x-8'>
              {[
                { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
                { id: 'recommendations', label: 'Clinical Recommendations', icon: Stethoscope },
                { id: 'export', label: 'Export & Share', icon: FileText },
              ].map(tab => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;

                return (
                  <button
                    key={tab.id}
                    onClick={() => handleTabChange(tab.id as typeof activeTab)}
                    role='tab'
                    aria-selected={isActive}
                    aria-controls={`${tab.id}-panel`}
                    className={`flex items-center gap-2 border-b-2 px-1 py-4 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                      isActive
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                    } `}
                  >
                    <Icon className='h-4 w-4' aria-hidden='true' />
                    {tab.label}
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main
          id='main-content'
          ref={targetRef}
          tabIndex={-1}
          className='mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8'
        >
          {/* Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <div id='dashboard-panel' role='tabpanel' aria-labelledby='dashboard-tab'>
              <ResultsDashboard
                results={MOCK_RESULTS}
                isRealTime={isRealTime}
                onExport={format => {
                  announce(`Exporting as ${format}`);
                  // Handle quick export
                }}
                onShare={handleShare}
                onScheduleFollowUp={handleScheduleFollowUp}
              />
            </div>
          )}

          {/* Recommendations Tab */}
          {activeTab === 'recommendations' && (
            <div id='recommendations-panel' role='tabpanel' aria-labelledby='recommendations-tab'>
              <ClinicalRecommendations
                results={MOCK_RESULTS}
                audience='both'
                onScheduleFollowUp={handleScheduleFollowUp}
                onMarkComplete={handleMarkRecommendationComplete}
              />
            </div>
          )}

          {/* Export Tab */}
          {activeTab === 'export' && (
            <div id='export-panel' role='tabpanel' aria-labelledby='export-tab'>
              <ExportInterface
                results={MOCK_RESULTS}
                recommendations={recommendations}
                onExportComplete={handleExportComplete}
              />
            </div>
          )}
        </main>

        {/* Notifications Panel */}
        {notifications.length > 0 && (
          <div
            className='fixed bottom-4 right-4 w-80 rounded-lg border border-gray-200 bg-white p-4 shadow-lg'
            role='log'
            aria-live='polite'
            aria-label='System notifications'
          >
            <div className='mb-3 flex items-center justify-between'>
              <h3 className='font-medium text-gray-900'>Recent Activity</h3>
              <button
                onClick={() => {
                  setNotifications([]);
                  announce('Notifications cleared');
                }}
                className='rounded text-sm text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
              >
                Clear all
              </button>
            </div>

            <div className='max-h-40 space-y-2 overflow-y-auto'>
              {notifications.map((notification, index) => (
                <div key={index} className='rounded bg-gray-50 p-2 text-sm text-gray-600'>
                  {notification}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
