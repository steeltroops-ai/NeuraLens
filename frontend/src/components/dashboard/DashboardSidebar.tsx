'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Home,
  Mic, 
  Eye, 
  Hand, 
  Brain,
  Zap,
  TrendingUp,
  Activity,
  Clock,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';

import type { AssessmentType } from '@/app/dashboard/page';

interface DashboardSidebarProps {
  activeAssessment: AssessmentType;
  onAssessmentChange: (assessment: AssessmentType) => void;
  systemStatus: 'healthy' | 'warning' | 'error';
  performanceMetrics: {
    speechLatency: number;
    retinalLatency: number;
    motorLatency: number;
    cognitiveLatency: number;
    nriLatency: number;
    overallAccuracy: number;
  };
}

interface NavigationItem {
  id: AssessmentType;
  label: string;
  description: string;
  icon: React.ReactNode;
  latency?: number;
  accuracy?: number;
  category: 'overview' | 'individual' | 'advanced';
  conditions?: string[];
}

export default function DashboardSidebar({ 
  activeAssessment, 
  onAssessmentChange, 
  systemStatus,
  performanceMetrics 
}: DashboardSidebarProps) {
  
  const navigationItems: NavigationItem[] = [
    // Overview
    {
      id: 'overview',
      label: 'Dashboard Overview',
      description: 'System status and performance metrics',
      icon: <Home className="h-5 w-5" />,
      category: 'overview'
    },
    
    // Individual Assessments
    {
      id: 'speech',
      label: 'Speech Analysis',
      description: 'Voice patterns & speech biomarkers',
      icon: <Mic className="h-5 w-5" />,
      latency: performanceMetrics.speechLatency,
      accuracy: 95,
      category: 'individual',
      conditions: ['Parkinson\'s Disease', 'Alzheimer\'s', 'Stroke Recovery', 'Speech Disorders']
    },
    {
      id: 'retinal',
      label: 'Retinal Imaging',
      description: 'Fundus analysis & vascular health',
      icon: <Eye className="h-5 w-5" />,
      latency: performanceMetrics.retinalLatency,
      accuracy: 89,
      category: 'individual',
      conditions: ['Diabetic Retinopathy', 'Glaucoma', 'Hypertensive Retinopathy', 'Stroke Risk']
    },
    {
      id: 'motor',
      label: 'Motor Function',
      description: 'Movement patterns & coordination',
      icon: <Hand className="h-5 w-5" />,
      latency: performanceMetrics.motorLatency,
      accuracy: 92,
      category: 'individual',
      conditions: ['Parkinson\'s Disease', 'Essential Tremor', 'Ataxia', 'Motor Neuron Disease']
    },
    {
      id: 'cognitive',
      label: 'Cognitive Testing',
      description: 'Memory, attention & executive function',
      icon: <Brain className="h-5 w-5" />,
      latency: performanceMetrics.cognitiveLatency,
      accuracy: 94,
      category: 'individual',
      conditions: ['Alzheimer\'s Disease', 'Mild Cognitive Impairment', 'ADHD', 'Executive Dysfunction']
    },
    
    // Advanced Assessments
    {
      id: 'multimodal',
      label: 'Multi-Modal Assessment',
      description: 'Combined analysis across all modalities',
      icon: <Activity className="h-5 w-5" />,
      latency: Math.max(performanceMetrics.speechLatency, performanceMetrics.retinalLatency, performanceMetrics.motorLatency, performanceMetrics.cognitiveLatency),
      accuracy: 96,
      category: 'advanced',
      conditions: ['Comprehensive Screening', 'Early Detection', 'Risk Stratification']
    },
    {
      id: 'nri-fusion',
      label: 'NRI Fusion Engine',
      description: 'Advanced risk index calculation',
      icon: <TrendingUp className="h-5 w-5" />,
      latency: performanceMetrics.nriLatency,
      accuracy: 97,
      category: 'advanced',
      conditions: ['Neurological Risk Index', 'Predictive Analytics', 'Clinical Decision Support']
    }
  ];

  const getCategoryTitle = (category: string) => {
    switch (category) {
      case 'overview':
        return 'System Overview';
      case 'individual':
        return 'Individual Assessments';
      case 'advanced':
        return 'Advanced Analytics';
      default:
        return '';
    }
  };

  const getLatencyColor = (latency?: number) => {
    if (!latency) return 'text-slate-500';
    if (latency < 50) return 'text-green-600';
    if (latency < 100) return 'text-yellow-600';
    return 'text-red-600';
  };

  const renderNavigationItem = (item: NavigationItem) => {
    const isActive = activeAssessment === item.id;
    
    return (
      <motion.button
        key={item.id}
        onClick={() => onAssessmentChange(item.id)}
        className={`w-full text-left p-4 rounded-lg transition-all duration-200 ${
          isActive 
            ? 'bg-blue-50 border-blue-200 shadow-sm' 
            : 'hover:bg-slate-50 border-transparent'
        } border`}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="flex items-start space-x-3">
          <div className={`p-2 rounded-lg ${
            isActive 
              ? 'bg-blue-100 text-blue-600' 
              : 'bg-slate-100 text-slate-600'
          }`}>
            {item.icon}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-1">
              <h3 className={`font-medium text-sm ${
                isActive ? 'text-blue-900' : 'text-slate-900'
              }`}>
                {item.label}
              </h3>
              {item.latency && (
                <span className={`text-xs font-medium ${getLatencyColor(item.latency)}`}>
                  {item.latency}ms
                </span>
              )}
            </div>
            
            <p className="text-xs text-slate-600 mb-2 line-clamp-2">
              {item.description}
            </p>
            
            {item.accuracy && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">Accuracy</span>
                <span className="font-medium text-green-600">{item.accuracy}%</span>
              </div>
            )}
            
            {item.conditions && (
              <div className="mt-2">
                <div className="flex items-center space-x-1 mb-1">
                  <Info className="h-3 w-3 text-slate-400" />
                  <span className="text-xs text-slate-500">Screens for:</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {item.conditions.slice(0, 2).map((condition, index) => (
                    <span 
                      key={index}
                      className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-full"
                    >
                      {condition}
                    </span>
                  ))}
                  {item.conditions.length > 2 && (
                    <span className="text-xs text-slate-400">
                      +{item.conditions.length - 2} more
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.button>
    );
  };

  return (
    <aside className="w-80 bg-white border-r border-slate-200 h-screen overflow-y-auto">
      <div className="p-6">
        {/* System Status */}
        <div className="mb-6">
          <div className={`p-4 rounded-lg border ${
            systemStatus === 'healthy' 
              ? 'bg-green-50 border-green-200' 
              : systemStatus === 'warning'
              ? 'bg-yellow-50 border-yellow-200'
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              {systemStatus === 'healthy' && <CheckCircle className="h-4 w-4 text-green-600" />}
              {systemStatus !== 'healthy' && <AlertTriangle className="h-4 w-4 text-yellow-600" />}
              <span className={`text-sm font-medium ${
                systemStatus === 'healthy' ? 'text-green-800' : 'text-yellow-800'
              }`}>
                System {systemStatus === 'healthy' ? 'Healthy' : 'Status'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-slate-600">ML Models:</span>
                <span className="ml-1 font-medium text-green-600">4/4 Active</span>
              </div>
              <div>
                <span className="text-slate-600">Avg Latency:</span>
                <span className="ml-1 font-medium text-blue-600">
                  {((performanceMetrics.speechLatency + performanceMetrics.motorLatency + performanceMetrics.cognitiveLatency + performanceMetrics.nriLatency) / 4).toFixed(1)}ms
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="space-y-6">
          {['overview', 'individual', 'advanced'].map(category => (
            <div key={category}>
              <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                {getCategoryTitle(category)}
              </h2>
              <div className="space-y-2">
                {navigationItems
                  .filter(item => item.category === category)
                  .map(renderNavigationItem)}
              </div>
            </div>
          ))}
        </nav>

        {/* Performance Summary */}
        <div className="mt-6 p-4 bg-slate-50 rounded-lg">
          <h3 className="text-sm font-medium text-slate-900 mb-3">Performance Summary</h3>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-600">Overall Accuracy:</span>
              <span className="font-medium text-green-600">{performanceMetrics.overallAccuracy}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Fastest Model:</span>
              <span className="font-medium text-blue-600">NRI ({performanceMetrics.nriLatency}ms)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Real-time Ready:</span>
              <span className="font-medium text-green-600">Yes</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
