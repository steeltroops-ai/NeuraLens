'use client';

import {
  LayoutDashboard,
  Plus,
  Search,
  Filter,
  Download,
  Settings,
  TrendingUp,
  Users,
  Activity,
  Clock,
  AlertCircle,
  CheckCircle,
  Info,
} from 'lucide-react';
import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Import assessment components
import SpeechAssessment from './SpeechAssessment';
import RetinalAssessment from './RetinalAssessment';
import MotorAssessment from './MotorAssessment';
import AssessmentHistory from './AssessmentHistory';

interface DashboardCRUDProps {
  initialView?: 'overview' | 'speech' | 'retinal' | 'motor' | 'cognitive' | 'history';
}

interface DashboardStats {
  totalAssessments: number;
  completedToday: number;
  averageRiskScore: number;
  processingTime: number;
}

interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  action: () => void;
}

export default function DashboardCRUD({ initialView = 'overview' }: DashboardCRUDProps) {
  const [activeView, setActiveView] = useState(initialView);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats>({
    totalAssessments: 0,
    completedToday: 0,
    averageRiskScore: 0,
    processingTime: 0,
  });

  // Load dashboard statistics
  useEffect(() => {
    const loadDashboardStats = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));

        setDashboardStats({
          totalAssessments: 247,
          completedToday: 12,
          averageRiskScore: 0.23,
          processingTime: 156,
        });
      } catch (error) {
        console.error('Failed to load dashboard stats:', error);
      }
    };

    loadDashboardStats();
  }, []);

  // Quick actions configuration
  const quickActions: QuickAction[] = [
    {
      id: 'speech',
      title: 'Speech Analysis',
      description: 'Voice pattern analysis with Whisper-tiny',
      icon: (
        <div className='rounded-lg bg-blue-100 p-2'>
          <Activity className='h-5 w-5 text-blue-600' />
        </div>
      ),
      color: 'blue',
      action: () => setActiveView('speech'),
    },
    {
      id: 'retinal',
      title: 'Retinal Analysis',
      description: 'Fundus image analysis with EfficientNet-B0',
      icon: (
        <div className='rounded-lg bg-green-100 p-2'>
          <TrendingUp className='h-5 w-5 text-green-600' />
        </div>
      ),
      color: 'green',
      action: () => setActiveView('retinal'),
    },
    {
      id: 'motor',
      title: 'Motor Assessment',
      description: 'Interactive finger tapping test',
      icon: (
        <div className='rounded-lg bg-purple-100 p-2'>
          <Users className='h-5 w-5 text-purple-600' />
        </div>
      ),
      color: 'purple',
      action: () => setActiveView('motor'),
    },
    {
      id: 'history',
      title: 'Assessment History',
      description: 'View and manage past assessments',
      icon: (
        <div className='rounded-lg bg-indigo-100 p-2'>
          <Clock className='h-5 w-5 text-indigo-600' />
        </div>
      ),
      color: 'indigo',
      action: () => setActiveView('history'),
    },
  ];

  // Navigation items
  const navigationItems = [
    { id: 'overview' as const, label: 'Overview', icon: LayoutDashboard },
    { id: 'speech' as const, label: 'Speech Analysis', icon: Activity },
    { id: 'retinal' as const, label: 'Retinal Analysis', icon: TrendingUp },
    { id: 'motor' as const, label: 'Motor Assessment', icon: Users },
    { id: 'history' as const, label: 'Assessment History', icon: Clock },
  ];

  const handleProcessingChange = useCallback((processing: boolean) => {
    setIsProcessing(processing);
  }, []);

  const renderActiveView = () => {
    switch (activeView) {
      case 'speech':
        return <SpeechAssessment onProcessingChange={handleProcessingChange} />;
      case 'retinal':
        return <RetinalAssessment onProcessingChange={handleProcessingChange} />;
      case 'motor':
        return <MotorAssessment onProcessingChange={handleProcessingChange} />;
      case 'history':
        return <AssessmentHistory onProcessingChange={handleProcessingChange} />;
      default:
        return renderOverview();
    }
  };

  const renderOverview = () => (
    <div className='space-y-6'>
      {/* Dashboard Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <div className='mb-6 flex items-center justify-between'>
          <div>
            <h1 className='text-3xl font-bold text-slate-900'>MediLens Dashboard</h1>
            <p className='text-slate-600'>Comprehensive medical assessment platform</p>
          </div>
          <div className='flex space-x-3'>
            <button className='rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50'>
              <Download className='mr-2 h-4 w-4' />
              Export Data
            </button>
            <button className='rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700'>
              <Plus className='mr-2 h-4 w-4' />
              New Assessment
            </button>
          </div>
        </div>

        {/* Statistics Grid */}
        <div className='grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4'>
          <div className='rounded-lg border border-slate-200 p-4'>
            <div className='flex items-center justify-between'>
              <div>
                <p className='text-sm font-medium text-slate-600'>Total Assessments</p>
                <p className='text-2xl font-bold text-slate-900'>
                  {dashboardStats.totalAssessments}
                </p>
              </div>
              <div className='rounded-lg bg-blue-100 p-2'>
                <Activity className='h-5 w-5 text-blue-600' />
              </div>
            </div>
            <p className='mt-2 text-xs text-slate-500'>All time</p>
          </div>

          <div className='rounded-lg border border-slate-200 p-4'>
            <div className='flex items-center justify-between'>
              <div>
                <p className='text-sm font-medium text-slate-600'>Completed Today</p>
                <p className='text-2xl font-bold text-slate-900'>{dashboardStats.completedToday}</p>
              </div>
              <div className='rounded-lg bg-green-100 p-2'>
                <CheckCircle className='h-5 w-5 text-green-600' />
              </div>
            </div>
            <p className='mt-2 text-xs text-green-600'>+23% from yesterday</p>
          </div>

          <div className='rounded-lg border border-slate-200 p-4'>
            <div className='flex items-center justify-between'>
              <div>
                <p className='text-sm font-medium text-slate-600'>Average Risk Score</p>
                <p className='text-2xl font-bold text-slate-900'>
                  {(dashboardStats.averageRiskScore * 100).toFixed(1)}%
                </p>
              </div>
              <div className='rounded-lg bg-yellow-100 p-2'>
                <AlertCircle className='h-5 w-5 text-yellow-600' />
              </div>
            </div>
            <p className='mt-2 text-xs text-slate-500'>Low risk range</p>
          </div>

          <div className='rounded-lg border border-slate-200 p-4'>
            <div className='flex items-center justify-between'>
              <div>
                <p className='text-sm font-medium text-slate-600'>Avg Processing Time</p>
                <p className='text-2xl font-bold text-slate-900'>
                  {dashboardStats.processingTime}ms
                </p>
              </div>
              <div className='rounded-lg bg-purple-100 p-2'>
                <Clock className='h-5 w-5 text-purple-600' />
              </div>
            </div>
            <p className='mt-2 text-xs text-purple-600'>Within target</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>Quick Actions</h2>
        <div className='grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4'>
          {quickActions.map(action => (
            <motion.button
              key={action.id}
              onClick={action.action}
              className='rounded-lg border border-slate-200 p-4 text-left transition-all duration-200 hover:border-slate-300 hover:shadow-md'
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className='mb-3 flex items-center space-x-3'>
                {action.icon}
                <h3 className='font-medium text-slate-900'>{action.title}</h3>
              </div>
              <p className='text-sm text-slate-600'>{action.description}</p>
            </motion.button>
          ))}
        </div>
      </div>

      {/* System Status */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>System Status</h2>
        <div className='space-y-3'>
          <div className='flex items-center justify-between rounded-lg border border-green-200 bg-green-50 p-3'>
            <div className='flex items-center space-x-3'>
              <CheckCircle className='h-5 w-5 text-green-600' />
              <span className='font-medium text-green-900'>Speech Analysis Module</span>
            </div>
            <span className='text-sm text-green-700'>Operational</span>
          </div>

          <div className='flex items-center justify-between rounded-lg border border-green-200 bg-green-50 p-3'>
            <div className='flex items-center space-x-3'>
              <CheckCircle className='h-5 w-5 text-green-600' />
              <span className='font-medium text-green-900'>Retinal Analysis Module</span>
            </div>
            <span className='text-sm text-green-700'>Operational</span>
          </div>

          <div className='flex items-center justify-between rounded-lg border border-green-200 bg-green-50 p-3'>
            <div className='flex items-center space-x-3'>
              <CheckCircle className='h-5 w-5 text-green-600' />
              <span className='font-medium text-green-900'>Motor Assessment Module</span>
            </div>
            <span className='text-sm text-green-700'>Operational</span>
          </div>

          <div className='flex items-center justify-between rounded-lg border border-blue-200 bg-blue-50 p-3'>
            <div className='flex items-center space-x-3'>
              <Info className='h-5 w-5 text-blue-600' />
              <span className='font-medium text-blue-900'>Database Connection</span>
            </div>
            <span className='text-sm text-blue-700'>Connected</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className='flex h-screen bg-slate-50'>
      {/* Sidebar Navigation */}
      <div className='w-64 border-r border-slate-200 bg-white'>
        <div className='p-6'>
          <h2 className='text-lg font-semibold text-slate-900'>MediLens</h2>
          <p className='text-sm text-slate-600'>Dashboard</p>
        </div>

        <nav className='px-4'>
          {navigationItems.map(item => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

            return (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`mb-2 flex w-full items-center space-x-3 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${isActive ? 'bg-indigo-100 text-indigo-900' : 'text-slate-700 hover:bg-slate-100'
                  }`}
              >
                <Icon className='h-4 w-4' />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Main Content */}
      <div className='flex-1 overflow-auto'>
        <div className='p-6'>
          <AnimatePresence mode='wait'>
            <motion.div
              key={activeView}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
            >
              {renderActiveView()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Processing Overlay */}
      {isProcessing && (
        <div className='fixed inset-0 z-50 flex items-center justify-center bg-black/20'>
          <div className='rounded-lg bg-white p-6 shadow-xl'>
            <div className='flex items-center space-x-3'>
              <div className='h-6 w-6 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent'></div>
              <span className='font-medium text-slate-900'>Processing assessment...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
