'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Mic,
  Eye,
  Hand,
  Zap,
  Activity,
  Shield,
  Clock,
  AlertTriangle,
  CheckCircle,
  Home,
  Settings,
  BarChart3,
  Users,
  FileText,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  LogOut,
  UserCircle,
} from 'lucide-react';
import SpeechAssessment from '@/components/dashboard/SpeechAssessment';
import RetinalAssessment from '@/components/dashboard/RetinalAssessment';
import MotorAssessment from '@/components/dashboard/MotorAssessment';
import CognitiveAssessment from '@/components/dashboard/CognitiveAssessment';
import MultiModalAssessment from '@/components/dashboard/MultiModalAssessment';
import NRIFusionDashboard from '@/components/dashboard/NRIFusionDashboard';
import PerformanceMetrics from '@/components/dashboard/PerformanceMetrics';
import type { AssessmentType, DashboardState } from '@/types/dashboard';

const sidebarItems = [
  { id: 'overview', label: 'Dashboard', icon: Home },
  { id: 'speech', label: 'Speech Analysis', icon: Mic },
  { id: 'retinal', label: 'Retinal Imaging', icon: Eye },
  { id: 'motor', label: 'Motor Function', icon: Hand },
  { id: 'cognitive', label: 'Cognitive Tests', icon: Brain },
  { id: 'multimodal', label: 'Multi-Modal', icon: Activity },
  { id: 'nri-fusion', label: 'NRI Fusion', icon: Zap },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'patients', label: 'Patients', icon: Users },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'settings', label: 'Settings', icon: Settings },
  { id: 'help', label: 'Help', icon: HelpCircle },
];

export default function Dashboard() {
  // BULLETPROOF icon size constants - NEVER CHANGE, NEVER SHRINK
  const BULLETPROOF_ICON_STYLE = {
    width: '24px !important',
    height: '24px !important',
    minWidth: '24px !important',
    minHeight: '24px !important',
    maxWidth: '24px !important',
    maxHeight: '24px !important',
    flexShrink: '0 !important',
    flexGrow: '0 !important',
    display: 'block !important',
  };

  const [dashboardState, setDashboardState] = useState<DashboardState>({
    activeAssessment: 'overview',
    isProcessing: false,
    lastUpdate: null,
    systemStatus: 'healthy',
  });

  const [performanceMetrics] = useState({
    speechLatency: 11.7,
    retinalLatency: 145.2,
    motorLatency: 42.3,
    cognitiveLatency: 38.1,
    nriLatency: 0.3,
    overallAccuracy: 95.2,
  });

  // Sidebar collapse state with localStorage persistence
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('neuralens-sidebar-collapsed');
      return saved ? JSON.parse(saved) : false;
    }
    return false;
  });

  // Toggle sidebar and persist state
  const toggleSidebar = () => {
    const newState = !sidebarCollapsed;
    setSidebarCollapsed(newState);
    if (typeof window !== 'undefined') {
      localStorage.setItem(
        'neuralens-sidebar-collapsed',
        JSON.stringify(newState)
      );
    }
  };

  // System health check
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await fetch('/api/health');
        if (response.ok) {
          setDashboardState((prev) => ({
            ...prev,
            systemStatus: 'healthy',
            lastUpdate: new Date(),
          }));
        } else {
          setDashboardState((prev) => ({ ...prev, systemStatus: 'warning' }));
        }
      } catch (error) {
        setDashboardState((prev) => ({ ...prev, systemStatus: 'error' }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAssessmentChange = (assessment: AssessmentType) => {
    setDashboardState((prev) => ({ ...prev, activeAssessment: assessment }));
  };

  const handleProcessingStateChange = (isProcessing: boolean) => {
    setDashboardState((prev) => ({ ...prev, isProcessing }));
  };

  const renderMainContent = () => {
    switch (dashboardState.activeAssessment) {
      case 'speech':
        return (
          <SpeechAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'retinal':
        return (
          <RetinalAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'motor':
        return (
          <MotorAssessment onProcessingChange={handleProcessingStateChange} />
        );
      case 'cognitive':
        return (
          <CognitiveAssessment
            onProcessingChange={handleProcessingStateChange}
          />
        );
      case 'multimodal':
        return (
          <MultiModalAssessment
            onProcessingChange={handleProcessingStateChange}
          />
        );
      case 'nri-fusion':
        return (
          <NRIFusionDashboard
            onProcessingChange={handleProcessingStateChange}
          />
        );
      default:
        return (
          <DashboardOverview
            metrics={performanceMetrics}
            systemStatus={dashboardState.systemStatus}
          />
        );
    }
  };

  return (
    <div className="flex min-h-screen" style={{ backgroundColor: '#F5F5F7' }}>
      {/* Apple-Style Dynamic Top Navigation */}
      <div
        className="fixed right-0 top-0 z-40 flex h-16 items-center justify-between px-6 transition-all duration-200"
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid #F5F5F7',
          left: sidebarCollapsed ? '64px' : '256px',
        }}
      >
        {/* Current Section */}
        <div className="flex items-center">
          <span
            className="text-lg font-semibold"
            style={{
              color: '#1D1D1F',
              fontFamily:
                '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
          >
            {sidebarItems.find(
              (item) => item.id === dashboardState.activeAssessment
            )?.label || 'Dashboard'}
          </span>
        </div>

        {/* User Profile */}
        <div className="flex items-center space-x-3">
          <button className="flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-gray-100">
            <UserCircle className="h-6 w-6" style={{ color: '#1D1D1F' }} />
          </button>
        </div>
      </div>

      {/* Apple-Style Collapsible Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-50 border-r transition-all duration-200 ease-in-out ${
          sidebarCollapsed ? 'w-16' : 'w-64'
        }`}
        style={{
          borderColor: '#F5F5F7',
          backgroundColor: '#FFFFFF',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Apple-Style Logo Header with Inline Toggle */}
        <div
          className="flex items-center border-b p-4 transition-all duration-200"
          style={{ borderColor: '#F5F5F7', minHeight: '64px' }}
        >
          {!sidebarCollapsed ? (
            <div className="flex w-full items-center justify-between">
              {/* Logo and Title */}
              <div
                className="flex cursor-pointer items-center space-x-3"
                onClick={() => handleAssessmentChange('overview')}
              >
                <div className="flex h-8 w-8 items-center justify-center">
                  <span
                    className="text-xl font-semibold"
                    style={{
                      color: '#1D1D1F',
                      fontFamily:
                        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    }}
                  >
                    N
                  </span>
                </div>
                <div>
                  <h1
                    className="text-lg font-semibold"
                    style={{
                      color: '#1D1D1F',
                      fontFamily:
                        '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                    }}
                  >
                    NeuraLens
                  </h1>                
                </div>
              </div>

              {/* Inline Collapse Button */}
              <button
                onClick={toggleSidebar}
                className="flex h-4 w-4 items-center justify-center rounded transition-colors hover:bg-gray-100"
                style={{ minWidth: '16px', minHeight: '16px' }}
              >
                <ChevronLeft
                  style={{
                    width: '16px',
                    height: '16px',
                    color: '#86868B',
                  }}
                />
              </button>
            </div>
          ) : (
            <div
              className="mx-auto flex h-8 w-8 cursor-pointer items-center justify-center"
              onClick={toggleSidebar}
            >
              <span
                className="text-xl font-semibold"
                style={{
                  color: '#1D1D1F',
                  fontFamily:
                    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                N
              </span>
            </div>
          )}
        </div>

        {/* Apple-Style Navigation Menu */}
        <nav
          className="flex-1 overflow-y-auto p-4"
          style={{ paddingBottom: '120px' }}
        >
          <div className="space-y-1">
            {sidebarItems.map((item) => {
              const Icon = item.icon;
              const isActive = dashboardState.activeAssessment === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() =>
                    handleAssessmentChange(item.id as AssessmentType)
                  }
                  className={`hover:scale-98 flex w-full items-center rounded-lg px-3 py-3 text-left transition-all duration-200 ${
                    sidebarCollapsed ? 'justify-center' : 'space-x-4'
                  }`}
                  style={{
                    backgroundColor: isActive ? '#007AFF' : 'transparent',
                    color: isActive ? '#FFFFFF' : '#1D1D1F',
                    minHeight: '44px', // Apple minimum touch target
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = '#F5F5F7';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <Icon
                    style={{
                      ...BULLETPROOF_ICON_STYLE,
                      color: isActive ? '#FFFFFF' : '#1D1D1F',
                    }}
                  />
                  {!sidebarCollapsed && (
                    <span
                      className="text-sm font-medium"
                      style={{
                        color: isActive ? '#FFFFFF' : '#1D1D1F',
                        fontFamily:
                          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                      }}
                    >
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </nav>

        {/* Apple-Style Bottom-Fixed Profile and Logout Buttons */}
        <div
          className="absolute bottom-0 left-0 right-0 border-t"
          style={{
            borderColor: '#F5F5F7',
            backgroundColor: '#FFFFFF',
          }}
        >
          {/* User Profile Button */}
          <button
            onClick={() => {
              // Handle user profile logic here
              console.log('User profile clicked');
            }}
            className={`hover:scale-98 flex w-full items-center px-3 py-3 text-left transition-all duration-200 ${
              sidebarCollapsed ? 'justify-center' : 'space-x-4'
            }`}
            style={{
              backgroundColor: 'transparent',
              color: '#1D1D1F',
              minHeight: '44px',
              borderBottom: '1px solid #F5F5F7',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#F5F5F7';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <UserCircle
              style={{
                ...BULLETPROOF_ICON_STYLE,
                color: '#1D1D1F',
              }}
            />
            {!sidebarCollapsed && (
              <span
                className="text-sm font-medium"
                style={{
                  color: '#1D1D1F',
                  fontFamily:
                    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                Profile
              </span>
            )}
          </button>

          {/* Logout Button - Bottom-most with 0px margin */}
          <button
            onClick={() => {
              // Handle logout logic here
              console.log('Logout clicked');
            }}
            className={`hover:scale-98 flex w-full items-center px-3 py-3 text-left transition-all duration-200 ${
              sidebarCollapsed ? 'justify-center' : 'space-x-4'
            }`}
            style={{
              backgroundColor: 'transparent',
              color: '#FF3B30',
              minHeight: '44px',
              margin: 0,
              padding: '12px',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#FFF5F5';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <LogOut
              style={{
                ...BULLETPROOF_ICON_STYLE,
                color: '#FF3B30',
              }}
            />
            {!sidebarCollapsed && (
              <span
                className="text-sm font-medium"
                style={{
                  color: '#FF3B30',
                  fontFamily:
                    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                Logout
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Apple-Style Main Content */}
      <div
        className={`flex-1 transition-all duration-200 ${
          sidebarCollapsed ? 'ml-16' : 'ml-64'
        }`}
        style={{ backgroundColor: '#F5F5F7' }}
      >
        <main className="p-8 pt-24">
          <AnimatePresence mode="wait">
            <motion.div
              key={dashboardState.activeAssessment}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {renderMainContent()}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

// Apple-Style Dashboard Overview Component
function DashboardOverview({
  metrics,
  systemStatus,
}: {
  metrics: any;
  systemStatus: 'healthy' | 'warning' | 'error';
}) {
  return (
    <div className="space-y-12">
      {/* Apple-Style Welcome Section */}
      <section className="px-8 py-12" style={{ backgroundColor: '#FFFFFF' }}>
        <div className="mx-auto max-w-7xl">
          <div className="flex items-start justify-between">
            <div className="max-w-2xl">
              <h2
                className="mb-4 text-3xl font-bold"
                style={{
                  color: '#1D1D1F',
                  fontFamily:
                    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                Welcome to NeuraLens Dashboard
              </h2>
              <p
                className="mb-8 text-lg leading-relaxed"
                style={{ color: '#86868B' }}
              >
                Advanced neurological assessment platform with real-time AI
                analysis and multi-modal data fusion for comprehensive health
                insights.
              </p>
              <div className="flex flex-wrap items-center gap-8">
                <div className="flex items-center space-x-3">
                  <Clock className="h-5 w-5" style={{ color: '#007AFF' }} />
                  <span
                    className="text-base font-medium"
                    style={{ color: '#1D1D1F' }}
                  >
                    Real-time processing (&lt;100ms)
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <Shield className="h-5 w-5" style={{ color: '#34C759' }} />
                  <span
                    className="text-base font-medium"
                    style={{ color: '#1D1D1F' }}
                  >
                    90%+ clinical accuracy
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <Activity className="h-5 w-5" style={{ color: '#FF9500' }} />
                  <span
                    className="text-base font-medium"
                    style={{ color: '#1D1D1F' }}
                  >
                    Multi-modal fusion
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div
                className="mb-2 text-5xl font-bold"
                style={{ color: '#007AFF' }}
              >
                {metrics.overallAccuracy}%
              </div>
              <div
                className="mb-4 text-lg font-medium"
                style={{ color: '#86868B' }}
              >
                Overall Accuracy
              </div>
              <div className="flex items-center justify-end space-x-2">
                {systemStatus === 'healthy' && (
                  <>
                    <CheckCircle
                      className="h-5 w-5"
                      style={{ color: '#34C759' }}
                    />
                    <span
                      className="text-base font-medium"
                      style={{ color: '#34C759' }}
                    >
                      System Healthy
                    </span>
                  </>
                )}
                {systemStatus === 'warning' && (
                  <>
                    <AlertTriangle
                      className="h-5 w-5"
                      style={{ color: '#FF9500' }}
                    />
                    <span
                      className="text-base font-medium"
                      style={{ color: '#FF9500' }}
                    >
                      System Warning
                    </span>
                  </>
                )}
                {systemStatus === 'error' && (
                  <>
                    <AlertTriangle
                      className="h-5 w-5"
                      style={{ color: '#FF3B30' }}
                    />
                    <span
                      className="text-base font-medium"
                      style={{ color: '#FF3B30' }}
                    >
                      System Error
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Apple-Style Performance Metrics */}
      <section className="px-8 py-12" style={{ backgroundColor: '#F5F5F7' }}>
        <PerformanceMetrics metrics={metrics} />
      </section>
    </div>
  );
}
