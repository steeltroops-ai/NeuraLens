'use client';

import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Mic,
  Eye,
  Hand,
  Zap,
  Activity,
  Home,
  Settings,
  BarChart3,
  Users,
  FileText,
  HelpCircle,
  ChevronLeft,
  LogOut,
  UserCircle,
  Bell,
  X,
} from 'lucide-react';
import { useState, useEffect, useCallback, memo, lazy, Suspense } from 'react';
import { useLogout } from '@/hooks/useLogout';

// Lazy load ALL dashboard components for optimal performance
const CognitiveAssessment = lazy(() => import('@/components/dashboard/CognitiveAssessment'));
const MotorAssessment = lazy(() => import('@/components/dashboard/MotorAssessment'));

const QuickActionButtons = lazy(() => import('@/components/dashboard/QuickActionButtons'));
const RecentActivityFeed = lazy(() => import('@/components/dashboard/RecentActivityFeed'));
const RetinalAssessment = lazy(() => import('@/components/dashboard/RetinalAssessment'));
const SpeechAssessment = lazy(() => import('@/components/dashboard/SpeechAssessment'));
const SystemStatusCards = lazy(() => import('@/components/dashboard/SystemStatusCards'));
const UserHealthOverview = lazy(() => import('@/components/dashboard/UserHealthOverview'));

import type { AssessmentType, DashboardState } from '@/types/dashboard';

// Lazy load non-critical components for better performance
const AIInsightsPanel = lazy(() =>
  import('@/components/dashboard/AIInsightsPanel').then(module => ({
    default: module.default,
  })),
);

// Lazy load assessment components for code splitting
const LazyMultiModalAssessment = lazy(() =>
  import('@/components/dashboard/MultiModalAssessment').then(module => ({
    default: module.default,
  })),
);
const LazyNRIFusionDashboard = lazy(() =>
  import('@/components/dashboard/NRIFusionDashboard').then(module => ({
    default: module.default,
  })),
);

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
  // Enhanced logout functionality
  const { isLoggingOut, error: logoutError, logout, clearError } = useLogout();

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

  // Sidebar collapse state with localStorage persistence (hydration-safe)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Notifications panel state
  const [notificationsOpen, setNotificationsOpen] = useState(false);

  // Close notifications when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (notificationsOpen && !target.closest('[data-notifications-panel]')) {
        setNotificationsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [notificationsOpen]);

  // Initialize client-side state after hydration
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('neuralens-sidebar-collapsed');
      if (saved) {
        setSidebarCollapsed(JSON.parse(saved));
      }
    }
  }, []);

  // Optimized toggle sidebar with useCallback (hydration-safe)
  const toggleSidebar = useCallback(() => {
    const newState = !sidebarCollapsed;
    setSidebarCollapsed(newState);
    if (isClient && typeof window !== 'undefined') {
      localStorage.setItem('neuralens-sidebar-collapsed', JSON.stringify(newState));
    }
  }, [sidebarCollapsed, isClient]);

  // System health check with optimized error handling
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await fetch('/api/health', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-cache',
        });
        if (response.ok) {
          setDashboardState(prev => ({
            ...prev,
            systemStatus: 'healthy',
            lastUpdate: new Date(),
          }));
        } else {
          setDashboardState(prev => ({ ...prev, systemStatus: 'warning' }));
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setDashboardState(prev => ({ ...prev, systemStatus: 'error' }));
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Preload components on hover for instant navigation
  const preloadComponent = useCallback((assessment: AssessmentType) => {
    switch (assessment) {
      case 'multimodal':
        import('@/components/dashboard/MultiModalAssessment');
        break;
      case 'nri-fusion':
        import('@/components/dashboard/NRIFusionDashboard');
        break;
      default:
        break;
    }
  }, []);

  // Optimized handlers with useCallback for performance
  const handleAssessmentChange = useCallback((assessment: AssessmentType) => {
    setDashboardState(prev => ({ ...prev, activeAssessment: assessment }));
  }, []);

  const handleProcessingStateChange = useCallback((isProcessing: boolean) => {
    setDashboardState(prev => ({ ...prev, isProcessing }));
  }, []);

  const renderMainContent = () => {
    const LoadingFallback = () => (
      <div className='flex h-64 items-center justify-center'>
        <div className='animate-pulse text-gray-500'>Loading...</div>
      </div>
    );

    switch (dashboardState.activeAssessment) {
      case 'speech':
        return <SpeechAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'retinal':
        return <RetinalAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'motor':
        return <MotorAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'cognitive':
        return <CognitiveAssessment onProcessingChange={handleProcessingStateChange} />;
      case 'multimodal':
        return (
          <Suspense fallback={<LoadingFallback />}>
            <LazyMultiModalAssessment onProcessingChange={handleProcessingStateChange} />
          </Suspense>
        );
      case 'nri-fusion':
        return (
          <Suspense fallback={<LoadingFallback />}>
            <LazyNRIFusionDashboard onProcessingChange={handleProcessingStateChange} />
          </Suspense>
        );
      default:
        return <DashboardOverview systemStatus={dashboardState.systemStatus} />;
    }
  };

  return (
    <div className='flex min-h-screen' style={{ backgroundColor: '#F5F5F7' }}>
      {/* Apple-Style Dynamic Top Navigation */}
      <div
        className='fixed right-0 top-0 z-40 flex h-16 items-center justify-between px-6 transition-all duration-200'
        style={{
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid #F5F5F7',
          left: sidebarCollapsed ? '64px' : '256px',
        }}
      >
        {/* Current Section */}
        <div className='flex items-center'>
          <span
            className='text-lg font-semibold'
            style={{
              color: '#1D1D1F',
              fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
          >
            {sidebarItems.find(item => item.id === dashboardState.activeAssessment)?.label ||
              'Dashboard'}
          </span>
        </div>

        {/* Notifications & User Profile */}
        <div className='flex items-center space-x-3'>
          {/* Notifications Button */}
          <div className='relative' data-notifications-panel>
            <button
              onClick={() => setNotificationsOpen(!notificationsOpen)}
              className='flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-gray-100'
            >
              <Bell className='h-5 w-5' style={{ color: '#1D1D1F' }} />
              {/* Notification Badge */}
              <div className='absolute -right-1 -top-1 h-3 w-3 rounded-full bg-red-500'>
                <div className='flex h-full w-full items-center justify-center'>
                  <span className='text-xs font-bold text-white'>3</span>
                </div>
              </div>
            </button>

            {/* Notifications Dropdown Panel */}
            {notificationsOpen && (
              <div
                className='absolute right-0 top-10 z-50 w-80 rounded-2xl border shadow-lg backdrop-blur-xl'
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderColor: 'rgba(0, 0, 0, 0.1)',
                  backdropFilter: 'blur(20px)',
                }}
              >
                {/* Notifications Header */}
                <div
                  className='flex items-center justify-between border-b p-4'
                  style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}
                >
                  <h3 className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
                    AI Insights
                  </h3>
                  <button
                    onClick={() => setNotificationsOpen(false)}
                    className='flex h-6 w-6 items-center justify-center rounded-full transition-colors hover:bg-gray-100'
                  >
                    <X className='h-4 w-4' style={{ color: '#86868B' }} />
                  </button>
                </div>

                {/* AI Insights Panel Content */}
                <div className='max-h-96 overflow-y-auto p-4'>
                  <Suspense
                    fallback={
                      <div className='animate-pulse space-y-3'>
                        <div className='h-4 w-3/4 rounded bg-gray-200' />
                        <div className='h-4 w-1/2 rounded bg-gray-200' />
                        <div className='h-4 w-2/3 rounded bg-gray-200' />
                      </div>
                    }
                  >
                    <AIInsightsPanel />
                  </Suspense>
                </div>
              </div>
            )}
          </div>

          {/* User Profile Button */}
          <button className='flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-gray-100'>
            <UserCircle className='h-6 w-6' style={{ color: '#1D1D1F' }} />
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
          className='flex items-center border-b p-4 transition-all duration-200'
          style={{ borderColor: '#F5F5F7', minHeight: '64px' }}
        >
          {!sidebarCollapsed ? (
            <div className='flex w-full items-center justify-between'>
              {/* Logo and Title */}
              <div
                className='flex cursor-pointer items-center space-x-3'
                onClick={() => handleAssessmentChange('overview')}
              >
                <div className='flex h-8 w-8 items-center justify-center'>
                  <span
                    className='text-xl font-semibold'
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
                    className='text-lg font-semibold'
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
                className='flex h-4 w-4 items-center justify-center rounded transition-colors hover:bg-gray-100'
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
              className='mx-auto flex h-8 w-8 cursor-pointer items-center justify-center'
              onClick={toggleSidebar}
            >
              <span
                className='text-xl font-semibold'
                style={{
                  color: '#1D1D1F',
                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                N
              </span>
            </div>
          )}
        </div>

        {/* Apple-Style Navigation Menu */}
        <nav className='flex-1 overflow-y-auto p-4' style={{ paddingBottom: '120px' }}>
          <div className='space-y-1'>
            {sidebarItems.map(item => {
              const Icon = item.icon;
              const isActive = dashboardState.activeAssessment === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => handleAssessmentChange(item.id as AssessmentType)}
                  className={`hover:scale-98 flex w-full items-center rounded-lg px-3 py-3 text-left transition-all duration-200 ${
                    sidebarCollapsed ? 'justify-center' : 'space-x-4'
                  }`}
                  style={{
                    backgroundColor: isActive ? '#007AFF' : 'transparent',
                    color: isActive ? '#FFFFFF' : '#1D1D1F',
                    minHeight: '44px', // Apple minimum touch target
                  }}
                  onMouseEnter={e => {
                    // Preload component for instant navigation
                    preloadComponent(item.id as AssessmentType);
                    // Visual hover effect
                    if (!isActive) {
                      e.currentTarget.style.backgroundColor = '#F5F5F7';
                    }
                  }}
                  onMouseLeave={e => {
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
                      className='text-sm font-medium'
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
          className='absolute bottom-0 left-0 right-0 border-t'
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
            onMouseEnter={e => {
              e.currentTarget.style.backgroundColor = '#F5F5F7';
            }}
            onMouseLeave={e => {
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
                className='text-sm font-medium'
                style={{
                  color: '#1D1D1F',
                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                Profile
              </span>
            )}
          </button>

          {/* Logout Button - Bottom-most with 0px margin */}
          <button
            onClick={logout}
            disabled={isLoggingOut}
            className={`hover:scale-98 flex w-full items-center px-3 py-3 text-left transition-all duration-200 ${
              sidebarCollapsed ? 'justify-center' : 'space-x-4'
            } ${isLoggingOut ? 'cursor-not-allowed opacity-50' : ''}`}
            style={{
              backgroundColor: 'transparent',
              color: '#FF3B30',
              minHeight: '44px',
              margin: 0,
              padding: '12px',
              cursor: isLoggingOut ? 'not-allowed' : 'pointer',
            }}
            onMouseEnter={e => {
              if (!isLoggingOut) {
                e.currentTarget.style.backgroundColor = '#FFF5F5';
              }
            }}
            onMouseLeave={e => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <LogOut
              style={{
                ...BULLETPROOF_ICON_STYLE,
                color: '#FF3B30',
              }}
              className={isLoggingOut ? 'animate-spin' : ''}
            />
            {!sidebarCollapsed && (
              <span
                className='text-sm font-medium'
                style={{
                  color: '#FF3B30',
                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                {isLoggingOut ? 'Logging out...' : 'Logout'}
              </span>
            )}
          </button>

          {/* Logout Error Display */}
          {logoutError && !sidebarCollapsed && (
            <div className='px-3 py-2 text-xs' style={{ color: '#FF3B30' }}>
              {logoutError}
              <button onClick={clearError} className='ml-2 underline hover:no-underline'>
                Dismiss
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Apple-Style Main Content */}
      <div
        className={`flex-1 transition-all duration-200 ${sidebarCollapsed ? 'ml-16' : 'ml-64'}`}
        style={{ backgroundColor: '#F5F5F7' }}
      >
        <main className='p-8 pt-24'>
          <div className='h-full'>
            <AnimatePresence mode='wait' initial={false}>
              <motion.div
                key={dashboardState.activeAssessment}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{
                  duration: 0.2,
                  ease: 'easeInOut',
                  layout: { duration: 0.2 },
                }}
                className='h-full'
                layout
              >
                <Suspense
                  fallback={
                    <div className='animate-pulse space-y-4 p-6'>
                      <div className='h-8 w-1/4 rounded bg-gray-200'></div>
                      <div className='grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4'>
                        {Array.from({ length: 4 }).map((_, i) => (
                          <div key={i} className='h-32 rounded bg-gray-200'></div>
                        ))}
                      </div>
                      <div className='h-64 rounded bg-gray-200'></div>
                    </div>
                  }
                >
                  {renderMainContent()}
                </Suspense>
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}

// Enhanced Apple-Style Dashboard Overview Component (Memoized for Performance)
const DashboardOverview = memo(
  ({ systemStatus: _systemStatus }: { systemStatus: 'healthy' | 'warning' | 'error' }) => {
    return (
      <div className='space-y-6'>
        {/* Primary Section - Critical Health Data */}
        <section className='space-y-6'>
          <UserHealthOverview />
          <SystemStatusCards />
        </section>

        {/* Secondary Section - Quick Actions & Recent Activity */}
        <section className='grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3'>
          <div className='md:col-span-2 lg:col-span-2'>
            <QuickActionButtons />
          </div>
          <div className='md:col-span-2 lg:col-span-1'>
            <RecentActivityFeed />
          </div>
        </section>
      </div>
    );
  },
);

DashboardOverview.displayName = 'DashboardOverview';
