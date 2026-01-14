'use client';

import { useState, useEffect, Suspense } from 'react';
import { usePathname } from 'next/navigation';
import { Bell, UserCircle, X } from 'lucide-react';
import { DashboardSidebar, sidebarItems, SIDEBAR_COLLAPSED_KEY } from '@/components/dashboard/DashboardSidebar';

// Lazy load AI Insights Panel for notifications
import dynamic from 'next/dynamic';
const AIInsightsPanel = dynamic(() => import('@/components/dashboard/AIInsightsPanel'), {
  ssr: false,
  loading: () => (
    <div className="animate-pulse space-y-3">
      <div className="h-4 w-3/4 rounded bg-gray-200" />
      <div className="h-4 w-1/2 rounded bg-gray-200" />
      <div className="h-4 w-2/3 rounded bg-gray-200" />
    </div>
  ),
});

interface DashboardLayoutProps {
  children: React.ReactNode;
}

/**
 * Dashboard Layout Component
 * 
 * Implements MediLens Layout Patterns:
 * - 280px sidebar width (Requirements 18.2)
 * - grid-cols-[280px_1fr] layout pattern
 * - Responsive behavior for mobile
 * 
 * Requirements: 18.2
 */
export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const pathname = usePathname();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);

  // Initialize client-side state after hydration
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
    }
  }, []);

  // Listen for sidebar collapse state changes
  useEffect(() => {
    if (!isClient) return;

    const handleStorageChange = () => {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
    };

    // Check for changes periodically (for same-tab updates)
    const interval = setInterval(handleStorageChange, 100);

    // Also listen for storage events (cross-tab)
    window.addEventListener('storage', handleStorageChange);

    return () => {
      clearInterval(interval);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [isClient]);

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

  // Get current page title from sidebar items
  const getCurrentPageTitle = () => {
    const currentItem = sidebarItems.find(item => {
      if (item.route === '/dashboard') {
        return pathname === '/dashboard';
      }
      return pathname.startsWith(item.route);
    });
    return currentItem?.label || 'Dashboard';
  };

  // Sidebar width based on collapse state (Requirements 18.2)
  const sidebarWidth = sidebarCollapsed ? '64px' : '280px';

  return (
    <div
      className="min-h-screen"
      style={{
        backgroundColor: '#F5F5F7',
        // Use CSS Grid for dashboard layout (Requirements 18.2)
        display: 'grid',
        gridTemplateColumns: `${sidebarWidth} 1fr`,
      }}
    >
      {/* Sidebar Navigation - 280px width (Requirements 18.2) */}
      <DashboardSidebar />

      {/* Main Content Area with Header */}
      <div className="flex flex-col min-h-screen lg:col-start-2">
        {/* Top Navigation Bar */}
        <header
          className="sticky top-0 z-40 flex h-16 items-center justify-between px-6"
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid #F5F5F7',
          }}
          role="banner"
          aria-label="Dashboard header"
        >
          {/* Current Section Title */}
          <div className="flex items-center pl-12 lg:pl-0">
            <h2
              className="text-lg font-semibold"
              style={{
                color: '#1D1D1F',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              }}
              id="page-title"
            >
              {getCurrentPageTitle()}
            </h2>
          </div>

          {/* Notifications & User Profile */}
          <div className="flex items-center space-x-3" role="group" aria-label="User actions">
            {/* Notifications Button */}
            <div className="relative" data-notifications-panel>
              <button
                onClick={() => setNotificationsOpen(!notificationsOpen)}
                className="flex h-10 w-10 items-center justify-center rounded-full transition-colors hover:bg-gray-100 focus-visible:ring-3 focus-visible:ring-[#007AFF]/40"
                aria-label={`Open notifications. You have 3 unread notifications`}
                aria-expanded={notificationsOpen}
                aria-haspopup="dialog"
                aria-controls="notifications-panel"
              >
                <Bell className="h-5 w-5" style={{ color: '#1D1D1F' }} aria-hidden="true" />
                {/* Notification Badge */}
                <span
                  className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-bold text-white"
                  aria-hidden="true"
                >
                  3
                </span>
              </button>

              {/* Notifications Dropdown Panel */}
              {notificationsOpen && (
                <div
                  id="notifications-panel"
                  className="absolute right-0 top-12 z-50 w-80 rounded-2xl border shadow-lg"
                  style={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                    backdropFilter: 'blur(20px)',
                  }}
                  role="dialog"
                  aria-label="AI Insights notifications"
                  aria-modal="false"
                >
                  {/* Notifications Header */}
                  <div
                    className="flex items-center justify-between border-b p-4"
                    style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}
                  >
                    <h3 className="text-lg font-semibold" style={{ color: '#1D1D1F' }} id="notifications-title">
                      AI Insights
                    </h3>
                    <button
                      onClick={() => setNotificationsOpen(false)}
                      className="flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-gray-100 focus-visible:ring-3 focus-visible:ring-[#007AFF]/40"
                      aria-label="Close notifications panel"
                    >
                      <X className="h-4 w-4" style={{ color: '#86868B' }} aria-hidden="true" />
                    </button>
                  </div>

                  {/* AI Insights Panel Content */}
                  <div className="max-h-96 overflow-y-auto p-4" role="region" aria-labelledby="notifications-title">
                    <Suspense
                      fallback={
                        <div className="animate-pulse space-y-3" role="status" aria-label="Loading notifications">
                          <div className="h-4 w-3/4 rounded bg-gray-200" />
                          <div className="h-4 w-1/2 rounded bg-gray-200" />
                          <div className="h-4 w-2/3 rounded bg-gray-200" />
                          <span className="sr-only">Loading notifications...</span>
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
            <button
              className="flex h-10 w-10 items-center justify-center rounded-full transition-colors hover:bg-gray-100 focus-visible:ring-3 focus-visible:ring-[#007AFF]/40"
              aria-label="Open user profile menu"
              aria-haspopup="menu"
            >
              <UserCircle className="h-6 w-6" style={{ color: '#1D1D1F' }} aria-hidden="true" />
            </button>
          </div>
        </header>

        {/* Main Content Area */}
        <main
          id="main-content"
          className="flex-1"
          style={{ backgroundColor: '#F5F5F7', maxWidth: '100%', overflowX: 'hidden' }}
          role="main"
          aria-labelledby="page-title"
        >
          <div className="p-4 sm:p-6 lg:p-8" style={{ maxWidth: '100%', overflowX: 'hidden' }}>
            {children}
          </div>
        </main>
      </div>

      {/* Mobile: Hide grid layout and use flex */}
      <style jsx>{`
                @media (max-width: 1023px) {
                    div[style*="grid-template-columns"] {
                        display: flex !important;
                        flex-direction: column !important;
                    }
                }
            `}</style>
    </div>
  );
}
