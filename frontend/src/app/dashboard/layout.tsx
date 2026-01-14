'use client';

import { useState, useEffect, useCallback } from 'react';
import { DashboardSidebar, SIDEBAR_COLLAPSED_KEY } from '@/components/dashboard/DashboardSidebar';
import { DashboardHeader } from '@/components/dashboard/DashboardHeader';
import { CommandPalette } from '@/components/dashboard/CommandPalette';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

/**
 * Dashboard Layout Component
 * 
 * Implements MediLens Layout Patterns:
 * - Fixed sidebar with main content offset
 * - Responsive behavior: hidden sidebar on mobile (<1024px), visible on desktop
 * - Smooth sidebar transitions using CSS variables
 * 
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 7.6
 */
export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);

  // Initialize client-side state after hydration
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
      // Check if desktop
      setIsDesktop(window.innerWidth >= 1024);
    }
  }, []);

  // Listen for sidebar collapse state changes and window resize
  useEffect(() => {
    if (!isClient) return;

    const handleStorageChange = () => {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
    };

    const handleResize = () => {
      setIsDesktop(window.innerWidth >= 1024);
    };

    // Check for changes periodically (for same-tab updates)
    const interval = setInterval(handleStorageChange, 100);

    // Also listen for storage events (cross-tab) and resize
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('resize', handleResize);

    return () => {
      clearInterval(interval);
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('resize', handleResize);
    };
  }, [isClient]);

  // Handle Cmd/Ctrl+K keyboard shortcut for command palette
  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
      event.preventDefault();
      setCommandPaletteOpen(prev => !prev);
    }
    // Close on Escape
    if (event.key === 'Escape' && commandPaletteOpen) {
      setCommandPaletteOpen(false);
    }
  }, [commandPaletteOpen]);

  // Register keyboard shortcut
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Handle search button click
  const handleSearchClick = () => {
    setCommandPaletteOpen(true);
  };

  // Sidebar width for margin calculation (Requirements 1.3)
  // On mobile (<1024px), sidebar is hidden/overlay so no margin needed
  // On desktop (>=1024px), offset by sidebar width
  const sidebarWidth = sidebarCollapsed ? 60 : 240;
  const mainMarginLeft = isDesktop ? sidebarWidth : 0;

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      {/* Sidebar Navigation - Fixed position (Requirements 1.1, 1.3) */}
      <DashboardSidebar />

      {/* Main Content Area - Offset by sidebar width on desktop */}
      <div
        className="flex flex-col min-h-screen transition-[margin-left] duration-200"
        style={{
          marginLeft: `${mainMarginLeft}px`,
          transitionTimingFunction: 'ease-out',
        }}
      >
        {/* Dashboard Header - 56px height (Requirements 3.1, 3.6) */}
        <DashboardHeader
          showSearch={true}
          onSearchClick={handleSearchClick}
        />

        {/* Main Content Area (Requirements 1.4, 7.6) */}
        <main
          id="main-content"
          className="flex-1 bg-[#f8fafc] w-full max-w-full overflow-x-hidden overflow-y-auto scrollbar-hide"
          role="main"
          aria-labelledby="page-title"
        >
          <div className="p-4 lg:p-6 w-full max-w-full overflow-x-hidden">
            {children}
          </div>
        </main>
      </div>

      {/* Command Palette - Requirements 3.4 */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />
    </div>
  );
}
