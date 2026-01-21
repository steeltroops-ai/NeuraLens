"use client";

import { useState, useEffect, useCallback } from "react";
import { usePathname } from "next/navigation";
import dynamic from "next/dynamic";
import {
  DashboardSidebar,
  SIDEBAR_COLLAPSED_KEY,
} from "./_components/DashboardSidebar";
import { DashboardHeader } from "./_components/DashboardHeader";
import {
  PipelineStatusBar,
  PipelineStatusProvider,
} from "@/components/pipeline";

// Lazy load non-critical components for faster initial load
const MedicalChatbot = dynamic(
  () => import("@/components/chatbot").then((m) => m.MedicalChatbot),
  { ssr: false },
);

const CommandPalette = dynamic(
  () => import("./_components/CommandPalette").then((m) => m.CommandPalette),
  { ssr: false },
);

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
  const pathname = usePathname();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Get chatbot context from current route
  const getChatbotContext = (): string => {
    if (pathname.includes("/retinal")) return "retinal";
    if (pathname.includes("/cardiology")) return "cardiology";
    if (pathname.includes("/radiology")) return "radiology";
    if (pathname.includes("/speech")) return "speech";
    if (pathname.includes("/motor")) return "motor";
    if (pathname.includes("/cognitive")) return "cognitive";
    return "dashboard";
  };

  // Initialize client-side state after hydration
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== "undefined") {
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

    const handleStorageChange = (e?: StorageEvent) => {
      // Only respond to sidebar key changes
      if (e && e.key !== SIDEBAR_COLLAPSED_KEY && e.key !== null) return;

      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
    };

    // Debounced resize handler
    let resizeTimer: NodeJS.Timeout;
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        setIsDesktop(window.innerWidth >= 1024);
      }, 100);
    };

    // Custom event for same-tab sidebar updates (instead of polling)
    const handleSidebarChange = () => handleStorageChange();

    // Listen for storage events (cross-tab) and resize
    window.addEventListener("storage", handleStorageChange);
    window.addEventListener("resize", handleResize);
    window.addEventListener("sidebar-change", handleSidebarChange);

    return () => {
      clearTimeout(resizeTimer);
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("sidebar-change", handleSidebarChange);
    };
  }, [isClient]);

  // Handle Cmd/Ctrl+K keyboard shortcut for command palette
  // Handle Ctrl+/ for chatbot
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "k") {
        event.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
      // Ctrl+/ to toggle chatbot
      if ((event.metaKey || event.ctrlKey) && event.key === "/") {
        event.preventDefault();
        // Dispatch custom event to toggle chatbot
        window.dispatchEvent(new CustomEvent("toggle-chatbot"));
      }
      // Close on Escape
      if (event.key === "Escape" && commandPaletteOpen) {
        setCommandPaletteOpen(false);
      }
    },
    [commandPaletteOpen],
  );

  // Register keyboard shortcut
  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
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
    <PipelineStatusProvider>
      <div className="min-h-screen bg-[#09090b]">
        {/* Sidebar Navigation - Fixed position (Requirements 1.1, 1.3) */}
        <DashboardSidebar
          mobileOpen={mobileOpen}
          setMobileOpen={setMobileOpen}
        />

        {/* Main Content Area - Offset by sidebar width on desktop */}
        <div
          className="flex flex-col min-h-screen"
          style={{
            marginLeft: `${mainMarginLeft}px`,
            transition: "margin-left 350ms cubic-bezier(0.32, 0.72, 0, 1)",
          }}
        >
          {/* Dashboard Header - Fixed at top, 56px height (Requirements 3.1, 3.6) */}
          <DashboardHeader
            showSearch={true}
            onSearchClick={handleSearchClick}
            onMenuClick={() => setMobileOpen(true)}
          />

          {/* Main Content Area - Offset by header height, leave room for status bar */}
          <main
            id="main-content"
            className="flex-1 bg-[#09090b] w-full max-w-full overflow-x-hidden overflow-y-auto scrollbar-hide pt-14 pb-8"
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

        {/* Medical Chatbot - Floating assistant */}
        <MedicalChatbot context={getChatbotContext()} />

        {/* Pipeline Status Bar - VS Code style status bar at bottom */}
        <PipelineStatusBar />
      </div>
    </PipelineStatusProvider>
  );
}
