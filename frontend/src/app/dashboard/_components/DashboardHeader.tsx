"use client";

import { useState, useEffect, Suspense } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Bell,
  X,
  ChevronRight,
  Search,
  Menu,
  UserPlus,
  User,
  Phone,
  Hash,
  Plus,
  LogOut,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  sidebarItems,
  sidebarGroups,
  SIDEBAR_COLLAPSED_KEY,
} from "./DashboardSidebar";
import dynamic from "next/dynamic";
import { usePatient } from "@/context/PatientContext";
import { PatientSelector } from "@/components/patient/PatientSelector";
import { NewPatientModal } from "@/components/patient/NewPatientModal";

// Lazy load AI Insights Panel for notifications
const AIInsightsPanel = dynamic(() => import("./shared/AIInsightsPanel"), {
  ssr: false,
  loading: () => (
    <div className="animate-pulse space-y-3">
      <div className="h-4 w-3/4 rounded bg-gray-100" />
      <div className="h-4 w-1/2 rounded bg-gray-100" />
      <div className="h-4 w-2/3 rounded bg-gray-100" />
    </div>
  ),
});

// Breadcrumb interface
interface Breadcrumb {
  label: string;
  href?: string;
}

export interface DashboardHeaderProps {
  title?: string;
  showSearch?: boolean;
  onSearchClick?: () => void;
  onMenuClick?: () => void;
}

/**
 * Dashboard Header Component
 *
 * Fixed header that stays at the top and adjusts position based on sidebar state
 * Height: 56px (h-14) - same as sidebar logo tab
 *
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */
export function DashboardHeader({
  title,
  showSearch = true,
  onSearchClick,
  onMenuClick,
}: DashboardHeaderProps) {
  const pathname = usePathname();
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notificationCount] = useState(0); // No static fake count - connect to real notifications when available
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);

  // Patient Context
  const { activePatient, setActivePatient } = usePatient();
  const [isNewPatientModalOpen, setIsNewPatientModalOpen] = useState(false);
  const [isPatientDropdownOpen, setIsPatientDropdownOpen] = useState(false);

  // Initialize client-side state and listen for sidebar changes
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setSidebarCollapsed(JSON.parse(saved));
      }
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
    window.addEventListener("storage", handleStorageChange);
    window.addEventListener("resize", handleResize);

    return () => {
      clearInterval(interval);
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener("resize", handleResize);
    };
  }, [isClient]);

  // Close notifications when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (notificationsOpen && !target.closest("[data-notifications-panel]")) {
        setNotificationsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [notificationsOpen]);

  // Generate breadcrumbs from current route
  const getBreadcrumbs = (): Breadcrumb[] => {
    const breadcrumbs: Breadcrumb[] = [
      { label: "Dashboard", href: "/dashboard" },
    ];

    if (pathname === "/dashboard") {
      return breadcrumbs;
    }

    const currentItem = sidebarItems.find((item) => {
      if (item.route === "/dashboard") return false;
      return pathname.startsWith(item.route);
    });

    if (currentItem) {
      const parentGroup = sidebarGroups.find((group) =>
        group.items.some((item) => item.id === currentItem.id),
      );

      if (parentGroup && parentGroup.id !== "overview") {
        breadcrumbs.push({ label: parentGroup.label });
      }

      breadcrumbs.push({ label: currentItem.label });
    }

    return breadcrumbs;
  };

  // Get current page title
  const getCurrentPageTitle = (): string => {
    if (title) return title;

    const currentItem = sidebarItems.find((item) => {
      if (item.route === "/dashboard") {
        return pathname === "/dashboard";
      }
      return pathname.startsWith(item.route);
    });

    return currentItem?.label || "Dashboard";
  };

  const breadcrumbs = getBreadcrumbs();
  const pageTitle = getCurrentPageTitle();

  // Calculate header left offset based on sidebar state
  const headerLeftOffset = isDesktop ? (sidebarCollapsed ? 60 : 240) : 0;

  return (
    <header
      className="fixed top-0 right-0 z-40 flex items-center justify-between px-4 lg:px-6 bg-black border-b border-[#27272a] h-14"
      style={{
        left: `${headerLeftOffset}px`,
        transition: "left 350ms cubic-bezier(0.32, 0.72, 0, 1)",
      }}
      role="banner"
      aria-label="Dashboard header"
      data-testid="dashboard-header"
    >
      {/* Left Section: Breadcrumbs */}
      <div className="flex items-center min-w-0 pl-0">
        {/* Mobile Menu Button - Integrated */}
        <button
          onClick={onMenuClick}
          className="mr-2 lg:hidden flex h-8 w-8 items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-white/10 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white"
          aria-label="Open menu"
          aria-controls="main-navigation"
        >
          <Menu className="h-5 w-5" strokeWidth={1.5} />
        </button>

        {/* Breadcrumb Navigation */}
        <nav className="flex items-center" aria-label="Breadcrumb">
          <ol className="flex items-center">
            {breadcrumbs.map((crumb, index) => (
              <li key={index} className="flex items-center">
                {index > 0 && (
                  <ChevronRight
                    size={14}
                    className="mx-1.5 text-zinc-500"
                    aria-hidden="true"
                  />
                )}
                {crumb.href && index < breadcrumbs.length - 1 ? (
                  <Link
                    href={crumb.href}
                    className="text-[13px] text-zinc-400 hover:text-white transition-colors"
                  >
                    {crumb.label}
                  </Link>
                ) : (
                  <span
                    className={`text-[13px] ${index === breadcrumbs.length - 1 ? "text-white font-medium" : "text-zinc-400"}`}
                  >
                    {crumb.label}
                  </span>
                )}
              </li>
            ))}
          </ol>
        </nav>
      </div>

      {/* Right Section: Actions */}
      <div
        className="flex items-center gap-1"
        role="group"
        aria-label="User actions"
      >
        {/* Patient Selection (Desktop) */}
        <div className="hidden md:flex items-center gap-2 mr-2 border-r border-zinc-800 pr-2">
          <PatientSelector />
          {activePatient ? (
            <div className="relative">
              <button
                onClick={() => setIsPatientDropdownOpen(!isPatientDropdownOpen)}
                className={`flex h-8 w-8 items-center justify-center rounded-md transition-all ${isPatientDropdownOpen ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white hover:bg-white/10"}`}
                title="Active Patient Profile"
              >
                <div className="relative">
                  <User size={16} strokeWidth={1.5} />
                  <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-emerald-500 ring-2 ring-black" />
                </div>
              </button>

              <AnimatePresence>
                {isPatientDropdownOpen && (
                  <>
                    <div
                      className="fixed inset-0 z-40"
                      onClick={() => setIsPatientDropdownOpen(false)}
                    />
                    <motion.div
                      initial={{ opacity: 0, y: 8, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8, scale: 0.95 }}
                      transition={{ duration: 0.15, ease: "easeOut" }}
                      className="absolute right-0 top-full mt-2 w-64 bg-zinc-950/90 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl z-50 overflow-hidden"
                    >
                      {/* Grid Pattern Background */}
                      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

                      <div className="relative z-10 p-1">
                        <div className="px-3 py-2.5 border-b border-white/5 bg-white/[0.02] rounded-t-lg">
                          <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
                            Current Patient
                          </p>
                          <p className="text-[13px] font-medium text-white truncate">
                            {activePatient.full_name}
                          </p>
                          <div className="flex items-center gap-2 mt-0.5">
                            {activePatient.phone_number && (
                              <div className="flex items-center gap-1 text-[11px] text-zinc-400">
                                <Phone size={10} />
                                <span>{activePatient.phone_number}</span>
                              </div>
                            )}
                            <div className="flex items-center gap-1 text-[11px] text-zinc-500">
                              <Hash size={10} />
                              <span className="font-mono">
                                ID-{activePatient.id.slice(0, 4)}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="p-1 space-y-0.5">
                          <button
                            onClick={() => {
                              setIsPatientDropdownOpen(false);
                              setIsNewPatientModalOpen(true);
                            }}
                            className="w-full flex items-center gap-2 px-2 py-2 text-[12px] text-zinc-300 hover:text-white hover:bg-white/10 rounded-md transition-colors text-left"
                          >
                            <Plus size={14} />
                            Add New Patient
                          </button>

                          <button
                            onClick={() => {
                              setActivePatient(null);
                              setIsPatientDropdownOpen(false);
                            }}
                            className="w-full flex items-center gap-2 px-2 py-2 text-[12px] text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-md transition-colors text-left"
                          >
                            <LogOut size={14} />
                            Clear Selection
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <button
              onClick={() => setIsNewPatientModalOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-md transition-all text-zinc-400 hover:text-white hover:bg-white/10"
              title="Add New Patient"
            >
              <UserPlus size={16} strokeWidth={1.5} />
            </button>
          )}
        </div>

        {/* Search Trigger */}
        {/* Search Trigger */}
        {showSearch && (
          <button
            onClick={onSearchClick}
            className="flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-white/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white"
            aria-label="Open search (⌘K)"
            data-testid="search-trigger"
          >
            <Search
              size={16}
              strokeWidth={1.5}
              className="text-zinc-300"
              aria-hidden="true"
            />
          </button>
        )}

        {/* Notifications */}
        <div className="relative" data-notifications-panel>
          <button
            onClick={() => setNotificationsOpen(!notificationsOpen)}
            className="flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-white/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white"
            aria-label={`Notifications (${notificationCount} unread)`}
            aria-expanded={notificationsOpen}
            aria-haspopup="dialog"
            aria-controls="notifications-panel"
            data-testid="notifications-trigger"
          >
            <Bell
              size={16}
              strokeWidth={1.5}
              className="text-zinc-300"
              aria-hidden="true"
            />
            {notificationCount > 0 && (
              <span
                className="absolute top-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-white text-[9px] font-bold text-black border border-[#27272a]"
                aria-hidden="true"
                data-testid="notification-badge"
              >
                {notificationCount > 9 ? "9+" : notificationCount}
              </span>
            )}
          </button>

          {/* Notifications Panel - Dark theme design */}
          <div
            id="notifications-panel"
            className={`
                            absolute right-0 top-10 z-50 w-80 max-h-[420px]
                            rounded-xl bg-zinc-900 border border-zinc-700
                            shadow-2xl shadow-black/50 overflow-hidden
                            transition-all duration-200 ease-out
                            origin-top-right
                            ${
                              notificationsOpen
                                ? "opacity-100 scale-100"
                                : "opacity-0 scale-95 pointer-events-none"
                            }
                        `}
            role="dialog"
            aria-label="Notifications"
            aria-modal="false"
            aria-hidden={!notificationsOpen}
            data-testid="notifications-panel"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-zinc-100">
                  Notifications
                </span>
                {notificationCount > 0 && (
                  <span className="px-1.5 py-0.5 text-[10px] font-medium bg-blue-500/20 text-blue-400 rounded">
                    {notificationCount} new
                  </span>
                )}
              </div>
              <button
                onClick={() => setNotificationsOpen(false)}
                className="flex h-6 w-6 items-center justify-center rounded hover:bg-zinc-800 transition-colors"
                aria-label="Close"
              >
                <X size={14} className="text-zinc-400" aria-hidden="true" />
              </button>
            </div>

            {/* Content */}
            <div className="max-h-[320px] overflow-y-auto scrollbar-hide p-2">
              <Suspense
                fallback={
                  <div
                    className="space-y-2 p-2"
                    role="status"
                    aria-label="Loading"
                  >
                    {[1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className="animate-pulse rounded-lg bg-zinc-800 p-3"
                      >
                        <div className="flex items-start gap-2">
                          <div className="h-6 w-6 rounded-full bg-zinc-700" />
                          <div className="flex-1 space-y-1.5">
                            <div className="h-3 w-3/4 rounded bg-zinc-700" />
                            <div className="h-2.5 w-1/2 rounded bg-zinc-700" />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                }
              >
                <AIInsightsPanel maxInsights={4} />
              </Suspense>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-4 py-2.5 border-t border-zinc-800 bg-zinc-900/80">
              <button
                className="text-[12px] font-medium text-blue-400 hover:text-blue-300 transition-colors"
                onClick={() => setNotificationsOpen(false)}
              >
                View all
              </button>
              <button
                className="text-[11px] text-zinc-500 hover:text-zinc-400 transition-colors"
                onClick={() => setNotificationsOpen(false)}
              >
                Mark all read
              </button>
            </div>
          </div>
        </div>

        {/* User Profile removed - relocated to sidebar */}
      </div>

      {/* Patient Modal */}
      <NewPatientModal
        isOpen={isNewPatientModalOpen}
        onClose={() => setIsNewPatientModalOpen(false)}
      />
    </header>
  );
}

export default DashboardHeader;
