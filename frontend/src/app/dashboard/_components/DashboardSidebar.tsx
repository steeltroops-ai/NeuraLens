"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Mic,
  Eye,
  Hand,
  Brain,
  Activity,
  Zap,
  BarChart3,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  LogOut,
  User,
  LucideIcon,
  Scan,
  Stethoscope,
  Heart,
  Sparkles,
  Microscope,
  Wind,
  Footprints,
  Bone,
} from "lucide-react";
import { UserButton, useUser } from "@clerk/nextjs";
import { Logo } from "@/components/common/Logo";

// Sidebar item interface
interface SidebarItem {
  id: string;
  label: string;
  icon: LucideIcon;
  route: string;
  badge?: number | "new";
}

// Sidebar group interface for visual grouping
interface SidebarGroup {
  id: string;
  label: string;
  items: SidebarItem[];
}

// Grouped navigation structure - minimal professional design
const sidebarGroups: SidebarGroup[] = [
  {
    id: "overview",
    label: "Overview",
    items: [
      { id: "dashboard", label: "Dashboard", icon: Home, route: "/dashboard" },
    ],
  },
  {
    id: "diagnostics",
    label: "AI Diagnostics",
    items: [
      // Core Diagnostic Modules (Live)
      {
        id: "retinal",
        label: "RetinaScan AI",
        icon: Eye,
        route: "/dashboard/retinal",
      },
      {
        id: "radiology",
        label: "ChestXplorer AI",
        icon: Scan,
        route: "/dashboard/radiology",
      },
      {
        id: "cardiology",
        label: "CardioPredict AI",
        icon: Heart,
        route: "/dashboard/cardiology",
      },
      {
        id: "speech",
        label: "SpeechMD AI",
        icon: Mic,
        route: "/dashboard/speech",
      },
      // Extended Diagnostic Modules (Live)
      {
        id: "dermatology",
        label: "SkinSense AI",
        icon: Sparkles,
        route: "/dashboard/dermatology",
      },
      {
        id: "motor",
        label: "Motor Assessment",
        icon: Hand,
        route: "/dashboard/motor",
      },
      {
        id: "cognitive",
        label: "Cognitive Testing",
        icon: Brain,
        route: "/dashboard/cognitive",
      },
      {
        id: "multimodal",
        label: "Multi-Modal",
        icon: Activity,
        route: "/dashboard/multimodal",
      },
      {
        id: "nri-fusion",
        label: "NRI Fusion",
        icon: Zap,
        route: "/dashboard/nri-fusion",
      },
    ],
  },
  {
    id: "coming-soon",
    label: "Coming Soon",
    items: [
      {
        id: "pathology",
        label: "HistoVision AI",
        icon: Microscope,
        route: "/dashboard/pathology",
      },
      {
        id: "neurology",
        label: "NeuroScan AI",
        icon: Brain,
        route: "/dashboard/neurology",
      },
      {
        id: "pulmonology",
        label: "RespiRate AI",
        icon: Wind,
        route: "/dashboard/pulmonology",
      },
      {
        id: "diabetic-foot",
        label: "FootCare AI",
        icon: Footprints,
        route: "/dashboard/diabetic-foot",
      },
      {
        id: "orthopedics",
        label: "BoneScan AI",
        icon: Bone,
        route: "/dashboard/orthopedics",
      },
    ],
  },
  {
    id: "insights",
    label: "Insights",
    items: [
      {
        id: "analytics",
        label: "Analytics",
        icon: BarChart3,
        route: "/dashboard/analytics",
      },
      {
        id: "reports",
        label: "Reports",
        icon: FileText,
        route: "/dashboard/reports",
      },
    ],
  },
  {
    id: "account",
    label: "Account",
    items: [
      {
        id: "settings",
        label: "Settings",
        icon: Settings,
        route: "/dashboard/settings",
      },
    ],
  },
];

// Flat list of all sidebar items for backwards compatibility
const sidebarItems = sidebarGroups.flatMap((group) => group.items);

// LocalStorage key for sidebar state persistence
const SIDEBAR_COLLAPSED_KEY = "medilens-sidebar-collapsed";

export interface DashboardSidebarProps {
  className?: string;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
}

// Minimal tooltip for collapsed state
interface TooltipProps {
  content: string;
  children: React.ReactNode;
  visible: boolean;
}

function Tooltip({ content, children, visible }: TooltipProps) {
  return (
    <div className="relative">
      {children}
      {visible && (
        <div
          className="absolute left-full ml-3 top-1/2 -translate-y-1/2 z-[1070] px-2.5 py-1.5 rounded-md bg-[#1a1a1a] text-white text-xs font-medium shadow-lg pointer-events-none whitespace-nowrap"
          role="tooltip"
          data-testid="sidebar-tooltip"
        >
          {content}
        </div>
      )}
    </div>
  );
}

// Minimal SidebarGroup component
interface SidebarGroupComponentProps {
  group: SidebarGroup;
  collapsed: boolean;
  isActiveRoute: (route: string) => boolean;
  hoveredItem: string | null;
  setHoveredItem: (id: string | null) => void;
}

function SidebarGroupComponent({
  group,
  collapsed,
  isActiveRoute,
  hoveredItem,
  setHoveredItem,
}: SidebarGroupComponentProps) {
  return (
    <div className={collapsed ? "mb-3" : "mb-4"}>
      {/* Group Label - Matches homepage card secondary text - Robotic/Technical */}
      {!collapsed && (
        <div className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 font-mono">
          {group.label}
        </div>
      )}

      {/* Group Items */}
      <ul className="space-y-0.5" role="list">
        {group.items.map((item) => {
          const Icon = item.icon;
          const isActive = isActiveRoute(item.route);
          const showTooltip = collapsed && hoveredItem === item.id;

          return (
            <li key={item.id}>
              <Tooltip content={item.label} visible={showTooltip}>
                <Link
                  href={item.route}
                  className={`
                                            flex items-center w-full rounded-md transition-all duration-200 group
                                            focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-500
                                            ${
                                              collapsed
                                                ? "justify-center px-0 py-2.5 mx-auto w-10 h-10"
                                                : "gap-3 px-3 py-2"
                                            }
                                            ${
                                              isActive
                                                ? "bg-[#18181b] text-white shadow-sm ring-1 ring-[#27272a]"
                                                : "text-zinc-400 hover:bg-white/10 hover:text-white"
                                            }
                                        `}
                  aria-current={isActive ? "page" : undefined}
                  aria-label={collapsed ? item.label : undefined}
                  onMouseEnter={() => setHoveredItem(item.id)}
                  onMouseLeave={() => setHoveredItem(null)}
                  data-testid={`sidebar-item-${item.id}`}
                  data-active={isActive}
                >
                  <Icon
                    size={16}
                    strokeWidth={1.5}
                    className={
                      isActive
                        ? "text-white"
                        : "text-zinc-500 group-hover:text-zinc-300"
                    }
                    aria-hidden="true"
                  />
                  {!collapsed && (
                    <>
                      <span
                        className={`text-[13px] font-mono tracking-tight flex-1 ${isActive ? "font-medium" : "font-normal"}`}
                      >
                        {item.label}
                      </span>
                      {item.badge === "new" && (
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wider bg-blue-500/20 text-blue-400 border border-blue-500/30">
                          New
                        </span>
                      )}
                    </>
                  )}
                </Link>
              </Tooltip>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function DashboardSidebar({
  className = "",
  mobileOpen,
  setMobileOpen,
}: DashboardSidebarProps) {
  const pathname = usePathname();
  const { user } = useUser();

  // Sidebar collapse state with localStorage persistence
  const [collapsed, setCollapsed] = useState(false);
  // mobileOpen state is now controlled by parent
  const [isClient, setIsClient] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [hoveredBottomItem, setHoveredBottomItem] = useState<string | null>(
    null,
  );

  // Initialize client-side state after hydration
  useEffect(() => {
    setIsClient(true);
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
      if (saved !== null) {
        setCollapsed(JSON.parse(saved));
      }
    }
  }, []);

  // Toggle sidebar collapse with localStorage persistence
  const toggleCollapse = useCallback(() => {
    const newState = !collapsed;
    setCollapsed(newState);
    if (isClient && typeof window !== "undefined") {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, JSON.stringify(newState));
    }
  }, [collapsed, isClient]);

  // Toggle mobile menu
  const toggleMobileMenu = useCallback(() => {
    setMobileOpen(!mobileOpen);
  }, [mobileOpen, setMobileOpen]);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Close mobile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Element;
      if (mobileOpen && !target.closest("[data-sidebar]")) {
        setMobileOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [mobileOpen]);

  // Check if a route is active
  const isActiveRoute = (route: string) => {
    if (route === "/dashboard") {
      return pathname === "/dashboard";
    }
    return pathname.startsWith(route);
  };

  return (
    <>
      {/* Mobile Menu Button - Removed (Integrated in Header) */}

      {/* Mobile Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
          data-testid="mobile-overlay"
        />
      )}

      <aside
        id="main-navigation"
        data-sidebar
        data-collapsed={collapsed}
        className={`
                    fixed inset-y-0 left-0 z-50 flex flex-col bg-black border-r border-[#27272a]
                    ${collapsed ? "w-[60px]" : "w-[240px]"}
                    ${mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
                    ${className}
                `}
        style={{
          transition:
            "width 350ms cubic-bezier(0.32, 0.72, 0, 1), transform 350ms cubic-bezier(0.32, 0.72, 0, 1)",
        }}
        role="navigation"
        aria-label="Dashboard navigation"
      >
        {/* Header - Logo area */}
        <div
          className={`flex items-center h-14 border-b border-[#27272a] ${collapsed ? "justify-center" : "justify-between px-3"}`}
        >
          {!collapsed ? (
            <>
              <Link
                href="/dashboard"
                className="flex items-center gap-3 px-3 py-2"
                aria-label="MediLens Home"
              >
                <Logo showText={true} size="md" />
              </Link>
              <button
                onClick={toggleCollapse}
                className="hidden lg:flex h-7 w-7 items-center justify-center rounded-md hover:bg-white/5 transition-colors"
                aria-label="Collapse sidebar"
              >
                <ChevronLeft size={16} className="text-zinc-400" />
              </button>
            </>
          ) : (
            <button
              onClick={toggleCollapse}
              className="flex items-center justify-center w-10 h-10"
              aria-label="Expand sidebar"
            >
              <Logo showText={false} size="md" />
            </button>
          )}
        </div>

        {/* Navigation */}
        <nav
          className={`flex-1 overflow-y-auto scrollbar-hide ${collapsed ? "px-2 py-3" : "px-3 py-4"}`}
          aria-label="Dashboard modules"
        >
          {sidebarGroups.map((group, index) => (
            <div key={group.id}>
              <SidebarGroupComponent
                group={group}
                collapsed={collapsed}
                isActiveRoute={isActiveRoute}
                hoveredItem={hoveredItem}
                setHoveredItem={setHoveredItem}
              />
              {/* Subtle divider */}
              {index < sidebarGroups.length - 1 && (
                <div
                  className={`border-t border-[#27272a] ${collapsed ? "mx-1 my-2" : "mx-2 my-3"}`}
                  aria-hidden="true"
                />
              )}
            </div>
          ))}
        </nav>

        {/* Bottom Section - User Profile */}
        <div className="border-t border-[#27272a] p-3">
          <div
            className={`flex items-center ${collapsed ? "justify-center" : "gap-3"}`}
          >
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  avatarBox: "h-8 w-8",
                  userButtonTrigger: "focus:shadow-none focus:outline-none",
                },
              }}
            />

            {!collapsed && user && (
              <div className="flex flex-col overflow-hidden">
                <span className="text-sm font-medium text-zinc-200 truncate">
                  {user.fullName || user.username || "User"}
                </span>
                <span className="text-xs text-zinc-500 truncate">
                  {user.primaryEmailAddress?.emailAddress}
                </span>
              </div>
            )}

            {!collapsed && !user && (
              <div className="flex flex-col">
                <span className="text-sm font-medium text-zinc-200">Guest</span>
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}

// Export sidebar items and groups for use in other components
export { sidebarItems, sidebarGroups, SIDEBAR_COLLAPSED_KEY };
export type { SidebarItem, SidebarGroup };

export default DashboardSidebar;
