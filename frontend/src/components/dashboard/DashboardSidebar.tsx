'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
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
} from 'lucide-react';
import { useLogout } from '@/hooks/useLogout';

// Sidebar item interface
interface SidebarItem {
    id: string;
    label: string;
    icon: LucideIcon;
    route: string;
    badge?: number | 'new';
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
        id: 'overview',
        label: 'Overview',
        items: [
            { id: 'dashboard', label: 'Dashboard', icon: Home, route: '/dashboard' },
        ],
    },
    {
        id: 'diagnostics',
        label: 'Diagnostics',
        items: [
            { id: 'speech', label: 'Speech Analysis', icon: Mic, route: '/dashboard/speech' },
            { id: 'retinal', label: 'Retinal Imaging', icon: Eye, route: '/dashboard/retinal' },
            { id: 'motor', label: 'Motor Assessment', icon: Hand, route: '/dashboard/motor' },
            { id: 'cognitive', label: 'Cognitive Testing', icon: Brain, route: '/dashboard/cognitive' },
            { id: 'multimodal', label: 'Multi-Modal', icon: Activity, route: '/dashboard/multimodal' },
            { id: 'nri-fusion', label: 'NRI Fusion', icon: Zap, route: '/dashboard/nri-fusion' },
        ],
    },
    {
        id: 'insights',
        label: 'Insights',
        items: [
            { id: 'analytics', label: 'Analytics', icon: BarChart3, route: '/dashboard/analytics' },
            { id: 'reports', label: 'Reports', icon: FileText, route: '/dashboard/reports' },
        ],
    },
    {
        id: 'account',
        label: 'Account',
        items: [
            { id: 'settings', label: 'Settings', icon: Settings, route: '/dashboard/settings' },
        ],
    },
];

// Flat list of all sidebar items for backwards compatibility
const sidebarItems = sidebarGroups.flatMap(group => group.items);

// LocalStorage key for sidebar state persistence
const SIDEBAR_COLLAPSED_KEY = 'medilens-sidebar-collapsed';

export interface DashboardSidebarProps {
    className?: string;
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
        <div className={collapsed ? 'mb-3' : 'mb-4'}>
            {/* Group Label - minimal style */}
            {!collapsed && (
                <div className="px-3 mb-1.5 text-[10px] font-medium uppercase tracking-widest text-[#9ca3af]">
                    {group.label}
                </div>
            )}

            {/* Group Items */}
            <ul className="space-y-0.5" role="list">
                {group.items.map(item => {
                    const Icon = item.icon;
                    const isActive = isActiveRoute(item.route);
                    const showTooltip = collapsed && hoveredItem === item.id;

                    return (
                        <li key={item.id}>
                            <Tooltip content={item.label} visible={showTooltip}>
                                <Link
                                    href={item.route}
                                    className={`
                                        flex items-center w-full rounded-lg transition-all duration-150
                                        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3b82f6]/50
                                        ${collapsed
                                            ? 'justify-center px-0 py-2.5 mx-auto w-10 h-10'
                                            : 'gap-2.5 px-3 py-2'
                                        }
                                        ${isActive
                                            ? 'bg-[#f0f9ff] text-[#0369a1]'
                                            : 'text-[#4b5563] hover:bg-[#f9fafb] hover:text-[#111827]'
                                        }
                                    `}
                                    aria-current={isActive ? 'page' : undefined}
                                    aria-label={collapsed ? item.label : undefined}
                                    onMouseEnter={() => setHoveredItem(item.id)}
                                    onMouseLeave={() => setHoveredItem(null)}
                                    data-testid={`sidebar-item-${item.id}`}
                                    data-active={isActive}
                                >
                                    <Icon
                                        size={18}
                                        strokeWidth={isActive ? 2 : 1.5}
                                        className={isActive ? 'text-[#0369a1]' : 'text-[#6b7280]'}
                                        aria-hidden="true"
                                    />
                                    {!collapsed && (
                                        <span className={`text-[13px] ${isActive ? 'font-medium' : 'font-normal'}`}>
                                            {item.label}
                                        </span>
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

export function DashboardSidebar({ className = '' }: DashboardSidebarProps) {
    const pathname = usePathname();
    const { isLoggingOut, error: logoutError, logout, clearError } = useLogout();

    // Sidebar collapse state with localStorage persistence
    const [collapsed, setCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const [isClient, setIsClient] = useState(false);
    const [hoveredItem, setHoveredItem] = useState<string | null>(null);
    const [hoveredBottomItem, setHoveredBottomItem] = useState<string | null>(null);

    // Initialize client-side state after hydration
    useEffect(() => {
        setIsClient(true);
        if (typeof window !== 'undefined') {
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
        if (isClient && typeof window !== 'undefined') {
            localStorage.setItem(SIDEBAR_COLLAPSED_KEY, JSON.stringify(newState));
        }
    }, [collapsed, isClient]);

    // Toggle mobile menu
    const toggleMobileMenu = useCallback(() => {
        setMobileOpen(prev => !prev);
    }, []);

    // Close mobile menu on route change
    useEffect(() => {
        setMobileOpen(false);
    }, [pathname]);

    // Close mobile menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            const target = event.target as Element;
            if (mobileOpen && !target.closest('[data-sidebar]')) {
                setMobileOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [mobileOpen]);

    // Check if a route is active
    const isActiveRoute = (route: string) => {
        if (route === '/dashboard') {
            return pathname === '/dashboard';
        }
        return pathname.startsWith(route);
    };

    return (
        <>
            {/* Mobile Menu Button - minimal */}
            <button
                onClick={toggleMobileMenu}
                className="fixed left-4 top-4 z-50 flex h-9 w-9 items-center justify-center rounded-lg bg-white border border-[#e5e7eb] shadow-sm lg:hidden hover:bg-[#f9fafb] transition-colors"
                aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
                aria-expanded={mobileOpen}
                aria-controls="main-navigation"
            >
                {mobileOpen ? (
                    <X size={18} className="text-[#374151]" aria-hidden="true" />
                ) : (
                    <Menu size={18} className="text-[#374151]" aria-hidden="true" />
                )}
            </button>

            {/* Mobile Overlay */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm lg:hidden"
                    onClick={() => setMobileOpen(false)}
                    aria-hidden="true"
                    data-testid="mobile-overlay"
                />
            )}

            {/* Sidebar - Clean minimal design */}
            <aside
                id="main-navigation"
                data-sidebar
                data-collapsed={collapsed}
                className={`
                    fixed inset-y-0 left-0 z-50 flex flex-col bg-white border-r border-[#f0f0f0]
                    transition-all duration-200 ease-out
                    ${collapsed ? 'w-[60px]' : 'w-[240px]'}
                    ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
                    ${className}
                `}
                role="navigation"
                aria-label="Dashboard navigation"
            >
                {/* Header - Logo area */}
                <div className={`flex items-center h-14 border-b border-[#f0f0f0] ${collapsed ? 'justify-center px-2' : 'justify-between px-4'}`}>
                    {!collapsed ? (
                        <>
                            <Link href="/dashboard" className="flex items-center gap-2.5" aria-label="MediLens Home">
                                <span className="text-[15px] font-semibold text-[#0f172a]">M</span>
                                <span className="text-[14px] font-medium text-[#334155]">MediLens</span>
                            </Link>
                            <button
                                onClick={toggleCollapse}
                                className="hidden lg:flex h-7 w-7 items-center justify-center rounded-md hover:bg-[#f3f4f6] transition-colors"
                                aria-label="Collapse sidebar"
                            >
                                <ChevronLeft size={16} className="text-[#9ca3af]" />
                            </button>
                        </>
                    ) : (
                        <button
                            onClick={toggleCollapse}
                            className="flex items-center justify-center"
                            aria-label="Expand sidebar"
                        >
                            <span className="text-[15px] font-semibold text-[#0f172a]">M</span>
                        </button>
                    )}
                </div>

                {/* Navigation */}
                <nav className={`flex-1 overflow-y-auto scrollbar-hide ${collapsed ? 'px-2 py-3' : 'px-3 py-4'}`} aria-label="Dashboard modules">
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
                                <div className={`border-t border-[#f0f0f0] ${collapsed ? 'mx-1 my-2' : 'mx-2 my-3'}`} aria-hidden="true" />
                            )}
                        </div>
                    ))}
                </nav>

                {/* Bottom Section - Profile & Logout */}
                <div className="border-t border-[#f0f0f0]">
                    {/* Profile */}
                    <Tooltip content="Profile" visible={collapsed && hoveredBottomItem === 'profile'}>
                        <button
                            className={`
                                flex items-center w-full transition-colors hover:bg-[#f9fafb]
                                ${collapsed ? 'justify-center px-2 py-2.5' : 'gap-2.5 px-4 py-2.5'}
                            `}
                            aria-label="User profile"
                            onMouseEnter={() => setHoveredBottomItem('profile')}
                            onMouseLeave={() => setHoveredBottomItem(null)}
                        >
                            <User size={18} strokeWidth={1.5} className="text-[#6b7280]" />
                            {!collapsed && <span className="text-[13px] text-[#4b5563]">Profile</span>}
                        </button>
                    </Tooltip>

                    {/* Logout */}
                    <Tooltip content={isLoggingOut ? 'Logging out...' : 'Logout'} visible={collapsed && hoveredBottomItem === 'logout'}>
                        <button
                            onClick={logout}
                            disabled={isLoggingOut}
                            className={`
                                flex items-center w-full transition-colors
                                ${collapsed ? 'justify-center px-2 py-2.5' : 'gap-2.5 px-4 py-2.5'}
                                ${isLoggingOut ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#fef2f2]'}
                            `}
                            aria-label={isLoggingOut ? 'Logging out...' : 'Logout'}
                            onMouseEnter={() => setHoveredBottomItem('logout')}
                            onMouseLeave={() => setHoveredBottomItem(null)}
                        >
                            <LogOut size={18} strokeWidth={1.5} className="text-[#ef4444]" />
                            {!collapsed && (
                                <span className="text-[13px] text-[#ef4444]">
                                    {isLoggingOut ? 'Logging out...' : 'Logout'}
                                </span>
                            )}
                        </button>
                    </Tooltip>

                    {/* Error display */}
                    {logoutError && !collapsed && (
                        <div className="px-4 py-2 text-xs text-[#ef4444]">
                            {logoutError}
                            <button onClick={clearError} className="ml-2 underline hover:no-underline">
                                Dismiss
                            </button>
                        </div>
                    )}
                </div>
            </aside>
        </>
    );
}

// Export sidebar items and groups for use in other components
export { sidebarItems, sidebarGroups, SIDEBAR_COLLAPSED_KEY };
export type { SidebarItem, SidebarGroup };

export default DashboardSidebar;
