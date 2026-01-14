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
    Menu,
    X,
    LogOut,
    UserCircle,
} from 'lucide-react';
import { useLogout } from '@/hooks/useLogout';

// Sidebar navigation items configuration
const sidebarItems = [
    { id: 'overview', label: 'Dashboard', icon: Home, route: '/dashboard' },
    { id: 'speech', label: 'Speech Analysis', icon: Mic, route: '/dashboard/speech' },
    { id: 'retinal', label: 'Retinal Imaging', icon: Eye, route: '/dashboard/retinal' },
    { id: 'motor', label: 'Motor Assessment', icon: Hand, route: '/dashboard/motor' },
    { id: 'cognitive', label: 'Cognitive Testing', icon: Brain, route: '/dashboard/cognitive' },
    { id: 'multimodal', label: 'Multi-Modal', icon: Activity, route: '/dashboard/multimodal' },
    { id: 'nri-fusion', label: 'NRI Fusion', icon: Zap, route: '/dashboard/nri-fusion' },
    { id: 'analytics', label: 'Analytics', icon: BarChart3, route: '/dashboard/analytics' },
    { id: 'reports', label: 'Reports', icon: FileText, route: '/dashboard/reports' },
    { id: 'settings', label: 'Settings', icon: Settings, route: '/dashboard/settings' },
];

// LocalStorage key for sidebar state persistence
const SIDEBAR_COLLAPSED_KEY = 'medilens-sidebar-collapsed';

export interface DashboardSidebarProps {
    className?: string;
}

export function DashboardSidebar({ className = '' }: DashboardSidebarProps) {
    const pathname = usePathname();
    const { isLoggingOut, error: logoutError, logout, clearError } = useLogout();

    // Sidebar collapse state with localStorage persistence
    const [collapsed, setCollapsed] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const [isClient, setIsClient] = useState(false);

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

    // Get current page title
    const getCurrentPageTitle = () => {
        const currentItem = sidebarItems.find(item => isActiveRoute(item.route));
        return currentItem?.label || 'Dashboard';
    };

    // Icon style for consistent sizing
    const iconStyle = {
        width: '24px',
        height: '24px',
        minWidth: '24px',
        minHeight: '24px',
        flexShrink: 0,
    };

    return (
        <>
            {/* Mobile Hamburger Button - 48px minimum touch target */}
            <button
                onClick={toggleMobileMenu}
                className="fixed left-3 top-3 z-50 flex h-12 w-12 min-h-[48px] min-w-[48px] items-center justify-center rounded-xl bg-white shadow-lg lg:hidden focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40"
                aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
                aria-expanded={mobileOpen}
                aria-controls="main-navigation"
                style={{ touchAction: 'manipulation' }}
            >
                {mobileOpen ? (
                    <X className="h-6 w-6" style={{ color: '#1D1D1F' }} aria-hidden="true" />
                ) : (
                    <Menu className="h-6 w-6" style={{ color: '#1D1D1F' }} aria-hidden="true" />
                )}
            </button>

            {/* Mobile Overlay */}
            {mobileOpen && (
                <div
                    className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden"
                    onClick={() => setMobileOpen(false)}
                    aria-hidden="true"
                />
            )}

            {/* Sidebar */}
            <aside
                id="main-navigation"
                data-sidebar
                className={`
          fixed inset-y-0 left-0 z-50 flex flex-col border-r bg-white transition-all duration-200 ease-out-quint
          ${collapsed ? 'w-16' : 'w-[280px] max-w-[calc(100vw-48px)]'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          ${className}
        `}
                style={{ borderColor: '#F5F5F7' }}
                role="navigation"
                aria-label="Dashboard navigation"
            >
                {/* Logo Header */}
                <div
                    className="flex items-center border-b p-4 transition-all duration-200"
                    style={{ borderColor: '#F5F5F7', minHeight: '64px' }}
                >
                    {!collapsed ? (
                        <div className="flex w-full items-center justify-between">
                            <Link
                                href="/dashboard"
                                className="flex items-center space-x-3"
                                aria-label="MediLens Dashboard Home"
                            >
                                <div className="flex h-8 w-8 items-center justify-center">
                                    <span
                                        className="text-xl font-semibold"
                                        style={{
                                            color: '#1D1D1F',
                                            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                        }}
                                    >
                                        M
                                    </span>
                                </div>
                                <h1
                                    className="text-lg font-semibold"
                                    style={{
                                        color: '#1D1D1F',
                                        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                    }}
                                >
                                    MediLens
                                </h1>
                            </Link>

                            {/* Collapse Button */}
                            <button
                                onClick={toggleCollapse}
                                className="hidden h-8 w-8 items-center justify-center rounded transition-colors hover:bg-gray-100 lg:flex"
                                aria-label="Collapse sidebar"
                            >
                                <ChevronLeft style={{ width: '16px', height: '16px', color: '#86868B' }} />
                            </button>
                        </div>
                    ) : (
                        <button
                            onClick={toggleCollapse}
                            className="mx-auto flex h-8 w-8 items-center justify-center"
                            aria-label="Expand sidebar"
                        >
                            <span
                                className="text-xl font-semibold"
                                style={{
                                    color: '#1D1D1F',
                                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                }}
                            >
                                M
                            </span>
                        </button>
                    )}
                </div>

                {/* Navigation Menu */}
                <nav className="flex-1 overflow-y-auto p-4" style={{ paddingBottom: '120px' }} aria-label="Dashboard modules">
                    <ul className="space-y-1" role="list">
                        {sidebarItems.map(item => {
                            const Icon = item.icon;
                            const isActive = isActiveRoute(item.route);

                            return (
                                <li key={item.id}>
                                    <Link
                                        href={item.route}
                                        className={`
                      flex w-full items-center rounded-lg px-3 py-3 text-left transition-all duration-200
                      focus-visible:ring-3 focus-visible:ring-[#007AFF]/40 focus-visible:outline-none
                      ${collapsed ? 'justify-center' : 'space-x-4'}
                      ${isActive ? 'bg-[#007AFF] text-white' : 'text-[#1D1D1F] hover:bg-[#F5F5F7]'}
                    `}
                                        style={{ minHeight: '48px' }}
                                        aria-current={isActive ? 'page' : undefined}
                                        aria-label={collapsed ? `${item.label}${isActive ? ' (current page)' : ''}` : undefined}
                                    >
                                        <Icon style={{ ...iconStyle, color: isActive ? '#FFFFFF' : '#1D1D1F' }} aria-hidden="true" />
                                        {!collapsed && (
                                            <span
                                                className="text-sm font-medium"
                                                style={{
                                                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                                }}
                                            >
                                                {item.label}
                                            </span>
                                        )}
                                    </Link>
                                </li>
                            );
                        })}
                    </ul>
                </nav>

                {/* Bottom Section - Profile and Logout */}
                <div
                    className="absolute bottom-0 left-0 right-0 border-t"
                    style={{ borderColor: '#F5F5F7', backgroundColor: '#FFFFFF' }}
                >
                    {/* User Profile Button */}
                    <button
                        className={`
              flex w-full items-center px-3 py-3 text-left transition-all duration-200 hover:bg-[#F5F5F7]
              ${collapsed ? 'justify-center' : 'space-x-4'}
            `}
                        style={{
                            minHeight: '48px',
                            borderBottom: '1px solid #F5F5F7',
                        }}
                        aria-label="User profile"
                    >
                        <UserCircle style={{ ...iconStyle, color: '#1D1D1F' }} />
                        {!collapsed && (
                            <span
                                className="text-sm font-medium"
                                style={{
                                    color: '#1D1D1F',
                                    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                                }}
                            >
                                Profile
                            </span>
                        )}
                    </button>

                    {/* Logout Button */}
                    <button
                        onClick={logout}
                        disabled={isLoggingOut}
                        className={`
              flex w-full items-center px-3 py-3 text-left transition-all duration-200
              ${collapsed ? 'justify-center' : 'space-x-4'}
              ${isLoggingOut ? 'cursor-not-allowed opacity-50' : 'hover:bg-[#FFF5F5]'}
            `}
                        style={{ minHeight: '48px' }}
                        aria-label={isLoggingOut ? 'Logging out...' : 'Logout'}
                    >
                        <LogOut
                            style={{ ...iconStyle, color: '#FF3B30' }}
                            className={isLoggingOut ? 'animate-spin' : ''}
                        />
                        {!collapsed && (
                            <span
                                className="text-sm font-medium"
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
                    {logoutError && !collapsed && (
                        <div className="px-3 py-2 text-xs" style={{ color: '#FF3B30' }}>
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

// Export sidebar items for use in other components
export { sidebarItems, SIDEBAR_COLLAPSED_KEY };

export default DashboardSidebar;
