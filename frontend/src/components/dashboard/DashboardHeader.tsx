'use client';

import { useState, useEffect, Suspense } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Bell, User, X, ChevronRight, Search } from 'lucide-react';
import { sidebarItems, sidebarGroups } from './DashboardSidebar';
import dynamic from 'next/dynamic';

// Lazy load AI Insights Panel for notifications
const AIInsightsPanel = dynamic(() => import('@/components/dashboard/AIInsightsPanel'), {
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
}

/**
 * Dashboard Header Component
 * 
 * Minimal, professional design matching sidebar aesthetic
 * Height: 56px (h-14) - same as sidebar logo tab
 * 
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
 */
export function DashboardHeader({
    title,
    showSearch = true,
    onSearchClick
}: DashboardHeaderProps) {
    const pathname = usePathname();
    const [notificationsOpen, setNotificationsOpen] = useState(false);
    const [notificationCount] = useState(3);

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

    // Generate breadcrumbs from current route
    const getBreadcrumbs = (): Breadcrumb[] => {
        const breadcrumbs: Breadcrumb[] = [
            { label: 'Dashboard', href: '/dashboard' }
        ];

        if (pathname === '/dashboard') {
            return breadcrumbs;
        }

        const currentItem = sidebarItems.find(item => {
            if (item.route === '/dashboard') return false;
            return pathname.startsWith(item.route);
        });

        if (currentItem) {
            const parentGroup = sidebarGroups.find(group =>
                group.items.some(item => item.id === currentItem.id)
            );

            if (parentGroup && parentGroup.id !== 'overview') {
                breadcrumbs.push({ label: parentGroup.label });
            }

            breadcrumbs.push({ label: currentItem.label });
        }

        return breadcrumbs;
    };

    // Get current page title
    const getCurrentPageTitle = (): string => {
        if (title) return title;

        const currentItem = sidebarItems.find(item => {
            if (item.route === '/dashboard') {
                return pathname === '/dashboard';
            }
            return pathname.startsWith(item.route);
        });

        return currentItem?.label || 'Dashboard';
    };

    const breadcrumbs = getBreadcrumbs();
    const pageTitle = getCurrentPageTitle();

    return (
        <header
            className="sticky top-0 z-40 flex items-center justify-between px-4 lg:px-6 bg-white border-b border-[#f0f0f0] h-14"
            role="banner"
            aria-label="Dashboard header"
            data-testid="dashboard-header"
        >
            {/* Left Section: Breadcrumbs */}
            <div className="flex items-center min-w-0 pl-12 lg:pl-0">
                {/* Breadcrumb Navigation */}
                <nav className="flex items-center" aria-label="Breadcrumb">
                    <ol className="flex items-center">
                        {breadcrumbs.map((crumb, index) => (
                            <li key={index} className="flex items-center">
                                {index > 0 && (
                                    <ChevronRight
                                        size={14}
                                        className="mx-1.5 text-[#d1d5db]"
                                        aria-hidden="true"
                                    />
                                )}
                                {crumb.href && index < breadcrumbs.length - 1 ? (
                                    <Link
                                        href={crumb.href}
                                        className="text-[13px] text-[#9ca3af] hover:text-[#6b7280] transition-colors"
                                    >
                                        {crumb.label}
                                    </Link>
                                ) : (
                                    <span className={`text-[13px] ${index === breadcrumbs.length - 1 ? 'text-[#111827] font-medium' : 'text-[#9ca3af]'}`}>
                                        {crumb.label}
                                    </span>
                                )}
                            </li>
                        ))}
                    </ol>
                </nav>
            </div>

            {/* Right Section: Actions */}
            <div className="flex items-center gap-1" role="group" aria-label="User actions">
                {/* Search Trigger */}
                {showSearch && (
                    <button
                        onClick={onSearchClick}
                        className="flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-[#f3f4f6] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3b82f6]/50"
                        aria-label="Open search (⌘K)"
                        data-testid="search-trigger"
                    >
                        <Search size={16} strokeWidth={1.5} className="text-[#6b7280]" aria-hidden="true" />
                    </button>
                )}

                {/* Notifications */}
                <div className="relative" data-notifications-panel>
                    <button
                        onClick={() => setNotificationsOpen(!notificationsOpen)}
                        className="flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-[#f3f4f6] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3b82f6]/50"
                        aria-label={`Notifications (${notificationCount} unread)`}
                        aria-expanded={notificationsOpen}
                        aria-haspopup="dialog"
                        aria-controls="notifications-panel"
                        data-testid="notifications-trigger"
                    >
                        <Bell size={16} strokeWidth={1.5} className="text-[#6b7280]" aria-hidden="true" />
                        {notificationCount > 0 && (
                            <span
                                className="absolute top-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-[#ef4444] text-[10px] font-medium text-white"
                                aria-hidden="true"
                                data-testid="notification-badge"
                            >
                                {notificationCount > 9 ? '9+' : notificationCount}
                            </span>
                        )}
                    </button>

                    {/* Notifications Panel - Clean minimal design */}
                    <div
                        id="notifications-panel"
                        className={`
                            absolute right-0 top-10 z-50 w-80 max-h-[420px]
                            rounded-lg bg-white border border-[#e5e7eb]
                            shadow-lg overflow-hidden
                            transition-all duration-200 ease-out
                            origin-top-right
                            ${notificationsOpen
                                ? 'opacity-100 scale-100'
                                : 'opacity-0 scale-95 pointer-events-none'
                            }
                        `}
                        role="dialog"
                        aria-label="Notifications"
                        aria-modal="false"
                        aria-hidden={!notificationsOpen}
                        data-testid="notifications-panel"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between px-4 py-3 border-b border-[#f0f0f0]">
                            <div className="flex items-center gap-2">
                                <span className="text-[13px] font-medium text-[#111827]">Notifications</span>
                                {notificationCount > 0 && (
                                    <span className="px-1.5 py-0.5 text-[10px] font-medium bg-[#fef2f2] text-[#ef4444] rounded">
                                        {notificationCount} new
                                    </span>
                                )}
                            </div>
                            <button
                                onClick={() => setNotificationsOpen(false)}
                                className="flex h-6 w-6 items-center justify-center rounded hover:bg-[#f3f4f6] transition-colors"
                                aria-label="Close"
                            >
                                <X size={14} className="text-[#9ca3af]" aria-hidden="true" />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="max-h-[320px] overflow-y-auto scrollbar-hide p-2">
                            <Suspense
                                fallback={
                                    <div className="space-y-2 p-2" role="status" aria-label="Loading">
                                        {[1, 2, 3].map((i) => (
                                            <div key={i} className="animate-pulse rounded-lg bg-[#f9fafb] p-3">
                                                <div className="flex items-start gap-2">
                                                    <div className="h-6 w-6 rounded-full bg-[#e5e7eb]" />
                                                    <div className="flex-1 space-y-1.5">
                                                        <div className="h-3 w-3/4 rounded bg-[#e5e7eb]" />
                                                        <div className="h-2.5 w-1/2 rounded bg-[#e5e7eb]" />
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
                        <div className="flex items-center justify-between px-4 py-2.5 border-t border-[#f0f0f0] bg-[#fafafa]">
                            <button
                                className="text-[12px] font-medium text-[#3b82f6] hover:text-[#2563eb] transition-colors"
                                onClick={() => setNotificationsOpen(false)}
                            >
                                View all
                            </button>
                            <button
                                className="text-[11px] text-[#9ca3af] hover:text-[#6b7280] transition-colors"
                                onClick={() => setNotificationsOpen(false)}
                            >
                                Mark all read
                            </button>
                        </div>
                    </div>
                </div>

                {/* User Profile */}
                <button
                    className="flex h-8 w-8 items-center justify-center rounded-md transition-colors hover:bg-[#f3f4f6] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3b82f6]/50"
                    aria-label="Profile"
                    aria-haspopup="menu"
                    data-testid="user-profile-trigger"
                >
                    <User size={16} strokeWidth={1.5} className="text-[#6b7280]" aria-hidden="true" />
                </button>
            </div>
        </header>
    );
}

export default DashboardHeader;
