/**
 * Dashboard Components Index
 *
 * Exports all dashboard-level components:
 * - Layout: Sidebar, Header, CommandPalette
 * - Home Page: DiagnosticGrid, StatusCards, etc.
 * - Clinical: AlertsPanel, PipelineHealth
 */

// Layout Components
export { DashboardSidebar, SIDEBAR_COLLAPSED_KEY } from "./DashboardSidebar";
export type { SidebarItem, SidebarGroup } from "./DashboardSidebar";
export { DashboardHeader } from "./DashboardHeader";
export { CommandPalette } from "./CommandPalette";

// Home Page Components
export { default as DiagnosticGrid } from "./DiagnosticGrid";
export { default as DiagnosticCard } from "./DiagnosticCard";
export { default as SystemStatusCards } from "./SystemStatusCards";
export { default as UserHealthOverview } from "./UserHealthOverview";
export { default as QuickActionButtons } from "./QuickActionButtons";
export { default as RecentActivityFeed } from "./RecentActivityFeed";

// Clinical Dashboard Components
export { default as AlertsPanel } from "./AlertsPanel";
export { default as PipelineHealthIndicator } from "./PipelineHealthIndicator";
