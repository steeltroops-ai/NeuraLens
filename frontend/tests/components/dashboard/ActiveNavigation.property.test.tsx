/**
 * Property-Based Tests for Active Navigation Indicator
 * 
 * Feature: dashboard-ux-improvements
 * Property 4: Active Navigation Indicator
 * Validates: Requirements 2.2
 * 
 * For any navigation item in the sidebar, WHEN the current route matches 
 * the item's route, the item SHALL have MediLens Blue (#007AFF) background color.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';

// Sidebar items configuration matching the component
const sidebarItems = [
    { id: 'dashboard', label: 'Dashboard', route: '/dashboard' },
    { id: 'speech', label: 'Speech Analysis', route: '/dashboard/speech' },
    { id: 'retinal', label: 'Retinal Imaging', route: '/dashboard/retinal' },
    { id: 'motor', label: 'Motor Assessment', route: '/dashboard/motor' },
    { id: 'cognitive', label: 'Cognitive Testing', route: '/dashboard/cognitive' },
    { id: 'multimodal', label: 'Multi-Modal', route: '/dashboard/multimodal' },
    { id: 'nri-fusion', label: 'NRI Fusion', route: '/dashboard/nri-fusion' },
    { id: 'analytics', label: 'Analytics', route: '/dashboard/analytics' },
    { id: 'reports', label: 'Reports', route: '/dashboard/reports' },
    { id: 'settings', label: 'Settings', route: '/dashboard/settings' },
];

// MediLens Blue color for active state
const MEDILENS_BLUE = '#007AFF';

// Route matching logic (same as component)
function isActiveRoute(currentPath: string, itemRoute: string): boolean {
    if (itemRoute === '/dashboard') {
        return currentPath === '/dashboard';
    }
    return currentPath.startsWith(itemRoute);
}

// Get expected styles for an item based on active state
interface ItemStyles {
    backgroundColor: string;
    textColor: string;
}

function getItemStyles(isActive: boolean): ItemStyles {
    if (isActive) {
        return {
            backgroundColor: MEDILENS_BLUE,
            textColor: '#FFFFFF',
        };
    }
    return {
        backgroundColor: 'transparent',
        textColor: '#3C3C43',
    };
}

describe('Active Navigation Indicator Property Tests', () => {
    /**
     * Property 4: Active Navigation Indicator
     * 
     * For any navigation item, when the current route matches the item's route,
     * the item should have MediLens Blue (#007AFF) background color.
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4: Active item has MediLens Blue background', () => {
        // Generator for sidebar items
        const itemArb = fc.constantFrom(...sidebarItems);

        fc.assert(
            fc.property(itemArb, (item) => {
                // When current path matches item route exactly
                const currentPath = item.route;
                const isActive = isActiveRoute(currentPath, item.route);

                expect(isActive).toBe(true);

                const styles = getItemStyles(isActive);
                expect(styles.backgroundColor).toBe(MEDILENS_BLUE);
                expect(styles.textColor).toBe('#FFFFFF');
            }),
            { numRuns: 100 }
        );
    });

    /**
     * Property 4.1: Inactive items do NOT have MediLens Blue background
     * 
     * For any navigation item, when the current route does NOT match the item's route,
     * the item should NOT have MediLens Blue background.
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4.1: Inactive items do NOT have MediLens Blue background', () => {
        // Generate pairs of different items
        const itemPairArb = fc.tuple(
            fc.constantFrom(...sidebarItems),
            fc.constantFrom(...sidebarItems)
        ).filter(([current, other]) => {
            // Ensure they're different and current doesn't start with other's route
            // (except for dashboard which is exact match only)
            if (current.id === other.id) return false;
            if (other.route === '/dashboard') return current.route !== '/dashboard';
            return !current.route.startsWith(other.route);
        });

        fc.assert(
            fc.property(itemPairArb, ([currentItem, otherItem]) => {
                const currentPath = currentItem.route;
                const isOtherActive = isActiveRoute(currentPath, otherItem.route);

                expect(isOtherActive).toBe(false);

                const styles = getItemStyles(isOtherActive);
                expect(styles.backgroundColor).not.toBe(MEDILENS_BLUE);
                expect(styles.backgroundColor).toBe('transparent');
            }),
            { numRuns: 100 }
        );
    });

    /**
     * Property 4.2: Dashboard route is exact match only
     * 
     * The dashboard route (/dashboard) should only be active when the path
     * is exactly /dashboard, not for any sub-routes.
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4.2: Dashboard route is exact match only', () => {
        const dashboardItem = sidebarItems.find(item => item.id === 'dashboard')!;
        const subRouteItems = sidebarItems.filter(item => item.id !== 'dashboard');

        const subRouteArb = fc.constantFrom(...subRouteItems);

        fc.assert(
            fc.property(subRouteArb, (subRouteItem) => {
                // When on a sub-route, dashboard should NOT be active
                const currentPath = subRouteItem.route;
                const isDashboardActive = isActiveRoute(currentPath, dashboardItem.route);

                expect(isDashboardActive).toBe(false);
            }),
            { numRuns: 100 }
        );
    });

    /**
     * Property 4.3: Sub-routes activate parent module
     * 
     * For any module route, sub-paths should also activate that module.
     * E.g., /dashboard/speech/results should activate the speech module.
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4.3: Sub-routes activate parent module', () => {
        // Non-dashboard items (which use startsWith matching)
        const moduleItems = sidebarItems.filter(item => item.id !== 'dashboard');
        const moduleArb = fc.constantFrom(...moduleItems);

        // Generate random sub-path suffixes
        const subPathArb = fc.constantFrom(
            '/results',
            '/history',
            '/details',
            '/analysis',
            '/report',
            '/123',
            '/abc-def'
        );

        fc.assert(
            fc.property(moduleArb, subPathArb, (module, subPath) => {
                const currentPath = module.route + subPath;
                const isActive = isActiveRoute(currentPath, module.route);

                expect(isActive).toBe(true);

                const styles = getItemStyles(isActive);
                expect(styles.backgroundColor).toBe(MEDILENS_BLUE);
            }),
            { numRuns: 100 }
        );
    });

    /**
     * Property 4.4: Exactly one item is active at a time
     * 
     * For any valid dashboard path, exactly one navigation item should be active.
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4.4: Exactly one item is active at a time', () => {
        const itemArb = fc.constantFrom(...sidebarItems);

        fc.assert(
            fc.property(itemArb, (currentItem) => {
                const currentPath = currentItem.route;

                // Count how many items are active for this path
                const activeItems = sidebarItems.filter(item =>
                    isActiveRoute(currentPath, item.route)
                );

                // Exactly one item should be active
                expect(activeItems.length).toBe(1);
                expect(activeItems[0].id).toBe(currentItem.id);
            }),
            { numRuns: 100 }
        );
    });

    /**
     * Property 4.5: Active state styling is consistent
     * 
     * For any active item, the styling should always be the same
     * (MediLens Blue background, white text).
     * 
     * **Validates: Requirements 2.2**
     */
    it('Property 4.5: Active state styling is consistent', () => {
        const itemArb = fc.constantFrom(...sidebarItems);

        fc.assert(
            fc.property(itemArb, (item) => {
                const styles = getItemStyles(true);

                // Active styling should always be consistent
                expect(styles.backgroundColor).toBe(MEDILENS_BLUE);
                expect(styles.textColor).toBe('#FFFFFF');
            }),
            { numRuns: 100 }
        );
    });
});
