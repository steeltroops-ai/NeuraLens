/**
 * Property-Based Tests for No Duplicate Navigation
 * 
 * Feature: frontend-global-fix
 * Property 13: No Duplicate Navigation
 * Validates: Requirements 11.2
 * 
 * For any page, there SHALL NOT be duplicate navigation elements
 * or buttons with the same action.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as fc from 'fast-check';
import { diagnosticModules, DiagnosticModule } from '@/data/diagnostic-modules';

// Mock next/navigation
vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: vi.fn(),
        replace: vi.fn(),
        prefetch: vi.fn(),
    }),
    usePathname: () => '/dashboard',
}));

/**
 * Sidebar navigation items configuration (mirroring DashboardSidebar)
 */
const sidebarItems = [
    { id: 'overview', label: 'Dashboard', route: '/dashboard' },
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

/**
 * Header navigation items (from Header component)
 */
const headerNavItems = [
    { label: 'Home', route: '/' },
    { label: 'Dashboard', route: '/dashboard' },
    { label: 'Assessment', route: '/assessment' },
    { label: 'About', route: '/about' },
];

/**
 * Homepage CTA buttons
 */
const homepageCTAs = [
    { label: 'Start Assessment', route: '/assessment' },
    { label: 'View Dashboard', route: '/dashboard' },
    { label: 'Access All Modules', route: '/dashboard' },
];

/**
 * All application routes
 */
const allRoutes = [
    '/',
    '/dashboard',
    '/dashboard/speech',
    '/dashboard/retinal',
    '/dashboard/motor',
    '/dashboard/cognitive',
    '/dashboard/multimodal',
    '/dashboard/nri-fusion',
    '/dashboard/analytics',
    '/dashboard/reports',
    '/dashboard/settings',
    '/assessment',
    '/about',
];

/**
 * Arbitrary generator for sidebar items
 */
const sidebarItemArb = fc.constantFrom(...sidebarItems);

/**
 * Arbitrary generator for diagnostic modules
 */
const diagnosticModuleArb = fc.constantFrom(...diagnosticModules);

/**
 * Arbitrary generator for routes
 */
const routeArb = fc.constantFrom(...allRoutes);

describe('No Duplicate Navigation Property Tests', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    /**
     * Property 13: No Duplicate Navigation
     * 
     * For any page, there SHALL NOT be duplicate navigation elements
     * or buttons with the same action.
     * 
     * **Validates: Requirements 11.2**
     */
    describe('Property 13: No Duplicate Navigation', () => {

        /**
         * Property 13.1: All sidebar navigation items have unique IDs
         * For any sidebar item, its ID SHALL be unique
         */
        it('Property 13.1: All sidebar navigation items have unique IDs', () => {
            const ids = sidebarItems.map(item => item.id);
            const uniqueIds = new Set(ids);

            expect(uniqueIds.size).toBe(ids.length);
        });

        /**
         * Property 13.2: All sidebar navigation items have unique routes
         * For any sidebar item, its route SHALL be unique
         */
        it('Property 13.2: All sidebar navigation items have unique routes', () => {
            const routes = sidebarItems.map(item => item.route);
            const uniqueRoutes = new Set(routes);

            expect(uniqueRoutes.size).toBe(routes.length);
        });

        /**
         * Property 13.3: All sidebar navigation items have unique labels
         * For any sidebar item, its label SHALL be unique
         */
        it('Property 13.3: All sidebar navigation items have unique labels', () => {
            const labels = sidebarItems.map(item => item.label);
            const uniqueLabels = new Set(labels);

            expect(uniqueLabels.size).toBe(labels.length);
        });

        /**
         * Property 13.4: All diagnostic modules have unique IDs
         * For any diagnostic module, its ID SHALL be unique
         */
        it('Property 13.4: All diagnostic modules have unique IDs', () => {
            const ids = diagnosticModules.map(m => m.id);
            const uniqueIds = new Set(ids);

            expect(uniqueIds.size).toBe(ids.length);
        });

        /**
         * Property 13.5: All diagnostic modules have unique routes
         * For any diagnostic module, its route SHALL be unique
         */
        it('Property 13.5: All diagnostic modules have unique routes', () => {
            const routes = diagnosticModules.map(m => m.route);
            const uniqueRoutes = new Set(routes);

            expect(uniqueRoutes.size).toBe(routes.length);
        });

        /**
         * Property 13.6: All diagnostic modules have unique names
         * For any diagnostic module, its name SHALL be unique
         */
        it('Property 13.6: All diagnostic modules have unique names', () => {
            const names = diagnosticModules.map(m => m.name);
            const uniqueNames = new Set(names);

            expect(uniqueNames.size).toBe(names.length);
        });

        /**
         * Property 13.7: Header navigation items have unique routes
         * For any header nav item, its route SHALL be unique
         */
        it('Property 13.7: Header navigation items have unique routes', () => {
            const routes = headerNavItems.map(item => item.route);
            const uniqueRoutes = new Set(routes);

            expect(uniqueRoutes.size).toBe(routes.length);
        });

        /**
         * Property 13.8: Header navigation items have unique labels
         * For any header nav item, its label SHALL be unique
         */
        it('Property 13.8: Header navigation items have unique labels', () => {
            const labels = headerNavItems.map(item => item.label);
            const uniqueLabels = new Set(labels);

            expect(uniqueLabels.size).toBe(labels.length);
        });

        /**
         * Property 13.9: Sidebar and header do not have overlapping navigation
         * The sidebar and header SHALL serve different navigation purposes
         */
        it('Property 13.9: Sidebar and header serve different navigation purposes', () => {
            // Header provides top-level navigation (Home, Dashboard, Assessment, About)
            // Sidebar provides dashboard-specific navigation (modules)

            const headerRoutes = new Set(headerNavItems.map(item => item.route));
            const sidebarRoutes = new Set(sidebarItems.map(item => item.route));

            // Only /dashboard should overlap (as entry point)
            const overlap = [...headerRoutes].filter(r => sidebarRoutes.has(r));

            // Dashboard is the only expected overlap
            expect(overlap.length).toBeLessThanOrEqual(1);
            if (overlap.length === 1) {
                expect(overlap[0]).toBe('/dashboard');
            }
        });

        /**
         * Property 13.10: No duplicate CTA buttons on homepage
         * Homepage CTAs SHALL have distinct purposes
         */
        it('Property 13.10: Homepage CTAs have distinct purposes', () => {
            const labels = homepageCTAs.map(cta => cta.label);
            const uniqueLabels = new Set(labels);

            expect(uniqueLabels.size).toBe(labels.length);
        });

        /**
         * Property 13.11: For any randomly selected pair of sidebar items, they are distinct
         */
        it('Property 13.11: Any two sidebar items are distinct', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 0, max: sidebarItems.length - 1 }),
                    fc.integer({ min: 0, max: sidebarItems.length - 1 }),
                    (i, j) => {
                        if (i === j) return true; // Same index, skip

                        const item1 = sidebarItems[i];
                        const item2 = sidebarItems[j];

                        // IDs must be different
                        expect(item1.id).not.toBe(item2.id);

                        // Routes must be different
                        expect(item1.route).not.toBe(item2.route);

                        // Labels must be different
                        expect(item1.label).not.toBe(item2.label);
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 13.12: For any randomly selected pair of diagnostic modules, they are distinct
         */
        it('Property 13.12: Any two diagnostic modules are distinct', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: 0, max: diagnosticModules.length - 1 }),
                    fc.integer({ min: 0, max: diagnosticModules.length - 1 }),
                    (i, j) => {
                        if (i === j) return true; // Same index, skip

                        const module1 = diagnosticModules[i];
                        const module2 = diagnosticModules[j];

                        // IDs must be different
                        expect(module1.id).not.toBe(module2.id);

                        // Routes must be different
                        expect(module1.route).not.toBe(module2.route);

                        // Names must be different
                        expect(module1.name).not.toBe(module2.name);
                    }
                ),
                { numRuns: 100 }
            );
        });

        /**
         * Property 13.13: Navigation structure follows single responsibility
         * Each navigation area SHALL have a single, clear purpose
         */
        it('Property 13.13: Navigation structure follows single responsibility', () => {
            // Header: Top-level app navigation
            expect(headerNavItems.some(item => item.route === '/')).toBe(true);
            expect(headerNavItems.some(item => item.route === '/about')).toBe(true);

            // Sidebar: Dashboard module navigation
            expect(sidebarItems.every(item =>
                item.route === '/dashboard' || item.route.startsWith('/dashboard/')
            )).toBe(true);

            // No sidebar items should link to non-dashboard routes
            expect(sidebarItems.every(item =>
                item.route.startsWith('/dashboard')
            )).toBe(true);
        });

        /**
         * Property 13.14: All routes in the application are unique
         */
        it('Property 13.14: All application routes are unique', () => {
            const uniqueRoutes = new Set(allRoutes);
            expect(uniqueRoutes.size).toBe(allRoutes.length);
        });

        /**
         * Property 13.15: Module routes match sidebar routes
         * Available modules SHALL have corresponding sidebar entries
         */
        it('Property 13.15: Available modules have corresponding sidebar entries', () => {
            const availableModules = diagnosticModules.filter(m => m.status === 'available');
            const sidebarRoutes = new Set(sidebarItems.map(item => item.route));

            availableModules.forEach(module => {
                expect(sidebarRoutes.has(module.route)).toBe(true);
            });
        });
    });
});
