/**
 * Property-Based Tests for Module Route Existence
 * 
 * Feature: frontend-global-fix
 * Property 2: Module Route Existence
 * Validates: Requirements 4.1, 4.3
 * 
 * For all diagnostic modules defined in the configuration, there SHALL exist
 * a corresponding route under /dashboard/[module-id] that renders the
 * appropriate assessment component.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import * as fs from 'fs';
import * as path from 'path';
import { diagnosticModules, DiagnosticModule, getAvailableModules } from '@/data/diagnostic-modules';

/**
 * Helper function to check if a page file exists for a given route
 */
const pageExistsForRoute = (route: string): boolean => {
    // Convert route to file path
    // e.g., /dashboard/speech -> frontend/src/app/dashboard/speech/page.tsx
    const routePath = route.replace(/^\//, ''); // Remove leading slash
    const pagePath = path.join(process.cwd(), 'src', 'app', routePath, 'page.tsx');

    try {
        return fs.existsSync(pagePath);
    } catch {
        return false;
    }
};

/**
 * Arbitrary generator for diagnostic modules from the actual configuration
 */
const diagnosticModuleArb = fc.constantFrom(...diagnosticModules);

/**
 * Arbitrary generator for available modules only
 */
const availableModuleArb = fc.constantFrom(...getAvailableModules());

describe('Module Route Existence Property Tests', () => {
    /**
     * Property 2: Module Route Existence
     * 
     * For all diagnostic modules, there SHALL exist a corresponding route
     * under /dashboard/[module-id].
     * 
     * **Validates: Requirements 4.1, 4.3**
     */
    describe('Property 2: Module Route Existence', () => {
        it('Property 2.1: All available modules have corresponding page files', () => {
            fc.assert(
                fc.property(availableModuleArb, (module: DiagnosticModule) => {
                    // Check that the page file exists for this module's route
                    const pageExists = pageExistsForRoute(module.route);

                    expect(pageExists).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 2.2: All module routes follow the /dashboard/[id] pattern', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Route must start with /dashboard/
                    expect(module.route.startsWith('/dashboard/')).toBe(true);

                    // Route must contain the module id as the path segment
                    const routeSegments = module.route.split('/').filter(Boolean);
                    expect(routeSegments[0]).toBe('dashboard');
                    expect(routeSegments.length).toBe(2);
                    expect(routeSegments[1]).toBe(module.id);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 2.3: Module routes are unique', () => {
            const routes = diagnosticModules.map(m => m.route);
            const uniqueRoutes = new Set(routes);

            // All routes must be unique
            expect(uniqueRoutes.size).toBe(routes.length);
        });

        it('Property 2.4: Module IDs are valid URL path segments', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Module ID should only contain valid URL characters
                    // (lowercase letters, numbers, hyphens)
                    const validPathRegex = /^[a-z0-9-]+$/;
                    expect(validPathRegex.test(module.id)).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 2.5: Dashboard overview page exists', () => {
            const dashboardPagePath = path.join(process.cwd(), 'src', 'app', 'dashboard', 'page.tsx');
            expect(fs.existsSync(dashboardPagePath)).toBe(true);
        });

        it('Property 2.6: Dashboard layout exists for shared navigation', () => {
            const layoutPath = path.join(process.cwd(), 'src', 'app', 'dashboard', 'layout.tsx');
            expect(fs.existsSync(layoutPath)).toBe(true);
        });
    });

    /**
     * Property 2.7: Verify all expected module pages exist
     * 
     * This is a comprehensive check that all available modules have pages.
     * 
     * **Validates: Requirements 4.1, 4.3**
     */
    describe('Property 2.7: Comprehensive Module Page Verification', () => {
        const availableModules = getAvailableModules();

        availableModules.forEach((module) => {
            it(`Module page exists for: ${module.name} (${module.route})`, () => {
                const pageExists = pageExistsForRoute(module.route);
                expect(pageExists).toBe(true);
            });
        });
    });

    /**
     * Property 2.8: Utility pages exist
     * 
     * Verify that utility pages (analytics, reports, settings) exist.
     * 
     * **Validates: Requirements 4.1**
     */
    describe('Property 2.8: Utility Pages Existence', () => {
        const utilityPages = [
            { name: 'Analytics', route: '/dashboard/analytics' },
            { name: 'Reports', route: '/dashboard/reports' },
            { name: 'Settings', route: '/dashboard/settings' },
        ];

        utilityPages.forEach((page) => {
            it(`Utility page exists: ${page.name} (${page.route})`, () => {
                const pageExists = pageExistsForRoute(page.route);
                expect(pageExists).toBe(true);
            });
        });
    });

    /**
     * Property 2.9: Additional assessment pages exist
     * 
     * Verify that additional assessment pages (multimodal, nri-fusion) exist.
     * 
     * **Validates: Requirements 4.1**
     */
    describe('Property 2.9: Additional Assessment Pages', () => {
        const additionalPages = [
            { name: 'Multi-Modal Assessment', route: '/dashboard/multimodal' },
            { name: 'NRI Fusion', route: '/dashboard/nri-fusion' },
        ];

        additionalPages.forEach((page) => {
            it(`Additional assessment page exists: ${page.name} (${page.route})`, () => {
                const pageExists = pageExistsForRoute(page.route);
                expect(pageExists).toBe(true);
            });
        });
    });
});
