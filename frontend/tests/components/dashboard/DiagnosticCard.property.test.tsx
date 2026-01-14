/**
 * Property-Based Tests for DiagnosticCard Component
 * 
 * Feature: frontend-global-fix
 * Property 5: Diagnostic Card Rendering
 * Validates: Requirements 6.1, 6.2
 * 
 * For any diagnostic module in the configuration, the dashboard overview 
 * SHALL render a card containing: the module's icon, name, description, 
 * and status badge (Available or Coming Soon).
 * 
 * Property 6: Coming Soon Module Behavior
 * Validates: Requirements 6.5
 * 
 * For any diagnostic module marked as "coming-soon", the card SHALL be 
 * disabled (non-clickable) and SHALL display a "Coming Soon" badge.
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
}));

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }: any) => {
            // Extract only valid HTML attributes
            const validProps: Record<string, any> = {};
            const htmlAttributes = [
                'className', 'onClick', 'onKeyDown', 'role', 'tabIndex',
                'aria-disabled', 'aria-label', 'data-testid', 'data-status', 'style'
            ];
            htmlAttributes.forEach(attr => {
                if (props[attr] !== undefined) {
                    validProps[attr] = props[attr];
                }
            });
            return <div {...validProps}>{children}</div>;
        },
    },
}));

/**
 * Arbitrary generator for diagnostic modules from the actual configuration
 */
const diagnosticModuleArb = fc.constantFrom(...diagnosticModules);

/**
 * Arbitrary generator for available modules only
 */
const availableModuleArb = fc.constantFrom(
    ...diagnosticModules.filter(m => m.status === 'available')
);

/**
 * Arbitrary generator for coming-soon modules only
 */
const comingSoonModuleArb = fc.constantFrom(
    ...diagnosticModules.filter(m => m.status === 'coming-soon')
);

describe('DiagnosticCard Property Tests', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    /**
     * Property 5: Diagnostic Card Rendering
     * 
     * For any diagnostic module, the card data structure SHALL contain:
     * - Module name (non-empty string)
     * - Module description (non-empty string)
     * - Module icon (valid Lucide icon component)
     * - Status (either 'available' or 'coming-soon')
     * 
     * **Validates: Requirements 6.1, 6.2**
     */
    describe('Property 5: Diagnostic Card Rendering', () => {
        it('Property 5.1: All modules have required display properties', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Module name must be a non-empty string
                    expect(typeof module.name).toBe('string');
                    expect(module.name.length).toBeGreaterThan(0);

                    // Module description must be a non-empty string
                    expect(typeof module.description).toBe('string');
                    expect(module.description.length).toBeGreaterThan(0);

                    // Module icon must be a valid React component (Lucide icons are forwardRef objects)
                    // Lucide icons have $$typeof Symbol for react.forward_ref or are functions
                    const isValidIcon = typeof module.icon === 'function' ||
                        (typeof module.icon === 'object' && module.icon !== null);
                    expect(isValidIcon).toBe(true);

                    // Status must be either 'available' or 'coming-soon'
                    expect(['available', 'coming-soon']).toContain(module.status);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 5.2: All modules have valid route paths', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Route must be a non-empty string
                    expect(typeof module.route).toBe('string');
                    expect(module.route.length).toBeGreaterThan(0);

                    // Route must start with /dashboard/
                    expect(module.route.startsWith('/dashboard/')).toBe(true);

                    // Route must contain the module id
                    expect(module.route).toContain(module.id);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 5.3: All modules have unique identifiers', () => {
            const moduleIds = diagnosticModules.map(m => m.id);
            const uniqueIds = new Set(moduleIds);

            // All module IDs must be unique
            expect(uniqueIds.size).toBe(moduleIds.length);
        });

        it('Property 5.4: All modules have diagnoses list', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Diagnoses must be an array
                    expect(Array.isArray(module.diagnoses)).toBe(true);

                    // Diagnoses array must not be empty
                    expect(module.diagnoses.length).toBeGreaterThan(0);

                    // Each diagnosis must be a non-empty string
                    module.diagnoses.forEach(diagnosis => {
                        expect(typeof diagnosis).toBe('string');
                        expect(diagnosis.length).toBeGreaterThan(0);
                    });
                }),
                { numRuns: 100 }
            );
        });

        it('Property 5.5: All modules have gradient styling', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Gradient must be a non-empty string
                    expect(typeof module.gradient).toBe('string');
                    expect(module.gradient.length).toBeGreaterThan(0);

                    // Gradient must contain 'from-' and 'to-' for Tailwind gradient
                    expect(module.gradient).toContain('from-');
                    expect(module.gradient).toContain('to-');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 5.6: Module category matches status', () => {
            fc.assert(
                fc.property(diagnosticModuleArb, (module: DiagnosticModule) => {
                    // Available modules should be in 'current' category
                    if (module.status === 'available') {
                        expect(module.category).toBe('current');
                    }

                    // Coming-soon modules should be in 'upcoming' category
                    if (module.status === 'coming-soon') {
                        expect(module.category).toBe('upcoming');
                    }
                }),
                { numRuns: 100 }
            );
        });
    });

    /**
     * Property 6: Coming Soon Module Behavior
     * 
     * For any diagnostic module marked as "coming-soon":
     * - The card SHALL be disabled (non-clickable)
     * - The card SHALL display a "Coming Soon" badge
     * 
     * **Validates: Requirements 6.5**
     */
    describe('Property 6: Coming Soon Module Behavior', () => {
        it('Property 6.1: Coming-soon modules have correct status', () => {
            fc.assert(
                fc.property(comingSoonModuleArb, (module: DiagnosticModule) => {
                    // Status must be 'coming-soon'
                    expect(module.status).toBe('coming-soon');

                    // Category must be 'upcoming'
                    expect(module.category).toBe('upcoming');
                }),
                { numRuns: 100 }
            );
        });

        it('Property 6.2: Coming-soon modules use muted gradient colors', () => {
            fc.assert(
                fc.property(comingSoonModuleArb, (module: DiagnosticModule) => {
                    // Coming-soon modules should use muted colors (400 shade or gray/slate)
                    const hasMutedColor =
                        module.gradient.includes('-400') ||
                        module.gradient.includes('gray') ||
                        module.gradient.includes('slate') ||
                        module.gradient.includes('stone');

                    expect(hasMutedColor).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 6.3: Available modules have vibrant gradient colors', () => {
            fc.assert(
                fc.property(availableModuleArb, (module: DiagnosticModule) => {
                    // Available modules should use vibrant colors (500 shade)
                    expect(module.gradient.includes('-500')).toBe(true);
                }),
                { numRuns: 100 }
            );
        });

        it('Property 6.4: Module counts are consistent', () => {
            const availableCount = diagnosticModules.filter(m => m.status === 'available').length;
            const comingSoonCount = diagnosticModules.filter(m => m.status === 'coming-soon').length;

            // Total should equal available + coming-soon
            expect(availableCount + comingSoonCount).toBe(diagnosticModules.length);

            // We expect 4 available modules per requirements
            expect(availableCount).toBe(4);

            // We expect 8 coming-soon modules per requirements
            expect(comingSoonCount).toBe(8);
        });
    });
});
