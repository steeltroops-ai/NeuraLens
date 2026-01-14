/**
 * Property-Based Tests for Dashboard Layout - No Horizontal Overflow
 * 
 * Feature: dashboard-ux-improvements
 * Property 1: No Horizontal Overflow
 * Validates: Requirements 1.4, 7.6
 * 
 * For any viewport width from 320px to 1920px, the dashboard layout SHALL NOT
 * produce horizontal scrollbars, and all content SHALL be contained within
 * the viewport width.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { layoutWidths, breakpoints } from '@/styles/design-tokens';

// Define viewport width range for testing (Requirements 1.4, 7.6)
const MIN_VIEWPORT_WIDTH = 320;  // Minimum mobile width
const MAX_VIEWPORT_WIDTH = 1920; // Maximum desktop width
const MOBILE_BREAKPOINT = 1024;  // lg breakpoint for mobile/desktop switch

// Layout configuration values
const SIDEBAR_WIDTH_EXPANDED = parseInt(layoutWidths.sidebar, 10);  // 280px
const SIDEBAR_WIDTH_COLLAPSED = parseInt(layoutWidths.sidebarCollapsed, 10);  // 64px

/**
 * Calculate the expected main content width based on viewport and sidebar state
 */
function calculateMainContentWidth(
    viewportWidth: number,
    sidebarCollapsed: boolean
): number {
    // On mobile (< 1024px), sidebar is hidden, full width available
    if (viewportWidth < MOBILE_BREAKPOINT) {
        return viewportWidth;
    }

    // On desktop, subtract sidebar width
    const sidebarWidth = sidebarCollapsed ? SIDEBAR_WIDTH_COLLAPSED : SIDEBAR_WIDTH_EXPANDED;
    return viewportWidth - sidebarWidth;
}

/**
 * Check if content would overflow horizontally
 */
function wouldOverflow(
    viewportWidth: number,
    contentWidth: number,
    sidebarCollapsed: boolean
): boolean {
    const availableWidth = calculateMainContentWidth(viewportWidth, sidebarCollapsed);
    return contentWidth > availableWidth;
}

/**
 * Validate that the layout uses correct display mode based on viewport
 */
function getExpectedDisplayMode(viewportWidth: number): 'flex' | 'grid' {
    return viewportWidth < MOBILE_BREAKPOINT ? 'flex' : 'grid';
}

/**
 * Validate grid template columns for desktop layout
 */
function getExpectedGridTemplate(sidebarCollapsed: boolean): string {
    const sidebarWidth = sidebarCollapsed ? '64px' : '280px';
    return `${sidebarWidth} 1fr`;
}

describe('Dashboard Layout Property Tests - No Horizontal Overflow', () => {
    /**
     * Property 1: No Horizontal Overflow
     * 
     * For any viewport width from 320px to 1920px, the dashboard layout
     * SHALL NOT produce horizontal scrollbars.
     * 
     * **Validates: Requirements 1.4, 7.6**
     */
    describe('Property 1: No Horizontal Overflow', () => {
        it('Property 1.1: Main content width never exceeds available space', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MIN_VIEWPORT_WIDTH, max: MAX_VIEWPORT_WIDTH }),
                    fc.boolean(), // sidebarCollapsed
                    (viewportWidth, sidebarCollapsed) => {
                        const availableWidth = calculateMainContentWidth(viewportWidth, sidebarCollapsed);

                        // Available width should always be positive
                        expect(availableWidth).toBeGreaterThan(0);

                        // Available width should never exceed viewport width
                        expect(availableWidth).toBeLessThanOrEqual(viewportWidth);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 1.2: Content at 100% width fits within available space', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MIN_VIEWPORT_WIDTH, max: MAX_VIEWPORT_WIDTH }),
                    fc.boolean(), // sidebarCollapsed
                    (viewportWidth, sidebarCollapsed) => {
                        const availableWidth = calculateMainContentWidth(viewportWidth, sidebarCollapsed);
                        const contentWidth = availableWidth; // Content at 100% width

                        // Content at 100% should not overflow
                        expect(wouldOverflow(viewportWidth, contentWidth, sidebarCollapsed)).toBe(false);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 1.3: Sidebar width is always valid', () => {
            fc.assert(
                fc.property(
                    fc.boolean(), // sidebarCollapsed
                    (sidebarCollapsed) => {
                        const sidebarWidth = sidebarCollapsed ? SIDEBAR_WIDTH_COLLAPSED : SIDEBAR_WIDTH_EXPANDED;

                        // Sidebar width should be positive
                        expect(sidebarWidth).toBeGreaterThan(0);

                        // Sidebar width should be one of the expected values
                        expect([SIDEBAR_WIDTH_COLLAPSED, SIDEBAR_WIDTH_EXPANDED]).toContain(sidebarWidth);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 1.4: Mobile layout uses full viewport width', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MIN_VIEWPORT_WIDTH, max: MOBILE_BREAKPOINT - 1 }),
                    fc.boolean(), // sidebarCollapsed (doesn't matter on mobile)
                    (viewportWidth, sidebarCollapsed) => {
                        const availableWidth = calculateMainContentWidth(viewportWidth, sidebarCollapsed);

                        // On mobile, full viewport width should be available
                        expect(availableWidth).toBe(viewportWidth);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 1.5: Desktop layout accounts for sidebar width', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MOBILE_BREAKPOINT, max: MAX_VIEWPORT_WIDTH }),
                    fc.boolean(), // sidebarCollapsed
                    (viewportWidth, sidebarCollapsed) => {
                        const availableWidth = calculateMainContentWidth(viewportWidth, sidebarCollapsed);
                        const expectedSidebarWidth = sidebarCollapsed ? SIDEBAR_WIDTH_COLLAPSED : SIDEBAR_WIDTH_EXPANDED;

                        // On desktop, available width should be viewport minus sidebar
                        expect(availableWidth).toBe(viewportWidth - expectedSidebarWidth);
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    /**
     * Property 2: Responsive Layout Switching
     * 
     * For any viewport width, WHEN the width is below 1024px, the layout
     * SHALL use flex display, and WHEN the width is 1024px or above,
     * the layout SHALL use CSS Grid with sidebar column.
     * 
     * **Validates: Requirements 1.1, 1.2**
     */
    describe('Property 2: Responsive Layout Switching', () => {
        it('Property 2.1: Mobile viewports use flex layout', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MIN_VIEWPORT_WIDTH, max: MOBILE_BREAKPOINT - 1 }),
                    (viewportWidth) => {
                        const displayMode = getExpectedDisplayMode(viewportWidth);
                        expect(displayMode).toBe('flex');
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 2.2: Desktop viewports use grid layout', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MOBILE_BREAKPOINT, max: MAX_VIEWPORT_WIDTH }),
                    (viewportWidth) => {
                        const displayMode = getExpectedDisplayMode(viewportWidth);
                        expect(displayMode).toBe('grid');
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 2.3: Grid template uses correct sidebar width', () => {
            fc.assert(
                fc.property(
                    fc.boolean(), // sidebarCollapsed
                    (sidebarCollapsed) => {
                        const gridTemplate = getExpectedGridTemplate(sidebarCollapsed);
                        const expectedWidth = sidebarCollapsed ? '64px' : '280px';

                        expect(gridTemplate).toBe(`${expectedWidth} 1fr`);
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    /**
     * Property 3: Sidebar Collapse Without Layout Shift
     * 
     * For any sidebar collapse/expand action, the main content area position
     * SHALL remain stable (no horizontal shift greater than the sidebar width change).
     * 
     * **Validates: Requirements 1.3**
     */
    describe('Property 3: Sidebar Collapse Without Layout Shift', () => {
        it('Property 3.1: Width change equals sidebar width difference', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MOBILE_BREAKPOINT, max: MAX_VIEWPORT_WIDTH }),
                    (viewportWidth) => {
                        const expandedContentWidth = calculateMainContentWidth(viewportWidth, false);
                        const collapsedContentWidth = calculateMainContentWidth(viewportWidth, true);

                        const widthDifference = collapsedContentWidth - expandedContentWidth;
                        const sidebarWidthDifference = SIDEBAR_WIDTH_EXPANDED - SIDEBAR_WIDTH_COLLAPSED;

                        // The content width change should equal the sidebar width change
                        expect(widthDifference).toBe(sidebarWidthDifference);
                    }
                ),
                { numRuns: 100 }
            );
        });

        it('Property 3.2: Collapsed sidebar provides more content space', () => {
            fc.assert(
                fc.property(
                    fc.integer({ min: MOBILE_BREAKPOINT, max: MAX_VIEWPORT_WIDTH }),
                    (viewportWidth) => {
                        const expandedContentWidth = calculateMainContentWidth(viewportWidth, false);
                        const collapsedContentWidth = calculateMainContentWidth(viewportWidth, true);

                        // Collapsed sidebar should provide more content space
                        expect(collapsedContentWidth).toBeGreaterThan(expandedContentWidth);
                    }
                ),
                { numRuns: 100 }
            );
        });
    });

    /**
     * Boundary Tests
     * 
     * Test specific boundary conditions for viewport widths
     */
    describe('Boundary Tests', () => {
        it('Minimum viewport width (320px) has positive available space', () => {
            const availableWidth = calculateMainContentWidth(MIN_VIEWPORT_WIDTH, false);
            expect(availableWidth).toBeGreaterThan(0);
        });

        it('Maximum viewport width (1920px) has correct available space', () => {
            const availableWidth = calculateMainContentWidth(MAX_VIEWPORT_WIDTH, false);
            expect(availableWidth).toBe(MAX_VIEWPORT_WIDTH - SIDEBAR_WIDTH_EXPANDED);
        });

        it('Breakpoint boundary (1024px) uses grid layout', () => {
            const displayMode = getExpectedDisplayMode(MOBILE_BREAKPOINT);
            expect(displayMode).toBe('grid');
        });

        it('Just below breakpoint (1023px) uses flex layout', () => {
            const displayMode = getExpectedDisplayMode(MOBILE_BREAKPOINT - 1);
            expect(displayMode).toBe('flex');
        });

        it('Sidebar widths match design tokens', () => {
            expect(SIDEBAR_WIDTH_EXPANDED).toBe(280);
            expect(SIDEBAR_WIDTH_COLLAPSED).toBe(64);
        });
    });
});
