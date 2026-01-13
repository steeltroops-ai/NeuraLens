# Frontend Global Fix Plan

## Overview

This document outlines all global frontend fixes including PWA issues, UI/UX improvements, naming consistency, and layout fixes.

## Issues to Fix

### PWA Issues

#### F-PWA-001: Remove Install Prompt Popup
**Priority**: P0
**Description**: PWA install prompt popup appearing unexpectedly and annoying users.
**Files**: 
- `frontend/src/app/layout.tsx`
- `frontend/public/manifest.json`
**Fix**:
- Remove or hide the `#pwa-install-prompt` div
- Disable automatic install prompt
- Only show install option in settings/menu if needed
- Remove service worker registration in development

```tsx
// Remove this from layout.tsx
<div id='pwa-install-prompt' className='pwa-install-prompt' />

// Or hide with CSS
.pwa-install-prompt {
  display: none !important;
}
```

#### F-PWA-002: Service Worker Issues
**Priority**: P1
**Description**: Service worker causing caching issues in development.
**Files**: `frontend/src/app/layout.tsx`
**Fix**:
- Only register service worker in production
- Add cache busting for development
- Provide manual cache clear option

### Naming Consistency

#### F-NAME-001: Brand Name Inconsistency
**Priority**: P0
**Description**: Multiple variations of brand name used (NeuraLens, NeuroLens, Neuralens, NeuroLens-X).
**Files**: Multiple files across frontend
**Fix**:
- Standardize to "NeuraLens" everywhere
- Update all occurrences in:
  - `frontend/src/app/layout.tsx` (metadata)
  - `frontend/public/manifest.json`
  - `frontend/src/components/layout/Header.tsx`
  - `frontend/src/components/layout/Footer.tsx`
  - All page titles and descriptions

**Search and Replace**:
```
NeuroLens-X → NeuraLens
NeuroLens → NeuraLens
Neuralens → NeuraLens
neurolens → neuralens
```

#### F-NAME-002: Feature Names
**Priority**: P1
**Description**: Inconsistent feature naming across UI.
**Files**: Various component files
**Fix**:
- Standardize modality names:
  - "Speech Analysis" (not "Voice Analysis")
  - "Retinal Imaging" (not "Eye Scan")
  - "Motor Assessment" (not "Movement Test")
  - "Cognitive Testing" (not "Brain Test")
- Update all UI labels consistently

### UI/UX Issues

#### F-UX-001: Loading States
**Priority**: P0
**Description**: Missing or inconsistent loading states.
**Files**: Various component files
**Fix**:
- Add skeleton loaders for all data-fetching components
- Show spinner during API calls
- Disable buttons during submission
- Add progress indicators for long operations

#### F-UX-002: Error States
**Priority**: P0
**Description**: Error messages not user-friendly.
**Files**: Various component files
**Fix**:
- Replace technical errors with user-friendly messages
- Add retry buttons for failed operations
- Show helpful suggestions for common errors
- Log technical details for debugging

#### F-UX-003: Empty States
**Priority**: P1
**Description**: No empty state designs for lists/dashboards.
**Files**: Dashboard components
**Fix**:
- Add empty state illustrations
- Provide call-to-action for empty states
- Show helpful tips for getting started

#### F-UX-004: Mobile Responsiveness
**Priority**: P0
**Description**: Some components not responsive on mobile.
**Files**: Various component files
**Fix**:
- Test all pages on mobile viewport
- Fix overflow issues
- Adjust font sizes for mobile
- Ensure touch targets are 48px minimum

#### F-UX-005: Navigation
**Priority**: P1
**Description**: Navigation not intuitive.
**Files**: `frontend/src/components/layout/Header.tsx`
**Fix**:
- Add breadcrumbs for nested pages
- Highlight current page in nav
- Add back button where appropriate
- Improve mobile menu

#### F-UX-006: Form Validation
**Priority**: P1
**Description**: Form validation feedback not clear.
**Files**: Various form components
**Fix**:
- Show inline validation errors
- Highlight invalid fields
- Provide clear error messages
- Add success feedback

### Layout Issues

#### F-LAYOUT-001: Header Overlap
**Priority**: P0
**Description**: Fixed header overlapping content.
**Files**: `frontend/src/app/layout.tsx`, `frontend/src/components/layout/Layout.tsx`
**Fix**:
- Add proper padding-top to main content
- Ensure header height is consistent
- Test with different content lengths

#### F-LAYOUT-002: Footer Position
**Priority**: P1
**Description**: Footer not sticking to bottom on short pages.
**Files**: `frontend/src/components/layout/Footer.tsx`
**Fix**:
- Implement sticky footer
- Use flexbox min-height approach
- Test on all page lengths

#### F-LAYOUT-003: Container Width
**Priority**: P1
**Description**: Content width inconsistent across pages.
**Files**: Various page components
**Fix**:
- Standardize max-width containers
- Use consistent padding
- Ensure responsive breakpoints match

### Accessibility Issues

#### F-A11Y-001: Color Contrast
**Priority**: P0
**Description**: Some text doesn't meet WCAG contrast requirements.
**Files**: `frontend/src/app/globals.css`, various components
**Fix**:
- Audit all color combinations
- Fix low contrast text
- Ensure 4.5:1 ratio for normal text
- Ensure 3:1 ratio for large text

#### F-A11Y-002: Focus Indicators
**Priority**: P0
**Description**: Focus indicators not visible on all interactive elements.
**Files**: Various component files
**Fix**:
- Add visible focus rings
- Ensure focus order is logical
- Test keyboard navigation

#### F-A11Y-003: Screen Reader Support
**Priority**: P1
**Description**: Missing ARIA labels and roles.
**Files**: Various component files
**Fix**:
- Add aria-labels to buttons
- Add aria-describedby for form fields
- Use semantic HTML elements
- Test with screen reader

#### F-A11Y-004: Skip Links
**Priority**: P1
**Description**: Skip links exist but may not work correctly.
**Files**: `frontend/src/app/layout.tsx`
**Fix**:
- Verify skip link targets exist
- Test skip link functionality
- Ensure visible on focus

### Performance Issues

#### F-PERF-001: Image Optimization
**Priority**: P1
**Description**: Images not optimized.
**Files**: Various component files
**Fix**:
- Use Next.js Image component
- Add proper width/height
- Use appropriate formats (WebP)
- Implement lazy loading

#### F-PERF-002: Bundle Size
**Priority**: P2
**Description**: Bundle size could be reduced.
**Files**: `frontend/package.json`, various imports
**Fix**:
- Audit dependencies
- Remove unused packages
- Use dynamic imports for large components
- Tree-shake unused code

## Files to Update

### High Priority
1. `frontend/src/app/layout.tsx` - PWA, naming, layout
2. `frontend/public/manifest.json` - PWA, naming
3. `frontend/src/components/layout/Header.tsx` - naming, navigation
4. `frontend/src/components/layout/Footer.tsx` - naming, layout
5. `frontend/src/app/globals.css` - accessibility, styling

### Medium Priority
1. All assessment step components - loading, errors
2. All dashboard components - empty states, loading
3. Form components - validation

## Kiro Spec Template

```
Feature: Fix Frontend Global Issues

As a developer, I want to fix all global frontend issues
so that users have a consistent, accessible, and polished experience.

Requirements:
1. Remove PWA install prompt popup
2. Standardize brand name to "NeuraLens"
3. Add loading states to all data-fetching components
4. Fix mobile responsiveness issues
5. Meet WCAG 2.1 AA accessibility standards
6. All tests must pass
```

## Verification Checklist

- [ ] PWA install prompt removed/hidden
- [ ] Brand name consistent everywhere
- [ ] All loading states implemented
- [ ] All error states user-friendly
- [ ] Mobile responsive on all pages
- [ ] Color contrast meets WCAG
- [ ] Focus indicators visible
- [ ] Screen reader tested
- [ ] Skip links working
- [ ] Images optimized
