# Assessment Flow Fix Plan

## Overview

This document outlines all fixes needed for the end-to-end assessment workflow, from welcome screen through results display.

## Current State

### Assessment Flow Components
- `frontend/src/components/assessment/AssessmentFlow.tsx` - Main flow controller
- `frontend/src/components/assessment/AccessibleAssessmentWorkflow.tsx` - Accessible version
- `frontend/src/components/assessment/steps/WelcomeStep.tsx` - Welcome/consent
- `frontend/src/components/assessment/steps/SpeechAssessmentStep.tsx` - Speech
- `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx` - Retinal
- `frontend/src/components/assessment/steps/MotorAssessmentStep.tsx` - Motor
- `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx` - Cognitive
- `frontend/src/components/assessment/steps/ProcessingStep.tsx` - Processing
- `frontend/src/components/assessment/steps/RiskAssessmentStep.tsx` - Risk display
- `frontend/src/components/assessment/steps/ResultsStep.tsx` - Final results

### Assessment Page
- `frontend/src/app/assessment/page.tsx` - Assessment route

## Issues to Fix

### Flow Control Issues

#### AF-FLOW-001: Step Navigation
**Priority**: P0
**Description**: Step navigation not working correctly.
**Files**: `frontend/src/components/assessment/AssessmentFlow.tsx`
**Fix**:
- Fix next/previous step logic
- Prevent skipping required steps
- Handle browser back button
- Save progress on navigation

#### AF-FLOW-002: Progress Persistence
**Priority**: P0
**Description**: Progress lost on page refresh.
**Files**: `frontend/src/components/assessment/AssessmentFlow.tsx`
**Fix**:
- Save progress to localStorage
- Restore progress on mount
- Clear progress on completion
- Handle stale progress

#### AF-FLOW-003: Step Validation
**Priority**: P0
**Description**: Can proceed without completing step.
**Files**: All step components
**Fix**:
- Validate step completion
- Disable next button until valid
- Show validation errors
- Require consent before proceeding

#### AF-FLOW-004: Error Recovery
**Priority**: P0
**Description**: No recovery from step errors.
**Files**: All step components
**Fix**:
- Add retry functionality
- Allow skipping failed steps
- Save partial results
- Show error context

### Welcome Step Issues

#### AF-WELCOME-001: Consent Form
**Priority**: P0
**Description**: Consent form not comprehensive.
**Files**: `frontend/src/components/assessment/steps/WelcomeStep.tsx`
**Fix**:
- Add full consent text
- Require checkbox acknowledgment
- Store consent timestamp
- Allow consent review later

#### AF-WELCOME-002: Instructions
**Priority**: P1
**Description**: Assessment instructions not clear.
**Files**: `frontend/src/components/assessment/steps/WelcomeStep.tsx`
**Fix**:
- Add step-by-step overview
- Show estimated time
- List required equipment
- Provide preparation tips

#### AF-WELCOME-003: Modality Selection
**Priority**: P1
**Description**: Cannot select specific modalities.
**Files**: `frontend/src/components/assessment/steps/WelcomeStep.tsx`
**Fix**:
- Add modality checkboxes
- Allow partial assessment
- Show modality descriptions
- Indicate required vs optional

### Processing Step Issues

#### AF-PROC-001: Progress Indication
**Priority**: P0
**Description**: Processing progress not shown.
**Files**: `frontend/src/components/assessment/steps/ProcessingStep.tsx`
**Fix**:
- Show processing stages
- Display progress percentage
- Indicate current modality
- Show estimated time remaining

#### AF-PROC-002: Background Processing
**Priority**: P1
**Description**: Processing blocks UI.
**Files**: `frontend/src/components/assessment/steps/ProcessingStep.tsx`
**Fix**:
- Process in background
- Allow cancellation
- Show partial results
- Handle timeouts

#### AF-PROC-003: Error Display
**Priority**: P0
**Description**: Processing errors not shown clearly.
**Files**: `frontend/src/components/assessment/steps/ProcessingStep.tsx`
**Fix**:
- Show which step failed
- Provide error details
- Offer retry option
- Allow continuing with partial results

### Results Step Issues

#### AF-RESULTS-001: Results Display
**Priority**: P0
**Description**: Results not displayed comprehensively.
**Files**: `frontend/src/components/assessment/steps/ResultsStep.tsx`
**Fix**:
- Show NRI score prominently
- Display all modality results
- Add risk interpretation
- Include recommendations

#### AF-RESULTS-002: Export Options
**Priority**: P1
**Description**: Cannot export results.
**Files**: `frontend/src/components/assessment/steps/ResultsStep.tsx`
**Fix**:
- Add PDF export
- Add print option
- Support email sharing
- Generate shareable link

#### AF-RESULTS-003: Next Steps
**Priority**: P1
**Description**: No guidance on next steps.
**Files**: `frontend/src/components/assessment/steps/ResultsStep.tsx`
**Fix**:
- Show recommended actions
- Provide healthcare resources
- Suggest follow-up timeline
- Link to dashboard

### Accessibility Issues

#### AF-A11Y-001: Keyboard Navigation
**Priority**: P0
**Description**: Cannot navigate flow with keyboard.
**Files**: All step components
**Fix**:
- Add keyboard shortcuts
- Focus management between steps
- Announce step changes
- Support escape to cancel

#### AF-A11Y-002: Screen Reader Support
**Priority**: P0
**Description**: Flow not accessible to screen readers.
**Files**: All step components
**Fix**:
- Add aria-labels
- Announce progress
- Describe step content
- Provide text alternatives

#### AF-A11Y-003: Reduced Motion
**Priority**: P1
**Description**: Animations may cause issues.
**Files**: All step components
**Fix**:
- Respect prefers-reduced-motion
- Provide static alternatives
- Reduce transition durations

### Mobile Issues

#### AF-MOBILE-001: Touch Interactions
**Priority**: P0
**Description**: Touch interactions not optimized.
**Files**: All step components
**Fix**:
- Increase touch targets
- Add swipe navigation
- Optimize for thumb reach
- Handle orientation changes

#### AF-MOBILE-002: Camera/Mic Access
**Priority**: P0
**Description**: Camera/mic access issues on mobile.
**Files**: Speech and Retinal steps
**Fix**:
- Request permissions properly
- Handle permission denied
- Provide fallback options
- Test on iOS and Android

## Assessment Flow Diagram

```
┌─────────────┐
│   Welcome   │
│  (Consent)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Modality   │
│  Selection  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Speech    │────▶│   Retinal   │
│ Assessment  │     │ Assessment  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│    Motor    │────▶│  Cognitive  │
│ Assessment  │     │ Assessment  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
         ┌─────────────┐
         │ Processing  │
         │   (NRI)     │
         └──────┬──────┘
                │
                ▼
         ┌─────────────┐
         │   Results   │
         │  (Report)   │
         └─────────────┘
```

## State Management

```typescript
interface AssessmentState {
  currentStep: number;
  completedSteps: string[];
  selectedModalities: string[];
  consentGiven: boolean;
  consentTimestamp: string | null;
  results: {
    speech?: SpeechResult;
    retinal?: RetinalResult;
    motor?: MotorResult;
    cognitive?: CognitiveResult;
    nri?: NRIResult;
  };
  errors: {
    [step: string]: string;
  };
  startTime: string;
  endTime: string | null;
}
```

## Kiro Spec Template

```
Feature: Fix Assessment Flow

As a developer, I want to fix all assessment flow issues
so that users can complete assessments smoothly from start to finish.

Requirements:
1. Step navigation must work correctly
2. Progress must persist across page refreshes
3. Consent must be properly captured
4. Processing must show progress
5. Results must be exportable
6. Flow must be accessible
7. All tests must pass
```

## Verification Checklist

- [ ] Step navigation works correctly
- [ ] Progress persists on refresh
- [ ] Cannot skip required steps
- [ ] Consent form complete
- [ ] Modality selection works
- [ ] Processing shows progress
- [ ] Errors handled gracefully
- [ ] Results display all data
- [ ] Export functionality works
- [ ] Keyboard navigation works
- [ ] Screen reader accessible
- [ ] Mobile touch optimized
- [ ] Camera/mic permissions work
- [ ] All tests pass
