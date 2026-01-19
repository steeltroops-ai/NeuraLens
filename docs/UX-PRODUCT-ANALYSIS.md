# MediLens Product-Level UX Analysis

**Date:** 2026-01-19  
**Version:** 1.0  
**Prepared By:** Product Design & UX Analysis

---

## Executive Summary

This document provides a comprehensive product-level analysis of MediLens, an AI-powered medical diagnostics platform. The analysis covers user journeys, UX breakdowns, visual design audits, dashboard usefulness, and alignment with industry-grade AI product standards.

---

## 1. User and Role Modeling

### 1.1 Identified User Personas

#### Persona A: **Clinician / Domain Expert (Primary)**
| Attribute | Details |
|-----------|---------|
| **Primary Goals** | Quickly analyze patient data, get AI-assisted diagnosis, generate clinical reports |
| **Time Sensitivity** | HIGH - needs results in under 2 seconds |
| **Tolerance for Complexity** | MEDIUM - wants depth but not at cost of speed |
| **Trust Requirements** | VERY HIGH - needs explainability, confidence scores, citations |
| **Key Screens** | Dashboard, Retinal/Speech/Cardiology pages, AI Explanation, Reports |

#### Persona B: **Technical Reviewer / AI Specialist**
| Attribute | Details |
|-----------|---------|
| **Primary Goals** | Validate model accuracy, understand pipeline stages, debug failures |
| **Time Sensitivity** | MEDIUM - can wait for detailed diagnostics |
| **Tolerance for Complexity** | HIGH - wants technical details, logs, stage timings |
| **Trust Requirements** | HIGH - needs transparency in model behavior |
| **Key Screens** | Pipeline status bar, Console logs, Biomarker details, Heatmaps |

#### Persona C: **Demo / Hackathon Judge**
| Attribute | Details |
|-----------|---------|
| **Primary Goals** | Quickly understand value proposition, see impressive demo, evaluate innovation |
| **Time Sensitivity** | VERY HIGH - needs "wow" in 30 seconds |
| **Tolerance for Complexity** | LOW - wants polished, simple interface |
| **Trust Requirements** | MODERATE - needs professional appearance |
| **Key Screens** | Homepage, Dashboard overview, One complete pipeline flow |

#### Persona D: **Internal Developer**
| Attribute | Details |
|-----------|---------|
| **Primary Goals** | Debug issues, add features, understand architecture |
| **Time Sensitivity** | LOW - detailed exploration acceptable |
| **Tolerance for Complexity** | VERY HIGH - expects access to all internals |
| **Trust Requirements** | LOW - understands demo nature |
| **Key Screens** | All screens, Console, API docs, Pipeline indicators |

### 1.2 Screen Relevance by Persona

```
                    Clinician  Technical  Demo Judge  Developer
Homepage              ★★☆☆☆      ★☆☆☆☆     ★★★★★       ★☆☆☆☆
Dashboard Main        ★★★★★      ★★☆☆☆     ★★★★☆       ★★★☆☆
Pipeline Pages        ★★★★★      ★★★★★     ★★★★☆       ★★★★★
AI Explanation        ★★★★★      ★★★☆☆     ★★★★★       ★★☆☆☆
Pipeline Status Bar   ★★★☆☆      ★★★★★     ★★★★☆       ★★★★★
Analytics             ★★★☆☆      ★★★★☆     ★★☆☆☆       ★★★☆☆
Reports               ★★★★★      ★★☆☆☆     ★★☆☆☆       ★★☆☆☆
Settings              ★★★☆☆      ★☆☆☆☆     ★☆☆☆☆       ★★☆☆☆
```

---

## 2. End-to-End Flow Mapping

### 2.1 Flow Diagram: Landing → Upload → Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LANDING → ANALYSIS FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐     ┌────────────┐     ┌─────────────┐     ┌──────────────┐
    │ Homepage │────▶│ Dashboard  │────▶│ Select      │────▶│ Pipeline     │
    │          │     │ Overview   │     │ Pipeline    │     │ Page         │
    └──────────┘     └────────────┘     └─────────────┘     └──────────────┘
         │                 │                   │                   │
         │                 │                   │                   ▼
         │           Click "Start    Click Module         ┌──────────────┐
         │            Diagnosing"       Card              │ Upload Zone  │
         │                                                │ (Drag/Drop)  │
         │                                                └──────┬───────┘
         │                                                       │
         ▼                                                       ▼
    ┌──────────┐                                          ┌──────────────┐
    │ Auth     │  ◄─── Clerk Authentication               │ File Select/ │
    │ (Clerk)  │                                          │ Recording    │
    └──────────┘                                          └──────┬───────┘
                                                                 │
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ Processing   │
                                                          │ (Status Bar) │
                                                          └──────┬───────┘
                                                                 │
    BACKEND DEPENDENCIES:                                        ▼
    - /api/{pipeline}/analyze                             ┌──────────────┐
    - /api/explain                                        │ Results      │
    - /api/voice                                          │ Display      │
                                                          └──────┬───────┘
                                                                 │
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ AI Explain   │
                                                          │ + Voice      │
                                                          └──────────────┘
```

### 2.2 Flow Diagram: Dashboard Monitoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DASHBOARD MONITORING FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
    │ Dashboard    │────▶│ View Stats     │────▶│ Recent Cases    │
    │ Entry        │     │ Cards (4)      │     │ List            │
    └──────────────┘     └────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                         ┌────────────────┐     ┌─────────────────┐
                         │ Click "View    │◄────│ See Priority    │
                         │ All Cases"     │     │ Indicators      │
                         └────────────────┘     └─────────────────┘
                                │
                                ▼
                         ┌────────────────┐
                         │ [NO ACTION]    │  ◄─── BREAKPOINT: Button 
                         │ No navigation  │       non-functional
                         └────────────────┘

    PIPELINE MONITORING:
    ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
    │ Status Bar   │────▶│ Active Jobs    │────▶│ Stage Progress  │
    │ (Bottom)     │     │ Indicator      │     │ Dots            │
    └──────────────┘     └────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                 ┌─────────────────┐
                                                 │ Completion      │
                                                 │ Message         │
                                                 └─────────────────┘
                                                        │
                                                        ▼
                                                 ┌─────────────────┐
                                                 │ Auto-Clear      │
                                                 │ (8 seconds)     │
                                                 └─────────────────┘
```

### 2.3 Cognitive Load Analysis by Flow Step

| Step | Cognitive Load | Issues Identified |
|------|----------------|-------------------|
| Homepage → Dashboard | LOW | Clear CTA, good contrast |
| Dashboard → Pipeline | MEDIUM | Many module cards, no clear priority |
| Upload Area | LOW | Clean drag/drop, clear instructions |
| Processing Wait | MEDIUM | Status bar small, main content shows only spinner |
| Results Display | HIGH | Dense biomarker grid, many sections at once |
| AI Explanation | MEDIUM | Auto-generates, streaming text helps |
| Voice Playback | LOW | Simple play/pause icon |

### 2.4 Critical Flow Breakdowns Identified

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FLOW BREAKDOWNS & ISSUES                              │
└─────────────────────────────────────────────────────────────────────────────┘

1. UNCLEAR STATE TRANSITIONS
   ├── Upload → Processing: Only spinner shown, no stage details in main view
   ├── Processing → Results: Abrupt transition, no animation
   └── Success → Next Action: "Analyze Another" buried at bottom

2. MISSING FEEDBACK
   ├── Error boundaries exist but error messages are generic
   ├── No toast notifications for async operations
   └── No progress percentage in main view (only status bar)

3. MISLEADING SUCCESS INDICATORS
   ├── Dashboard stats are STATIC (hardcoded values)
   ├── "Recent Cases" list is STATIC (demo data)
   └── Analytics page numbers are STATIC

4. SILENT FAILURES
   ├── If backend is down, frontend shows generic error
   ├── No retry mechanism for failed uploads
   └── Voice generation failure only shows console error

5. NON-FUNCTIONAL UI ELEMENTS
   ├── "View All Cases" button - no navigation
   ├── Settings toggles - non-functional
   ├── Reports "Generate" buttons - disabled but present
   └── Analytics charts - static placeholders
```

---

## 3. Dashboard Audit

### 3.1 Current Dashboard Components Analysis

| Component | Status | Value | Recommendation |
|-----------|--------|-------|----------------|
| **Stats Cards (4)** | STATIC | LOW | REMOVE or connect to real data |
| **Recent Patient Cases** | STATIC | LOW-MEDIUM | REMOVE or connect to job history |
| **Diagnostic Modules Grid** | FUNCTIONAL | HIGH | KEEP - primary navigation |
| **"View All Cases" Button** | BROKEN | NEGATIVE | REMOVE until implemented |

### 3.2 Industry-Standard Components: SHOULD EXIST

| Component | Priority | Rationale |
|-----------|----------|-----------|
| **Pipeline Health Monitor** | HIGH | Real-time backend status indicators |
| **Recent Jobs List** | HIGH | Actual job history from backend |
| **Failure Rate Widget** | MEDIUM | Last 24h success/fail ratio |
| **Latency Distribution** | MEDIUM | P50/P95/P99 response times |
| **Model Version Tracker** | MEDIUM | Active model versions per pipeline |
| **Data Quality Alerts** | LOW | Warnings for poor input quality |
| **Quick Action Buttons** | HIGH | Start most-used pipelines directly |

### 3.3 Components Currently USELESS or MISLEADING

| Component | Issue | Action |
|-----------|-------|--------|
| **Today's Statistics (4 cards)** | Hardcoded demo data | REMOVE or replace with real metrics |
| **Recent Patient Cases** | Static array, not real jobs | REMOVE or connect to backend history |
| **Analytics "Coming Soon"** | Mixed with real content | CONSOLIDATE into clear "planned" section |
| **Notification Badge (3)** | Static number | REMOVE or connect to real events |

### 3.4 Dashboard Component Recommendations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DASHBOARD: ADD / REMOVE / MERGE                       │
└─────────────────────────────────────────────────────────────────────────────┘

ADD:
  ✅ Backend Health Indicator (real /health check)
  ✅ Recent Job History (from actual API calls)  
  ✅ Quick Pipeline Launchers (4 available modules)
  ✅ System Status Summary (APIs, models loaded)

REMOVE:
  ❌ Static "Today's Statistics" cards
  ❌ Static "Recent Patient Cases" component
  ❌ Static notification count badge
  ❌ "View All Cases" non-functional button

MERGE:
  🔀 Available Modules + Coming Soon → Single grid with clear visual separation
  🔀 User Profile → Already in sidebar (remove from header completely)

DEFER TO FUTURE:
  📅 Analytics charts (mark clearly as "Coming in v2")
  📅 Reports generation (mark clearly as "Coming in v2")
  📅 Patient management (mark clearly as "Coming in v2")
```

---

## 4. Pipeline Page Audit

### 4.1 Retinal Analysis Page

**Strengths:**
- Excellent pipeline stage indicator with timing
- Clear biomarker cards with status coloring
- Heatmap toggle for explainability
- Console log for debugging
- AI Explanation integration

**Issues:**
| Issue | Severity | Fix |
|-------|----------|-----|
| Results layout too dense at once | MEDIUM | Progressive disclosure - tabs or accordion |
| Biomarker grid 2x4 overwhelming | MEDIUM | Group by clinical significance |
| Clinical Summary easily missed | HIGH | Move to top, make more prominent |
| Reset button at bottom | LOW | Add reset at top too |
| No clear "next step" after results | MEDIUM | Add prominent referral/save action |

### 4.2 Speech Analysis Page

**Strengths:**
- Clean recording interface
- Good processing feedback
- AI Explanation side panel
- Clear instructions

**Issues:**
| Issue | Severity | Fix |
|-------|----------|-----|
| Max 30s recording may be too short | LOW | Make configurable |
| No waveform visualization during recording | MEDIUM | Add visual feedback |
| Results panel separate from explanation | LOW | Consider unified view |
| Patient context hardcoded (age: 65, female) | HIGH | Make dynamic or remove |

### 4.3 Cardiology Page

**Issues:**
| Issue | Severity | Fix |
|-------|----------|-----|
| Minimal wrapper - actual component in _components | LOW | N/A |
| Demo data if no real file uploaded | MEDIUM | Clarify demo mode |

### 4.4 Coming Soon Pages (Motor, Cognitive, etc.)

**Issues:**
| Issue | Severity | Fix |
|-------|----------|-----|
| Error skeleton shows, then Coming Soon - confusing | HIGH | Show Coming Soon immediately |
| Sidebar shows "Coming Soon" group mixing with available | MEDIUM | Better visual separation |
| Pages are accessible but non-functional | MEDIUM | Consider blocking navigation |

### 4.5 Common Pipeline Page Issues

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE PAGE ISSUES & FIXES                             │
└─────────────────────────────────────────────────────────────────────────────┘

MISSING INTERMEDIATE STATES:
  ├── No "validating input" state before processing
  ├── No "connecting to backend" state  
  ├── No "model loading" state
  └── Fix: Add more granular stage updates to status bar

UNCLEAR ERROR MESSAGING:
  ├── Generic "Analysis failed" messages
  ├── No distinction between: network error, validation error, model error
  └── Fix: Use ErrorDisplay component with specific error codes

POOR EXPLAINABILITY SURFACES:
  ├── Heatmaps available but toggle is subtle
  ├── No segmentation overlay option
  └── Fix: Add prominent "Explain" tab with all visualizations

CLUTTERED RESULT SECTIONS:
  ├── All biomarkers shown at once
  ├── Medical terminology without tooltips
  └── Fix: Add progressive disclosure, tooltips, "Learn more" links
```

---

## 5. Visual Design and Color System Audit

### 5.1 Current Color Usage

| Color | Semantic Meaning | Consistency |
|-------|------------------|-------------|
| Green (`#10b981`, `#22c55e`) | Success, Normal, Low Risk | GOOD |
| Yellow/Amber (`#f59e0b`, `#eab308`) | Warning, Borderline, Moderate | GOOD |
| Red (`#ef4444`, `#dc2626`) | Error, Abnormal, High Risk | GOOD |
| Blue (`#3b82f6`, `#06b6d4`) | Info, Processing, Interactive | INCONSISTENT - too many shades |
| Purple (`#8b5cf6`) | AI/ML features | GOOD |

### 5.2 Identified Color Issues

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Cyan vs Blue inconsistency | Pipeline indicators, buttons | Standardize on one blue family |
| Red used for both errors AND cardiology | Sidebar, pipeline icons | Use red for severity only |
| Processing state uses cyan AND blue | Different components | Standardize to cyan for processing |
| Purple for both "AI" and "Coming Soon" | Various | Differentiate these concepts |

### 5.3 Accessibility Concerns

| Issue | Severity | Location |
|-------|----------|----------|
| Text size 10px for biomarker labels | HIGH | BiomarkerCard component |
| Text size 9px for pipeline stage timing | MEDIUM | PipelineStageIndicator |
| LowCcontrast: zinc-400 on white | MEDIUM | Various descriptive text |
| No focus indicators on some buttons | MEDIUM | Upload zones, reset buttons |

### 5.4 Color System Recommendations

```css
/* RECOMMENDED SEMANTIC COLOR TOKENS */

/* Status Colors - Medical Standards */
--color-normal: #22c55e;      /* Green 500 - Normal, Success */
--color-borderline: #eab308;  /* Yellow 500 - Borderline, Warning */
--color-abnormal: #ef4444;    /* Red 500 - Abnormal, Error */
--color-critical: #dc2626;    /* Red 600 - Critical, Urgent */

/* Processing States */
--color-processing: #06b6d4;  /* Cyan 500 - Active Processing */
--color-pending: #94a3b8;     /* Slate 400 - Pending, Inactive */
--color-complete: #22c55e;    /* Green 500 - Complete */

/* Pipeline Identity Colors */
--color-speech: #3b82f6;      /* Blue 500 */
--color-retinal: #06b6d4;     /* Cyan 500 */
--color-cardiology: #ef4444;  /* Red 500 */
--color-radiology: #8b5cf6;   /* Violet 500 */

/* AI Features */
--color-ai-primary: #a855f7;  /* Purple 500 */
--color-ai-secondary: #818cf8; /* Indigo 400 */

/* Typography Minimum Sizes */
--font-size-min: 11px;        /* Absolute minimum */
--font-size-body: 13px;       /* Body text */
--font-size-label: 12px;      /* Form labels */
```

---

## 6. Frontend Technical Analysis

### 6.1 State Management Flow

**Current Architecture:**
- React Context for PipelineStatus (global)
- useState for local component state
- No centralized state management (Redux, Zustand)

**Issues:**
| Issue | Impact | Recommendation |
|-------|--------|----------------|
| Pipeline status polls localStorage every 100ms | Performance | Use React Context or custom events |
| No state persistence between sessions | UX | Add localStorage for incomplete jobs |
| Results lost on page refresh | UX Critical | Persist last N results |

### 6.2 Loading vs Idle States

| Component | Loading State | Idle State | Issue |
|-----------|---------------|------------|-------|
| Dashboard | Skeleton | Content | GOOD |
| Pipeline Pages | Spinner | Upload zone | Spinner too generic |
| AI Explanation | Text + cursor | Static | Auto-generates - no idle after first |
| Results | N/A | Full display | Abrupt appearance |

### 6.3 Error Boundary Handling

**Current:** ErrorBoundary components exist but fallbacks are generic.

**Recommendation:**
```tsx
// Each pipeline should have specific error recovery
<ErrorBoundary 
  fallback={<PipelineErrorRecovery 
    pipeline="retinal" 
    actions={['retry', 'change-file', 'contact-support']}
  />}
>
```

### 6.4 Frontend Technical Fixes Needed

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRONTEND TECHNICAL FIXES                                │
└─────────────────────────────────────────────────────────────────────────────┘

STATE MODEL CHANGES:
  1. Replace localStorage polling with custom event listeners
  2. Add React Query or SWR for API state management
  3. Persist analysis results to localStorage for session recovery

POLLING vs WEBSOCKET:
  Current: No real-time updates, one-shot API calls
  Recommendation: Keep current for simplicity (sub-2s responses)
  Future: WebSocket for long-running jobs if needed

CACHE AND REFRESH:
  1. Add result caching per session (max 10 results)
  2. Add clear cache button in settings
  3. Show cached results indicator

RACE CONDITIONS:
  1. Add abort controllers for in-flight requests
  2. Disable re-submit during processing
  3. Show "already processing" message if duplicate submit
```

---

## 7. Backend Behavior Affecting UX

### 7.1 Current Backend Support Analysis

| Feature | Status | UX Impact |
|---------|--------|-----------|
| Job lifecycle tracking | PARTIAL | No job history API |
| Partial failures | GOOD | Pipeline returns errors array |
| Progress reporting | NONE | No streaming progress |
| Result versioning | NONE | No way to compare results |
| Health check | GOOD | /health endpoint exists |

### 7.2 Missing Endpoints for UX

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `GET /api/jobs` | List recent analysis jobs | HIGH |
| `GET /api/jobs/:id` | Get specific job result | HIGH |
| `GET /api/system/status` | All pipeline health status | MEDIUM |
| `WebSocket /ws/job/:id` | Real-time job progress | LOW |
| `GET /api/analytics/summary` | Real dashboard metrics | MEDIUM |

### 7.3 Backend Gaps Affecting Trust

| Gap | UX Impact | Recommendation |
|-----|-----------|----------------|
| No job persistence | Can't show job history | Add SQLite job logging |
| No model version in API | Can't show "analyzed with v4.0" | Add model_version header |
| No processing time in all responses | Inconsistent UX | Standardize response format |
| Generic error messages | User confusion | Return error codes + messages |

---

## 8. Legacy Cleanup and Scope Control

### 8.1 Unused Pages Identified

| Page/Component | Status | Action |
|----------------|--------|--------|
| `/dashboard/dermatology` | Coming Soon placeholder | KEEP but add clear banner |
| `/dashboard/multimodal` | Coming Soon placeholder | KEEP but add clear banner |
| `/dashboard/nri-fusion` | Coming Soon placeholder | KEEP but add clear banner |
| `/dashboard/pathology` (route exists in data) | Missing page | REMOVE from sidebar |
| `/dashboard/neurology` (route exists in data) | Missing page | REMOVE from sidebar |

### 8.2 Half-Implemented Features

| Feature | Current State | Action |
|---------|---------------|--------|
| Notification system | Badge shows "3", no real notifications | REMOVE badge |
| Settings page toggles | UI only, no persistence | REMOVE toggles or add functionality |
| Reports generation | All buttons disabled | REMOVE whole section, add "v2" banner |
| "View All Cases" button | Non-functional | REMOVE |
| Analytics charts | Static data only | ADD clear "demo data" indicator |

### 8.3 Misleading Labels/Elements

| Element | Issue | Fix |
|---------|-------|-----|
| "Total Patients: 23" | Static number implies real data | Change to "Sample Data" or remove |
| "Completed Assessments: 18" | Static | Change to "Sample Data" or remove |
| "Critical Alerts: 2" | Static, creates false urgency | Remove |
| "4+ years" in About (not seen in code) | N/A | Verify accuracy |

### 8.4 Demo-Only Artifacts

| Artifact | Location | Action |
|----------|----------|--------|
| Hardcoded patient cases | dashboard/page.tsx | MARK clearly as demo |
| Hardcoded analytics data | analytics/page.tsx | MARK clearly as demo |
| Hardcoded settings values | settings/page.tsx | MARK clearly as demo |
| patientContext hardcoded in speech | speech/page.tsx | REMOVE or make dynamic |

---

## 9. Prioritized Improvement Plan

### 9.1 Quick Wins (UI Only) - 1-2 days

| Change | File | Impact | Status |
|--------|------|--------|--------|
| Remove static notification badge | DashboardHeader.tsx | Trust | **DONE** |
| Remove "View All Cases" button | dashboard/page.tsx | Trust | **DONE** |
| Replace static stats with System Health | dashboard/page.tsx | Trust | **DONE** |
| Increase minimum font size to 11px | Various components | Accessibility | **DONE** |
| Add prominent Clinical Summary box styling | retinal/page.tsx | Clarity | **DONE** |
| Remove disabled Settings toggles | settings/page.tsx | Trust | Pending |
| Standardize processing color to cyan-500 | All pipelines | Consistency | **DONE** |

### 9.2 Medium Effort (Frontend + Backend) - 1-2 weeks

| Change | Frontend | Backend | Impact |
|--------|----------|---------|--------|
| Real job history | New JobHistory component | `GET /api/jobs` endpoint | HIGH |
| Pipeline health status | SystemStatus widget | Expose health per pipeline | MEDIUM |
| Better error messages | ErrorDisplay enhancements | Error code system | HIGH |
| Progress streaming | SSE support | Progress events | MEDIUM |
| Result persistence | localStorage caching | Add session_id tracking | MEDIUM |
| Real notification system | Toast/notification component | Event logging | MEDIUM |

### 9.3 Strategic Upgrades - 2-4 weeks

| Change | Scope | Impact |
|--------|-------|--------|
| Observability layer | Add OpenTelemetry traces | Debug, Demo Quality |
| Audit trails | Log all analyses with timestamp | Compliance, Trust |
| Longitudinal result views | Compare results over time | Clinical Value |
| Progressive disclosure UI | Tabs/accordion for results | Clarity |
| Accessibility audit | WCAG 2.1 AA compliance | Compliance |

### 9.4 Ranked Impact Matrix

```
                    HIGH IMPACT
                         │
    ┌────────────────────┼────────────────────┐
    │  QUICK WINS        │  STRATEGIC FOCUS   │
    │                    │                    │
    │  - Remove static   │  - Job history     │
    │    dashboard data  │  - Health monitors │
    │  - Better fonts    │  - Real analytics  │
    │  - Error clarity   │  - Result persist  │
    │                    │                    │
LOW ──────────────────────────────────────────── HIGH
EFFORT │                    │                    │ EFFORT
    │  DEFER             │  FUTURE ROADMAP    │
    │                    │                    │
    │  - Settings toggle │  - WebSocket       │
    │  - Subtle colors   │  - Full audit      │
    │                    │  - EMR integration │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT
```

---

## 10. Summary Tables

### 10.1 Dashboard Component Add/Remove List

| Action | Component | Priority |
|--------|-----------|----------|
| REMOVE | Static "Today's Statistics" cards | P0 |
| REMOVE | Static "Recent Patient Cases" | P0 |
| REMOVE | Static notification badge (3) | P0 |
| REMOVE | "View All Cases" button | P0 |
| ADD | "Demo Mode" banner/indicator | P0 |
| ADD | Real pipeline health status | P1 |
| ADD | Recent job history (real) | P1 |
| ADD | Quick action launch buttons | P2 |
| KEEP | Diagnostic modules grid | - |
| KEEP | Page header with breadcrumbs | - |

### 10.2 Pipeline Page Issues & Fixes

| Pipeline | Issue | Fix Priority |
|----------|-------|--------------|
| All | Generic spinner during processing | P1 |
| Retinal | Dense biomarker grid | P2 |
| Retinal | Clinical summary not prominent | P1 |
| Speech | Hardcoded patient context | P0 |
| Speech | No waveform visualization | P3 |
| Cardiology | Demo mode confusion | P1 |
| Coming Soon | Pages accessible but broken | P1 |

### 10.3 Visual Design Corrections

| Issue | Location | Fix |
|-------|----------|-----|
| 9-10px text | Biomarker cards, pipeline stages | Min 11px |
| Cyan vs Blue inconsistency | Various | Standardize |
| Low contrast description text | zinc-400 on white | Use zinc-600 |
| No focus indicators | Upload zones, buttons | Add ring styles |

### 10.4 Frontend Technical Fixes

| Issue | Current | Fix |
|-------|---------|-----|
| LocalStorage polling | 100ms interval | Custom events |
| No result persistence | Lost on refresh | LocalStorage cache |
| No abort controllers | Race conditions possible | Add abort logic |
| Generic error display | Same message for all | Error code mapping |

### 10.5 Backend Support Gaps

| Gap | Impact | Fix Priority |
|-----|--------|--------------|
| No job history API | Can't show real jobs | P1 |
| No streaming progress | User waits blind | P2 |
| Generic errors | User confusion | P1 |
| No model version exposure | Missing trust signal | P3 |

### 10.6 Legacy Cleanup List

| Item | Action | Priority |
|------|--------|----------|
| Static dashboard stats | Remove or label "Demo" | P0 |
| Non-functional buttons | Remove | P0 |
| Disabled settings UI | Remove until implemented | P1 |
| Misleading notification count | Remove | P0 |
| Coming Soon pages in main nav | Add clear visual separation | P1 |

---

## Appendix A: File Reference

Key frontend files analyzed:
- `frontend/src/app/dashboard/page.tsx` - Main dashboard
- `frontend/src/app/dashboard/layout.tsx` - Dashboard layout
- `frontend/src/app/dashboard/_components/DashboardSidebar.tsx` - Navigation
- `frontend/src/app/dashboard/_components/DashboardHeader.tsx` - Header
- `frontend/src/app/dashboard/retinal/page.tsx` - Retinal pipeline
- `frontend/src/app/dashboard/speech/page.tsx` - Speech pipeline
- `frontend/src/components/pipeline/PipelineStatusBar.tsx` - Status bar
- `frontend/src/components/explanation/ExplanationPanel.tsx` - AI explanation
- `frontend/src/data/diagnostic-modules.ts` - Module definitions

Key backend files analyzed:
- `backend/app/main.py` - FastAPI entry
- `backend/app/routers/api.py` - Router hub

---

*End of Product Analysis Document*
