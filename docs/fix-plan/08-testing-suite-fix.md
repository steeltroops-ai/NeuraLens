# Testing Suite Fix Plan

## Overview

This document outlines all fixes needed for the testing infrastructure, including unit tests, integration tests, and end-to-end tests for both backend and frontend.

## Current State

### Backend Tests
- `backend/test_all_endpoints.py` - Main test file
- `backend/test_complete_integration.py` - Integration tests
- `backend/test_retinal_analysis.py` - Retinal-specific tests

### Frontend Tests
- `frontend/tests/setup.ts` - Test configuration
- `frontend/tests/components/` - Component tests
- `frontend/tests/e2e/` - End-to-end tests
- `frontend/tests/integration/` - Integration tests

## Issues to Fix

### Backend Testing Issues

#### B-TST-001: Test Runner Configuration
**Priority**: P0
**Description**: pytest not configured properly.
**Files**: `backend/pytest.ini` (create), `backend/conftest.py` (create)
**Fix**:
- Create pytest.ini with proper configuration
- Create conftest.py with fixtures
- Configure test database
- Add coverage reporting

```ini
# backend/pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
asyncio_mode = auto
addopts = --cov=app --cov-report=html --cov-report=term-missing
```

#### B-TST-002: Test Database
**Priority**: P0
**Description**: Tests using production database.
**Files**: `backend/conftest.py`
**Fix**:
- Create separate test database
- Reset database between tests
- Use transactions for isolation

#### B-TST-003: Missing Unit Tests
**Priority**: P0
**Description**: Many functions lack unit tests.
**Files**: `backend/tests/` (create directory structure)
**Fix**:
- Create test files for each module:
  - `tests/test_speech.py`
  - `tests/test_retinal.py`
  - `tests/test_motor.py`
  - `tests/test_cognitive.py`
  - `tests/test_nri.py`
  - `tests/test_validation.py`

#### B-TST-004: Mock External Services
**Priority**: P1
**Description**: Tests calling real ML models.
**Files**: Test files
**Fix**:
- Mock ML model calls
- Create test fixtures for model outputs
- Add mock for file uploads

#### B-TST-005: Async Test Support
**Priority**: P0
**Description**: Async tests not running correctly.
**Files**: `backend/conftest.py`
**Fix**:
- Install pytest-asyncio
- Configure async test mode
- Use async fixtures

#### B-TST-006: API Integration Tests
**Priority**: P1
**Description**: API integration tests incomplete.
**Files**: `backend/tests/test_api_integration.py`
**Fix**:
- Test all API endpoints
- Test error responses
- Test authentication
- Test rate limiting

### Frontend Testing Issues

#### F-TST-001: Jest Configuration
**Priority**: P0
**Description**: Jest not configured for Next.js 15.
**Files**: `frontend/jest.config.js` (create)
**Fix**:
- Create Jest configuration
- Configure module aliases
- Set up test environment

```javascript
// frontend/jest.config.js
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',
});

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  testEnvironment: 'jest-environment-jsdom',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
};

module.exports = createJestConfig(customJestConfig);
```

#### F-TST-002: Component Tests
**Priority**: P0
**Description**: Component tests missing or incomplete.
**Files**: `frontend/tests/components/`
**Fix**:
- Add tests for all UI components
- Test user interactions
- Test accessibility
- Test error states

#### F-TST-003: Hook Tests
**Priority**: P1
**Description**: Custom hooks not tested.
**Files**: `frontend/tests/hooks/`
**Fix**:
- Test useApi hook
- Test assessment hooks
- Test state management hooks

#### F-TST-004: E2E Tests
**Priority**: P1
**Description**: E2E tests not comprehensive.
**Files**: `frontend/tests/e2e/`
**Fix**:
- Add Playwright configuration
- Test complete assessment flow
- Test navigation
- Test form submissions

#### F-TST-005: Mock API Responses
**Priority**: P0
**Description**: API mocks not comprehensive.
**Files**: `frontend/tests/__mocks__/`
**Fix**:
- Mock all API endpoints
- Create response fixtures
- Handle error scenarios

#### F-TST-006: Snapshot Tests
**Priority**: P2
**Description**: No snapshot tests for UI consistency.
**Files**: Component test files
**Fix**:
- Add snapshot tests for key components
- Update snapshots on intentional changes
- Review snapshot diffs in PRs

### Test Coverage Requirements

| Area | Current | Target |
|------|---------|--------|
| Backend API | ~40% | 80% |
| Backend ML | ~30% | 70% |
| Frontend Components | ~20% | 70% |
| Frontend Hooks | ~10% | 80% |
| E2E Flows | ~10% | 60% |

## Test Structure

### Backend Test Structure
```
backend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_speech.py
│   │   ├── test_retinal.py
│   │   ├── test_motor.py
│   │   ├── test_cognitive.py
│   │   └── test_nri.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   └── test_database.py
│   └── fixtures/
│       ├── audio_samples/
│       ├── image_samples/
│       └── mock_responses.py
├── pytest.ini
└── .coveragerc
```

### Frontend Test Structure
```
frontend/
├── tests/
│   ├── setup.ts
│   ├── __mocks__/
│   │   ├── api.ts
│   │   └── next-router.ts
│   ├── components/
│   │   ├── ui/
│   │   ├── assessment/
│   │   └── dashboard/
│   ├── hooks/
│   ├── integration/
│   └── e2e/
│       ├── assessment-flow.spec.ts
│       └── dashboard.spec.ts
├── jest.config.js
└── playwright.config.ts
```

## Test Cases to Add

### Backend Unit Tests

```python
# tests/unit/test_speech.py
class TestSpeechAnalyzer:
    def test_analyze_valid_audio(self):
        """Test analysis with valid audio data"""
        pass
    
    def test_analyze_invalid_format(self):
        """Test rejection of invalid audio format"""
        pass
    
    def test_biomarker_extraction(self):
        """Test all biomarkers are extracted"""
        pass
    
    def test_risk_score_calculation(self):
        """Test risk score is in valid range"""
        pass
```

### Frontend Component Tests

```typescript
// tests/components/assessment/SpeechAssessmentStep.test.tsx
describe('SpeechAssessmentStep', () => {
  it('renders recording button', () => {});
  it('starts recording on button click', () => {});
  it('stops recording and submits', () => {});
  it('displays error on failure', () => {});
  it('shows loading state during analysis', () => {});
  it('displays results after analysis', () => {});
});
```

### E2E Tests

```typescript
// tests/e2e/assessment-flow.spec.ts
test.describe('Assessment Flow', () => {
  test('complete speech assessment', async ({ page }) => {
    await page.goto('/assessment');
    // Navigate through steps
    // Complete speech recording
    // Verify results displayed
  });
  
  test('complete full assessment', async ({ page }) => {
    // Test all modalities
  });
});
```

## Kiro Spec Template

```
Feature: Fix Testing Suite

As a developer, I want to fix all testing infrastructure issues
so that we have comprehensive test coverage and reliable CI/CD.

Requirements:
1. Configure pytest for backend with async support
2. Configure Jest for frontend with Next.js 15
3. Add unit tests for all modules
4. Add integration tests for API endpoints
5. Add E2E tests for critical flows
6. Achieve 70%+ code coverage
```

## Verification Checklist

- [ ] pytest configured and running
- [ ] Jest configured and running
- [ ] Test database isolated
- [ ] All API endpoints tested
- [ ] All components have tests
- [ ] Custom hooks tested
- [ ] E2E tests for assessment flow
- [ ] Mocks comprehensive
- [ ] Coverage > 70%
- [ ] CI/CD runs all tests
