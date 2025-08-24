/**
 * Accessibility Testing Page
 * Comprehensive WCAG 2.1 AA compliance testing and demonstration
 */

import React, { useState, useEffect, useRef } from 'react';
import { accessibilityTester, AccessibilityTestResult } from '@/lib/accessibility/testing';
import { AccessibleAssessmentWorkflow } from '@/components/assessment/AccessibleAssessmentWorkflow';
import { useScreenReader, useSkipLink } from '@/hooks/useAccessibility';
import { LoadingButton } from '@/components/ui/LoadingStates';
import { CheckCircle, AlertTriangle, Play, Download, Eye, Keyboard, Volume2 } from 'lucide-react';

export default function AccessibilityTestPage() {
  const [testResults, setTestResults] = useState<AccessibilityTestResult[]>([]);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [sessionId] = useState(() => `accessibility-test-${Date.now()}`);

  const { announce } = useScreenReader();
  const { SkipLink, targetRef } = useSkipLink();
  const testContainerRef = useRef<HTMLDivElement>(null);

  // Run accessibility tests
  const runAccessibilityTests = async () => {
    setIsRunningTests(true);
    announce('Starting accessibility tests');

    try {
      const results = await accessibilityTester.runAllTests(
        testContainerRef.current || document.body,
      );
      setTestResults(results);

      const passedCount = results.filter(r => r.passed).length;
      const totalCount = results.length;
      announce(`Accessibility tests completed. ${passedCount} of ${totalCount} tests passed.`);
    } catch (error) {
      console.error('Accessibility tests failed:', error);
      announce('Accessibility tests failed');
    } finally {
      setIsRunningTests(false);
    }
  };

  // Download test report
  const downloadReport = () => {
    const report = accessibilityTester.generateReport();
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'neuralens-accessibility-report.md';
    link.click();
    URL.revokeObjectURL(url);
    announce('Accessibility report downloaded');
  };

  // Calculate compliance percentages
  const getComplianceStats = () => {
    if (testResults.length === 0) return { overall: 0, levelA: 0, levelAA: 0, levelAAA: 0 };

    const total = testResults.length;
    const passed = testResults.filter(r => r.passed).length;

    const levelA = testResults.filter(r => r.level === 'A');
    const levelAA = testResults.filter(r => r.level === 'AA');
    const levelAAA = testResults.filter(r => r.level === 'AAA');

    const passedA = levelA.filter(r => r.passed).length;
    const passedAA = levelAA.filter(r => r.passed).length;
    const passedAAA = levelAAA.filter(r => r.passed).length;

    return {
      overall: (passed / total) * 100,
      levelA: levelA.length > 0 ? (passedA / levelA.length) * 100 : 100,
      levelAA: levelAA.length > 0 ? (passedAA / levelAA.length) * 100 : 100,
      levelAAA: levelAAA.length > 0 ? (passedAAA / levelAAA.length) * 100 : 100,
    };
  };

  const stats = getComplianceStats();

  return (
    <div className='min-h-screen bg-gray-50 py-8' ref={testContainerRef}>
      {/* Skip link */}
      <SkipLink href='#main-content'>Skip to main content</SkipLink>

      <div className='mx-auto max-w-6xl px-4'>
        <div className='rounded-lg bg-white shadow-lg'>
          {/* Header */}
          <header className='border-b border-gray-200 p-6'>
            <h1 className='text-3xl font-bold text-gray-900'>NeuraLens Accessibility Testing</h1>
            <p className='mt-2 text-gray-600'>
              Comprehensive WCAG 2.1 AA compliance testing and demonstration
            </p>
          </header>

          {/* Main content */}
          <main id='main-content' ref={targetRef} tabIndex={-1}>
            {/* Test Controls */}
            <section
              className='border-b border-gray-200 p-6'
              aria-labelledby='test-controls-heading'
            >
              <h2 id='test-controls-heading' className='mb-4 text-xl font-semibold text-gray-900'>
                Accessibility Test Controls
              </h2>

              <div className='flex flex-wrap items-center gap-4'>
                <LoadingButton
                  loading={isRunningTests}
                  onClick={runAccessibilityTests}
                  loadingText='Running tests...'
                  className='flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                  aria-describedby='run-tests-description'
                >
                  <Play className='h-4 w-4' aria-hidden='true' />
                  Run Accessibility Tests
                </LoadingButton>
                <div id='run-tests-description' className='sr-only'>
                  Run comprehensive WCAG 2.1 accessibility tests on the current page
                </div>

                {testResults.length > 0 && (
                  <button
                    onClick={downloadReport}
                    className='flex items-center gap-2 rounded-lg border border-blue-600 px-4 py-2 text-blue-600 transition-colors hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                    aria-label='Download accessibility test report'
                  >
                    <Download className='h-4 w-4' aria-hidden='true' />
                    Download Report
                  </button>
                )}

                <button
                  onClick={() => {
                    setShowWorkflow(!showWorkflow);
                    announce(
                      showWorkflow ? 'Assessment workflow hidden' : 'Assessment workflow shown',
                    );
                  }}
                  className='flex items-center gap-2 rounded-lg border border-green-600 px-4 py-2 text-green-600 transition-colors hover:bg-green-50 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2'
                  aria-expanded={showWorkflow}
                  aria-controls='workflow-demo'
                >
                  <Eye className='h-4 w-4' aria-hidden='true' />
                  {showWorkflow ? 'Hide' : 'Show'} Workflow Demo
                </button>
              </div>
            </section>

            {/* Test Results */}
            {testResults.length > 0 && (
              <section
                className='border-b border-gray-200 p-6'
                aria-labelledby='test-results-heading'
              >
                <h2 id='test-results-heading' className='mb-6 text-xl font-semibold text-gray-900'>
                  Test Results
                </h2>

                {/* Compliance Summary */}
                <div className='mb-6 grid grid-cols-1 gap-4 md:grid-cols-4'>
                  <div className='rounded-lg bg-gray-50 p-4 text-center'>
                    <div className='text-2xl font-bold text-gray-900'>
                      {stats.overall.toFixed(1)}%
                    </div>
                    <div className='text-sm text-gray-600'>Overall Compliance</div>
                  </div>

                  <div className='rounded-lg bg-green-50 p-4 text-center'>
                    <div className='text-2xl font-bold text-green-900'>
                      {stats.levelA.toFixed(1)}%
                    </div>
                    <div className='text-sm text-green-700'>WCAG Level A</div>
                  </div>

                  <div className='rounded-lg bg-blue-50 p-4 text-center'>
                    <div className='text-2xl font-bold text-blue-900'>
                      {stats.levelAA.toFixed(1)}%
                    </div>
                    <div className='text-sm text-blue-700'>WCAG Level AA</div>
                  </div>

                  <div className='rounded-lg bg-purple-50 p-4 text-center'>
                    <div className='text-2xl font-bold text-purple-900'>
                      {stats.levelAAA.toFixed(1)}%
                    </div>
                    <div className='text-sm text-purple-700'>WCAG Level AAA</div>
                  </div>
                </div>

                {/* Detailed Results */}
                <div className='space-y-4'>
                  <h3 className='text-lg font-semibold text-gray-900'>
                    Detailed Test Results ({testResults.length} tests)
                  </h3>

                  <div className='max-h-96 overflow-y-auto rounded-lg border border-gray-200'>
                    <table className='w-full' role='table' aria-label='Accessibility test results'>
                      <thead className='sticky top-0 bg-gray-50'>
                        <tr>
                          <th
                            scope='col'
                            className='px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500'
                          >
                            Status
                          </th>
                          <th
                            scope='col'
                            className='px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500'
                          >
                            Level
                          </th>
                          <th
                            scope='col'
                            className='px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500'
                          >
                            Criterion
                          </th>
                          <th
                            scope='col'
                            className='px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500'
                          >
                            Description
                          </th>
                          <th
                            scope='col'
                            className='px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500'
                          >
                            Recommendation
                          </th>
                        </tr>
                      </thead>
                      <tbody className='divide-y divide-gray-200 bg-white'>
                        {testResults.map((result, index) => (
                          <tr key={index} className={result.passed ? 'bg-green-50' : 'bg-red-50'}>
                            <td className='whitespace-nowrap px-4 py-3'>
                              {result.passed ? (
                                <CheckCircle
                                  className='h-5 w-5 text-green-600'
                                  aria-label='Passed'
                                />
                              ) : (
                                <AlertTriangle
                                  className='h-5 w-5 text-red-600'
                                  aria-label='Failed'
                                />
                              )}
                            </td>
                            <td className='whitespace-nowrap px-4 py-3'>
                              <span
                                className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${
                                  result.level === 'A'
                                    ? 'bg-green-100 text-green-800'
                                    : result.level === 'AA'
                                      ? 'bg-blue-100 text-blue-800'
                                      : 'bg-purple-100 text-purple-800'
                                }`}
                              >
                                {result.level}
                              </span>
                            </td>
                            <td className='px-4 py-3 text-sm font-medium text-gray-900'>
                              {result.criterion}
                            </td>
                            <td className='px-4 py-3 text-sm text-gray-700'>
                              {result.description}
                            </td>
                            <td className='px-4 py-3 text-sm text-gray-600'>
                              {result.recommendation || 'No issues found'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>
            )}

            {/* Accessibility Features Demo */}
            <section className='border-b border-gray-200 p-6' aria-labelledby='features-heading'>
              <h2 id='features-heading' className='mb-6 text-xl font-semibold text-gray-900'>
                Accessibility Features Demonstration
              </h2>

              <div className='grid grid-cols-1 gap-6 md:grid-cols-3'>
                {/* Keyboard Navigation */}
                <div className='rounded-lg border border-gray-200 p-4'>
                  <div className='mb-3 flex items-center gap-3'>
                    <Keyboard className='h-6 w-6 text-blue-600' aria-hidden='true' />
                    <h3 className='font-semibold text-gray-900'>Keyboard Navigation</h3>
                  </div>
                  <ul className='space-y-1 text-sm text-gray-600'>
                    <li>• Tab to navigate between elements</li>
                    <li>• Enter/Space to activate buttons</li>
                    <li>• Arrow keys for lists and menus</li>
                    <li>• Escape to close modals</li>
                    <li>• Skip links for main content</li>
                  </ul>
                </div>

                {/* Screen Reader Support */}
                <div className='rounded-lg border border-gray-200 p-4'>
                  <div className='mb-3 flex items-center gap-3'>
                    <Volume2 className='h-6 w-6 text-green-600' aria-hidden='true' />
                    <h3 className='font-semibold text-gray-900'>Screen Reader Support</h3>
                  </div>
                  <ul className='space-y-1 text-sm text-gray-600'>
                    <li>• ARIA labels and descriptions</li>
                    <li>• Live region announcements</li>
                    <li>• Semantic HTML structure</li>
                    <li>• Progress announcements</li>
                    <li>• Error message association</li>
                  </ul>
                </div>

                {/* Visual Accessibility */}
                <div className='rounded-lg border border-gray-200 p-4'>
                  <div className='mb-3 flex items-center gap-3'>
                    <Eye className='h-6 w-6 text-purple-600' aria-hidden='true' />
                    <h3 className='font-semibold text-gray-900'>Visual Accessibility</h3>
                  </div>
                  <ul className='space-y-1 text-sm text-gray-600'>
                    <li>• WCAG AA color contrast ratios</li>
                    <li>• Focus indicators on all elements</li>
                    <li>• Reduced motion support</li>
                    <li>• Scalable text and UI elements</li>
                    <li>• Alternative text for images</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Assessment Workflow Demo */}
            {showWorkflow && (
              <section id='workflow-demo' className='p-6' aria-labelledby='workflow-heading'>
                <h2 id='workflow-heading' className='mb-6 text-xl font-semibold text-gray-900'>
                  Accessible Assessment Workflow Demo
                </h2>
                <AccessibleAssessmentWorkflow
                  sessionId={sessionId}
                  onComplete={results => {
                    announce('Assessment workflow demonstration completed');
                    console.log('Demo results:', results);
                  }}
                />
              </section>
            )}
          </main>

          {/* Footer */}
          <footer className='border-t border-gray-200 bg-gray-50 p-6'>
            <div className='text-center text-sm text-gray-600'>
              <p>
                This page demonstrates WCAG 2.1 AA compliance features including keyboard
                navigation, screen reader support, color contrast, and semantic HTML structure.
              </p>
              <p className='mt-2'>
                Use keyboard navigation (Tab, Enter, Arrow keys) and screen readers (NVDA, JAWS,
                VoiceOver) to test accessibility.
              </p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}
