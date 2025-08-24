'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Play, CheckCircle, XCircle, Clock, Zap, AlertCircle } from 'lucide-react';
import {
  runComprehensiveMLIntegrationTest,
  generateMLTestReport,
  type ComprehensiveMLTestResults,
  type MLIntegrationTestResult,
} from '../../lib/ml/ml-integration-test';

interface MLIntegrationTestRunnerProps {
  className?: string;
}

export default function MLIntegrationTestRunner({ className = '' }: MLIntegrationTestRunnerProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<ComprehensiveMLTestResults | null>(null);
  const [currentTest, setCurrentTest] = useState<string>('');

  const runTests = async () => {
    setIsRunning(true);
    setResults(null);
    setCurrentTest('Initializing ML integration tests...');

    try {
      // Simulate test progress updates
      const testSteps = [
        'Testing session generation...',
        'Testing request validation...',
        'Testing speech analysis integration...',
        'Testing retinal analysis integration...',
        'Testing motor analysis integration...',
        'Testing cognitive analysis integration...',
        'Testing multi-modal assessment...',
        'Testing progress tracking...',
        'Testing error handling...',
        'Generating comprehensive report...',
      ];

      for (let i = 0; i < testSteps.length - 1; i++) {
        setCurrentTest(testSteps[i] ?? 'Unknown test');
        await new Promise(resolve => setTimeout(resolve, 300));
      }

      setCurrentTest(testSteps[testSteps.length - 1] ?? 'Test completed');
      const testResults = await runComprehensiveMLIntegrationTest();
      setResults(testResults);
      setCurrentTest('ML integration tests completed!');
    } catch (error) {
      console.error('ML integration test execution failed:', error);
      setCurrentTest('ML integration test execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  const getStatusIcon = (result: MLIntegrationTestResult) => {
    if (result.success) {
      return <CheckCircle className='h-5 w-5 text-green-500' />;
    }
    return <XCircle className='h-5 w-5 text-red-500' />;
  };

  const getStatusColor = (success: boolean) => {
    return success ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50';
  };

  const getCategoryIcon = (testKey: string) => {
    const icons = {
      sessionGeneration: '🔑',
      requestValidation: '✅',
      speechAnalysis: '🎤',
      retinalAnalysis: '👁️',
      motorAnalysis: '🏃',
      cognitiveAnalysis: '🧠',
      multiModalAssessment: '🔄',
      progressTracking: '📊',
      errorHandling: '⚠️',
    };
    return icons[testKey as keyof typeof icons] || '🔧';
  };

  return (
    <div className={`rounded-xl bg-white p-6 shadow-lg ${className}`}>
      <div className='mb-6 flex items-center justify-between'>
        <div>
          <div className='mb-2 flex items-center gap-3'>
            <Brain className='h-8 w-8 text-purple-600' />
            <h2 className='text-2xl font-bold text-gray-900'>ML Integration Test</h2>
          </div>
          <p className='text-gray-600'>
            Test complete ML integration layer with backend API communication
          </p>
        </div>
        <button
          onClick={runTests}
          disabled={isRunning}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 font-medium transition-all ${
            isRunning
              ? 'cursor-not-allowed bg-gray-100 text-gray-400'
              : 'bg-purple-600 text-white hover:bg-purple-700 hover:shadow-md'
          } `}
        >
          {isRunning ? (
            <>
              <Clock className='h-4 w-4 animate-spin' />
              Running Tests...
            </>
          ) : (
            <>
              <Play className='h-4 w-4' />
              Run ML Tests
            </>
          )}
        </button>
      </div>

      {/* Current Test Status */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className='mb-6 rounded-lg border border-purple-200 bg-purple-50 p-4'
        >
          <div className='flex items-center gap-3'>
            <div className='h-2 w-2 animate-pulse rounded-full bg-purple-500' />
            <span className='font-medium text-purple-700'>{currentTest}</span>
          </div>
        </motion.div>
      )}

      {/* Test Results */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className='space-y-6'
        >
          {/* Overall Results */}
          <div className={`rounded-lg border p-4 ${getStatusColor(results.overall.success)}`}>
            <div className='flex items-center justify-between'>
              <div className='flex items-center gap-3'>
                {results.overall.success ? (
                  <CheckCircle className='h-6 w-6 text-green-500' />
                ) : (
                  <AlertCircle className='h-6 w-6 text-red-500' />
                )}
                <div>
                  <h3 className='text-lg font-bold'>
                    {results.overall.success
                      ? 'All ML Integration Tests Passed!'
                      : 'Some ML Tests Failed'}
                  </h3>
                  <p className='text-sm opacity-75'>
                    {results.overall.passedTests}/{results.overall.totalTests} tests passed in{' '}
                    {results.overall.totalDuration}ms
                  </p>
                </div>
              </div>
              <div className='flex items-center gap-2 text-sm'>
                <Zap className='h-4 w-4' />
                <span>
                  {Math.round(results.overall.totalDuration / results.overall.totalTests)}ms avg
                </span>
              </div>
            </div>
          </div>

          {/* Individual Test Results */}
          <div className='grid gap-4'>
            <h4 className='font-semibold text-gray-900'>Individual ML Integration Test Results</h4>
            {Object.entries(results.results).map(([testKey, result]) => (
              <motion.div
                key={testKey}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className='flex items-center justify-between rounded-lg border bg-gray-50 p-3'
              >
                <div className='flex items-center gap-3'>
                  <span className='text-2xl'>{getCategoryIcon(testKey)}</span>
                  {getStatusIcon(result)}
                  <div>
                    <span className='font-medium text-gray-900'>{result.testName}</span>
                    {result.error && (
                      <div className='max-w-xs truncate text-sm text-red-600'>{result.error}</div>
                    )}
                  </div>
                </div>
                <div className='text-right'>
                  <div className='text-sm font-medium text-gray-900'>{result.duration}ms</div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Test Report */}
          <div className='mt-6'>
            <h4 className='mb-3 font-semibold text-gray-900'>ML Integration Test Report</h4>
            <div className='max-h-96 overflow-auto rounded-lg bg-gray-900 p-4 font-mono text-sm text-green-400'>
              <pre>{generateMLTestReport(results)}</pre>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className='grid grid-cols-2 gap-4 md:grid-cols-4'>
            <div className='rounded-lg bg-purple-50 p-3 text-center'>
              <div className='text-2xl font-bold text-purple-600'>{results.overall.totalTests}</div>
              <div className='text-sm text-purple-700'>Total Tests</div>
            </div>
            <div className='rounded-lg bg-green-50 p-3 text-center'>
              <div className='text-2xl font-bold text-green-600'>{results.overall.passedTests}</div>
              <div className='text-sm text-green-700'>Passed</div>
            </div>
            <div className='rounded-lg bg-red-50 p-3 text-center'>
              <div className='text-2xl font-bold text-red-600'>{results.overall.failedTests}</div>
              <div className='text-sm text-red-700'>Failed</div>
            </div>
            <div className='rounded-lg bg-blue-50 p-3 text-center'>
              <div className='text-2xl font-bold text-blue-600'>
                {results.overall.totalDuration}ms
              </div>
              <div className='text-sm text-blue-700'>Total Time</div>
            </div>
          </div>

          {/* Integration Status */}
          <div className='rounded-lg border border-blue-200 bg-blue-50 p-4'>
            <h4 className='mb-2 font-semibold text-blue-900'>ML Integration Status</h4>
            <div className='grid grid-cols-2 gap-4 text-sm'>
              <div>
                <span className='font-medium text-blue-800'>Session Management:</span>
                <span className='ml-2 text-blue-700'>
                  {results.results.sessionGeneration.success ? '✅ Working' : '❌ Failed'}
                </span>
              </div>
              <div>
                <span className='font-medium text-blue-800'>Request Validation:</span>
                <span className='ml-2 text-blue-700'>
                  {results.results.requestValidation.success ? '✅ Working' : '❌ Failed'}
                </span>
              </div>
              <div>
                <span className='font-medium text-blue-800'>API Integration:</span>
                <span className='ml-2 text-blue-700'>
                  {results.results.multiModalAssessment.success ? '✅ Ready' : '❌ Issues'}
                </span>
              </div>
              <div>
                <span className='font-medium text-blue-800'>Error Handling:</span>
                <span className='ml-2 text-blue-700'>
                  {results.results.errorHandling.success ? '✅ Robust' : '❌ Needs Work'}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      {!results && !isRunning && (
        <div className='py-8 text-center text-gray-500'>
          <Brain className='mx-auto mb-4 h-12 w-12 opacity-50' />
          <p>Click "Run ML Tests" to verify the complete ML integration layer.</p>
          <p className='mt-2 text-sm'>
            This will test session management, API communication, progress tracking, and error
            handling.
          </p>
        </div>
      )}
    </div>
  );
}
