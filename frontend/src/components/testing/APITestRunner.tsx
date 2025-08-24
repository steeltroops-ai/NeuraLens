'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, CheckCircle, XCircle, Clock, Zap } from 'lucide-react';
import { 
  runComprehensiveIntegrationTest, 
  generateTestReport,
  type ComprehensiveTestResults,
  type IntegrationTestResult 
} from '../../lib/api/integration-test';

interface APITestRunnerProps {
  className?: string;
}

export default function APITestRunner({ className = '' }: APITestRunnerProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<ComprehensiveTestResults | null>(null);
  const [currentTest, setCurrentTest] = useState<string>('');

  const runTests = async () => {
    setIsRunning(true);
    setResults(null);
    setCurrentTest('Initializing tests...');

    try {
      // Simulate test progress updates
      const testSteps = [
        'Testing health endpoint...',
        'Testing speech analysis...',
        'Testing retinal analysis...',
        'Testing motor assessment...',
        'Testing cognitive assessment...',
        'Testing NRI fusion...',
        'Generating report...',
      ];

      for (let i = 0; i < testSteps.length - 1; i++) {
        setCurrentTest(testSteps[i]);
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      setCurrentTest(testSteps[testSteps.length - 1]);
      const testResults = await runComprehensiveIntegrationTest();
      setResults(testResults);
      setCurrentTest('Tests completed!');
    } catch (error) {
      console.error('Test execution failed:', error);
      setCurrentTest('Test execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  const getStatusIcon = (result: IntegrationTestResult) => {
    if (result.success) {
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    }
    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const getStatusColor = (success: boolean) => {
    return success ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50';
  };

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">API Integration Test</h2>
          <p className="text-gray-600 mt-1">Test all frontend API endpoints</p>
        </div>
        <button
          onClick={runTests}
          disabled={isRunning}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
            ${isRunning 
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
              : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-md'
            }
          `}
        >
          {isRunning ? (
            <>
              <Clock className="w-4 h-4 animate-spin" />
              Running Tests...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Tests
            </>
          )}
        </button>
      </div>

      {/* Current Test Status */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200"
        >
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            <span className="text-blue-700 font-medium">{currentTest}</span>
          </div>
        </motion.div>
      )}

      {/* Test Results */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Overall Results */}
          <div className={`p-4 rounded-lg border ${getStatusColor(results.overall.success)}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {results.overall.success ? (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                ) : (
                  <XCircle className="w-6 h-6 text-red-500" />
                )}
                <div>
                  <h3 className="font-bold text-lg">
                    {results.overall.success ? 'All Tests Passed!' : 'Some Tests Failed'}
                  </h3>
                  <p className="text-sm opacity-75">
                    {results.overall.passedTests}/{results.overall.totalTests} tests passed in {results.overall.totalTime}ms
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Zap className="w-4 h-4" />
                <span>{Math.round(results.overall.totalTime / results.overall.totalTests)}ms avg</span>
              </div>
            </div>
          </div>

          {/* Individual Test Results */}
          <div className="grid gap-4">
            <h4 className="font-semibold text-gray-900">Individual Test Results</h4>
            {Object.entries(results.results).map(([endpoint, result]) => (
              <motion.div
                key={endpoint}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border"
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(result)}
                  <div>
                    <span className="font-medium text-gray-900 capitalize">
                      {endpoint} Endpoint
                    </span>
                    <div className="text-sm text-gray-600">
                      {result.endpoint}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900">
                    {result.responseTime}ms
                  </div>
                  {result.error && (
                    <div className="text-xs text-red-600 max-w-xs truncate">
                      {result.error}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Test Report */}
          <div className="mt-6">
            <h4 className="font-semibold text-gray-900 mb-3">Test Report</h4>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
              <pre>{generateTestReport(results)}</pre>
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">
                {results.overall.totalTests}
              </div>
              <div className="text-sm text-blue-700">Total Tests</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">
                {results.overall.passedTests}
              </div>
              <div className="text-sm text-green-700">Passed</div>
            </div>
            <div className="bg-red-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-red-600">
                {results.overall.failedTests}
              </div>
              <div className="text-sm text-red-700">Failed</div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">
                {results.overall.totalTime}ms
              </div>
              <div className="text-sm text-purple-700">Total Time</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      {!results && !isRunning && (
        <div className="text-center py-8 text-gray-500">
          <Play className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>Click "Run Tests" to verify all API endpoints are working correctly.</p>
          <p className="text-sm mt-2">
            This will test speech, retinal, motor, cognitive, and NRI fusion endpoints.
          </p>
        </div>
      )}
    </div>
  );
}
