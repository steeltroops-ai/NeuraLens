/**
 * API Integration Test Page
 * Demonstrates frontend-backend communication with all assessment types
 */

import React, { useState } from 'react';
import {
  SpeechAnalysisService,
  RetinalAnalysisService,
  MotorAssessmentService,
  CognitiveAssessmentService,
  SystemService,
} from '@/lib/api/services';
import { useApi, useFileUpload } from '@/hooks/useApi';
import { LoadingButton } from '@/components/ui/LoadingStates';
import { ErrorDisplay } from '@/components/ui/ErrorDisplay';

export default function ApiTestPage() {
  const [results, setResults] = useState<any>({});
  const healthApi = useApi();
  const speechUpload = useFileUpload();
  const retinalUpload = useFileUpload();
  const motorApi = useApi();
  const cognitiveApi = useApi();

  // Test health check
  const testHealthCheck = async () => {
    await healthApi.execute(
      async () => {
        return await SystemService.healthCheck();
      },
      {
        onSuccess: data => {
          setResults((prev: any) => ({ ...prev, health: data }));
        },
      },
    );
  };

  // Test speech analysis
  const testSpeechAnalysis = async () => {
    // Create a mock audio file
    const canvas = document.createElement('canvas');
    canvas.toBlob(async blob => {
      if (blob) {
        const audioFile = new File([blob], 'test-audio.wav', { type: 'audio/wav' });

        await speechUpload.upload(
          async () => {
            return await SpeechAnalysisService.analyze({
              session_id: 'test-speech-' + Date.now(),
              audio_file: audioFile,
            });
          },
          {
            onSuccess: data => {
              setResults((prev: any) => ({ ...prev, speech: data }));
            },
          },
        );
      }
    });
  };

  // Test retinal analysis
  const testRetinalAnalysis = async () => {
    // Create a mock image file
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#ff0000';
      ctx.fillRect(0, 0, 512, 512);
    }

    canvas.toBlob(async blob => {
      if (blob) {
        const imageFile = new File([blob], 'test-retinal.jpg', { type: 'image/jpeg' });

        await retinalUpload.upload(
          async () => {
            return await RetinalAnalysisService.analyze({
              session_id: 'test-retinal-' + Date.now(),
              image_file: imageFile,
            });
          },
          {
            onSuccess: data => {
              setResults((prev: any) => ({ ...prev, retinal: data }));
            },
          },
        );
      }
    });
  };

  // Test motor assessment
  const testMotorAssessment = async () => {
    await motorApi.execute(
      async () => {
        return await MotorAssessmentService.analyze({
          session_id: 'test-motor-' + Date.now(),
          sensor_data: {
            accelerometer: [
              { x: 0.1, y: 0.2, z: 9.8 },
              { x: 0.15, y: 0.18, z: 9.85 },
              { x: 0.12, y: 0.22, z: 9.75 },
            ],
          },
          assessment_type: 'tremor',
        });
      },
      {
        onSuccess: data => {
          setResults((prev: any) => ({ ...prev, motor: data }));
        },
      },
    );
  };

  // Test cognitive assessment
  const testCognitiveAssessment = async () => {
    await cognitiveApi.execute(
      async () => {
        return await CognitiveAssessmentService.analyze({
          session_id: 'test-cognitive-' + Date.now(),
          test_results: {
            response_times: [1200, 1150, 1300, 1100, 1250],
            accuracy: [1, 1, 0, 1, 1],
            memory: { immediate_recall: 0.8, delayed_recall: 0.75 },
          },
          test_battery: ['memory', 'attention'],
          difficulty_level: 'standard',
        });
      },
      {
        onSuccess: data => {
          setResults((prev: any) => ({ ...prev, cognitive: data }));
        },
      },
    );
  };

  return (
    <div className='min-h-screen bg-gray-50 py-8'>
      <div className='mx-auto max-w-4xl px-4'>
        <div className='rounded-lg bg-white p-8 shadow-lg'>
          <h1 className='mb-8 text-3xl font-bold text-gray-900'>NeuraLens API Integration Test</h1>

          <div className='mb-8 grid grid-cols-1 gap-6 md:grid-cols-2'>
            {/* Health Check */}
            <div className='rounded-lg border border-gray-200 p-6'>
              <h2 className='mb-4 text-xl font-semibold'>Health Check</h2>
              <LoadingButton
                loading={healthApi.loading}
                onClick={testHealthCheck}
                className='w-full rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700'
              >
                Test Health Check
              </LoadingButton>

              {healthApi.error && <ErrorDisplay error={healthApi.error} className='mt-4' />}

              {results.health && (
                <div className='mt-4 rounded border border-green-200 bg-green-50 p-3'>
                  <p className='text-sm text-green-800'>Status: {results.health.status}</p>
                </div>
              )}
            </div>

            {/* Speech Analysis */}
            <div className='rounded-lg border border-gray-200 p-6'>
              <h2 className='mb-4 text-xl font-semibold'>Speech Analysis</h2>
              <LoadingButton
                loading={speechUpload.loading}
                onClick={testSpeechAnalysis}
                className='w-full rounded bg-green-600 px-4 py-2 text-white hover:bg-green-700'
              >
                Test Speech Analysis
              </LoadingButton>

              {speechUpload.uploadProgress > 0 && (
                <div className='mt-2 h-2 rounded-full bg-gray-200'>
                  <div
                    className='h-2 rounded-full bg-green-600 transition-all'
                    style={{ width: `${speechUpload.uploadProgress}%` }}
                  />
                </div>
              )}

              {speechUpload.error && <ErrorDisplay error={speechUpload.error} className='mt-4' />}

              {results.speech && (
                <div className='mt-4 rounded border border-green-200 bg-green-50 p-3'>
                  <p className='text-sm text-green-800'>
                    Risk Score: {results.speech.risk_score?.toFixed(3)}
                  </p>
                  <p className='text-sm text-green-800'>
                    Confidence: {results.speech.confidence?.toFixed(3)}
                  </p>
                </div>
              )}
            </div>

            {/* Retinal Analysis */}
            <div className='rounded-lg border border-gray-200 p-6'>
              <h2 className='mb-4 text-xl font-semibold'>Retinal Analysis</h2>
              <LoadingButton
                loading={retinalUpload.loading}
                onClick={testRetinalAnalysis}
                className='w-full rounded bg-purple-600 px-4 py-2 text-white hover:bg-purple-700'
              >
                Test Retinal Analysis
              </LoadingButton>

              {retinalUpload.uploadProgress > 0 && (
                <div className='mt-2 h-2 rounded-full bg-gray-200'>
                  <div
                    className='h-2 rounded-full bg-purple-600 transition-all'
                    style={{ width: `${retinalUpload.uploadProgress}%` }}
                  />
                </div>
              )}

              {retinalUpload.error && <ErrorDisplay error={retinalUpload.error} className='mt-4' />}

              {results.retinal && (
                <div className='mt-4 rounded border border-purple-200 bg-purple-50 p-3'>
                  <p className='text-sm text-purple-800'>
                    Risk Score: {results.retinal.risk_score?.toFixed(3)}
                  </p>
                  <p className='text-sm text-purple-800'>
                    Quality: {results.retinal.quality_score?.toFixed(3)}
                  </p>
                </div>
              )}
            </div>

            {/* Motor Assessment */}
            <div className='rounded-lg border border-gray-200 p-6'>
              <h2 className='mb-4 text-xl font-semibold'>Motor Assessment</h2>
              <LoadingButton
                loading={motorApi.loading}
                onClick={testMotorAssessment}
                className='w-full rounded bg-orange-600 px-4 py-2 text-white hover:bg-orange-700'
              >
                Test Motor Assessment
              </LoadingButton>

              {motorApi.error && <ErrorDisplay error={motorApi.error} className='mt-4' />}

              {results.motor && (
                <div className='mt-4 rounded border border-orange-200 bg-orange-50 p-3'>
                  <p className='text-sm text-orange-800'>
                    Risk Score: {results.motor.risk_score?.toFixed(3)}
                  </p>
                  <p className='text-sm text-orange-800'>
                    Quality: {results.motor.movement_quality}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Cognitive Assessment */}
          <div className='mb-8 rounded-lg border border-gray-200 p-6'>
            <h2 className='mb-4 text-xl font-semibold'>Cognitive Assessment</h2>
            <LoadingButton
              loading={cognitiveApi.loading}
              onClick={testCognitiveAssessment}
              className='w-full rounded bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700'
            >
              Test Cognitive Assessment
            </LoadingButton>

            {cognitiveApi.error && <ErrorDisplay error={cognitiveApi.error} className='mt-4' />}

            {results.cognitive && (
              <div className='mt-4 rounded border border-indigo-200 bg-indigo-50 p-3'>
                <p className='text-sm text-indigo-800'>
                  Risk Score: {results.cognitive.risk_score?.toFixed(3)}
                </p>
                <p className='text-sm text-indigo-800'>
                  Overall Score: {results.cognitive.overall_score?.toFixed(3)}
                </p>
              </div>
            )}
          </div>

          {/* Results Summary */}
          {Object.keys(results).length > 0 && (
            <div className='rounded-lg border border-gray-200 bg-gray-50 p-6'>
              <h3 className='mb-4 text-lg font-semibold'>Test Results Summary</h3>
              <pre className='max-h-96 overflow-auto rounded border bg-white p-4 text-sm'>
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
