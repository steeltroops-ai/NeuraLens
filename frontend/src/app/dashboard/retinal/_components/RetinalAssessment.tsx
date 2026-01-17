'use client';

import {
  Eye,
  Upload,
  Activity,
  Clock,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface RetinalAssessmentProps {
  onProcessingChange: (isProcessing: boolean) => void;
}

interface RetinalAnalysisResult {
  sessionId: string;
  processingTime: number;
  confidence: number;
  riskScore: number;
  qualityScore: number;
  biomarkers: {
    vesselTortuosity: number;
    avRatio: number;
    cupDiscRatio: number;
    vesselDensity: number;
  };
  recommendations: string[];
  timestamp: Date;
}

interface UploadState {
  isDragActive: boolean;
  isUploading: boolean;
  isProcessing: boolean;
  uploadProgress: number;
  error: string | null;
  selectedFile: File | null;
  previewUrl: string | null;
}

export default function RetinalAssessment({ onProcessingChange }: RetinalAssessmentProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    isDragActive: false,
    isUploading: false,
    isProcessing: false,
    uploadProgress: 0,
    error: null,
    selectedFile: null,
    previewUrl: null,
  });

  const [analysisResult, setAnalysisResult] = useState<RetinalAnalysisResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // File validation
  const validateFile = useCallback((file: File): string | null => {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];

    if (!allowedTypes.includes(file.type)) {
      return 'Please upload a JPG or PNG image file.';
    }

    if (file.size > maxSize) {
      return 'File size must be less than 10MB.';
    }

    if (file.size < 1024) {
      return 'File is too small. Please upload a valid retinal image.';
    }

    return null;
  }, []);

  // Handle file selection
  const handleFileSelect = useCallback(
    async (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setUploadState(prev => ({ ...prev, error: validationError }));
        return;
      }

      // Create preview URL
      const previewUrl = URL.createObjectURL(file);

      setUploadState(prev => ({
        ...prev,
        selectedFile: file,
        previewUrl,
        error: null,
      }));
    },
    [validateFile],
  );

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setUploadState(prev => ({ ...prev, isDragActive: true }));
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setUploadState(prev => ({ ...prev, isDragActive: false }));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setUploadState(prev => ({ ...prev, isDragActive: false }));

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0 && files[0]) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect],
  );

  // Handle file input change
  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files[0]) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect],
  );

  // Process retinal analysis
  const processRetinalAnalysis = useCallback(async () => {
    if (!uploadState.selectedFile) return;

    setUploadState(prev => ({ ...prev, isProcessing: true, error: null }));
    onProcessingChange(true);

    try {
      // Simulate upload progress
      for (let progress = 0; progress <= 100; progress += 10) {
        setUploadState(prev => ({ ...prev, uploadProgress: progress }));
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image_file', uploadState.selectedFile);

      // Call retinal analysis API
      const response = await fetch('/api/v1/retinal/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();

      // Transform API response to our format
      const analysisResult: RetinalAnalysisResult = {
        sessionId: result.session_id,
        processingTime: result.processing_time,
        confidence: result.confidence,
        riskScore: result.risk_score,
        qualityScore: result.quality_score,
        biomarkers: {
          vesselTortuosity: result.biomarkers.vessel_tortuosity,
          avRatio: result.biomarkers.av_ratio,
          cupDiscRatio: result.biomarkers.cup_disc_ratio,
          vesselDensity: result.biomarkers.vessel_density,
        },
        recommendations: result.recommendations || [],
        timestamp: new Date(result.timestamp),
      };

      setAnalysisResult(analysisResult);
    } catch (error) {
      console.error('Retinal analysis failed:', error);
      setUploadState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Analysis failed. Please try again.',
      }));
    } finally {
      setUploadState(prev => ({ ...prev, isProcessing: false, uploadProgress: 0 }));
      onProcessingChange(false);
    }
  }, [uploadState.selectedFile, onProcessingChange]);

  // Reset analysis
  const resetAnalysis = useCallback(() => {
    if (uploadState.previewUrl) {
      URL.revokeObjectURL(uploadState.previewUrl);
    }

    setUploadState({
      isDragActive: false,
      isUploading: false,
      isProcessing: false,
      uploadProgress: 0,
      error: null,
      selectedFile: null,
      previewUrl: null,
    });

    setAnalysisResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [uploadState.previewUrl]);

  return (
    <div className='space-y-6'>
      {/* Header - Enhanced with ophthalmology theme */}
      <div className='relative overflow-hidden bg-white rounded-2xl border border-zinc-200/80 p-8'>
        {/* Gradient background */}
        <div className='absolute inset-0 bg-gradient-to-br from-cyan-50/40 via-transparent to-blue-50/30 pointer-events-none' />

        <div className='relative'>
          <div className='flex items-start justify-between'>
            <div className='flex items-start gap-4'>
              <div className='p-3 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20'>
                <Eye className='h-7 w-7 text-white' strokeWidth={2} />
              </div>
              <div>
                <div className='flex items-center gap-3 mb-2'>
                  <h1 className='text-[24px] font-semibold text-zinc-900'>Retinal Analysis</h1>
                  <span className='px-2.5 py-1 bg-cyan-100 text-cyan-700 text-[11px] font-medium rounded-full'>
                    EfficientNet-B0
                  </span>
                </div>
                <p className='text-[14px] text-zinc-600 max-w-xl'>
                  Advanced fundus image analysis with deep learning vessel detection
                </p>
              </div>
            </div>
          </div>

          {/* Feature pills */}
          <div className='flex flex-wrap gap-2 mt-6'>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <Clock className='h-4 w-4 text-cyan-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Processing: &lt;200ms</span>
            </div>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <Activity className='h-4 w-4 text-blue-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Computer Vision</span>
            </div>
            <div className='flex items-center gap-2 px-3 py-2 bg-white border border-zinc-200 rounded-lg'>
              <TrendingUp className='h-4 w-4 text-purple-600' strokeWidth={2} />
              <span className='text-[12px] font-medium text-zinc-700'>Vessel Analysis</span>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Interface */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <h2 className='mb-4 text-lg font-semibold text-slate-900'>Fundus Image Upload</h2>

        <AnimatePresence mode='wait'>
          {!uploadState.selectedFile ? (
            <motion.div
              key='upload'
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300 ${uploadState.isDragActive
                ? 'scale-105 border-green-500 bg-green-50'
                : 'border-slate-300 hover:border-green-400 hover:bg-slate-50'
                }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload
                className={`mx-auto mb-4 h-12 w-12 transition-colors ${uploadState.isDragActive ? 'text-green-500' : 'text-slate-400'
                  }`}
              />
              <h3 className='mb-2 text-lg font-medium text-slate-900'>
                {uploadState.isDragActive ? 'Drop image here' : 'Upload Retinal Image'}
              </h3>
              <p className='mb-4 text-slate-600'>
                Drag and drop a fundus photograph or click to browse
              </p>
              <input
                ref={fileInputRef}
                type='file'
                accept='image/jpeg,image/jpg,image/png'
                onChange={handleFileInputChange}
                className='hidden'
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className='rounded-lg bg-green-600 px-6 py-2 font-medium text-white transition-all duration-200 hover:scale-105 hover:bg-green-700'
              >
                Choose Image File
              </button>
              <p className='mt-2 text-xs text-slate-500'>Supports JPG, PNG files up to 10MB</p>
            </motion.div>
          ) : (
            <motion.div
              key='preview'
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className='space-y-4'
            >
              {/* File Preview */}
              <div className='flex items-start space-x-4 rounded-lg border border-slate-200 p-4'>
                {uploadState.previewUrl && (
                  <img
                    src={uploadState.previewUrl}
                    alt='Retinal image preview'
                    className='h-20 w-20 rounded-lg object-cover'
                  />
                )}
                <div className='flex-1'>
                  <h3 className='font-medium text-slate-900'>{uploadState.selectedFile.name}</h3>
                  <p className='text-sm text-slate-600'>
                    Size: {(uploadState.selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  <p className='text-sm text-slate-600'>Type: {uploadState.selectedFile.type}</p>
                </div>
                <div className='flex space-x-2'>
                  {!uploadState.isProcessing && !analysisResult && (
                    <button
                      onClick={processRetinalAnalysis}
                      className='rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700'
                    >
                      Analyze Image
                    </button>
                  )}
                  <button
                    onClick={resetAnalysis}
                    disabled={uploadState.isProcessing}
                    className='rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 disabled:opacity-50'
                  >
                    Remove
                  </button>
                </div>
              </div>

              {/* Processing State */}
              {uploadState.isProcessing && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className='rounded-lg border border-blue-200 bg-blue-50 p-4'
                >
                  <div className='flex items-center space-x-3'>
                    <Loader2 className='h-5 w-5 animate-spin text-blue-600' />
                    <div className='flex-1'>
                      <p className='font-medium text-blue-900'>Processing retinal image...</p>
                      <p className='text-sm text-blue-700'>
                        Running EfficientNet-B0 analysis and vessel detection
                      </p>
                    </div>
                  </div>
                  {uploadState.uploadProgress > 0 && (
                    <div className='mt-3'>
                      <div className='flex justify-between text-sm text-blue-700'>
                        <span>Upload Progress</span>
                        <span>{uploadState.uploadProgress}%</span>
                      </div>
                      <div className='mt-1 h-2 rounded-full bg-blue-200'>
                        <div
                          className='h-2 rounded-full bg-blue-600 transition-all duration-300'
                          style={{ width: `${uploadState.uploadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {/* Error State */}
              {uploadState.error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className='rounded-lg border border-red-200 bg-red-50 p-4'
                >
                  <div className='flex items-center space-x-3'>
                    <AlertCircle className='h-5 w-5 text-red-600' />
                    <div>
                      <p className='font-medium text-red-900'>Analysis Failed</p>
                      <p className='text-sm text-red-700'>{uploadState.error}</p>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Analysis Results */}
      {analysisResult && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'
        >
          <div className='mb-6 flex items-center space-x-3'>
            <div className='rounded-lg bg-gradient-to-r from-green-500 to-green-600 p-2'>
              <CheckCircle className='h-5 w-5 text-white' />
            </div>
            <div>
              <h2 className='text-lg font-semibold text-slate-900'>Analysis Complete</h2>
              <p className='text-sm text-slate-600'>
                Processed in {analysisResult.processingTime.toFixed(1)}ms • Confidence:{' '}
                {(analysisResult.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Biomarkers Grid */}
          <div className='mb-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4'>
            <div className='rounded-lg border border-slate-200 p-4'>
              <h3 className='text-sm font-medium text-slate-600'>Vessel Tortuosity</h3>
              <p className='text-2xl font-bold text-slate-900'>
                {(analysisResult.biomarkers.vesselTortuosity * 100).toFixed(1)}%
              </p>
              <div className='mt-2 h-2 rounded-full bg-slate-200'>
                <div
                  className='h-2 rounded-full bg-gradient-to-r from-green-500 to-green-600'
                  style={{ width: `${analysisResult.biomarkers.vesselTortuosity * 100}%` }}
                />
              </div>
            </div>

            <div className='rounded-lg border border-slate-200 p-4'>
              <h3 className='text-sm font-medium text-slate-600'>A/V Ratio</h3>
              <p className='text-2xl font-bold text-slate-900'>
                {analysisResult.biomarkers.avRatio.toFixed(2)}
              </p>
              <p className='mt-1 text-xs text-slate-500'>Normal: 0.6-0.8</p>
            </div>

            <div className='rounded-lg border border-slate-200 p-4'>
              <h3 className='text-sm font-medium text-slate-600'>Cup-Disc Ratio</h3>
              <p className='text-2xl font-bold text-slate-900'>
                {analysisResult.biomarkers.cupDiscRatio.toFixed(2)}
              </p>
              <p className='mt-1 text-xs text-slate-500'>Normal: &lt;0.3</p>
            </div>

            <div className='rounded-lg border border-slate-200 p-4'>
              <h3 className='text-sm font-medium text-slate-600'>Vessel Density</h3>
              <p className='text-2xl font-bold text-slate-900'>
                {(analysisResult.biomarkers.vesselDensity * 100).toFixed(1)}%
              </p>
              <div className='mt-2 h-2 rounded-full bg-slate-200'>
                <div
                  className='h-2 rounded-full bg-gradient-to-r from-blue-500 to-blue-600'
                  style={{ width: `${analysisResult.biomarkers.vesselDensity * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Risk Assessment */}
          <div className='mb-6 rounded-lg border border-slate-200 p-4'>
            <h3 className='mb-3 text-sm font-medium text-slate-600'>Risk Assessment</h3>
            <div className='flex items-center space-x-4'>
              <div className='flex-1'>
                <div className='flex justify-between text-sm'>
                  <span>Overall Risk Score</span>
                  <span className='font-medium'>
                    {(analysisResult.riskScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className='mt-2 h-3 rounded-full bg-slate-200'>
                  <div
                    className={`h-3 rounded-full ${analysisResult.riskScore < 0.3
                      ? 'bg-gradient-to-r from-green-500 to-green-600'
                      : analysisResult.riskScore < 0.7
                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600'
                        : 'bg-gradient-to-r from-red-500 to-red-600'
                      }`}
                    style={{ width: `${analysisResult.riskScore * 100}%` }}
                  />
                </div>
              </div>
              <div
                className={`rounded-full px-3 py-1 text-xs font-medium ${analysisResult.riskScore < 0.3
                  ? 'bg-green-100 text-green-800'
                  : analysisResult.riskScore < 0.7
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                  }`}
              >
                {analysisResult.riskScore < 0.3
                  ? 'Low Risk'
                  : analysisResult.riskScore < 0.7
                    ? 'Moderate Risk'
                    : 'High Risk'}
              </div>
            </div>
          </div>

          {/* Recommendations */}
          {analysisResult.recommendations.length > 0 && (
            <div className='rounded-lg border border-blue-200 bg-blue-50 p-4'>
              <h3 className='mb-3 text-sm font-medium text-blue-900'>Recommendations</h3>
              <ul className='space-y-2'>
                {analysisResult.recommendations.map((recommendation, index) => (
                  <li key={index} className='flex items-start space-x-2 text-sm text-blue-800'>
                    <span className='mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-blue-600' />
                    <span>{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action Buttons */}
          <div className='mt-6 flex justify-between'>
            <button
              onClick={resetAnalysis}
              className='rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50'
            >
              Analyze Another Image
            </button>
            <div className='flex space-x-3'>
              <button className='rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50'>
                Export Results
              </button>
              <button className='rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700'>
                Save to History
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
