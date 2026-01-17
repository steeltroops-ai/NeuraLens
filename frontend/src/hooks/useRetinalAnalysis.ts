/**
 * useRetinalAnalysis Hook
 * 
 * Comprehensive React hook for retinal analysis functionality:
 * - Image upload and validation
 * - Analysis execution
 * - Results management
 * - Patient history
 * - Report downloading
 * - Loading and error states
 * 
 * Requirements: 1.1-1.10, 3.1-3.12
 * 
 * @module hooks/useRetinalAnalysis
 */

'use client';

import { useState, useCallback, useRef } from 'react';
import { 
  analyzeRetinalImage, 
  validateRetinalImage, 
  generateRetinalReport,
  getRetinalResults,
  getPatientHistory,
  getPatientTrends,
  deleteRetinalResult,
  downloadReport,
  getVisualizationUrl,
  getNRIDashboardData,
} from '@/lib/api/endpoints/retinal';
import { 
  RetinalAnalysisResult, 
  ImageValidationResult,
  PatientHistoryResponse,
  TrendAnalysisResponse,
  RiskCategory,
  RetinalAnalysisOptions,
  NRIDashboardData,
} from '@/types/retinal-analysis';

// ============================================================================
// Types
// ============================================================================

/**
 * Analysis state stages
 */
export type AnalysisStage = 
  | 'idle'
  | 'validating'
  | 'uploading'
  | 'processing'
  | 'complete'
  | 'error';

/**
 * Hook return type
 */
export interface UseRetinalAnalysisReturn {
  // State
  result: RetinalAnalysisResult | null;
  validation: ImageValidationResult | null;
  history: PatientHistoryResponse | null;
  trends: TrendAnalysisResponse[] | null;
  nriData: NRIDashboardData | null;
  
  // Loading states
  loading: boolean;
  stage: AnalysisStage;
  uploadProgress: number;
  error: string | null;
  
  // Actions
  validate: (file: File) => Promise<ImageValidationResult | null>;
  analyze: (file: File, patientId?: string, options?: RetinalAnalysisOptions) => Promise<RetinalAnalysisResult | null>;
  loadResult: (assessmentId: string) => Promise<RetinalAnalysisResult | null>;
  loadHistory: (patientId: string, limit?: number, offset?: number) => Promise<PatientHistoryResponse | null>;
  loadTrends: (patientId: string, biomarker?: string) => Promise<TrendAnalysisResponse[] | null>;
  loadNRIData: (assessmentId: string) => Promise<NRIDashboardData | null>;
  downloadReport: (assessmentId: string, filename?: string) => Promise<void>;
  deleteResult: (assessmentId: string) => Promise<boolean>;
  reset: () => void;
  clearError: () => void;
  
  // Utilities
  getVisualizationUrl: (assessmentId: string, type: 'heatmap' | 'segmentation' | 'gauge' | 'measurements') => string;
  isHighRisk: boolean;
  riskColor: string;
}

/**
 * Risk category color mapping
 */
const RISK_COLORS: Record<RiskCategory, string> = {
  minimal: '#22c55e',
  low: '#84cc16',
  moderate: '#eab308',
  elevated: '#f97316',
  high: '#ef4444',
  critical: '#991b1b',
};

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Custom hook for retinal analysis functionality.
 * 
 * Provides state management and actions for the complete retinal analysis workflow.
 * 
 * @example
 * ```tsx
 * const { 
 *   result, 
 *   loading, 
 *   analyze, 
 *   validate,
 *   downloadReport 
 * } = useRetinalAnalysis();
 * 
 * const handleUpload = async (file: File) => {
 *   const validation = await validate(file);
 *   if (validation?.is_valid) {
 *     await analyze(file, 'PATIENT-001');
 *   }
 * };
 * ```
 */
export function useRetinalAnalysis(): UseRetinalAnalysisReturn {
  // ============================================================================
  // State
  // ============================================================================
  
  const [result, setResult] = useState<RetinalAnalysisResult | null>(null);
  const [validation, setValidation] = useState<ImageValidationResult | null>(null);
  const [history, setHistory] = useState<PatientHistoryResponse | null>(null);
  const [trends, setTrends] = useState<TrendAnalysisResponse[] | null>(null);
  const [nriData, setNriData] = useState<NRIDashboardData | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<AnalysisStage>('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // Track active request for cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // ============================================================================
  // Validation
  // ============================================================================

  /**
   * Validate an image file before analysis.
   */
  const validate = useCallback(async (file: File): Promise<ImageValidationResult | null> => {
    try {
      setLoading(true);
      setStage('validating');
      setError(null);
      
      const result = await validateRetinalImage(file);
      setValidation(result);
      
      if (!result.is_valid) {
        setStage('error');
        setError(result.issues.join(', ') || 'Image validation failed');
      } else {
        setStage('idle');
      }
      
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Image validation failed';
      setError(message);
      setStage('error');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // Analysis
  // ============================================================================

  /**
   * Run full retinal analysis on an image.
   */
  const analyze = useCallback(async (
    file: File, 
    patientId: string = 'DEMO-PATIENT',
    options?: RetinalAnalysisOptions
  ): Promise<RetinalAnalysisResult | null> => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    
    setLoading(true);
    setError(null);
    setUploadProgress(0);
    setStage('uploading');
    
    try {
      // Simulate upload progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 30) {
            setStage('processing');
          }
          return Math.min(prev + 5, 90);
        });
      }, 200);

      const data = await analyzeRetinalImage(file, patientId, options);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      setResult(data);
      setStage('complete');
      
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      setStage('error');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // Load Existing Results
  // ============================================================================

  /**
   * Load an existing analysis result by ID.
   */
  const loadResult = useCallback(async (assessmentId: string): Promise<RetinalAnalysisResult | null> => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getRetinalResults(assessmentId);
      setResult(data);
      setStage('complete');
      
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load results';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // Patient History
  // ============================================================================

  /**
   * Load patient assessment history.
   */
  const loadHistory = useCallback(async (
    patientId: string,
    limit: number = 10,
    offset: number = 0
  ): Promise<PatientHistoryResponse | null> => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getPatientHistory(patientId, limit, offset);
      setHistory(data);
      
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load history';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Load biomarker trends for a patient.
   */
  const loadTrends = useCallback(async (
    patientId: string,
    biomarker?: string
  ): Promise<TrendAnalysisResponse[] | null> => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getPatientTrends(patientId, biomarker);
      setTrends(data);
      
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load trends';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // NRI Data
  // ============================================================================

  /**
   * Load NRI dashboard data.
   */
  const loadNRIData = useCallback(async (assessmentId: string): Promise<NRIDashboardData | null> => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getNRIDashboardData(assessmentId);
      setNriData(data);
      
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load NRI data';
      setError(message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // Report Download
  // ============================================================================

  /**
   * Download PDF report for an assessment.
   */
  const handleDownloadReport = useCallback(async (
    assessmentId: string,
    filename?: string
  ): Promise<void> => {
    try {
      setLoading(true);
      setError(null);
      
      await downloadReport(assessmentId, filename);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to download report';
      setError(message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // ============================================================================
  // Delete Result
  // ============================================================================

  /**
   * Delete an analysis result.
   */
  const deleteResult = useCallback(async (assessmentId: string): Promise<boolean> => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await deleteRetinalResult(assessmentId);
      
      if (response.success) {
        // Clear local state if deleted result was the current one
        if (result?.assessment_id === assessmentId) {
          setResult(null);
          setStage('idle');
        }
      }
      
      return response.success;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete result';
      setError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [result]);

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Reset all state to initial values.
   */
  const reset = useCallback(() => {
    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    setResult(null);
    setValidation(null);
    setHistory(null);
    setTrends(null);
    setNriData(null);
    setError(null);
    setUploadProgress(0);
    setStage('idle');
  }, []);

  /**
   * Clear current error.
   */
  const clearError = useCallback(() => {
    setError(null);
    if (stage === 'error') {
      setStage('idle');
    }
  }, [stage]);

  // ============================================================================
  // Derived State
  // ============================================================================

  /**
   * Check if current result is high risk.
   */
  const isHighRisk = result?.risk_assessment?.risk_category === 'high' || 
                      result?.risk_assessment?.risk_category === 'critical';

  /**
   * Get color for current risk category.
   */
  const riskColor = result?.risk_assessment?.risk_category 
    ? RISK_COLORS[result.risk_assessment.risk_category]
    : '#6b7280';

  // ============================================================================
  // Return
  // ============================================================================

  return {
    // State
    result,
    validation,
    history,
    trends,
    nriData,
    
    // Loading states
    loading,
    stage,
    uploadProgress,
    error,
    
    // Actions
    validate,
    analyze,
    loadResult,
    loadHistory,
    loadTrends,
    loadNRIData,
    downloadReport: handleDownloadReport,
    deleteResult,
    reset,
    clearError,
    
    // Utilities
    getVisualizationUrl,
    isHighRisk,
    riskColor,
  };
}

export default useRetinalAnalysis;
