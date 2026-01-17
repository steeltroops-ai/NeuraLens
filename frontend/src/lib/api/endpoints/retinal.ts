/**
 * Retinal Analysis API Client
 * 
 * Comprehensive API client for retinal analysis pipeline:
 * - Image analysis and validation
 * - Results retrieval
 * - Patient history and trends
 * - Report generation
 * - Visualization fetching
 * - Queue status tracking
 * 
 * Requirements: 1.1-1.10, 2.1-2.12, 8.8
 * 
 * @module lib/api/endpoints/retinal
 */

import { apiClient } from '../client';
import { 
  RetinalAnalysisResult, 
  ImageValidationResult,
  PatientHistoryResponse,
  TrendAnalysisResponse,
  VisualizationType,
  QueueStatus,
  ReportOptions,
  RetinalAnalysisOptions,
  ImageValidationOptions,
  NRIDashboardData,
} from '@/types/retinal-analysis';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = '/retinal';

/**
 * Default timeout for analysis requests (30 seconds)
 */
const ANALYSIS_TIMEOUT = 30000;

// ============================================================================
// Core Analysis Functions
// ============================================================================

/**
 * Analyze a retinal fundus image.
 * 
 * Uploads an image and runs full ML processing pipeline including:
 * - Image validation
 * - Vessel segmentation
 * - Optic disc analysis
 * - Amyloid-beta detection
 * - Risk score calculation
 * 
 * @param imageFile - Retinal fundus image file
 * @param patientId - Patient identifier
 * @param options - Optional analysis configuration
 * @returns Complete analysis results
 * @throws Error if analysis fails or times out
 * 
 * Requirements: 1.1-1.10, 3.1-3.12, 5.1-5.12
 */
export async function analyzeRetinalImage(
  imageFile: File,
  patientId: string,
  options?: RetinalAnalysisOptions
): Promise<RetinalAnalysisResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('patient_id', patientId);
  
  if (options?.include_visualizations !== undefined) {
    formData.append('include_visualizations', String(options.include_visualizations));
  }
  if (options?.priority !== undefined) {
    formData.append('priority', String(options.priority));
  }
  if (options?.metadata) {
    formData.append('metadata', JSON.stringify(options.metadata));
  }

  const response = await apiClient.post<RetinalAnalysisResult>(
    `${API_BASE}/analyze`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: ANALYSIS_TIMEOUT,
    }
  );
  
  if (!response.data) {
    throw new Error('No data returned from retinal analysis');
  }
  
  return response.data;
}

/**
 * Validate retinal image quality without full analysis.
 * 
 * Quick validation to check if image meets quality requirements:
 * - Resolution and format check
 * - Signal-to-noise ratio
 * - Focus quality
 * - Anatomical feature detection
 * 
 * @param imageFile - Image file to validate
 * @param options - Validation options
 * @returns Validation results with quality score and issues
 * 
 * Requirements: 2.1-2.12
 */
export async function validateRetinalImage(
  imageFile: File,
  options?: ImageValidationOptions
): Promise<ImageValidationResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  if (options?.detailed !== undefined) {
    formData.append('detailed', String(options.detailed));
  }
  if (options?.check_anatomy !== undefined) {
    formData.append('check_anatomy', String(options.check_anatomy));
  }
  
  const response = await apiClient.post<ImageValidationResult>(
    `${API_BASE}/validate`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' }
    }
  );
  
  if (!response.data) {
    throw new Error('No data returned from image validation');
  }
  
  return response.data;
}

// ============================================================================
// Results Retrieval
// ============================================================================

/**
 * Get stored analysis results by assessment ID.
 * 
 * @param assessmentId - Unique assessment identifier
 * @returns Complete analysis result
 * @throws Error if assessment not found
 * 
 * Requirement: 8.8
 */
export async function getRetinalResults(
  assessmentId: string
): Promise<RetinalAnalysisResult> {
  const response = await apiClient.get<RetinalAnalysisResult>(
    `${API_BASE}/results/${assessmentId}`
  );
  
  if (!response.data) {
    throw new Error('No data returned for assessment');
  }
  
  return response.data;
}

/**
 * Delete an analysis result.
 * 
 * Permanently removes assessment data for HIPAA compliance.
 * 
 * @param assessmentId - Assessment to delete
 * @returns Success confirmation
 */
export async function deleteRetinalResult(
  assessmentId: string
): Promise<{ success: boolean; message: string }> {
  const response = await apiClient.delete<{ success: boolean; message: string }>(
    `${API_BASE}/results/${assessmentId}`
  );
  
  return response.data || { success: false, message: 'Unknown error' };
}

// ============================================================================
// Patient History
// ============================================================================

/**
 * Get patient assessment history.
 * 
 * Retrieves historical assessments for trend analysis and comparison.
 * 
 * @param patientId - Patient identifier
 * @param limit - Maximum results to return
 * @param offset - Pagination offset
 * @returns Paginated history response
 * 
 * Requirement: 8.9
 */
export async function getPatientHistory(
  patientId: string,
  limit: number = 10,
  offset: number = 0
): Promise<PatientHistoryResponse> {
  const queryParams = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString()
  });
  
  const response = await apiClient.get<PatientHistoryResponse>(
    `${API_BASE}/history/${patientId}?${queryParams.toString()}`
  );
  
  if (!response.data) {
    throw new Error('No history data returned');
  }
  
  return response.data;
}

/**
 * Get biomarker trend data for a patient.
 * 
 * @param patientId - Patient identifier
 * @param biomarker - Specific biomarker to track (optional)
 * @returns Trend analysis with data points and direction
 */
export async function getPatientTrends(
  patientId: string,
  biomarker?: string
): Promise<TrendAnalysisResponse[]> {
  let url = `${API_BASE}/trends/${patientId}`;
  if (biomarker) {
    url += `?biomarker=${encodeURIComponent(biomarker)}`;
  }
  
  const response = await apiClient.get<TrendAnalysisResponse[]>(url);
  
  return response.data || [];
}

// ============================================================================
// Report Generation
// ============================================================================

/**
 * Generate and download PDF clinical report.
 * 
 * Creates a comprehensive clinical report including:
 * - Patient demographics
 * - All biomarker values with reference ranges
 * - Risk assessment with interpretation
 * - Visualizations
 * - Clinical recommendations
 * 
 * @param assessmentId - Assessment to generate report for
 * @param options - Optional patient/provider details
 * @returns PDF file as Blob
 * 
 * Requirements: 7.1-7.12
 */
export async function generateRetinalReport(
  assessmentId: string,
  options?: ReportOptions
): Promise<Blob> {
  const queryParams = new URLSearchParams();
  if (options?.patient_name) {
    queryParams.append('patient_name', options.patient_name);
  }
  if (options?.patient_dob) {
    queryParams.append('patient_dob', options.patient_dob);
  }
  if (options?.provider_name) {
    queryParams.append('provider_name', options.provider_name);
  }
  if (options?.provider_npi) {
    queryParams.append('provider_npi', options.provider_npi);
  }
  
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const queryString = queryParams.toString();
  const url = `${baseUrl}/api/v1${API_BASE}/report/${assessmentId}${queryString ? `?${queryString}` : ''}`;
  
  // Use fetch directly for blob response
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error('Failed to generate report');
  }
  
  return response.blob();
}

/**
 * Utility to download a report directly.
 * 
 * @param assessmentId - Assessment ID
 * @param filename - Optional custom filename
 * @param options - Report options
 */
export async function downloadReport(
  assessmentId: string,
  filename?: string,
  options?: ReportOptions
): Promise<void> {
  const blob = await generateRetinalReport(assessmentId, options);
  const url = window.URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename || `retinal-report-${assessmentId}.pdf`;
  
  document.body.appendChild(link);
  link.click();
  
  // Cleanup
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

// ============================================================================
// Visualization Functions
// ============================================================================

/**
 * Get visualization image URL.
 * 
 * @param assessmentId - Assessment ID
 * @param type - Type of visualization
 * @returns Full URL to visualization image
 */
export function getVisualizationUrl(
  assessmentId: string,
  type: VisualizationType
): string {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  return `${baseUrl}/api/v1${API_BASE}/visualizations/${assessmentId}/${type}`;
}

/**
 * Fetch visualization as Blob.
 * 
 * @param assessmentId - Assessment ID
 * @param type - Type of visualization
 * @returns Image as Blob
 */
export async function getVisualization(
  assessmentId: string,
  type: VisualizationType
): Promise<Blob> {
  const url = getVisualizationUrl(assessmentId, type);
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to get visualization: ${type}`);
  }
  
  return response.blob();
}

// ============================================================================
// Queue Status (for high-load scenarios)
// ============================================================================

/**
 * Get queue status for a pending request.
 * 
 * @param requestId - Request ID from initial submission
 * @returns Queue status with position and estimated wait
 * 
 * Requirement: 10.6
 */
export async function getQueueStatus(requestId: string): Promise<QueueStatus> {
  const response = await apiClient.get<QueueStatus>(
    `${API_BASE}/queue/${requestId}`
  );
  
  return response.data || { status: 'failed' };
}

/**
 * Poll queue status until completion.
 * 
 * @param requestId - Request ID
 * @param intervalMs - Polling interval (default 2000ms)
 * @param maxAttempts - Maximum polling attempts (default 60)
 * @param onStatusUpdate - Callback for status updates
 * @returns Final status when completed or failed
 */
export async function pollQueueStatus(
  requestId: string,
  intervalMs: number = 2000,
  maxAttempts: number = 60,
  onStatusUpdate?: (status: QueueStatus) => void
): Promise<QueueStatus> {
  let attempts = 0;
  
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getQueueStatus(requestId);
        
        if (onStatusUpdate) {
          onStatusUpdate(status);
        }
        
        if (status.status === 'completed' || status.status === 'failed') {
          resolve(status);
          return;
        }
        
        attempts++;
        if (attempts >= maxAttempts) {
          reject(new Error('Queue polling timeout'));
          return;
        }
        
        setTimeout(poll, intervalMs);
      } catch (error) {
        reject(error);
      }
    };
    
    poll();
  });
}

// ============================================================================
// NRI Integration
// ============================================================================

/**
 * Get NRI dashboard data with retinal contribution.
 * 
 * @param assessmentId - Assessment ID
 * @returns Dashboard data with NRI contribution details
 */
export async function getNRIDashboardData(
  assessmentId: string
): Promise<NRIDashboardData> {
  const response = await apiClient.get<NRIDashboardData>(
    `${API_BASE}/nri/${assessmentId}`
  );
  
  if (!response.data) {
    throw new Error('No NRI dashboard data returned');
  }
  
  return response.data;
}

// ============================================================================
// Health Check
// ============================================================================

/**
 * Check retinal analysis service health.
 * 
 * @returns Service health status
 */
export async function checkServiceHealth(): Promise<{
  status: string;
  models_loaded: boolean;
  version: string;
}> {
  const response = await apiClient.get<{
    status: string;
    models_loaded: boolean;
    version: string;
  }>(`${API_BASE}/health`);
  
  return response.data || { status: 'unhealthy', models_loaded: false, version: 'unknown' };
}

// ============================================================================
// Export all functions
// ============================================================================

export const retinalApi = {
  // Core analysis
  analyzeRetinalImage,
  validateRetinalImage,
  
  // Results
  getRetinalResults,
  deleteRetinalResult,
  
  // History & Trends
  getPatientHistory,
  getPatientTrends,
  
  // Reports
  generateRetinalReport,
  downloadReport,
  
  // Visualizations
  getVisualizationUrl,
  getVisualization,
  
  // Queue
  getQueueStatus,
  pollQueueStatus,
  
  // NRI
  getNRIDashboardData,
  
  // Health
  checkServiceHealth,
};

export default retinalApi;
