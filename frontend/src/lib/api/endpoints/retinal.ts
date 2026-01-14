
import { apiClient } from '../../client';
import { RetinalAnalysisResult, ImageValidationResult } from '../../../types/retinal-analysis';

export async function analyzeRetinalImage(
  imageFile: File,
  patientId: string
): Promise<RetinalAnalysisResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('patient_id', patientId);
  
  const response = await apiClient.post<RetinalAnalysisResult>(
    '/retinal/analyze',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' }
    }
  );
  
  return response.data;
}

export async function validateRetinalImage(
  imageFile: File
): Promise<ImageValidationResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await apiClient.post<ImageValidationResult>(
    '/retinal/validate',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' }
    }
  );
  
  return response.data;
}

export async function getRetinalResults(
  assessmentId: string
): Promise<RetinalAnalysisResult> {
  const response = await apiClient.get<RetinalAnalysisResult>(
    `/retinal/results/${assessmentId}`
  );
  
  return response.data;
}

export async function generateRetinalReport(
  assessmentId: string
): Promise<Blob> {
  const response = await apiClient.get(
    `/retinal/report/${assessmentId}`,
    { responseType: 'blob' }
  );
  
  return response.data;
}
