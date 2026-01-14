import { useState } from 'react';
import { 
  analyzeRetinalImage, 
  validateRetinalImage, 
  generateRetinalReport 
} from '../lib/api/endpoints/retinal';
import { 
  RetinalAnalysisResult, 
  ImageValidationResult 
} from '../types/retinal-analysis';

export function useRetinalAnalysis() {
  const [result, setResult] = useState<RetinalAnalysisResult | null>(null);
  const [validation, setValidation] = useState<ImageValidationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const validate = async (file: File) => {
    try {
      setLoading(true);
      const res = await validateRetinalImage(file);
      setValidation(res);
      return res;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed');
      return null;
    } finally {
      setLoading(false);
    }
  };

  const analyze = async (file: File, patientId: string = 'DEMO-PATIENT') => {
    setLoading(true);
    setError(null);
    setUploadProgress(0);
    
    try {
      // Simulate progress for UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 300);

      const data = await analyzeRetinalImage(file, patientId);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async (assessmentId: string) => {
    try {
      const blob = await generateRetinalReport(assessmentId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `retinal-report-${assessmentId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download report');
    }
  };

  const reset = () => {
    setResult(null);
    setValidation(null);
    setError(null);
    setUploadProgress(0);
  };

  return {
    result,
    validation,
    loading,
    error,
    uploadProgress,
    analyze,
    validate,
    downloadReport,
    reset
  };
}
