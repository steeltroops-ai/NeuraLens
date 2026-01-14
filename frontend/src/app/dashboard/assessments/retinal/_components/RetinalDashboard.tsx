
import React, { useState } from 'react';
import { useRetinalAnalysis } from '@/hooks/useRetinalAnalysis'; // Ensure alias usage
import { RetinalUpload } from './RetinalUpload';
import { RetinalViewer } from './RetinalViewer';
import { RetinalMetrics } from './RetinalMetrics';
import { Button } from '@/components/ui/Button'; // Global UI

export function RetinalDashboard() {
  const { 
    result, 
    loading, 
    error, 
    uploadProgress, 
    analyze, 
    downloadReport, 
    reset 
  } = useRetinalAnalysis();
  
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    analyze(file);
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    reset();
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] p-6 bg-black text-zinc-100 font-sans">
      <header className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-light tracking-tight text-white mb-1">
            Retinal Analysis <span className="text-zinc-500 mx-2">/</span> Neuro-Degenerative Assessment
          </h1>
          <p className="text-sm text-zinc-500">
            AI-powered detection of Alzheimer's and vascular dementia biomarkers via retinal imaging.
          </p>
        </div>
        {result && (
          <div className="flex space-x-3">
             <Button 
               variant="secondary"
               onClick={() => downloadReport(result.assessment_id)}
               className="flex items-center"
             >
               <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
               Export Report
             </Button>
             <Button 
               variant="primary" 
               onClick={handleReset}
             >
               New Analysis
             </Button>
          </div>
        )}
      </header>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
        
        {/* Left Column: Viewer / Upload */}
        <div className="lg:col-span-8 flex flex-col h-[75vh]">
           {!selectedImage ? (
             <div className="flex-1 flex items-center justify-center bg-zinc-950/50 rounded-2xl border border-zinc-800/50 backdrop-blur-sm">
                <div className="max-w-md w-full">
                   <RetinalUpload onFileSelect={handleFileSelect} disabled={loading} />
                </div>
             </div>
           ) : (
             <RetinalViewer imageUrl={previewUrl!} result={result} />
           )}
        </div>

        {/* Right Column: Metrics / Status */}
        <div className="lg:col-span-4 h-[75vh] overflow-y-auto pr-2 custom-scrollbar">
           {loading ? (
             <div className="h-full flex flex-col items-center justify-center p-8 text-center space-y-6 bg-zinc-900/10 rounded-2xl border border-dashed border-zinc-800">
                <div className="relative w-16 h-16">
                   {/* Scanning Animation */}
                   <div className="absolute inset-0 border-4 border-zinc-800 rounded-full"></div>
                   <div className="absolute inset-0 border-4 border-t-blue-500 rounded-full animate-spin"></div>
                </div>
                <div>
                   <h3 className="text-xl font-medium text-white">Analyzing Retina</h3>
                   <p className="text-zinc-500 text-sm mt-2">Extracting micro-vascular biomarkers...</p>
                </div>
                <div className="w-full max-w-xs bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                   <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
                </div>
             </div>
           ) : error ? (
             <div className="p-6 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
                <h3 className="font-semibold mb-2">Analysis Failed</h3>
                <p className="text-sm">{error}</p>
                <button onClick={handleReset} className="mt-4 text-xs underline hover:text-red-300">Try Again</button>
             </div>
           ) : result ? (
             <RetinalMetrics result={result} />
           ) : (
             <div className="h-full flex items-center justify-center text-zinc-600 text-sm">
               Upload a scan to view metrics
             </div>
           )}
        </div>

      </div>
    </div>
  );
}
