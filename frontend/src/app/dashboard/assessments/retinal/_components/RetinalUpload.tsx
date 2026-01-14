
import React, { useCallback, useState } from 'react';

interface RetinalUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function RetinalUpload({ onFileSelect, disabled }: RetinalUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  }, [onFileSelect]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`relative border-2 border-dashed rounded-xl p-12 transition-all duration-300 ease-in-out text-center
        ${isDragging 
          ? 'border-blue-500 bg-blue-50/10 scale-[1.02]' 
          : 'border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900 hover:border-zinc-700'
        }
        ${disabled ? 'opacity-50 pointer-events-none' : 'cursor-pointer'}
      `}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        onChange={handleChange}
        accept="image/jpeg,image/png,application/dicom"
        disabled={disabled}
      />
      
      <div className="flex flex-col items-center justify-center space-y-4">
        <div className={`p-4 rounded-full ${isDragging ? 'bg-blue-500/10' : 'bg-zinc-800'}`}>
          <svg className={`w-8 h-8 ${isDragging ? 'text-blue-500' : 'text-zinc-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        </div>
        <div>
          <p className="text-lg font-medium text-zinc-200">
            Upload Retinal Scan
          </p>
          <p className="text-sm text-zinc-500 mt-1">
            Drag and drop or click to select
          </p>
          <p className="text-xs text-zinc-600 mt-2">
            Supports JPEG, PNG, DICOM (Min 1024x1024)
          </p>
        </div>
      </div>
    </div>
  );
}
