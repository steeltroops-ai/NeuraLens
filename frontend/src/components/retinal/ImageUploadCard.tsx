/**
 * ImageUploadCard Component
 * 
 * Drag-and-drop image upload with validation feedback.
 * 
 * Features:
 * - Drag-and-drop support
 * - File type and size validation
 * - Upload progress indication
 * - Validation feedback with recommendations
 * 
 * Requirements: 1.1, 1.2, 1.4
 * 
 * @module components/retinal/ImageUploadCard
 */

'use client';

import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Image as ImageIcon,
  Loader2,
  FileImage
} from 'lucide-react';
import { ImageValidationResult } from '@/types/retinal-analysis';

// ============================================================================
// Types
// ============================================================================

interface ImageUploadCardProps {
  /** Called when file is selected */
  onFileSelect: (file: File) => void;
  /** Called on validation result */
  onValidationResult?: (result: ImageValidationResult) => void;
  /** Accepted file types */
  acceptedTypes?: string[];
  /** Maximum file size in bytes (default 10MB) */
  maxSizeBytes?: number;
  /** Whether upload is disabled */
  disabled?: boolean;
  /** Whether currently uploading */
  isUploading?: boolean;
  /** Upload progress (0-100) */
  progress?: number;
  /** Show validation inline */
  showValidation?: boolean;
  /** Validation result if available */
  validationResult?: ImageValidationResult | null;
  /** Title text */
  title?: string;
  /** Description text */
  description?: string;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
const DEFAULT_MAX_SIZE = 10 * 1024 * 1024; // 10MB

// ============================================================================
// Component
// ============================================================================

export function ImageUploadCard({
  onFileSelect,
  onValidationResult,
  acceptedTypes = DEFAULT_ACCEPTED_TYPES,
  maxSizeBytes = DEFAULT_MAX_SIZE,
  disabled = false,
  isUploading = false,
  progress = 0,
  showValidation = true,
  validationResult = null,
  title = 'Upload Retinal Image',
  description = 'Drag and drop a fundus photograph or click to browse',
}: ImageUploadCardProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ============================================================================
  // Validation
  // ============================================================================

  const validateFile = useCallback((file: File): string | null => {
    if (!acceptedTypes.includes(file.type)) {
      const extensions = acceptedTypes.map(t => t.split('/')[1]).join(', ');
      return `Please upload a ${extensions.toUpperCase()} image file.`;
    }

    if (file.size > maxSizeBytes) {
      const maxMB = (maxSizeBytes / 1024 / 1024).toFixed(0);
      return `File size must be less than ${maxMB}MB.`;
    }

    if (file.size < 1024) {
      return 'File is too small. Please upload a valid retinal image.';
    }

    return null;
  }, [acceptedTypes, maxSizeBytes]);

  // ============================================================================
  // File Selection
  // ============================================================================

  const handleFileSelect = useCallback((file: File) => {
    const error = validateFile(file);
    
    if (error) {
      setLocalError(error);
      return;
    }

    // Clean up previous preview
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const newPreviewUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(newPreviewUrl);
    setLocalError(null);
    
    onFileSelect(file);
  }, [validateFile, previewUrl, onFileSelect]);

  // ============================================================================
  // Drag and Drop Handlers
  // ============================================================================

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragActive(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0]) {
      handleFileSelect(files[0]);
    }
  }, [disabled, handleFileSelect]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  // ============================================================================
  // Clear Selection
  // ============================================================================

  const clearSelection = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setLocalError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [previewUrl]);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm">
      <AnimatePresence mode="wait">
        {!selectedFile ? (
          // Upload Zone
          <motion.div
            key="upload-zone"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`
              relative rounded-lg border-2 border-dashed p-8 text-center transition-all duration-300
              ${isDragActive 
                ? 'scale-[1.02] border-cyan-500 bg-cyan-50' 
                : 'border-zinc-300 hover:border-cyan-400 hover:bg-zinc-50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !disabled && fileInputRef.current?.click()}
          >
            {/* Icon */}
            <div className={`
              mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full transition-colors
              ${isDragActive ? 'bg-cyan-100' : 'bg-zinc-100'}
            `}>
              {isDragActive ? (
                <FileImage className="h-8 w-8 text-cyan-600" />
              ) : (
                <Upload className="h-8 w-8 text-zinc-400" />
              )}
            </div>

            {/* Text */}
            <h3 className="mb-2 text-lg font-semibold text-zinc-900">
              {isDragActive ? 'Drop image here' : title}
            </h3>
            <p className="mb-4 text-sm text-zinc-600">{description}</p>

            {/* Hidden Input */}
            <input
              ref={fileInputRef}
              type="file"
              accept={acceptedTypes.join(',')}
              onChange={handleFileInputChange}
              disabled={disabled}
              className="hidden"
            />

            {/* Button */}
            <button
              type="button"
              disabled={disabled}
              className="rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-2.5 font-medium text-white shadow-md shadow-cyan-500/20 transition-all duration-200 hover:scale-105 hover:shadow-lg disabled:opacity-50 disabled:hover:scale-100"
            >
              Choose Image File
            </button>

            {/* Format info */}
            <p className="mt-3 text-xs text-zinc-500">
              Supports JPG, PNG, TIFF files up to {(maxSizeBytes / 1024 / 1024).toFixed(0)}MB
            </p>

            {/* Local Error */}
            {localError && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 flex items-center justify-center gap-2 text-red-600"
              >
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm">{localError}</span>
              </motion.div>
            )}
          </motion.div>
        ) : (
          // Preview with Progress
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            {/* File Preview */}
            <div className="flex items-start gap-4 rounded-lg border border-zinc-200 bg-zinc-50 p-4">
              {previewUrl && (
                <div className="relative h-20 w-20 flex-shrink-0 overflow-hidden rounded-lg border border-zinc-200">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="h-full w-full object-cover"
                  />
                </div>
              )}
              
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-zinc-900 truncate">{selectedFile.name}</h4>
                <p className="text-sm text-zinc-600">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
                <div className="mt-1 flex items-center gap-2">
                  <ImageIcon className="h-4 w-4 text-zinc-400" />
                  <span className="text-xs text-zinc-500">{selectedFile.type}</span>
                </div>
              </div>

              {/* Remove Button */}
              {!isUploading && (
                <button
                  onClick={clearSelection}
                  className="flex-shrink-0 rounded-full p-1.5 text-zinc-400 transition-colors hover:bg-zinc-200 hover:text-zinc-600"
                >
                  <X className="h-5 w-5" />
                </button>
              )}
            </div>

            {/* Upload Progress */}
            {isUploading && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="rounded-lg border border-blue-200 bg-blue-50 p-4"
              >
                <div className="flex items-center gap-3 mb-3">
                  <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
                  <span className="font-medium text-blue-900">
                    {progress < 30 ? 'Uploading...' : progress < 90 ? 'Processing...' : 'Finalizing...'}
                  </span>
                </div>
                <div className="h-2 rounded-full bg-blue-200 overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="mt-2 text-right text-sm text-blue-700">{progress}%</p>
              </motion.div>
            )}

            {/* Validation Result */}
            {showValidation && validationResult && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className={`rounded-lg border p-4 ${
                  validationResult.is_valid
                    ? 'border-green-200 bg-green-50'
                    : 'border-amber-200 bg-amber-50'
                }`}
              >
                <div className="flex items-start gap-3">
                  {validationResult.is_valid ? (
                    <CheckCircle className="h-5 w-5 flex-shrink-0 text-green-600 mt-0.5" />
                  ) : (
                    <AlertTriangle className="h-5 w-5 flex-shrink-0 text-amber-600 mt-0.5" />
                  )}
                  <div className="flex-1">
                    <p className={`font-medium ${
                      validationResult.is_valid ? 'text-green-900' : 'text-amber-900'
                    }`}>
                      {validationResult.is_valid 
                        ? 'Image quality verified' 
                        : 'Quality issues detected'
                      }
                    </p>
                    <p className={`text-sm ${
                      validationResult.is_valid ? 'text-green-700' : 'text-amber-700'
                    }`}>
                      Quality Score: {validationResult.quality_score.toFixed(0)}/100
                    </p>

                    {/* Issues */}
                    {validationResult.issues.length > 0 && (
                      <ul className="mt-2 space-y-1">
                        {validationResult.issues.map((issue, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-amber-800">
                            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-amber-500 flex-shrink-0" />
                            {issue}
                          </li>
                        ))}
                      </ul>
                    )}

                    {/* Recommendations */}
                    {validationResult.recommendations.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-amber-200">
                        <p className="text-xs font-medium text-amber-800 mb-1">Recommendations:</p>
                        <ul className="space-y-1">
                          {validationResult.recommendations.map((rec, i) => (
                            <li key={i} className="text-xs text-amber-700">{rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ImageUploadCard;
