'use client';

import React, { useState } from 'react';
import { Button, Card } from '@/components/ui';

interface RetinalAssessmentStepProps {
  onComplete: (retinalImage: File) => void;
  onBack: () => void;
  onSkip: () => void;
}

export const RetinalAssessmentStep: React.FC<RetinalAssessmentStepProps> = ({
  onComplete,
  onBack,
  onSkip,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (file: File) => {
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
    } else {
      alert('Please select a valid image file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0]) {
      handleFileSelect(files[0]);
    }
  };

  const handleComplete = () => {
    if (selectedFile) {
      onComplete(selectedFile);
    }
  };

  return (
    <div className="min-h-screen bg-surface-background py-8">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-4xl space-y-8">
          {/* Header */}
          <div className="space-y-4 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600">
              <svg
                className="h-8 w-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-text-primary">
              Retinal Image Analysis
            </h1>
            <p className="mx-auto max-w-2xl text-lg text-text-secondary">
              Upload a retinal fundus image for vascular pattern analysis
            </p>
          </div>

          {/* Upload Interface */}
          <Card className="p-8">
            <div
              className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                dragActive
                  ? 'border-primary-500 bg-primary-500/10'
                  : selectedFile
                    ? 'border-success bg-success/10'
                    : 'border-neutral-600 hover:border-neutral-500'
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              {selectedFile ? (
                <div className="space-y-4">
                  <svg
                    className="mx-auto h-16 w-16 text-success"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                  </svg>
                  <div>
                    <h3 className="mb-2 text-xl font-semibold text-text-primary">
                      Image Selected
                    </h3>
                    <p className="mb-2 text-text-secondary">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-text-muted">
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Button onClick={handleComplete} size="lg" className="px-8">
                      Analyze Image
                    </Button>
                    <br />
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setSelectedFile(null)}
                    >
                      Select Different Image
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <svg
                    className="mx-auto h-16 w-16 text-neutral-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <div>
                    <h3 className="mb-2 text-xl font-semibold text-text-primary">
                      Upload Retinal Image
                    </h3>
                    <p className="mb-4 text-text-secondary">
                      Drag and drop your retinal fundus image here, or click to
                      browse
                    </p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileSelect(file);
                      }}
                      className="hidden"
                      id="retinal-upload"
                    />
                    <label htmlFor="retinal-upload">
                      <Button
                        as="span"
                        size="lg"
                        className="cursor-pointer px-8"
                      >
                        Browse Files
                      </Button>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Instructions */}
          <Card className="p-6">
            <h4 className="mb-3 font-semibold text-text-primary">
              Image Requirements
            </h4>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li>• High-quality fundus photograph (color preferred)</li>
              <li>• Clear view of optic disc and macula</li>
              <li>• Minimal artifacts or reflections</li>
              <li>• Supported formats: JPEG, PNG, TIFF</li>
              <li>• Maximum file size: 10 MB</li>
            </ul>
          </Card>

          {/* Navigation */}
          <div className="flex justify-between">
            <Button variant="secondary" onClick={onBack}>
              Back
            </Button>
            <Button variant="ghost" onClick={onSkip}>
              Skip This Step
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
