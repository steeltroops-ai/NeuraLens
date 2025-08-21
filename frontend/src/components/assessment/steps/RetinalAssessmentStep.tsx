'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui';

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
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-6">
        <div className="mx-auto max-w-4xl space-y-12">
          {/* Apple-Style Header */}
          <div className="animate-fade-in space-y-6 text-center">
            <div className="shadow-neural mx-auto flex h-20 w-20 items-center justify-center rounded-apple-xl bg-gradient-to-br from-neural-500 to-neural-600">
              <svg
                className="h-10 w-10 text-white"
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
            <h1 className="text-4xl font-bold tracking-tight text-text-primary">
              Eye Health Scan
            </h1>
            <p className="mx-auto max-w-2xl text-xl leading-relaxed text-text-secondary">
              Upload an eye photo for blood vessel health analysis
            </p>
          </div>

          {/* Apple-Style Upload Interface */}
          <div className="card-apple animate-slide-up p-12">
            <div
              className={`rounded-apple-lg border-2 border-dashed p-12 text-center transition-all duration-300 ${
                dragActive
                  ? 'scale-105 border-neural-500 bg-neural-50'
                  : selectedFile
                    ? 'border-success-500 bg-success-50'
                    : 'border-gray-300 hover:border-neural-400 hover:bg-gray-50'
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              {selectedFile ? (
                <div className="space-y-6">
                  <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-success-400 to-success-500 shadow-lg">
                    <svg
                      className="h-12 w-12 text-white"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                    </svg>
                  </div>
                  <div className="space-y-3">
                    <h3 className="text-2xl font-semibold text-text-primary">
                      Image Ready for Analysis
                    </h3>
                    <p className="text-lg text-text-secondary">
                      {selectedFile.name}
                    </p>
                    <p className="text-text-muted inline-block rounded-full bg-gray-100 px-4 py-2 text-sm">
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="space-y-4">
                    <Button
                      onClick={handleComplete}
                      size="xl"
                      className="shadow-neural hover:shadow-neural-hover"
                    >
                      Analyze Image
                      <svg
                        className="ml-3 h-6 w-6"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 7l5 5m0 0l-5 5m5-5H6"
                        />
                      </svg>
                    </Button>
                    <Button
                      variant="secondary"
                      size="lg"
                      onClick={() => setSelectedFile(null)}
                      className="px-8"
                    >
                      Select Different Image
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-8">
                  <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-neural-400 to-neural-500 shadow-lg">
                    <svg
                      className="h-12 w-12 text-white"
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
                  </div>
                  <div className="space-y-4">
                    <h3 className="text-2xl font-semibold text-text-primary">
                      Upload Retinal Image
                    </h3>
                    <p className="text-lg leading-relaxed text-text-secondary">
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
                        variant="primary"
                        size="xl"
                        className="shadow-neural hover:shadow-neural-hover cursor-pointer"
                      >
                        <span>
                          <svg
                            className="mr-3 h-6 w-6"
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
                          Browse Files
                        </span>
                      </Button>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Apple-Style Instructions */}
          <div className="animate-scale-in rounded-apple-lg border border-neural-100 bg-neural-50 p-8">
            <h4 className="mb-6 text-center text-xl font-semibold text-text-primary">
              Image Requirements
            </h4>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="flex items-start space-x-3">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-neural-100">
                  <svg
                    className="h-4 w-4 text-neural-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h5 className="font-medium text-text-primary">
                    High Quality
                  </h5>
                  <p className="text-sm text-text-secondary">
                    Color fundus photograph with clear details
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-neural-100">
                  <svg
                    className="h-4 w-4 text-neural-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h5 className="font-medium text-text-primary">Clear View</h5>
                  <p className="text-sm text-text-secondary">
                    Optic disc and macula clearly visible
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-neural-100">
                  <svg
                    className="h-4 w-4 text-neural-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h5 className="font-medium text-text-primary">
                    File Formats
                  </h5>
                  <p className="text-sm text-text-secondary">
                    JPEG, PNG, TIFF supported
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-apple bg-neural-100">
                  <svg
                    className="h-4 w-4 text-neural-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div>
                  <h5 className="font-medium text-text-primary">File Size</h5>
                  <p className="text-sm text-text-secondary">
                    Maximum 10 MB per image
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Apple-Style Navigation */}
          <div className="flex animate-fade-in justify-center gap-6">
            <Button
              variant="secondary"
              onClick={onSkip}
              size="lg"
              className="px-8"
            >
              Skip This Step
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
