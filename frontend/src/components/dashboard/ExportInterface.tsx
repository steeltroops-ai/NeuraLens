/**
 * Advanced Export Interface Component
 * Comprehensive export functionality with progress tracking and accessibility
 */

import React, { useState, useCallback } from 'react';
import {
  ExportService,
  ExportOptions,
  ExportFormat,
  ExportTemplate,
  ExportProgress,
} from '@/lib/export/exportService';
import { AssessmentResults } from '@/lib/assessment/workflow';
import { ClinicalRecommendation } from '@/lib/clinical/recommendations';
import { useScreenReader } from '@/hooks/useAccessibility';
import { LoadingButton } from '@/components/ui/LoadingStates';
import {
  Download,
  FileText,
  Database,
  Share2,
  Lock,
  Calendar,
  CheckCircle,
  AlertTriangle,
  Settings,
  Users,
} from 'lucide-react';

// Component props
interface ExportInterfaceProps {
  results: AssessmentResults;
  recommendations: ClinicalRecommendation[];
  onExportComplete?: (result: any) => void;
  className?: string;
}

// Export format configurations
const EXPORT_FORMATS: Array<{
  id: ExportFormat;
  label: string;
  description: string;
  icon: React.ComponentType<any>;
  fileExtension: string;
}> = [
  {
    id: 'pdf',
    label: 'PDF Report',
    description: 'Professional report suitable for clinical use',
    icon: FileText,
    fileExtension: '.pdf',
  },
  {
    id: 'json',
    label: 'JSON Data',
    description: 'Structured data for integration and analysis',
    icon: Database,
    fileExtension: '.json',
  },
  {
    id: 'csv',
    label: 'CSV Spreadsheet',
    description: 'Tabular data for statistical analysis',
    icon: Database,
    fileExtension: '.csv',
  },
  {
    id: 'hl7',
    label: 'HL7 Message',
    description: 'Healthcare interoperability standard',
    icon: Share2,
    fileExtension: '.hl7',
  },
  {
    id: 'fhir',
    label: 'FHIR Resource',
    description: 'Fast Healthcare Interoperability Resources',
    icon: Share2,
    fileExtension: '.json',
  },
];

// Export template configurations
const EXPORT_TEMPLATES: Array<{
  id: ExportTemplate;
  label: string;
  description: string;
  audience: string;
}> = [
  {
    id: 'clinical_summary',
    label: 'Clinical Summary',
    description: 'Concise report for healthcare providers',
    audience: 'Healthcare Providers',
  },
  {
    id: 'detailed_technical',
    label: 'Technical Report',
    description: 'Detailed technical analysis with biomarkers',
    audience: 'Researchers & Specialists',
  },
  {
    id: 'patient_friendly',
    label: 'Patient Report',
    description: 'Easy-to-understand report for patients',
    audience: 'Patients & Families',
  },
  {
    id: 'research_data',
    label: 'Research Data',
    description: 'Raw data for research and analysis',
    audience: 'Researchers',
  },
];

export function ExportInterface({
  results,
  recommendations,
  onExportComplete,
  className = '',
}: ExportInterfaceProps) {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('pdf');
  const [selectedTemplate, setSelectedTemplate] = useState<ExportTemplate>('clinical_summary');
  const [exportOptions, setExportOptions] = useState<Partial<ExportOptions>>({
    includeRecommendations: true,
    includeRawData: false,
    includeTrendData: false,
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const { announce } = useScreenReader();

  // Handle export
  const handleExport = useCallback(async () => {
    setIsExporting(true);
    setExportProgress(null);

    announce('Starting export process');

    const options: ExportOptions = {
      format: selectedFormat,
      template: selectedTemplate,
      ...exportOptions,
    };

    try {
      const result = await ExportService.exportAssessment(
        results,
        recommendations,
        options,
        progress => {
          setExportProgress(progress);
          announce(`Export progress: ${progress.message}`);
        },
      );

      if (result.success && result.downloadUrl) {
        // Trigger download
        const link = document.createElement('a');
        link.href = result.downloadUrl;
        link.download = result.fileName;
        link.click();

        // Clean up URL
        setTimeout(() => URL.revokeObjectURL(result.downloadUrl!), 1000);

        announce('Export completed successfully');
        onExportComplete?.(result);
      } else {
        throw new Error(result.error || 'Export failed');
      }
    } catch (error) {
      console.error('Export failed:', error);
      announce(`Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsExporting(false);
      setExportProgress(null);
    }
  }, [
    selectedFormat,
    selectedTemplate,
    exportOptions,
    results,
    recommendations,
    announce,
    onExportComplete,
  ]);

  // Handle format selection
  const handleFormatSelect = useCallback(
    (format: ExportFormat) => {
      setSelectedFormat(format);
      announce(`Selected ${format.toUpperCase()} format`);
    },
    [announce],
  );

  // Handle template selection
  const handleTemplateSelect = useCallback(
    (template: ExportTemplate) => {
      setSelectedTemplate(template);
      announce(`Selected ${template.replace('_', ' ')} template`);
    },
    [announce],
  );

  return (
    <div className={`rounded-lg bg-white shadow-lg ${className}`}>
      {/* Header */}
      <div className='border-b border-gray-200 p-6'>
        <h2 className='mb-2 text-xl font-semibold text-gray-900'>Export Assessment Results</h2>
        <p className='text-gray-600'>Choose format and template for your assessment report</p>
      </div>

      <div className='space-y-6 p-6'>
        {/* Format Selection */}
        <div>
          <h3 className='mb-4 text-lg font-medium text-gray-900'>Export Format</h3>
          <div className='grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3'>
            {EXPORT_FORMATS.map(format => {
              const Icon = format.icon;
              const isSelected = selectedFormat === format.id;

              return (
                <button
                  key={format.id}
                  onClick={() => handleFormatSelect(format.id)}
                  className={`rounded-lg border-2 p-4 text-left transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  } `}
                  aria-pressed={isSelected}
                  role='radio'
                  aria-describedby={`format-${format.id}-desc`}
                >
                  <div className='mb-2 flex items-center gap-3'>
                    <Icon className={`h-5 w-5 ${isSelected ? 'text-blue-600' : 'text-gray-600'}`} />
                    <span
                      className={`font-medium ${isSelected ? 'text-blue-900' : 'text-gray-900'}`}
                    >
                      {format.label}
                    </span>
                  </div>
                  <p id={`format-${format.id}-desc`} className='text-sm text-gray-600'>
                    {format.description}
                  </p>
                  <p className='mt-1 text-xs text-gray-500'>File type: {format.fileExtension}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Template Selection */}
        <div>
          <h3 className='mb-4 text-lg font-medium text-gray-900'>Report Template</h3>
          <div className='grid grid-cols-1 gap-4 md:grid-cols-2'>
            {EXPORT_TEMPLATES.map(template => {
              const isSelected = selectedTemplate === template.id;

              return (
                <button
                  key={template.id}
                  onClick={() => handleTemplateSelect(template.id)}
                  className={`rounded-lg border-2 p-4 text-left transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  } `}
                  aria-pressed={isSelected}
                  role='radio'
                  aria-describedby={`template-${template.id}-desc`}
                >
                  <div className='mb-2 flex items-center justify-between'>
                    <span
                      className={`font-medium ${isSelected ? 'text-blue-900' : 'text-gray-900'}`}
                    >
                      {template.label}
                    </span>
                    <Users
                      className={`h-4 w-4 ${isSelected ? 'text-blue-600' : 'text-gray-400'}`}
                    />
                  </div>
                  <p id={`template-${template.id}-desc`} className='mb-1 text-sm text-gray-600'>
                    {template.description}
                  </p>
                  <p className='text-xs text-gray-500'>Target audience: {template.audience}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Export Options */}
        <div>
          <div className='mb-4 flex items-center justify-between'>
            <h3 className='text-lg font-medium text-gray-900'>Export Options</h3>
            <button
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className='flex items-center gap-2 rounded text-sm text-blue-600 hover:text-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
              aria-expanded={showAdvancedOptions}
              aria-controls='advanced-options'
            >
              <Settings className='h-4 w-4' />
              {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
            </button>
          </div>

          <div className='space-y-3'>
            {/* Basic Options */}
            <label className='flex items-center gap-3'>
              <input
                type='checkbox'
                checked={exportOptions.includeRecommendations}
                onChange={e =>
                  setExportOptions(prev => ({
                    ...prev,
                    includeRecommendations: e.target.checked,
                  }))
                }
                className='h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500'
              />
              <span className='text-sm text-gray-700'>Include clinical recommendations</span>
            </label>

            <label className='flex items-center gap-3'>
              <input
                type='checkbox'
                checked={exportOptions.includeRawData}
                onChange={e =>
                  setExportOptions(prev => ({
                    ...prev,
                    includeRawData: e.target.checked,
                  }))
                }
                className='h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500'
              />
              <span className='text-sm text-gray-700'>Include raw biomarker data</span>
            </label>

            <label className='flex items-center gap-3'>
              <input
                type='checkbox'
                checked={exportOptions.includeTrendData}
                onChange={e =>
                  setExportOptions(prev => ({
                    ...prev,
                    includeTrendData: e.target.checked,
                  }))
                }
                className='h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500'
              />
              <span className='text-sm text-gray-700'>Include trend analysis data</span>
            </label>

            {/* Advanced Options */}
            {showAdvancedOptions && (
              <div id='advanced-options' className='mt-4 space-y-3 rounded-lg bg-gray-50 p-4'>
                <h4 className='font-medium text-gray-900'>Access Controls</h4>

                <label className='flex items-center gap-3'>
                  <input
                    type='checkbox'
                    onChange={e =>
                      setExportOptions(prev => ({
                        ...prev,
                        accessControls: {
                          ...prev.accessControls,
                          requirePassword: e.target.checked,
                        },
                      }))
                    }
                    className='h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500'
                  />
                  <span className='text-sm text-gray-700'>Require password for access</span>
                </label>

                <div className='flex items-center gap-3'>
                  <Calendar className='h-4 w-4 text-gray-500' />
                  <label className='flex-1'>
                    <span className='mb-1 block text-sm text-gray-700'>
                      Expiration date (optional)
                    </span>
                    <input
                      type='date'
                      onChange={e =>
                        setExportOptions(prev => ({
                          ...prev,
                          accessControls: {
                            ...prev.accessControls,
                            expirationDate: e.target.value,
                          },
                        }))
                      }
                      className='w-full rounded-lg border border-gray-300 px-3 py-2 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500'
                    />
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Export Progress */}
        {exportProgress && (
          <div
            className='rounded-lg border border-blue-200 bg-blue-50 p-4'
            role='status'
            aria-live='polite'
          >
            <div className='mb-2 flex items-center gap-3'>
              <div className='h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent' />
              <span className='font-medium text-blue-900'>{exportProgress.message}</span>
            </div>

            <div className='h-2 w-full rounded-full bg-blue-200'>
              <div
                className='h-2 rounded-full bg-blue-600 transition-all duration-300'
                style={{ width: `${exportProgress.progress}%` }}
              />
            </div>

            <div className='mt-1 flex justify-between text-sm text-blue-700'>
              <span>{exportProgress.progress.toFixed(0)}% complete</span>
              {exportProgress.estimatedTimeRemaining && (
                <span>{exportProgress.estimatedTimeRemaining}s remaining</span>
              )}
            </div>
          </div>
        )}

        {/* Export Button */}
        <div className='flex items-center justify-between border-t border-gray-200 pt-4'>
          <div className='text-sm text-gray-600'>
            <p>Export will include assessment data from session: {results.sessionId}</p>
            <p>Completed: {new Date(results.completionTime).toLocaleString()}</p>
          </div>

          <LoadingButton
            loading={isExporting}
            onClick={handleExport}
            loadingText='Exporting...'
            className='flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
            aria-describedby='export-description'
          >
            <Download className='h-4 w-4' />
            Export {selectedFormat.toUpperCase()}
          </LoadingButton>

          <div id='export-description' className='sr-only'>
            Export assessment results as {selectedFormat} using {selectedTemplate.replace('_', ' ')}{' '}
            template
          </div>
        </div>
      </div>
    </div>
  );
}
