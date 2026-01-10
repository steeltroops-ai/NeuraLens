'use client';

import {
  History,
  Search,
  Filter,
  Edit3,
  Trash2,
  Eye,
  Download,
  Calendar,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  MoreHorizontal,
  Plus,
} from 'lucide-react';
import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface AssessmentRecord {
  id: string;
  sessionId: string;
  type: 'speech' | 'retinal' | 'motor' | 'cognitive' | 'multimodal';
  timestamp: Date;
  status: 'completed' | 'processing' | 'failed';
  riskScore: number;
  confidence: number;
  processingTime: number;
  biomarkers: Record<string, number>;
  notes?: string;
  tags: string[];
}

interface AssessmentHistoryProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

interface FilterState {
  type: string;
  status: string;
  dateRange: string;
  riskLevel: string;
}

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText: string;
  cancelText: string;
  onConfirm: () => void;
  onCancel: () => void;
  variant: 'danger' | 'warning' | 'info';
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  title,
  message,
  confirmText,
  cancelText,
  onConfirm,
  onCancel,
  variant,
}) => {
  if (!isOpen) return null;

  const variantStyles = {
    danger: 'border-red-200 bg-red-50',
    warning: 'border-yellow-200 bg-yellow-50',
    info: 'border-blue-200 bg-blue-50',
  };

  const buttonStyles = {
    danger: 'bg-red-600 hover:bg-red-700',
    warning: 'bg-yellow-600 hover:bg-yellow-700',
    info: 'bg-blue-600 hover:bg-blue-700',
  };

  return (
    <div className='fixed inset-0 z-50 flex items-center justify-center bg-black/50'>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className={`mx-4 w-full max-w-md rounded-xl border p-6 shadow-xl ${variantStyles[variant]}`}
      >
        <div className='mb-4'>
          <h3 className='text-lg font-semibold text-slate-900'>{title}</h3>
          <p className='mt-2 text-sm text-slate-600'>{message}</p>
        </div>
        <div className='flex justify-end space-x-3'>
          <button
            onClick={onCancel}
            className='rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50'
          >
            {cancelText}
          </button>
          <button
            onClick={onConfirm}
            className={`rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors ${buttonStyles[variant]}`}
          >
            {confirmText}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default function AssessmentHistory({ onProcessingChange }: AssessmentHistoryProps) {
  const [assessments, setAssessments] = useState<AssessmentRecord[]>([]);
  const [filteredAssessments, setFilteredAssessments] = useState<AssessmentRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    type: 'all',
    status: 'all',
    dateRange: 'all',
    riskLevel: 'all',
  });
  const [selectedAssessment, setSelectedAssessment] = useState<AssessmentRecord | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    type: 'delete' | 'export' | null;
    assessmentId: string | null;
  }>({
    isOpen: false,
    type: null,
    assessmentId: null,
  });

  // Mock data - replace with actual API calls
  const mockAssessments: AssessmentRecord[] = [
    {
      id: '1',
      sessionId: 'sess_001',
      type: 'speech',
      timestamp: new Date('2024-01-15T10:30:00'),
      status: 'completed',
      riskScore: 0.25,
      confidence: 0.92,
      processingTime: 87,
      biomarkers: { fluency: 0.85, tremor: 0.15, articulation: 0.9 },
      notes: 'Patient showed good speech clarity with minimal tremor indicators.',
      tags: ['baseline', 'routine'],
    },
    {
      id: '2',
      sessionId: 'sess_002',
      type: 'retinal',
      timestamp: new Date('2024-01-14T14:15:00'),
      status: 'completed',
      riskScore: 0.45,
      confidence: 0.88,
      processingTime: 156,
      biomarkers: {
        vesselTortuosity: 0.35,
        avRatio: 0.72,
        cupDiscRatio: 0.28,
        vesselDensity: 0.65,
      },
      notes: 'Moderate vessel tortuosity detected. Recommend follow-up.',
      tags: ['follow-up', 'vascular'],
    },
    {
      id: '3',
      sessionId: 'sess_003',
      type: 'motor',
      timestamp: new Date('2024-01-13T09:45:00'),
      status: 'completed',
      riskScore: 0.15,
      confidence: 0.95,
      processingTime: 134,
      biomarkers: { tapFrequency: 0.85, coordination: 0.92, tremor: 0.08 },
      tags: ['baseline'],
    },
    {
      id: '4',
      sessionId: 'sess_004',
      type: 'cognitive',
      timestamp: new Date('2024-01-12T16:20:00'),
      status: 'processing',
      riskScore: 0,
      confidence: 0,
      processingTime: 0,
      biomarkers: {},
      tags: ['in-progress'],
    },
  ];

  // Load assessments
  useEffect(() => {
    const loadAssessments = async () => {
      setIsLoading(true);
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setAssessments(mockAssessments);
        setFilteredAssessments(mockAssessments);
      } catch (error) {
        console.error('Failed to load assessments:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadAssessments();
  }, []);

  // Filter and search assessments
  useEffect(() => {
    let filtered = assessments;

    // Apply search
    if (searchQuery) {
      filtered = filtered.filter(
        assessment =>
          assessment.sessionId.toLowerCase().includes(searchQuery.toLowerCase()) ||
          assessment.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
          assessment.notes?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          assessment.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())),
      );
    }

    // Apply filters
    if (filters.type !== 'all') {
      filtered = filtered.filter(assessment => assessment.type === filters.type);
    }

    if (filters.status !== 'all') {
      filtered = filtered.filter(assessment => assessment.status === filters.status);
    }

    if (filters.riskLevel !== 'all') {
      filtered = filtered.filter(assessment => {
        if (filters.riskLevel === 'low') return assessment.riskScore < 0.3;
        if (filters.riskLevel === 'moderate')
          return assessment.riskScore >= 0.3 && assessment.riskScore < 0.7;
        if (filters.riskLevel === 'high') return assessment.riskScore >= 0.7;
        return true;
      });
    }

    // Apply date range filter
    if (filters.dateRange !== 'all') {
      const now = new Date();
      const filterDate = new Date();

      switch (filters.dateRange) {
        case 'today':
          filterDate.setHours(0, 0, 0, 0);
          break;
        case 'week':
          filterDate.setDate(now.getDate() - 7);
          break;
        case 'month':
          filterDate.setMonth(now.getMonth() - 1);
          break;
      }

      if (filters.dateRange !== 'all') {
        filtered = filtered.filter(assessment => assessment.timestamp >= filterDate);
      }
    }

    setFilteredAssessments(filtered);
  }, [assessments, searchQuery, filters]);

  // CRUD Operations
  const handleDelete = useCallback(
    async (assessmentId: string) => {
      try {
        onProcessingChange?.(true);

        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));

        setAssessments(prev => prev.filter(a => a.id !== assessmentId));
        setConfirmDialog({ isOpen: false, type: null, assessmentId: null });

        // Show success notification (you can implement a toast system)
        console.log('Assessment deleted successfully');
      } catch (error) {
        console.error('Failed to delete assessment:', error);
      } finally {
        onProcessingChange?.(false);
      }
    },
    [onProcessingChange],
  );

  const handleExport = useCallback(
    async (assessmentId: string) => {
      try {
        onProcessingChange?.(true);

        const assessment = assessments.find(a => a.id === assessmentId);
        if (!assessment) return;

        // Simulate export
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Create and download file
        const data = JSON.stringify(assessment, null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `assessment_${assessment.sessionId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        setConfirmDialog({ isOpen: false, type: null, assessmentId: null });
      } catch (error) {
        console.error('Failed to export assessment:', error);
      } finally {
        onProcessingChange?.(false);
      }
    },
    [assessments, onProcessingChange],
  );

  const handleEdit = useCallback((assessment: AssessmentRecord) => {
    setSelectedAssessment(assessment);
    // Open edit modal (implement as needed)
    console.log('Edit assessment:', assessment);
  }, []);

  const handleView = useCallback((assessment: AssessmentRecord) => {
    setSelectedAssessment(assessment);
    // Open view modal (implement as needed)
    console.log('View assessment:', assessment);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className='h-4 w-4 text-green-600' />;
      case 'processing':
        return <Clock className='h-4 w-4 text-yellow-600' />;
      case 'failed':
        return <AlertCircle className='h-4 w-4 text-red-600' />;
      default:
        return <Clock className='h-4 w-4 text-slate-400' />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'speech':
        return 'bg-blue-100 text-blue-800';
      case 'retinal':
        return 'bg-green-100 text-green-800';
      case 'motor':
        return 'bg-purple-100 text-purple-800';
      case 'cognitive':
        return 'bg-orange-100 text-orange-800';
      case 'multimodal':
        return 'bg-indigo-100 text-indigo-800';
      default:
        return 'bg-slate-100 text-slate-800';
    }
  };

  const getRiskColor = (riskScore: number) => {
    if (riskScore < 0.3) return 'text-green-600';
    if (riskScore < 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className='space-y-6'>
      {/* Header */}
      <div className='rounded-xl border border-slate-200 bg-white p-6 shadow-sm'>
        <div className='mb-4 flex items-center justify-between'>
          <div className='flex items-center space-x-3'>
            <div className='rounded-lg bg-gradient-to-r from-indigo-500 to-indigo-600 p-3'>
              <History className='h-6 w-6 text-white' />
            </div>
            <div>
              <h1 className='text-2xl font-bold text-slate-900'>Assessment History</h1>
              <p className='text-slate-600'>Manage and review your neurological assessments</p>
            </div>
          </div>
          <button className='rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700'>
            <Plus className='mr-2 h-4 w-4' />
            New Assessment
          </button>
        </div>

        {/* Search and Filters */}
        <div className='grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5'>
          {/* Search */}
          <div className='relative lg:col-span-2'>
            <Search className='absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400' />
            <input
              type='text'
              placeholder='Search assessments...'
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className='w-full rounded-lg border border-slate-300 py-2 pl-10 pr-4 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500'
            />
          </div>

          {/* Type Filter */}
          <select
            value={filters.type}
            onChange={e => setFilters(prev => ({ ...prev, type: e.target.value }))}
            className='rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500'
          >
            <option value='all'>All Types</option>
            <option value='speech'>Speech</option>
            <option value='retinal'>Retinal</option>
            <option value='motor'>Motor</option>
            <option value='cognitive'>Cognitive</option>
            <option value='multimodal'>Multimodal</option>
          </select>

          {/* Status Filter */}
          <select
            value={filters.status}
            onChange={e => setFilters(prev => ({ ...prev, status: e.target.value }))}
            className='rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500'
          >
            <option value='all'>All Status</option>
            <option value='completed'>Completed</option>
            <option value='processing'>Processing</option>
            <option value='failed'>Failed</option>
          </select>

          {/* Date Range Filter */}
          <select
            value={filters.dateRange}
            onChange={e => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
            className='rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500'
          >
            <option value='all'>All Time</option>
            <option value='today'>Today</option>
            <option value='week'>This Week</option>
            <option value='month'>This Month</option>
          </select>
        </div>

        {/* Results Summary */}
        <div className='mt-4 flex items-center justify-between text-sm text-slate-600'>
          <span>
            Showing {filteredAssessments.length} of {assessments.length} assessments
          </span>
          <div className='flex items-center space-x-4'>
            <button className='flex items-center space-x-1 hover:text-slate-900'>
              <Download className='h-4 w-4' />
              <span>Export All</span>
            </button>
            <button className='flex items-center space-x-1 hover:text-slate-900'>
              <Filter className='h-4 w-4' />
              <span>Advanced Filters</span>
            </button>
          </div>
        </div>
      </div>

      {/* Assessment List */}
      <div className='rounded-xl border border-slate-200 bg-white shadow-sm'>
        {isLoading ? (
          <div className='flex items-center justify-center p-12'>
            <div className='text-center'>
              <div className='mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent'></div>
              <p className='text-slate-600'>Loading assessments...</p>
            </div>
          </div>
        ) : filteredAssessments.length === 0 ? (
          <div className='flex items-center justify-center p-12'>
            <div className='text-center'>
              <History className='mx-auto mb-4 h-12 w-12 text-slate-400' />
              <h3 className='mb-2 text-lg font-medium text-slate-900'>No assessments found</h3>
              <p className='text-slate-600'>
                {searchQuery || Object.values(filters).some(f => f !== 'all')
                  ? 'Try adjusting your search or filters'
                  : 'Start by creating your first assessment'}
              </p>
            </div>
          </div>
        ) : (
          <div className='divide-y divide-slate-200'>
            <AnimatePresence>
              {filteredAssessments.map(assessment => (
                <motion.div
                  key={assessment.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className='p-6 transition-colors hover:bg-slate-50'
                >
                  <div className='flex items-center justify-between'>
                    <div className='flex items-center space-x-4'>
                      {/* Status Icon */}
                      <div className='flex-shrink-0'>{getStatusIcon(assessment.status)}</div>

                      {/* Assessment Info */}
                      <div className='min-w-0 flex-1'>
                        <div className='flex items-center space-x-3'>
                          <h3 className='text-sm font-medium text-slate-900'>
                            {assessment.sessionId}
                          </h3>
                          <span
                            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${getTypeColor(assessment.type)}`}
                          >
                            {assessment.type}
                          </span>
                          {assessment.status === 'completed' && (
                            <span
                              className={`text-sm font-medium ${getRiskColor(assessment.riskScore)}`}
                            >
                              Risk: {(assessment.riskScore * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                        <div className='mt-1 flex items-center space-x-4 text-sm text-slate-500'>
                          <div className='flex items-center space-x-1'>
                            <Calendar className='h-3 w-3' />
                            <span>{assessment.timestamp.toLocaleDateString()}</span>
                          </div>
                          <div className='flex items-center space-x-1'>
                            <Clock className='h-3 w-3' />
                            <span>{assessment.timestamp.toLocaleTimeString()}</span>
                          </div>
                          {assessment.status === 'completed' && (
                            <div className='flex items-center space-x-1'>
                              <TrendingUp className='h-3 w-3' />
                              <span>{assessment.processingTime}ms</span>
                            </div>
                          )}
                        </div>
                        {assessment.notes && (
                          <p className='mt-2 line-clamp-2 text-sm text-slate-600'>
                            {assessment.notes}
                          </p>
                        )}
                        {assessment.tags.length > 0 && (
                          <div className='mt-2 flex flex-wrap gap-1'>
                            {assessment.tags.map((tag, index) => (
                              <span
                                key={index}
                                className='inline-flex items-center rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600'
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className='flex items-center space-x-2'>
                      <button
                        onClick={() => handleView(assessment)}
                        className='rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                        title='View Details'
                      >
                        <Eye className='h-4 w-4' />
                      </button>
                      <button
                        onClick={() => handleEdit(assessment)}
                        className='rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                        title='Edit Assessment'
                      >
                        <Edit3 className='h-4 w-4' />
                      </button>
                      <button
                        onClick={() =>
                          setConfirmDialog({
                            isOpen: true,
                            type: 'export',
                            assessmentId: assessment.id,
                          })
                        }
                        className='rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600'
                        title='Export Assessment'
                      >
                        <Download className='h-4 w-4' />
                      </button>
                      <button
                        onClick={() =>
                          setConfirmDialog({
                            isOpen: true,
                            type: 'delete',
                            assessmentId: assessment.id,
                          })
                        }
                        className='rounded-lg p-2 text-slate-400 hover:bg-red-100 hover:text-red-600'
                        title='Delete Assessment'
                      >
                        <Trash2 className='h-4 w-4' />
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Confirmation Dialogs */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen && confirmDialog.type === 'delete'}
        title='Delete Assessment'
        message='Are you sure you want to delete this assessment? This action cannot be undone.'
        confirmText='Delete'
        cancelText='Cancel'
        variant='danger'
        onConfirm={() => confirmDialog.assessmentId && handleDelete(confirmDialog.assessmentId)}
        onCancel={() => setConfirmDialog({ isOpen: false, type: null, assessmentId: null })}
      />

      <ConfirmDialog
        isOpen={confirmDialog.isOpen && confirmDialog.type === 'export'}
        title='Export Assessment'
        message='Export this assessment data as a JSON file?'
        confirmText='Export'
        cancelText='Cancel'
        variant='info'
        onConfirm={() => confirmDialog.assessmentId && handleExport(confirmDialog.assessmentId)}
        onCancel={() => setConfirmDialog({ isOpen: false, type: null, assessmentId: null })}
      />
    </div>
  );
}
