/**
 * Clinical Recommendations Component
 * Displays intelligent, actionable clinical recommendations with accessibility
 */

import React, { useState, useCallback } from 'react';
import {
  ClinicalRecommendation,
  ClinicalRecommendationEngine,
} from '@/lib/clinical/recommendations';
import { AssessmentResults } from '@/lib/assessment/workflow';
import { useScreenReader } from '@/hooks/useAccessibility';
import {
  AlertTriangle,
  Clock,
  Activity,
  Heart,
  Calendar,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Info,
  Zap,
} from 'lucide-react';

// Component props
interface ClinicalRecommendationsProps {
  results: AssessmentResults;
  audience?: 'clinician' | 'patient' | 'both';
  onScheduleFollowUp?: (recommendation: ClinicalRecommendation) => void;
  onMarkComplete?: (recommendationId: string) => void;
  className?: string;
}

// Priority styling
const getPriorityStyle = (priority: ClinicalRecommendation['priority']) => {
  switch (priority) {
    case 'critical':
      return 'border-red-500 bg-red-50 text-red-900';
    case 'high':
      return 'border-orange-500 bg-orange-50 text-orange-900';
    case 'medium':
      return 'border-yellow-500 bg-yellow-50 text-yellow-900';
    case 'low':
      return 'border-blue-500 bg-blue-50 text-blue-900';
    default:
      return 'border-gray-500 bg-gray-50 text-gray-900';
  }
};

// Priority icon
const getPriorityIcon = (priority: ClinicalRecommendation['priority']) => {
  switch (priority) {
    case 'critical':
      return <AlertTriangle className='h-5 w-5' aria-hidden='true' />;
    case 'high':
      return <Zap className='h-5 w-5' aria-hidden='true' />;
    case 'medium':
      return <Info className='h-5 w-5' aria-hidden='true' />;
    case 'low':
      return <CheckCircle className='h-5 w-5' aria-hidden='true' />;
    default:
      return <Info className='h-5 w-5' aria-hidden='true' />;
  }
};

// Category icon
const getCategoryIcon = (category: ClinicalRecommendation['category']) => {
  switch (category) {
    case 'immediate':
      return <AlertTriangle className='h-4 w-4' aria-hidden='true' />;
    case 'followup':
      return <Calendar className='h-4 w-4' aria-hidden='true' />;
    case 'monitoring':
      return <Activity className='h-4 w-4' aria-hidden='true' />;
    case 'lifestyle':
      return <Heart className='h-4 w-4' aria-hidden='true' />;
    default:
      return <Info className='h-4 w-4' aria-hidden='true' />;
  }
};

export function ClinicalRecommendations({
  results,
  audience = 'both',
  onScheduleFollowUp,
  onMarkComplete,
  className = '',
}: ClinicalRecommendationsProps) {
  const [expandedRecommendations, setExpandedRecommendations] = useState<Set<string>>(new Set());
  const [completedRecommendations, setCompletedRecommendations] = useState<Set<string>>(new Set());
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const { announce } = useScreenReader();

  // Generate recommendations
  const allRecommendations = ClinicalRecommendationEngine.generateRecommendations(results);
  const filteredRecommendations =
    audience === 'both'
      ? allRecommendations
      : ClinicalRecommendationEngine.filterByAudience(allRecommendations, audience);

  // Filter by category
  const displayRecommendations =
    selectedCategory === 'all'
      ? filteredRecommendations
      : filteredRecommendations.filter(rec => rec.category === selectedCategory);

  // Toggle recommendation expansion
  const toggleExpansion = useCallback(
    (recommendationId: string) => {
      setExpandedRecommendations(prev => {
        const newSet = new Set(prev);
        if (newSet.has(recommendationId)) {
          newSet.delete(recommendationId);
          announce('Recommendation collapsed');
        } else {
          newSet.add(recommendationId);
          announce('Recommendation expanded');
        }
        return newSet;
      });
    },
    [announce],
  );

  // Mark recommendation as complete
  const markComplete = useCallback(
    (recommendationId: string) => {
      setCompletedRecommendations(prev => new Set([...prev, recommendationId]));
      onMarkComplete?.(recommendationId);
      announce('Recommendation marked as complete');
    },
    [onMarkComplete, announce],
  );

  // Get category counts
  const categoryCounts = {
    all: filteredRecommendations.length,
    immediate: filteredRecommendations.filter(r => r.category === 'immediate').length,
    followup: filteredRecommendations.filter(r => r.category === 'followup').length,
    monitoring: filteredRecommendations.filter(r => r.category === 'monitoring').length,
    lifestyle: filteredRecommendations.filter(r => r.category === 'lifestyle').length,
  };

  return (
    <div className={`rounded-lg bg-white shadow-lg ${className}`}>
      {/* Header */}
      <div className='border-b border-gray-200 p-6'>
        <h2 className='mb-2 text-xl font-semibold text-gray-900'>Clinical Recommendations</h2>
        <p className='text-gray-600'>
          Personalized recommendations based on your assessment results and clinical guidelines
        </p>
      </div>

      {/* Category Filter */}
      <div className='border-b border-gray-200 p-6'>
        <div className='flex flex-wrap gap-2' role='tablist' aria-label='Recommendation categories'>
          {[
            { id: 'all', label: 'All Recommendations', count: categoryCounts.all },
            { id: 'immediate', label: 'Immediate Action', count: categoryCounts.immediate },
            { id: 'followup', label: 'Follow-up', count: categoryCounts.followup },
            { id: 'monitoring', label: 'Monitoring', count: categoryCounts.monitoring },
            { id: 'lifestyle', label: 'Lifestyle', count: categoryCounts.lifestyle },
          ].map(category => (
            <button
              key={category.id}
              onClick={() => {
                setSelectedCategory(category.id);
                announce(`Showing ${category.label.toLowerCase()} recommendations`);
              }}
              role='tab'
              aria-selected={selectedCategory === category.id}
              aria-controls='recommendations-list'
              className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                selectedCategory === category.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              } `}
            >
              {category.label}
              {category.count > 0 && (
                <span
                  className={`ml-2 rounded-full px-2 py-1 text-xs ${
                    selectedCategory === category.id ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                >
                  {category.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Recommendations List */}
      <div className='p-6'>
        {displayRecommendations.length === 0 ? (
          <div className='py-8 text-center'>
            <CheckCircle className='mx-auto mb-4 h-12 w-12 text-green-500' />
            <h3 className='mb-2 text-lg font-semibold text-gray-900'>No Recommendations</h3>
            <p className='text-gray-600'>
              {selectedCategory === 'all'
                ? 'No recommendations available for your current assessment.'
                : `No ${selectedCategory} recommendations at this time.`}
            </p>
          </div>
        ) : (
          <div id='recommendations-list' className='space-y-4' role='tabpanel'>
            {displayRecommendations.map(recommendation => {
              const isExpanded = expandedRecommendations.has(recommendation.id);
              const isCompleted = completedRecommendations.has(recommendation.id);

              return (
                <div
                  key={recommendation.id}
                  className={`rounded-lg border-l-4 p-4 transition-all ${getPriorityStyle(recommendation.priority)} ${isCompleted ? 'opacity-60' : ''} `}
                  role='article'
                  aria-labelledby={`rec-title-${recommendation.id}`}
                >
                  {/* Recommendation Header */}
                  <div className='mb-3 flex items-start justify-between'>
                    <div className='flex flex-1 items-start gap-3'>
                      <div className='mt-1 flex items-center gap-2'>
                        {getPriorityIcon(recommendation.priority)}
                        {getCategoryIcon(recommendation.category)}
                      </div>

                      <div className='flex-1'>
                        <div className='mb-1 flex items-center gap-2'>
                          <h3
                            id={`rec-title-${recommendation.id}`}
                            className='text-lg font-semibold'
                          >
                            {recommendation.title}
                          </h3>

                          <span
                            className={`rounded-full px-2 py-1 text-xs font-medium uppercase tracking-wide ${
                              recommendation.priority === 'critical'
                                ? 'bg-red-100 text-red-800'
                                : recommendation.priority === 'high'
                                  ? 'bg-orange-100 text-orange-800'
                                  : recommendation.priority === 'medium'
                                    ? 'bg-yellow-100 text-yellow-800'
                                    : 'bg-blue-100 text-blue-800'
                            } `}
                          >
                            {recommendation.priority}
                          </span>
                        </div>

                        <p className='mb-2 text-sm'>{recommendation.description}</p>

                        <div className='flex items-center gap-4 text-xs'>
                          <div className='flex items-center gap-1'>
                            <Clock className='h-3 w-3' aria-hidden='true' />
                            <span>Timeframe: {recommendation.timeframe}</span>
                          </div>

                          <div className='flex items-center gap-1'>
                            <span>Evidence Level: {recommendation.evidenceLevel}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className='ml-4 flex items-center gap-2'>
                      {!isCompleted && onMarkComplete && (
                        <button
                          onClick={() => markComplete(recommendation.id)}
                          className='rounded-lg p-2 text-green-600 transition-colors hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2'
                          aria-label='Mark recommendation as complete'
                          title='Mark as complete'
                        >
                          <CheckCircle className='h-4 w-4' />
                        </button>
                      )}

                      <button
                        onClick={() => toggleExpansion(recommendation.id)}
                        className='rounded-lg p-2 text-gray-600 transition-colors hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2'
                        aria-expanded={isExpanded}
                        aria-controls={`rec-details-${recommendation.id}`}
                        aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                      >
                        {isExpanded ? (
                          <ChevronUp className='h-4 w-4' />
                        ) : (
                          <ChevronDown className='h-4 w-4' />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {isExpanded && (
                    <div
                      id={`rec-details-${recommendation.id}`}
                      className='mt-4 border-t border-gray-200 pt-4'
                    >
                      {/* Rationale */}
                      <div className='mb-4'>
                        <h4 className='mb-2 text-sm font-medium'>Clinical Rationale:</h4>
                        <p className='text-sm text-gray-700'>{recommendation.rationale}</p>
                      </div>

                      {/* Action Items */}
                      <div className='mb-4'>
                        <h4 className='mb-2 text-sm font-medium'>Action Items:</h4>
                        <ul className='list-inside list-disc space-y-1 text-sm text-gray-700'>
                          {recommendation.actionItems.map((item, index) => (
                            <li key={index}>{item}</li>
                          ))}
                        </ul>
                      </div>

                      {/* Sources */}
                      {recommendation.sources.length > 0 && (
                        <div className='mb-4'>
                          <h4 className='mb-2 text-sm font-medium'>Evidence Sources:</h4>
                          <ul className='space-y-2'>
                            {recommendation.sources.map((source, index) => (
                              <li key={index} className='text-sm'>
                                <a
                                  href={source.url}
                                  target='_blank'
                                  rel='noopener noreferrer'
                                  className='flex items-center gap-1 text-blue-600 underline hover:text-blue-800'
                                >
                                  {source.title} ({source.year})
                                  <ExternalLink className='h-3 w-3' aria-hidden='true' />
                                </a>
                                <span className='ml-2 text-xs text-gray-500'>
                                  {source.type.replace('_', ' ')}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Actions */}
                      <div className='flex items-center gap-3 border-t border-gray-200 pt-3'>
                        {onScheduleFollowUp && recommendation.category === 'followup' && (
                          <button
                            onClick={() => onScheduleFollowUp(recommendation)}
                            className='flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
                          >
                            <Calendar className='h-4 w-4' />
                            Schedule Appointment
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
