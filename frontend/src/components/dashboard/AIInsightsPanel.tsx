'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Lightbulb,
  ChevronRight,
  Star,
  Target,
  Activity,
} from 'lucide-react';

interface AIInsight {
  id: string;
  type: 'recommendation' | 'alert' | 'trend' | 'achievement';
  title: string;
  description: string;
  confidence: number;
  priority: 'high' | 'medium' | 'low';
  actionable: boolean;
  timestamp: string;
}

interface AIInsightsPanelProps {
  insights?: AIInsight[];
  maxInsights?: number;
}

export default function AIInsightsPanel({ insights, maxInsights = 6 }: AIInsightsPanelProps) {
  const [selectedInsight, setSelectedInsight] = useState<string | null>(null);

  const defaultInsights: AIInsight[] = [
    {
      id: '1',
      type: 'recommendation',
      title: 'Optimize Speech Assessment Frequency',
      description:
        'Based on patient patterns, increasing speech assessments to twice weekly could improve early detection by 23%.',
      confidence: 94,
      priority: 'high',
      actionable: true,
      timestamp: '2025-08-23T14:30:00Z',
    },
    {
      id: '2',
      type: 'trend',
      title: 'Cognitive Scores Trending Upward',
      description:
        'Patient cognitive assessment scores have improved 15% over the last month, indicating positive treatment response.',
      confidence: 87,
      priority: 'medium',
      actionable: false,
      timestamp: '2025-08-23T14:15:00Z',
    },
    {
      id: '3',
      type: 'alert',
      title: 'Motor Function Variance Detected',
      description:
        'Unusual variance in motor function tests suggests need for additional evaluation or equipment calibration.',
      confidence: 91,
      priority: 'high',
      actionable: true,
      timestamp: '2025-08-23T14:00:00Z',
    },
    {
      id: '4',
      type: 'achievement',
      title: 'Assessment Accuracy Milestone',
      description:
        'Your assessment accuracy has reached 98.2%, placing you in the top 5% of healthcare professionals.',
      confidence: 99,
      priority: 'low',
      actionable: false,
      timestamp: '2025-08-23T13:45:00Z',
    },
    {
      id: '5',
      type: 'recommendation',
      title: 'Retinal Imaging Protocol Update',
      description:
        'New AI model suggests adjusting retinal imaging parameters for 12% improvement in diagnostic accuracy.',
      confidence: 89,
      priority: 'medium',
      actionable: true,
      timestamp: '2025-08-23T13:30:00Z',
    },
    {
      id: '6',
      type: 'trend',
      title: 'Multi-Modal Fusion Efficiency',
      description:
        'NRI fusion processing time has decreased by 18% while maintaining accuracy, indicating system optimization success.',
      confidence: 92,
      priority: 'low',
      actionable: false,
      timestamp: '2025-08-23T13:15:00Z',
    },
  ];

  const insightList = (insights || defaultInsights).slice(0, maxInsights);

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'recommendation':
        return <Lightbulb className='h-4 w-4' />;
      case 'alert':
        return <AlertTriangle className='h-4 w-4' />;
      case 'trend':
        return <TrendingUp className='h-4 w-4' />;
      case 'achievement':
        return <Star className='h-4 w-4' />;
      default:
        return <Brain className='h-4 w-4' />;
    }
  };

  const getInsightColor = (type: string, priority: string) => {
    if (priority === 'high') return '#FF3B30';

    switch (type) {
      case 'recommendation':
        return '#007AFF';
      case 'alert':
        return '#FF9500';
      case 'trend':
        return '#34C759';
      case 'achievement':
        return '#FFD60A';
      default:
        return '#86868B';
    }
  };

  const getPriorityBadge = (priority: string) => {
    const colors = {
      high: '#FF3B30',
      medium: '#FF9500',
      low: '#34C759',
    };

    return (
      <span
        className='rounded-full px-2 py-1 text-xs font-medium'
        style={{
          backgroundColor: `${colors[priority as keyof typeof colors]}20`,
          color: colors[priority as keyof typeof colors],
        }}
      >
        {priority.toUpperCase()}
      </span>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className='rounded-2xl border p-6 backdrop-blur-xl'
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.6)',
        borderColor: 'rgba(0, 0, 0, 0.1)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div className='mb-6 flex items-center justify-between'>
        <div className='flex items-center space-x-3'>
          <div
            className='flex h-10 w-10 items-center justify-center rounded-full'
            style={{ backgroundColor: '#007AFF20' }}
          >
            <Brain className='h-5 w-5' style={{ color: '#007AFF' }} />
          </div>
          <div>
            <h3
              className='text-lg font-semibold'
              style={{
                color: '#1D1D1F',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              }}
            >
              AI Insights
            </h3>
            <p className='text-sm' style={{ color: '#86868B' }}>
              Personalized recommendations and analysis
            </p>
          </div>
        </div>
        <button
          className='text-sm font-medium transition-opacity hover:opacity-70'
          style={{ color: '#007AFF' }}
        >
          View All
        </button>
      </div>

      {/* Insights List */}
      <div className='space-y-3'>
        {insightList.map((insight, index) => (
          <motion.div
            key={insight.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            className={`cursor-pointer rounded-xl border p-4 transition-all duration-300 ${
              selectedInsight === insight.id
                ? 'bg-white/80 shadow-lg'
                : 'bg-white/40 hover:bg-white/60'
            } `}
            style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}
            onClick={() => setSelectedInsight(selectedInsight === insight.id ? null : insight.id)}
          >
            <div className='flex items-start justify-between'>
              {/* Insight Content */}
              <div className='flex flex-1 items-start space-x-3'>
                <div
                  className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full'
                  style={{
                    backgroundColor: `${getInsightColor(insight.type, insight.priority)}20`,
                  }}
                >
                  {React.cloneElement(getInsightIcon(insight.type) as React.ReactElement, {
                    style: { color: getInsightColor(insight.type, insight.priority) },
                  })}
                </div>
                <div className='min-w-0 flex-1'>
                  <div className='mb-1 flex items-center space-x-2'>
                    <h4 className='text-sm font-medium' style={{ color: '#1D1D1F' }}>
                      {insight.title}
                    </h4>
                    {getPriorityBadge(insight.priority)}
                  </div>
                  <p className='line-clamp-2 text-xs' style={{ color: '#86868B' }}>
                    {insight.description}
                  </p>
                  <div className='mt-2 flex items-center justify-between'>
                    <div className='flex items-center space-x-3'>
                      <span className='text-xs' style={{ color: '#86868B' }}>
                        Confidence: {insight.confidence}%
                      </span>
                      {insight.actionable && (
                        <span className='text-xs font-medium' style={{ color: '#007AFF' }}>
                          Actionable
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Expand Icon */}
              <motion.div
                animate={{ rotate: selectedInsight === insight.id ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronRight className='h-4 w-4' style={{ color: '#86868B' }} />
              </motion.div>
            </div>

            {/* Expanded Content */}
            <AnimatePresence>
              {selectedInsight === insight.id && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className='mt-4 border-t pt-4'
                  style={{ borderColor: 'rgba(0, 0, 0, 0.1)' }}
                >
                  <div className='flex items-center justify-between'>
                    <div className='flex items-center space-x-4'>
                      <div className='flex items-center space-x-1'>
                        <Target className='h-3 w-3' style={{ color: '#86868B' }} />
                        <span className='text-xs' style={{ color: '#86868B' }}>
                          Generated: {new Date(insight.timestamp).toLocaleString()}
                        </span>
                      </div>
                    </div>
                    {insight.actionable && (
                      <button
                        className='rounded-full px-3 py-1 text-xs font-medium transition-opacity hover:opacity-80'
                        style={{
                          backgroundColor: '#007AFF',
                          color: '#FFFFFF',
                        }}
                      >
                        Take Action
                      </button>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
