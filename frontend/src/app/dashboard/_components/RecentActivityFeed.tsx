'use client';

import { motion } from 'framer-motion';
import {
  Mic,
  Eye,
  Hand,
  Brain,
  Zap,
  Activity,
  CheckCircle,
  Clock,
  AlertTriangle,
  User,
} from 'lucide-react';
import React from 'react';

interface ActivityItem {
  id: string;
  type: 'speech' | 'retinal' | 'motor' | 'cognitive' | 'nri' | 'multimodal' | 'system';
  title: string;
  description: string;
  timestamp: string;
  status: 'completed' | 'processing' | 'failed' | 'pending';
  user?: string;
  score?: number;
}

interface RecentActivityFeedProps {
  activities?: ActivityItem[];
  maxItems?: number;
}

export default function RecentActivityFeed({ activities, maxItems = 8 }: RecentActivityFeedProps) {
  const defaultActivities: ActivityItem[] = [
    {
      id: '1',
      type: 'speech',
      title: 'Speech Assessment Completed',
      description: 'Voice pattern analysis with 98.2% accuracy',
      timestamp: '2025-08-23T14:30:00Z',
      status: 'completed',
      user: 'Dr. Sarah Chen',
      score: 98.2,
    },
    {
      id: '2',
      type: 'retinal',
      title: 'Retinal Imaging Analysis',
      description: 'High-resolution fundus examination processed',
      timestamp: '2025-08-23T14:15:00Z',
      status: 'completed',
      user: 'Dr. Michael Rodriguez',
      score: 94.7,
    },
    {
      id: '3',
      type: 'motor',
      title: 'Motor Function Test',
      description: 'Hand kinematics and coordination assessment',
      timestamp: '2025-08-23T14:00:00Z',
      status: 'processing',
      user: 'Dr. Emily Watson',
    },
    {
      id: '4',
      type: 'cognitive',
      title: 'Cognitive Assessment',
      description: 'Memory and attention span evaluation',
      timestamp: '2025-08-23T13:45:00Z',
      status: 'completed',
      user: 'Dr. James Liu',
      score: 91.5,
    },
    {
      id: '5',
      type: 'nri',
      title: 'NRI Fusion Analysis',
      description: 'Multi-modal data integration completed',
      timestamp: '2025-08-23T13:30:00Z',
      status: 'completed',
      score: 96.8,
    },
    {
      id: '6',
      type: 'system',
      title: 'System Health Check',
      description: 'All services operational, 99.9% uptime',
      timestamp: '2025-08-23T13:00:00Z',
      status: 'completed',
    },
  ];

  const activityList = (activities || defaultActivities).slice(0, maxItems);

  const getActivityIcon = (type: string) => {
    const iconProps = { className: 'h-4 w-4' };
    switch (type) {
      case 'speech':
        return <Mic {...iconProps} />;
      case 'retinal':
        return <Eye {...iconProps} />;
      case 'motor':
        return <Hand {...iconProps} />;
      case 'cognitive':
        return <Brain {...iconProps} />;
      case 'nri':
        return <Zap {...iconProps} />;
      case 'multimodal':
        return <Activity {...iconProps} />;
      case 'system':
        return <Activity {...iconProps} />;
      default:
        return <Activity {...iconProps} />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className='h-4 w-4' style={{ color: '#34C759' }} />;
      case 'processing':
        return <Clock className='h-4 w-4' style={{ color: '#007AFF' }} />;
      case 'failed':
        return <AlertTriangle className='h-4 w-4' style={{ color: '#FF3B30' }} />;
      case 'pending':
        return <Clock className='h-4 w-4' style={{ color: '#86868B' }} />;
      default:
        return <Activity className='h-4 w-4' style={{ color: '#86868B' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#34C759';
      case 'processing':
        return '#007AFF';
      case 'failed':
        return '#FF3B30';
      case 'pending':
        return '#86868B';
      default:
        return '#86868B';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    // Use deterministic time for SSR compatibility
    const date = new Date(timestamp);
    const now = typeof window !== 'undefined' ? new Date() : new Date(1703000000000); // Fixed time for SSR
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className='rounded-2xl border border-zinc-200 bg-white/60 p-6 backdrop-blur-xl'
    >
      {/* Header */}
      <div className='mb-6 flex items-center justify-between'>
        <h3 className='text-lg font-semibold text-zinc-900'>
          Recent Activity
        </h3>
        <button className='text-sm font-medium text-blue-500 transition-opacity hover:opacity-70'>
          View All
        </button>
      </div>

      {/* Activity List */}
      <div className='max-h-96 space-y-4 overflow-y-auto'>
        {activityList.map((activity, index) => (
          <motion.div
            key={activity.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            className='flex items-start space-x-4 rounded-xl p-3 transition-colors hover:bg-white/50'
          >
            {/* Activity Icon */}
            <div
              className='flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full'
              style={{ backgroundColor: `${getStatusColor(activity.status)}20` }}
            >
              {React.cloneElement(getActivityIcon(activity.type) as React.ReactElement, {
                style: { color: getStatusColor(activity.status) },
              })}
            </div>

            {/* Activity Content */}
            <div className='min-w-0 flex-1'>
              <div className='flex items-start justify-between'>
                <div className='flex-1'>
                  <h4 className='truncate text-sm font-medium text-zinc-900'>
                    {activity.title}
                  </h4>
                  <p className='mt-1 line-clamp-2 text-xs text-zinc-500'>
                    {activity.description}
                  </p>
                  {activity.user && (
                    <div className='mt-2 flex items-center space-x-1'>
                      <User className='h-3 w-3 text-zinc-400' />
                      <span className='text-xs text-zinc-400'>
                        {activity.user}
                      </span>
                    </div>
                  )}
                </div>
                <div className='ml-2 flex flex-col items-end space-y-1'>
                  {getStatusIcon(activity.status)}
                  {activity.score && (
                    <span
                      className='text-xs font-medium'
                      style={{ color: getStatusColor(activity.status) }}
                    >
                      {activity.score}%
                    </span>
                  )}
                </div>
              </div>
              <div className='mt-2 text-xs text-zinc-400'>
                {formatTimestamp(activity.timestamp)}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
