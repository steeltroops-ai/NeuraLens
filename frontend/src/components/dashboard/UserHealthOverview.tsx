'use client';

import React, { memo } from 'react';
import { motion } from 'framer-motion';
import { User, TrendingUp, Calendar, Activity } from 'lucide-react';

interface UserHealthOverviewProps {
  userName?: string;
  lastAssessmentScore?: number;
  lastAssessmentDate?: string;
  healthTrend?: 'improving' | 'stable' | 'declining';
  totalAssessments?: number;
}

const UserHealthOverview = memo(
  ({
    userName = 'Dr. Sarah Chen',
    lastAssessmentScore = 94.2,
    lastAssessmentDate = '2025-08-23',
    healthTrend = 'improving',
    totalAssessments = 127,
  }: UserHealthOverviewProps) => {
    const getTrendColor = () => {
      switch (healthTrend) {
        case 'improving':
          return '#34C759';
        case 'stable':
          return '#007AFF';
        case 'declining':
          return '#FF9500';
        default:
          return '#86868B';
      }
    };

    const getTrendIcon = () => {
      switch (healthTrend) {
        case 'improving':
          return <TrendingUp className='h-4 w-4' style={{ color: getTrendColor() }} />;
        case 'stable':
          return <Activity className='h-4 w-4' style={{ color: getTrendColor() }} />;
        case 'declining':
          return <TrendingUp className='h-4 w-4 rotate-180' style={{ color: getTrendColor() }} />;
        default:
          return <Activity className='h-4 w-4' style={{ color: getTrendColor() }} />;
      }
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
        <div className='flex items-start justify-between'>
          {/* User Info Section */}
          <div className='flex items-start space-x-4'>
            <div
              className='flex h-12 w-12 items-center justify-center rounded-full'
              style={{ backgroundColor: '#007AFF' }}
            >
              <User className='h-6 w-6' style={{ color: '#FFFFFF' }} />
            </div>
            <div>
              <h3
                className='text-xl font-semibold'
                style={{
                  color: '#1D1D1F',
                  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                }}
              >
                {userName}
              </h3>
              <p className='text-sm' style={{ color: '#86868B' }}>
                Healthcare Professional
              </p>
              <div className='mt-2 flex items-center space-x-2'>
                <Calendar className='h-4 w-4' style={{ color: '#86868B' }} />
                <span className='text-sm' style={{ color: '#86868B' }}>
                  Last assessment: {new Date(lastAssessmentDate).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>

          {/* Health Metrics Section */}
          <div className='text-right'>
            <div className='text-3xl font-bold' style={{ color: '#007AFF' }}>
              {lastAssessmentScore}%
            </div>
            <div className='text-sm font-medium' style={{ color: '#86868B' }}>
              Latest Score
            </div>
            <div className='mt-2 flex items-center justify-end space-x-2'>
              {getTrendIcon()}
              <span className='text-sm font-medium capitalize' style={{ color: getTrendColor() }}>
                {healthTrend}
              </span>
            </div>
            <div className='mt-1 text-xs' style={{ color: '#86868B' }}>
              {totalAssessments} total assessments
            </div>
          </div>
        </div>

        {/* Quick Stats Row */}
        <div className='mt-6 grid grid-cols-3 gap-4'>
          <div className='text-center'>
            <div className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
              98.2%
            </div>
            <div className='text-xs' style={{ color: '#86868B' }}>
              Speech Accuracy
            </div>
          </div>
          <div className='text-center'>
            <div className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
              92.7%
            </div>
            <div className='text-xs' style={{ color: '#86868B' }}>
              Motor Function
            </div>
          </div>
          <div className='text-center'>
            <div className='text-lg font-semibold' style={{ color: '#1D1D1F' }}>
              96.1%
            </div>
            <div className='text-xs' style={{ color: '#86868B' }}>
              Cognitive Score
            </div>
          </div>
        </div>
      </motion.div>
    );
  },
);

export default UserHealthOverview;
