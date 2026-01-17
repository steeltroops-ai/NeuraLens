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
          return '#22c55e';
        case 'stable':
          return '#3b82f6';
        case 'declining':
          return '#f59e0b';
        default:
          return '#64748b';
      }
    };

    const getTrendIcon = () => {
      switch (healthTrend) {
        case 'improving':
          return <TrendingUp size={14} strokeWidth={1.5} style={{ color: getTrendColor() }} />;
        case 'stable':
          return <Activity size={14} strokeWidth={1.5} style={{ color: getTrendColor() }} />;
        case 'declining':
          return <TrendingUp size={14} strokeWidth={1.5} className="rotate-180" style={{ color: getTrendColor() }} />;
        default:
          return <Activity size={14} strokeWidth={1.5} style={{ color: getTrendColor() }} />;
      }
    };

    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.15 }}
        className="rounded-lg border border-zinc-200 bg-white p-4"
      >
        <div className="flex items-start justify-between">
          {/* User Info Section */}
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500">
              <User size={18} strokeWidth={1.5} className="text-white" />
            </div>
            <div>
              <h3 className="text-[14px] font-medium text-zinc-900">
                {userName}
              </h3>
              <p className="text-[12px] text-zinc-500">
                Healthcare Professional
              </p>
              <div className="mt-1.5 flex items-center gap-1.5">
                <Calendar size={12} strokeWidth={1.5} className="text-zinc-400" />
                <span className="text-[11px] text-zinc-500">
                  Last assessment: {new Date(lastAssessmentDate).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>

          {/* Health Metrics Section */}
          <div className="text-right">
            <div className="text-[24px] font-semibold text-blue-500">
              {lastAssessmentScore}%
            </div>
            <div className="text-[11px] text-zinc-500">
              Latest Score
            </div>
            <div className="mt-1.5 flex items-center justify-end gap-1">
              {getTrendIcon()}
              <span className="text-[11px] font-medium capitalize" style={{ color: getTrendColor() }}>
                {healthTrend}
              </span>
            </div>
            <div className="mt-0.5 text-[10px] text-zinc-400">
              {totalAssessments} total assessments
            </div>
          </div>
        </div>

        {/* Quick Stats Row */}
        <div className="mt-4 pt-4 border-t border-zinc-100 grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-[16px] font-semibold text-zinc-900">
              98.2%
            </div>
            <div className="text-[10px] text-zinc-500">
              Speech Accuracy
            </div>
          </div>
          <div className="text-center">
            <div className="text-[16px] font-semibold text-zinc-900">
              92.7%
            </div>
            <div className="text-[10px] text-zinc-500">
              Motor Function
            </div>
          </div>
          <div className="text-center">
            <div className="text-[16px] font-semibold text-zinc-900">
              96.1%
            </div>
            <div className="text-[10px] text-zinc-500">
              Cognitive Score
            </div>
          </div>
        </div>
      </motion.div>
    );
  },
);

UserHealthOverview.displayName = 'UserHealthOverview';

export default UserHealthOverview;
