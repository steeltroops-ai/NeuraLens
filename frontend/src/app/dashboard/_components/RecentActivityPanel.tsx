'use client';

/**
 * Recent Activity Panel Component
 * 
 * Displays recent diagnostic activities with timeline view
 * Following MediLens Light Mode Design Guidelines
 */

import { memo } from 'react';
import { motion } from 'framer-motion';
import { Clock, Eye, Mic, Brain, Hand, Activity, ChevronRight, CheckCircle, AlertCircle } from 'lucide-react';
import Link from 'next/link';

interface ActivityItem {
    id: string;
    type: 'retinal' | 'speech' | 'cognitive' | 'motor' | 'nri';
    title: string;
    patient: string;
    timestamp: string;
    status: 'completed' | 'in-progress' | 'flagged';
    score?: number;
    route: string;
}

const activityData: ActivityItem[] = [
    {
        id: '1',
        type: 'retinal',
        title: 'Retinal Scan Analysis',
        patient: 'Patient #4782',
        timestamp: '2 hours ago',
        status: 'completed',
        score: 94.2,
        route: '/dashboard/retinal',
    },
    {
        id: '2',
        type: 'speech',
        title: 'Speech Assessment',
        patient: 'Patient #4781',
        timestamp: '4 hours ago',
        status: 'completed',
        score: 91.8,
        route: '/dashboard/speech',
    },
    {
        id: '3',
        type: 'cognitive',
        title: 'Cognitive Testing',
        patient: 'Patient #4780',
        timestamp: '6 hours ago',
        status: 'flagged',
        score: 72.3,
        route: '/dashboard/cognitive',
    },
    {
        id: '4',
        type: 'motor',
        title: 'Motor Assessment',
        patient: 'Patient #4779',
        timestamp: '1 day ago',
        status: 'completed',
        score: 88.5,
        route: '/dashboard/motor',
    },
    {
        id: '5',
        type: 'nri',
        title: 'NRI Fusion Analysis',
        patient: 'Patient #4778',
        timestamp: '1 day ago',
        status: 'completed',
        score: 96.1,
        route: '/dashboard/nri-fusion',
    },
];

const getModuleIcon = (type: string) => {
    switch (type) {
        case 'retinal':
            return <Eye size={16} strokeWidth={1.5} />;
        case 'speech':
            return <Mic size={16} strokeWidth={1.5} />;
        case 'cognitive':
            return <Brain size={16} strokeWidth={1.5} />;
        case 'motor':
            return <Hand size={16} strokeWidth={1.5} />;
        case 'nri':
            return <Activity size={16} strokeWidth={1.5} />;
        default:
            return <Activity size={16} strokeWidth={1.5} />;
    }
};

const getModuleColor = (type: string) => {
    switch (type) {
        case 'retinal':
            return { bg: '#ecfdf5', text: '#059669', border: '#d1fae5' };
        case 'speech':
            return { bg: '#dbeafe', text: '#2563eb', border: '#bfdbfe' };
        case 'cognitive':
            return { bg: '#fef3c7', text: '#d97706', border: '#fde68a' };
        case 'motor':
            return { bg: '#f3e8ff', text: '#9333ea', border: '#e9d5ff' };
        case 'nri':
            return { bg: '#fce7f3', text: '#db2777', border: '#fbcfe8' };
        default:
            return { bg: '#f4f4f5', text: '#71717a', border: '#e4e4e7' };
    }
};

const getStatusBadge = (status: string, score?: number) => {
    if (status === 'flagged') {
        return (
            <div className="flex items-center gap-1 rounded-full bg-[#fef2f2] px-2 py-0.5 border border-[#fecaca]">
                <AlertCircle size={12} strokeWidth={1.5} className="text-[#dc2626]" />
                <span className="text-[10px] font-medium text-[#dc2626]">Flagged</span>
            </div>
        );
    }

    if (status === 'completed' && score) {
        return (
            <div className="flex items-center gap-1 rounded-full bg-[#dcfce7] px-2 py-0.5 border border-[#bbf7d0]">
                <CheckCircle size={12} strokeWidth={1.5} className="text-[#16a34a]" />
                <span className="text-[10px] font-medium text-[#166534]">{score}%</span>
            </div>
        );
    }

    return null;
};

export const RecentActivityPanel = memo(() => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.1 }}
            className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm"
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                    <Clock size={18} strokeWidth={1.5} className="text-[#3b82f6]" />
                    <h3 className="text-[16px] font-semibold text-zinc-900">Recent Activity</h3>
                </div>
                <Link
                    href="/dashboard/analytics"
                    className="text-[12px] font-medium text-[#3b82f6] hover:text-[#2563eb] transition-colors flex items-center gap-1"
                >
                    View All
                    <ChevronRight size={14} strokeWidth={1.5} />
                </Link>
            </div>

            {/* Activity List */}
            <div className="space-y-3">
                {activityData.map((activity, index) => {
                    const colors = getModuleColor(activity.type);

                    return (
                        <motion.div
                            key={activity.id}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.15, delay: index * 0.05 }}
                        >
                            <Link
                                href={activity.route}
                                className="flex items-center gap-3 p-3 rounded-lg border border-zinc-100 hover:border-zinc-200 hover:bg-zinc-50 transition-all duration-150 group"
                            >
                                {/* Icon */}
                                <div
                                    className="flex h-10 w-10 items-center justify-center rounded-lg flex-shrink-0"
                                    style={{ backgroundColor: colors.bg, borderColor: colors.border }}
                                >
                                    <div style={{ color: colors.text }}>
                                        {getModuleIcon(activity.type)}
                                    </div>
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between gap-2 mb-0.5">
                                        <h4 className="text-[13px] font-medium text-zinc-900 truncate">
                                            {activity.title}
                                        </h4>
                                        {getStatusBadge(activity.status, activity.score)}
                                    </div>
                                    <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                                        <span>{activity.patient}</span>
                                        <span>•</span>
                                        <span>{activity.timestamp}</span>
                                    </div>
                                </div>

                                {/* Arrow */}
                                <ChevronRight
                                    size={16}
                                    strokeWidth={1.5}
                                    className="text-zinc-300 group-hover:text-zinc-400 transition-colors flex-shrink-0"
                                />
                            </Link>
                        </motion.div>
                    );
                })}
            </div>

            {/* Empty State (if no activities) */}
            {activityData.length === 0 && (
                <div className="py-12 text-center">
                    <Clock size={40} strokeWidth={1.5} className="mx-auto text-zinc-300 mb-3" />
                    <p className="text-[13px] text-zinc-500">No recent activity</p>
                    <p className="text-[12px] text-zinc-400 mt-1">
                        Start a diagnostic assessment to see activity here
                    </p>
                </div>
            )}
        </motion.div>
    );
});

RecentActivityPanel.displayName = 'RecentActivityPanel';

export default RecentActivityPanel;
