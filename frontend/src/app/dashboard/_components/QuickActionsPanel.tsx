'use client';

/**
 * Quick Actions Panel Component
 * 
 * Provides quick access to common diagnostic actions
 * Following MediLens Light Mode Design Guidelines
 */

import { memo } from 'react';
import { motion } from 'framer-motion';
import { Zap, Eye, Mic, Brain, Hand, Activity, FileText, BarChart3 } from 'lucide-react';
import Link from 'next/link';

interface QuickAction {
    id: string;
    label: string;
    description: string;
    icon: React.ReactNode;
    route: string;
    color: {
        bg: string;
        text: string;
        border: string;
    };
}

const quickActions: QuickAction[] = [
    {
        id: 'retinal',
        label: 'Retinal Scan',
        description: 'Analyze retinal images',
        icon: <Eye size={20} strokeWidth={1.5} />,
        route: '/dashboard/retinal',
        color: { bg: '#ecfdf5', text: '#059669', border: '#d1fae5' },
    },
    {
        id: 'speech',
        label: 'Speech Analysis',
        description: 'Assess speech patterns',
        icon: <Mic size={20} strokeWidth={1.5} />,
        route: '/dashboard/speech',
        color: { bg: '#dbeafe', text: '#2563eb', border: '#bfdbfe' },
    },
    {
        id: 'cognitive',
        label: 'Cognitive Test',
        description: 'Evaluate cognitive function',
        icon: <Brain size={20} strokeWidth={1.5} />,
        route: '/dashboard/cognitive',
        color: { bg: '#fef3c7', text: '#d97706', border: '#fde68a' },
    },
    {
        id: 'motor',
        label: 'Motor Assessment',
        description: 'Test motor skills',
        icon: <Hand size={20} strokeWidth={1.5} />,
        route: '/dashboard/motor',
        color: { bg: '#f3e8ff', text: '#9333ea', border: '#e9d5ff' },
    },
    {
        id: 'nri',
        label: 'NRI Fusion',
        description: 'Multi-modal analysis',
        icon: <Activity size={20} strokeWidth={1.5} />,
        route: '/dashboard/nri-fusion',
        color: { bg: '#fce7f3', text: '#db2777', border: '#fbcfe8' },
    },
    {
        id: 'reports',
        label: 'View Reports',
        description: 'Access patient reports',
        icon: <FileText size={20} strokeWidth={1.5} />,
        route: '/dashboard/reports',
        color: { bg: '#fef2f2', text: '#dc2626', border: '#fecaca' },
    },
    {
        id: 'analytics',
        label: 'Analytics',
        description: 'View insights & trends',
        icon: <BarChart3 size={20} strokeWidth={1.5} />,
        route: '/dashboard/analytics',
        color: { bg: '#ede9fe', text: '#7c3aed', border: '#ddd6fe' },
    },
    {
        id: 'multimodal',
        label: 'Multi-Modal',
        description: 'Combined diagnostics',
        icon: <Zap size={20} strokeWidth={1.5} />,
        route: '/dashboard/multimodal',
        color: { bg: '#fef9c3', text: '#ca8a04', border: '#fef08a' },
    },
];

export const QuickActionsPanel = memo(() => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: 0.15 }}
            className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm"
        >
            {/* Header */}
            <div className="flex items-center gap-2 mb-5">
                <Zap size={18} strokeWidth={1.5} className="text-[#3b82f6]" />
                <h3 className="text-[16px] font-semibold text-zinc-900">Quick Actions</h3>
            </div>

            {/* Actions Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {quickActions.map((action, index) => (
                    <motion.div
                        key={action.id}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.15, delay: index * 0.03 }}
                    >
                        <Link
                            href={action.route}
                            className="flex flex-col items-center gap-2 p-4 rounded-lg border border-zinc-100 hover:border-zinc-200 hover:shadow-sm transition-all duration-150 group"
                        >
                            {/* Icon */}
                            <div
                                className="flex h-12 w-12 items-center justify-center rounded-xl transition-transform duration-150 group-hover:scale-110"
                                style={{
                                    backgroundColor: action.color.bg,
                                    borderWidth: '1px',
                                    borderStyle: 'solid',
                                    borderColor: action.color.border,
                                }}
                            >
                                <div style={{ color: action.color.text }}>
                                    {action.icon}
                                </div>
                            </div>

                            {/* Label */}
                            <div className="text-center">
                                <p className="text-[12px] font-medium text-zinc-900 mb-0.5">
                                    {action.label}
                                </p>
                                <p className="text-[10px] text-zinc-500 line-clamp-1">
                                    {action.description}
                                </p>
                            </div>
                        </Link>
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
});

QuickActionsPanel.displayName = 'QuickActionsPanel';

export default QuickActionsPanel;
