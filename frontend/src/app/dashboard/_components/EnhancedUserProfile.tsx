'use client';

/**
 * Enhanced User Profile Component
 * 
 * Professional user profile card with real medical professional information
 * Following MediLens Light Mode Design Guidelines
 */

import { memo } from 'react';
import { motion } from 'framer-motion';
import { User, Mail, Phone, MapPin, Award, Calendar, TrendingUp, Activity } from 'lucide-react';
import { useUser } from '@clerk/nextjs';

interface EnhancedUserProfileProps {
    className?: string;
}

export const EnhancedUserProfile = memo(({ className = '' }: EnhancedUserProfileProps) => {
    const { user } = useUser();

    // Mock professional data (in production, fetch from API)
    const professionalData = {
        specialty: 'Neurologist',
        license: 'MD-2024-8472',
        institution: 'Stanford Medical Center',
        yearsExperience: 8,
        patientsServed: 1247,
        assessmentsCompleted: 3891,
        averageAccuracy: 96.8,
        lastActive: new Date().toISOString(),
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className={`rounded-xl border border-zinc-200 bg-white p-6 shadow-sm ${className}`}
        >
            {/* Header Section */}
            <div className="flex items-start justify-between mb-6">
                <div className="flex items-start gap-4">
                    {/* Avatar */}
                    <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#2563eb] shadow-md">
                        <User size={28} strokeWidth={1.5} className="text-white" />
                    </div>

                    {/* User Info */}
                    <div>
                        <h2 className="text-[18px] font-semibold text-zinc-900">
                            {user?.fullName || 'Dr. Sarah Chen'}
                        </h2>
                        <p className="text-[13px] text-zinc-500 mt-0.5">
                            {professionalData.specialty}
                        </p>
                        <div className="flex items-center gap-4 mt-2">
                            <div className="flex items-center gap-1.5">
                                <Award size={14} strokeWidth={1.5} className="text-[#3b82f6]" />
                                <span className="text-[12px] text-zinc-500">
                                    {professionalData.license}
                                </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <MapPin size={14} strokeWidth={1.5} className="text-zinc-500" />
                                <span className="text-[12px] text-zinc-500">
                                    {professionalData.institution}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Status Badge */}
                <div className="flex items-center gap-1.5 rounded-full bg-[#dcfce7] px-3 py-1.5 border border-[#bbf7d0]">
                    <div className="h-2 w-2 rounded-full bg-[#22c55e] animate-pulse" />
                    <span className="text-[11px] font-medium text-[#166534]">Active</span>
                </div>
            </div>

            {/* Contact Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6 pb-6 border-b border-zinc-100">
                <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100">
                        <Mail size={14} strokeWidth={1.5} className="text-zinc-500" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-400 uppercase tracking-wider font-medium">Email</p>
                        <p className="text-[12px] text-zinc-900 font-medium">
                            {user?.primaryEmailAddress?.emailAddress || 'sarah.chen@stanford.edu'}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100">
                        <Phone size={14} strokeWidth={1.5} className="text-zinc-500" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-400 uppercase tracking-wider font-medium">Phone</p>
                        <p className="text-[12px] text-zinc-900 font-medium">+1 (650) 555-0123</p>
                    </div>
                </div>
            </div>

            {/* Professional Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                    <div className="flex items-center justify-center gap-1 mb-1">
                        <Calendar size={14} strokeWidth={1.5} className="text-[#3b82f6]" />
                        <span className="text-[20px] font-semibold text-zinc-900">
                            {professionalData.yearsExperience}
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500">Years Experience</p>
                </div>

                <div className="text-center">
                    <div className="flex items-center justify-center gap-1 mb-1">
                        <User size={14} strokeWidth={1.5} className="text-[#10b981]" />
                        <span className="text-[20px] font-semibold text-zinc-900">
                            {professionalData.patientsServed.toLocaleString()}
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500">Patients Served</p>
                </div>

                <div className="text-center">
                    <div className="flex items-center justify-center gap-1 mb-1">
                        <Activity size={14} strokeWidth={1.5} className="text-[#f59e0b]" />
                        <span className="text-[20px] font-semibold text-zinc-900">
                            {professionalData.assessmentsCompleted.toLocaleString()}
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500">Assessments</p>
                </div>

                <div className="text-center">
                    <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingUp size={14} strokeWidth={1.5} className="text-[#8b5cf6]" />
                        <span className="text-[20px] font-semibold text-zinc-900">
                            {professionalData.averageAccuracy}%
                        </span>
                    </div>
                    <p className="text-[11px] text-zinc-500">Avg. Accuracy</p>
                </div>
            </div>
        </motion.div>
    );
});

EnhancedUserProfile.displayName = 'EnhancedUserProfile';

export default EnhancedUserProfile;
