'use client';

/**
 * DiagnosticGrid Component
 * 
 * Displays diagnostic module cards in a responsive grid layout.
 * Industry-grade minimal design.
 */

import { motion } from 'framer-motion';
import { DiagnosticCard } from './DiagnosticCard';
import {
    diagnosticModules,
    getAvailableModules,
    getComingSoonModules,
    DiagnosticModule
} from '@/data/diagnostic-modules';

export interface DiagnosticGridProps {
    modules?: DiagnosticModule[];
    showSectionHeaders?: boolean;
    className?: string;
}

function SectionHeader({
    title,
    count,
    variant = 'available'
}: {
    title: string;
    count: number;
    variant?: 'available' | 'coming-soon';
}) {
    return (
        <div className="mb-4 flex items-center gap-2">
            <h2 className="text-[14px] font-medium text-zinc-900">
                {title}
            </h2>
            <span
                className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium"
                style={{
                    backgroundColor: variant === 'available' ? '#dbeafe' : '#f4f4f5',
                    color: variant === 'available' ? '#1d4ed8' : '#71717a',
                }}
            >
                {count}
            </span>
        </div>
    );
}

const containerVariants = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: {
            staggerChildren: 0.05,
        },
    },
};

export function DiagnosticGrid({
    modules,
    showSectionHeaders = true,
    className = '',
}: DiagnosticGridProps) {
    const allModules = modules || diagnosticModules;

    const availableModules = modules
        ? modules.filter(m => m.status === 'available')
        : getAvailableModules();

    const comingSoonModules = modules
        ? modules.filter(m => m.status === 'coming-soon')
        : getComingSoonModules();

    if (modules && !showSectionHeaders) {
        return (
            <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className={`grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 ${className}`}
                role="list"
                aria-label="Diagnostic modules"
            >
                {allModules.map((module, index) => (
                    <DiagnosticCard
                        key={module.id}
                        module={module}
                        animationDelay={index * 0.03}
                    />
                ))}
            </motion.div>
        );
    }

    return (
        <div className={`space-y-8 ${className}`}>
            {/* Available Modules Section */}
            {availableModules.length > 0 && (
                <section aria-labelledby="available-modules-heading">
                    {showSectionHeaders && (
                        <SectionHeader
                            title="Available Now"
                            count={availableModules.length}
                            variant="available"
                        />
                    )}
                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        animate="show"
                        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                        role="list"
                        aria-label="Available diagnostic modules"
                        id="available-modules-heading"
                    >
                        {availableModules.map((module, index) => (
                            <DiagnosticCard
                                key={module.id}
                                module={module}
                                animationDelay={index * 0.03}
                            />
                        ))}
                    </motion.div>
                </section>
            )}

            {/* Coming Soon Modules Section */}
            {comingSoonModules.length > 0 && (
                <section aria-labelledby="coming-soon-modules-heading">
                    {showSectionHeaders && (
                        <SectionHeader
                            title="Coming Soon"
                            count={comingSoonModules.length}
                            variant="coming-soon"
                        />
                    )}
                    <motion.div
                        variants={containerVariants}
                        initial="hidden"
                        animate="show"
                        className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                        role="list"
                        aria-label="Coming soon diagnostic modules"
                        id="coming-soon-modules-heading"
                    >
                        {comingSoonModules.map((module, index) => (
                            <DiagnosticCard
                                key={module.id}
                                module={module}
                                animationDelay={(availableModules.length + index) * 0.03}
                            />
                        ))}
                    </motion.div>
                </section>
            )}
        </div>
    );
}

export default DiagnosticGrid;
