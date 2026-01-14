'use client';

/**
 * DiagnosticGrid Component
 * 
 * Displays diagnostic module cards in a responsive grid layout.
 * Separates available and coming-soon modules into distinct sections.
 * Follows MediLens Design System patterns.
 * 
 * Requirements: 3.3, 6.1
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
    /** Optional filter to show only specific modules */
    modules?: DiagnosticModule[];
    /** Whether to show section headers */
    showSectionHeaders?: boolean;
    /** Optional additional CSS classes */
    className?: string;
}

/**
 * Section header component for module groups
 */
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
        <div className="mb-6 flex items-center gap-3">
            <h2
                className="text-xl font-semibold"
                style={{
                    color: '#000000',
                    fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", system-ui, sans-serif',
                }}
            >
                {title}
            </h2>
            <span
                className="inline-flex items-center rounded-full px-3 py-1 text-sm font-medium"
                style={{
                    backgroundColor: variant === 'available'
                        ? 'rgba(0, 122, 255, 0.1)'
                        : 'rgba(142, 142, 147, 0.1)',
                    color: variant === 'available' ? '#007AFF' : '#8E8E93',
                }}
            >
                {count} {count === 1 ? 'module' : 'modules'}
            </span>
        </div>
    );
}

/**
 * Container animation variants for staggered children
 */
const containerVariants = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: {
            staggerChildren: 0.1,
        },
    },
};

/**
 * DiagnosticGrid Component
 * 
 * Renders a responsive grid of diagnostic module cards.
 * By default, shows all modules separated into "Available Now" and "Coming Soon" sections.
 */
export function DiagnosticGrid({
    modules,
    showSectionHeaders = true,
    className = '',
}: DiagnosticGridProps) {
    // Use provided modules or default to all modules
    const allModules = modules || diagnosticModules;

    // Separate modules by availability
    const availableModules = modules
        ? modules.filter(m => m.status === 'available')
        : getAvailableModules();

    const comingSoonModules = modules
        ? modules.filter(m => m.status === 'coming-soon')
        : getComingSoonModules();

    // If custom modules provided without section headers, render flat grid
    if (modules && !showSectionHeaders) {
        return (
            <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="show"
                className={`grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 ${className}`}
                role="list"
                aria-label="Diagnostic modules"
            >
                {allModules.map((module, index) => (
                    <DiagnosticCard
                        key={module.id}
                        module={module}
                        animationDelay={index * 0.05}
                    />
                ))}
            </motion.div>
        );
    }

    return (
        <div className={`space-y-12 ${className}`}>
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
                        className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                        role="list"
                        aria-label="Available diagnostic modules"
                        id="available-modules-heading"
                    >
                        {availableModules.map((module, index) => (
                            <DiagnosticCard
                                key={module.id}
                                module={module}
                                animationDelay={index * 0.05}
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
                        className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                        role="list"
                        aria-label="Coming soon diagnostic modules"
                        id="coming-soon-modules-heading"
                    >
                        {comingSoonModules.map((module, index) => (
                            <DiagnosticCard
                                key={module.id}
                                module={module}
                                animationDelay={(availableModules.length + index) * 0.05}
                            />
                        ))}
                    </motion.div>
                </section>
            )}
        </div>
    );
}

export default DiagnosticGrid;
