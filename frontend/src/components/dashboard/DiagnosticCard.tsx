'use client';

/**
 * DiagnosticCard Component
 * 
 * Displays a diagnostic module as an interactive card on the dashboard.
 * Follows MediLens Design System patterns with:
 * - Standard card styling with hover lift animation
 * - Module icon with gradient background
 * - Status badge (Available/Coming Soon)
 * - Disabled state for coming-soon modules
 * - 48px minimum touch target
 * - Framer Motion fade-in-up animation
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.2
 */

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { DiagnosticModule } from '@/data/diagnostic-modules';

export interface DiagnosticCardProps {
    /** The diagnostic module to display */
    module: DiagnosticModule;
    /** Optional click handler (overrides default navigation) */
    onClick?: () => void;
    /** Optional additional CSS classes */
    className?: string;
    /** Animation delay for staggered animations */
    animationDelay?: number;
}

/**
 * Status badge component for displaying module availability
 * Uses Caption (12px) typography for labels
 */
function StatusBadge({ status }: { status: 'available' | 'coming-soon' }) {
    if (status === 'available') {
        return (
            <span
                className="inline-flex items-center rounded-full px-3 py-1 text-caption font-medium"
                style={{
                    backgroundColor: 'rgba(52, 199, 89, 0.1)',
                    color: '#34C759',
                }}
                aria-label="Available"
            >
                Available
            </span>
        );
    }

    return (
        <span
            className="inline-flex items-center rounded-full px-3 py-1 text-caption font-medium"
            style={{
                backgroundColor: 'rgba(142, 142, 147, 0.1)',
                color: '#8E8E93',
            }}
            aria-label="Coming Soon"
        >
            Coming Soon
        </span>
    );
}

/**
 * DiagnosticCard Component
 * 
 * Renders a diagnostic module card with icon, name, description, and status.
 * Available modules are clickable and navigate to their assessment page.
 * Coming-soon modules are disabled with reduced opacity.
 */
export function DiagnosticCard({
    module,
    onClick,
    className = '',
    animationDelay = 0,
}: DiagnosticCardProps) {
    const router = useRouter();
    const Icon = module.icon;
    const isAvailable = module.status === 'available';

    // Handle card click - navigate to module page if available
    const handleClick = () => {
        if (!isAvailable) return;

        if (onClick) {
            onClick();
        } else {
            router.push(module.route);
        }
    };

    // Handle keyboard navigation
    const handleKeyDown = (event: React.KeyboardEvent) => {
        if (!isAvailable) return;

        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            handleClick();
        }
    };

    // Animation variants following MediLens Design System
    const cardVariants = {
        initial: { opacity: 0, y: 20 },
        animate: {
            opacity: 1,
            y: 0,
            transition: {
                duration: 0.5,
                delay: animationDelay,
                ease: [0.22, 1, 0.36, 1] as const, // ease-out-quint
            },
        },
    };

    // Base card styles following MediLens Design System
    const baseCardStyles = `
        relative flex flex-col rounded-2xl p-4 sm:p-6
        border transition-all duration-300
        min-h-[180px]
        focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[#007AFF]/40
    `;

    // Available card styles with hover effects
    const availableCardStyles = `
        bg-white shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),0_2px_4px_-1px_rgba(0,0,0,0.06)]
        hover:shadow-[0_10px_15px_-3px_rgba(0,0,0,0.1),0_4px_6px_-2px_rgba(0,0,0,0.05)]
        border-black/5
        cursor-pointer
        hover:-translate-y-1
    `;

    // Disabled card styles for coming-soon modules
    const disabledCardStyles = `
        bg-[#F2F2F7] border-[#E5E5EA]
        opacity-60 cursor-not-allowed
    `;

    return (
        <motion.div
            variants={cardVariants}
            initial="initial"
            animate="animate"
            className={`
                ${baseCardStyles}
                ${isAvailable ? availableCardStyles : disabledCardStyles}
                ${className}
            `}
            onClick={handleClick}
            onKeyDown={handleKeyDown}
            role="button"
            tabIndex={isAvailable ? 0 : -1}
            aria-disabled={!isAvailable}
            aria-label={`${module.name}: ${module.description}. ${isAvailable ? 'Click to start assessment' : 'Coming soon'}`}
            data-testid={`diagnostic-card-${module.id}`}
            data-status={module.status}
            style={{ minHeight: '48px' }} // Ensure minimum touch target
        >
            {/* Icon with gradient background */}
            <div
                className={`
                    mb-4 flex h-12 w-12 items-center justify-center rounded-xl
                    bg-gradient-to-br ${module.gradient}
                `}
                aria-hidden="true"
            >
                <Icon
                    className="h-6 w-6 text-white"
                    strokeWidth={2}
                />
            </div>

            {/* Module name - Title 3 (22px) for card titles */}
            <h3
                className="mb-2 text-title3"
                style={{
                    color: isAvailable ? '#000000' : '#8E8E93',
                }}
            >
                {module.name}
            </h3>

            {/* Module description - Subhead (15px) for supporting text */}
            <p
                className="mb-4 flex-1 text-subhead leading-relaxed"
                style={{
                    color: isAvailable ? '#3C3C43' : '#8E8E93',
                }}
            >
                {module.description}
            </p>

            {/* Status badge */}
            <div className="mt-auto">
                <StatusBadge status={module.status} />
            </div>
        </motion.div>
    );
}

export default DiagnosticCard;
