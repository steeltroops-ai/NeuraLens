'use client';

/**
 * DiagnosticCard Component
 * 
 * Displays a diagnostic module as an interactive card on the dashboard.
 * Industry-grade minimal design with clean borders and subtle interactions.
 */

import { useRouter } from 'next/navigation';
import { motion, type Variants } from 'framer-motion';
import { DiagnosticModule } from '@/data/diagnostic-modules';

export interface DiagnosticCardProps {
    module: DiagnosticModule;
    onClick?: () => void;
    className?: string;
    animationDelay?: number;
}

function StatusBadge({ status }: { status: 'available' | 'coming-soon' }) {
    if (status === 'available') {
        return (
            <div className="flex items-center gap-1.5 rounded-full bg-green-50 px-3 py-1.5 border border-green-300">
                <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                <span className="text-[11px] font-medium text-green-700" aria-label="Available">
                    Live
                </span>
            </div>
        );
    }

    return (
        <span
            className="inline-flex items-center rounded-full px-3 py-1.5 text-[11px] font-medium bg-zinc-100 text-zinc-600 border border-zinc-300"
            aria-label="Coming Soon"
        >
            Coming Soon
        </span>
    );
}

export function DiagnosticCard({
    module,
    onClick,
    className = '',
    animationDelay = 0,
}: DiagnosticCardProps) {
    const router = useRouter();
    const Icon = module.icon;
    const isAvailable = module.status === 'available';

    const handleClick = () => {
        if (!isAvailable) return;
        if (onClick) {
            onClick();
        } else {
            router.push(module.route);
        }
    };

    const handleKeyDown = (event: React.KeyboardEvent) => {
        if (!isAvailable) return;
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            handleClick();
        }
    };

    const cardVariants: Variants = {
        initial: { opacity: 0, y: 8 },
        animate: {
            opacity: 1,
            y: 0,
            transition: {
                duration: 0.2,
                delay: animationDelay,
                ease: 'easeOut' as const,
            },
        },
    };

    return (
        <motion.div
            variants={cardVariants}
            initial="initial"
            animate="animate"
            className={`
                relative flex flex-col rounded-xl p-5
                border transition-all duration-200
                min-h-[160px]
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400
                ${isAvailable
                    ? 'bg-white border-gray-300 shadow-md cursor-pointer hover:border-gray-400 hover:shadow-lg hover:-translate-y-0.5'
                    : 'bg-gray-50 border-gray-200 opacity-70 cursor-not-allowed'
                }
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
        >
            {/* Top accent line */}
            {isAvailable && (
                <div
                    className={`absolute top-0 left-0 right-0 h-1 rounded-t-xl bg-gradient-to-r ${module.gradient}`}
                    aria-hidden="true"
                />
            )}

            {/* Icon */}
            <div
                className={`mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${module.gradient} shadow-sm`}
                aria-hidden="true"
            >
                <Icon size={20} strokeWidth={1.5} className="text-white" />
            </div>

            {/* Module name */}
            <h3 className={`mb-2 text-[15px] font-semibold ${isAvailable ? 'text-gray-900' : 'text-gray-500'}`}>
                {module.name}
            </h3>

            {/* Module description */}
            <p className={`mb-4 flex-1 text-[13px] leading-relaxed ${isAvailable ? 'text-gray-600' : 'text-gray-400'}`}>
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
