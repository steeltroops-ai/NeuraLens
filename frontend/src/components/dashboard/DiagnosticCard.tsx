'use client';

/**
 * DiagnosticCard Component
 * 
 * Displays a diagnostic module as an interactive card on the dashboard.
 * Industry-grade minimal design with clean borders and subtle interactions.
 */

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
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
            <span
                className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium"
                style={{ backgroundColor: '#dcfce7', color: '#166534' }}
                aria-label="Available"
            >
                Available
            </span>
        );
    }

    return (
        <span
            className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-medium"
            style={{ backgroundColor: '#f1f5f9', color: '#64748b' }}
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

    const cardVariants = {
        initial: { opacity: 0, y: 8 },
        animate: {
            opacity: 1,
            y: 0,
            transition: {
                duration: 0.2,
                delay: animationDelay,
                ease: 'easeOut',
            },
        },
    };

    return (
        <motion.div
            variants={cardVariants}
            initial="initial"
            animate="animate"
            className={`
                relative flex flex-col rounded-lg p-4
                border transition-all duration-150
                min-h-[140px]
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3b82f6]/50
                ${isAvailable
                    ? 'bg-white border-[#e2e8f0] cursor-pointer hover:border-[#cbd5e1] hover:shadow-sm'
                    : 'bg-[#f8fafc] border-[#f0f0f0] opacity-60 cursor-not-allowed'
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
            {/* Icon */}
            <div
                className={`mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br ${module.gradient}`}
                aria-hidden="true"
            >
                <Icon size={18} strokeWidth={1.5} className="text-white" />
            </div>

            {/* Module name */}
            <h3
                className="mb-1 text-[14px] font-medium"
                style={{ color: isAvailable ? '#0f172a' : '#64748b' }}
            >
                {module.name}
            </h3>

            {/* Module description */}
            <p
                className="mb-3 flex-1 text-[12px] leading-relaxed"
                style={{ color: isAvailable ? '#64748b' : '#94a3b8' }}
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
