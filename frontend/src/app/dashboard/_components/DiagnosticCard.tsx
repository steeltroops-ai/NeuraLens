"use client";

/**
 * DiagnosticCard Component
 *
 * Sleek, compact diagnostic module card with minimal design.
 * Features smooth hover animations and clean visual hierarchy.
 */

import { useRouter } from "next/navigation";
import { motion, type Variants } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { DiagnosticModule } from "@/data/diagnostic-modules";

export interface DiagnosticCardProps {
  module: DiagnosticModule;
  onClick?: () => void;
  className?: string;
  animationDelay?: number;
}

function StatusBadge({ status }: { status: "available" | "coming-soon" }) {
  if (status === "available") {
    return (
      <div className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 border border-emerald-200">
        <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
        <span className="text-[9px] font-semibold text-emerald-700 uppercase">
          Live
        </span>
      </div>
    );
  }

  return (
    <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase bg-zinc-100 text-zinc-500 border border-zinc-200">
      Soon
    </span>
  );
}

export function DiagnosticCard({
  module,
  onClick,
  className = "",
  animationDelay = 0,
}: DiagnosticCardProps) {
  const router = useRouter();
  const Icon = module.icon;
  const isAvailable = module.status === "available";

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
    if (event.key === "Enter" || event.key === " ") {
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
        ease: [0.22, 1, 0.36, 1],
      },
    },
  };

  return (
    <motion.div
      variants={cardVariants}
      initial="initial"
      animate="animate"
      whileHover={isAvailable ? { y: -2 } : undefined}
      className={`
                group relative flex items-center gap-4 p-4 rounded-xl
                border transition-all duration-200
                ${
                  isAvailable
                    ? "bg-white border-zinc-200 cursor-pointer hover:border-zinc-300 hover:shadow-md"
                    : "bg-zinc-50/50 border-zinc-100 opacity-60 cursor-not-allowed"
                }
                ${className}
            `}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={isAvailable ? 0 : -1}
      aria-disabled={!isAvailable}
      aria-label={`${module.name}: ${module.description}`}
      data-testid={`diagnostic-card-${module.id}`}
      data-status={module.status}
    >
      {/* Icon */}
      <div
        className={`flex-shrink-0 p-2.5 rounded-xl bg-gradient-to-br ${module.gradient} ${isAvailable ? "shadow-sm" : "opacity-50"}`}
      >
        <Icon size={18} strokeWidth={2} className="text-white" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <h3
            className={`text-[14px] font-semibold truncate ${isAvailable ? "text-zinc-900" : "text-zinc-500"}`}
          >
            {module.name}
          </h3>
          <StatusBadge status={module.status} />
        </div>
        <p
          className={`text-[12px] truncate ${isAvailable ? "text-zinc-500" : "text-zinc-400"}`}
        >
          {module.description}
        </p>
      </div>

      {/* Arrow */}
      {isAvailable && (
        <ArrowRight className="h-4 w-4 text-zinc-300 group-hover:text-zinc-500 group-hover:translate-x-1 transition-all duration-200 flex-shrink-0" />
      )}
    </motion.div>
  );
}

export default DiagnosticCard;
