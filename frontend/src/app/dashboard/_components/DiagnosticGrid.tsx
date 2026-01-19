"use client";

/**
 * DiagnosticGrid Component
 *
 * Displays diagnostic module cards in a sleek, compact layout.
 * Responsive design with row-based cards.
 */

import { motion } from "framer-motion";
import { DiagnosticCard } from "./DiagnosticCard";
import {
  diagnosticModules,
  getAvailableModules,
  getComingSoonModules,
  DiagnosticModule,
} from "@/data/diagnostic-modules";

export interface DiagnosticGridProps {
  modules?: DiagnosticModule[];
  showSectionHeaders?: boolean;
  className?: string;
}

function SectionHeader({
  title,
  count,
  variant = "available",
}: {
  title: string;
  count: number;
  variant?: "available" | "coming-soon";
}) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <h3 className="text-[13px] font-semibold text-zinc-900">{title}</h3>
      <span
        className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${
          variant === "available"
            ? "bg-blue-50 text-blue-600 border border-blue-100"
            : "bg-zinc-100 text-zinc-500 border border-zinc-200"
        }`}
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
      staggerChildren: 0.03,
    },
  },
};

export function DiagnosticGrid({
  modules,
  showSectionHeaders = true,
  className = "",
}: DiagnosticGridProps) {
  const allModules = modules || diagnosticModules;

  const availableModules = modules
    ? modules.filter((m) => m.status === "available")
    : getAvailableModules();

  const comingSoonModules = modules
    ? modules.filter((m) => m.status === "coming-soon")
    : getComingSoonModules();

  if (modules && !showSectionHeaders) {
    return (
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className={`grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 ${className}`}
        role="list"
        aria-label="Diagnostic modules"
      >
        {allModules.map((module, index) => (
          <DiagnosticCard
            key={module.id}
            module={module}
            animationDelay={index * 0.02}
          />
        ))}
      </motion.div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
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
            className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3"
            role="list"
            aria-label="Available diagnostic modules"
            id="available-modules-heading"
          >
            {availableModules.map((module, index) => (
              <DiagnosticCard
                key={module.id}
                module={module}
                animationDelay={index * 0.02}
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
            className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3"
            role="list"
            aria-label="Coming soon diagnostic modules"
            id="coming-soon-modules-heading"
          >
            {comingSoonModules.map((module, index) => (
              <DiagnosticCard
                key={module.id}
                module={module}
                animationDelay={(availableModules.length + index) * 0.02}
              />
            ))}
          </motion.div>
        </section>
      )}
    </div>
  );
}

export default DiagnosticGrid;
