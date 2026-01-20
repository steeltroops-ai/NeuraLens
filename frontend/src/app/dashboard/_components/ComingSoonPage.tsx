"use client";

import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

interface ComingSoonPageProps {
  title: string;
  subtitle?: string;
  accentColor?:
    | "cyan"
    | "violet"
    | "emerald"
    | "amber"
    | "rose"
    | "blue"
    | "red";
}

export default function ComingSoonPage({
  title,
  subtitle,
  accentColor = "cyan",
}: ComingSoonPageProps) {
  const [dots, setDots] = useState("");

  // Animated dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
    }, 400);
    return () => clearInterval(interval);
  }, []);

  // Color mapping
  const colors: Record<string, { gradient: string; glow: string }> = {
    cyan: { gradient: "from-cyan-400 to-cyan-600", glow: "bg-cyan-500/[0.05]" },
    violet: {
      gradient: "from-violet-400 to-violet-600",
      glow: "bg-violet-500/[0.05]",
    },
    emerald: {
      gradient: "from-emerald-400 to-emerald-600",
      glow: "bg-emerald-500/[0.05]",
    },
    amber: {
      gradient: "from-amber-400 to-amber-600",
      glow: "bg-amber-500/[0.05]",
    },
    rose: { gradient: "from-rose-400 to-rose-600", glow: "bg-rose-500/[0.05]" },
    blue: { gradient: "from-blue-400 to-blue-600", glow: "bg-blue-500/[0.05]" },
    red: { gradient: "from-red-400 to-red-600", glow: "bg-red-500/[0.05]" },
  };

  const defaultColor = {
    gradient: "from-cyan-400 to-cyan-600",
    glow: "bg-cyan-500/[0.05]",
  };
  const colorScheme = colors[accentColor] ?? defaultColor;

  return (
    <div className="relative min-h-[85vh] flex items-center justify-center overflow-hidden bg-black selection:bg-white/20">
      {/* Subtle spotlight glow */}
      <div
        className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] ${colorScheme.glow} blur-[120px] rounded-full pointer-events-none`}
      />

      {/* Very faint grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.015)_1px,transparent_1px)] bg-[size:48px_48px]" />

      {/* Content */}
      <div className="relative z-10 text-center px-6 max-w-2xl mx-auto">
        {/* Animated badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="inline-flex items-center gap-2.5 rounded-full bg-zinc-900/80 px-4 py-1.5 text-[11px] font-medium text-zinc-400 mb-8"
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-500 opacity-40"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-500"></span>
          </span>
          <span className="tracking-widest uppercase font-semibold text-[10px]">
            In Development{dots}
          </span>
        </motion.div>

        {/* Main title */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.7,
            delay: 0.1,
            ease: [0.25, 0.46, 0.45, 0.94],
          }}
          className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-semibold tracking-tight mb-6"
        >
          <span
            className={`text-transparent bg-clip-text bg-gradient-to-r ${colorScheme.gradient}`}
          >
            {title}
          </span>
        </motion.h1>

        {/* Subtitle */}
        {subtitle && (
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-[15px] sm:text-[16px] text-zinc-500 leading-relaxed mb-12 max-w-lg mx-auto"
          >
            {subtitle}
          </motion.p>
        )}

        {/* Coming Soon text */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="mb-12"
        >
          <p className="text-[13px] text-zinc-600 font-mono tracking-wider uppercase mb-2">
            Coming Soon
          </p>
          <div className="flex items-center justify-center gap-3">
            <div className="h-[1px] w-12 bg-gradient-to-r from-transparent to-zinc-800" />
            <span className="text-[11px] text-zinc-700 tracking-widest">
              Q2 2026
            </span>
            <div className="h-[1px] w-12 bg-gradient-to-l from-transparent to-zinc-800" />
          </div>
        </motion.div>

        {/* Back to Dashboard */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Link
            href="/dashboard"
            className="group inline-flex items-center gap-2 text-[13px] text-zinc-500 hover:text-white transition-colors duration-200"
          >
            <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
            <span>Back to Dashboard</span>
          </Link>
        </motion.div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#09090b] to-transparent pointer-events-none" />
    </div>
  );
}
