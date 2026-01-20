"use client";

/**
 * PipelineHealthIndicator Component
 *
 * Displays real-time health status of all AI diagnostic pipelines.
 * Shows last run time, success rate, and current status.
 *
 * Design follows enterprise observability tools (Datadog, Grafana).
 */

import { memo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  Eye,
  Mic,
  Heart,
  Scan,
  Sparkles,
  Brain,
  Hand,
  Zap,
} from "lucide-react";
import Link from "next/link";

type PipelineStatus = "healthy" | "degraded" | "failed" | "idle";

interface PipelineHealth {
  id: string;
  name: string;
  route: string;
  status: PipelineStatus;
  lastRun?: string;
  successRate?: number;
  avgProcessingMs?: number;
  icon: React.ElementType;
  color: string;
  bgColor: string;
}

interface PipelineHealthIndicatorProps {
  pipelines?: PipelineHealth[];
  compact?: boolean;
}

// Default pipeline status - in production, fetch from API
const defaultPipelines: PipelineHealth[] = [
  {
    id: "retinal",
    name: "RetinaScan",
    route: "/dashboard/retinal",
    status: "healthy",
    lastRun: "2m ago",
    successRate: 98.5,
    avgProcessingMs: 1240,
    icon: Eye,
    color: "#0891B2",
    bgColor: "#ECFEFF",
  },
  {
    id: "speech",
    name: "SpeechMD",
    route: "/dashboard/speech",
    status: "healthy",
    lastRun: "5m ago",
    successRate: 97.2,
    avgProcessingMs: 890,
    icon: Mic,
    color: "#4F46E5",
    bgColor: "#EEF2FF",
  },
  {
    id: "cardiology",
    name: "CardioPredict",
    route: "/dashboard/cardiology",
    status: "healthy",
    lastRun: "12m ago",
    successRate: 96.8,
    avgProcessingMs: 1560,
    icon: Heart,
    color: "#BE123C",
    bgColor: "#FFF1F2",
  },
  {
    id: "radiology",
    name: "ChestXplorer",
    route: "/dashboard/radiology",
    status: "idle",
    lastRun: "1h ago",
    successRate: 99.1,
    avgProcessingMs: 2100,
    icon: Scan,
    color: "#7C3AED",
    bgColor: "#F5F3FF",
  },
  {
    id: "dermatology",
    name: "SkinSense",
    route: "/dashboard/dermatology",
    status: "healthy",
    lastRun: "8m ago",
    successRate: 95.4,
    avgProcessingMs: 1180,
    icon: Sparkles,
    color: "#C026D3",
    bgColor: "#FDF4FF",
  },
];

const getStatusConfig = (status: PipelineStatus) => {
  switch (status) {
    case "healthy":
      return {
        label: "Healthy",
        color: "text-emerald-600",
        bgColor: "bg-emerald-50",
        borderColor: "border-emerald-200",
        icon: CheckCircle2,
        dotColor: "bg-emerald-500",
      };
    case "degraded":
      return {
        label: "Degraded",
        color: "text-amber-600",
        bgColor: "bg-amber-50",
        borderColor: "border-amber-200",
        icon: Activity,
        dotColor: "bg-amber-500 animate-pulse",
      };
    case "failed":
      return {
        label: "Failed",
        color: "text-red-600",
        bgColor: "bg-red-50",
        borderColor: "border-red-200",
        icon: XCircle,
        dotColor: "bg-red-500",
      };
    case "idle":
    default:
      return {
        label: "Idle",
        color: "text-slate-500",
        bgColor: "bg-slate-50",
        borderColor: "border-slate-200",
        icon: Clock,
        dotColor: "bg-slate-400",
      };
  }
};

function PipelineRow({ pipeline }: { pipeline: PipelineHealth }) {
  const statusConfig = getStatusConfig(pipeline.status);
  const Icon = pipeline.icon;

  return (
    <Link
      href={pipeline.route}
      className="flex items-center gap-3 p-3 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 hover:border-slate-300 transition-all duration-150 group"
    >
      {/* Module Icon */}
      <div
        className="flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center"
        style={{ backgroundColor: pipeline.bgColor }}
      >
        <Icon size={18} style={{ color: pipeline.color }} strokeWidth={1.5} />
      </div>

      {/* Name & Status */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-medium text-slate-900 truncate">
            {pipeline.name}
          </span>
          <span
            className={`flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded ${statusConfig.bgColor} ${statusConfig.color}`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${statusConfig.dotColor}`}
            />
            {statusConfig.label}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-slate-500 mt-0.5">
          {pipeline.lastRun && (
            <span className="flex items-center gap-1">
              <Clock size={10} strokeWidth={1.5} />
              {pipeline.lastRun}
            </span>
          )}
          {pipeline.successRate !== undefined && (
            <span className="text-emerald-600 font-medium">
              {pipeline.successRate.toFixed(1)}% success
            </span>
          )}
        </div>
      </div>

      {/* Processing Time */}
      {pipeline.avgProcessingMs && (
        <div className="hidden sm:block text-right">
          <div className="text-[12px] font-medium text-slate-700">
            {(pipeline.avgProcessingMs / 1000).toFixed(1)}s
          </div>
          <div className="text-[10px] text-slate-400">avg</div>
        </div>
      )}
    </Link>
  );
}

function CompactPipelineIndicator({ pipeline }: { pipeline: PipelineHealth }) {
  const statusConfig = getStatusConfig(pipeline.status);
  const Icon = pipeline.icon;

  return (
    <Link
      href={pipeline.route}
      className="flex items-center gap-2 p-2 rounded-md hover:bg-slate-50 transition-colors"
      title={`${pipeline.name} - ${statusConfig.label}`}
    >
      <div
        className="w-7 h-7 rounded flex items-center justify-center"
        style={{ backgroundColor: pipeline.bgColor }}
      >
        <Icon size={14} style={{ color: pipeline.color }} strokeWidth={1.5} />
      </div>
      <span className={`w-2 h-2 rounded-full ${statusConfig.dotColor}`} />
    </Link>
  );
}

export const PipelineHealthIndicator = memo(
  ({
    pipelines = defaultPipelines,
    compact = false,
  }: PipelineHealthIndicatorProps) => {
    const healthyCount = pipelines.filter((p) => p.status === "healthy").length;
    const totalCount = pipelines.length;

    if (compact) {
      return (
        <div className="flex items-center gap-1">
          {pipelines.slice(0, 5).map((pipeline) => (
            <CompactPipelineIndicator key={pipeline.id} pipeline={pipeline} />
          ))}
        </div>
      );
    }

    return (
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2, delay: 0.1 }}
        className="bg-white rounded-xl border border-slate-200 p-5"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-slate-600" strokeWidth={2} />
            <h2 className="text-[14px] font-semibold text-slate-900">
              Pipeline Status
            </h2>
            <span className="text-[11px] text-emerald-600 font-medium bg-emerald-50 px-1.5 py-0.5 rounded">
              {healthyCount}/{totalCount} healthy
            </span>
          </div>
          <button
            className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700 transition-colors"
            onClick={() => {
              // In production: refresh pipeline status
            }}
          >
            <RefreshCw size={12} strokeWidth={2} />
            Refresh
          </button>
        </div>

        {/* Pipeline List */}
        <div className="space-y-2">
          {pipelines.map((pipeline, index) => (
            <motion.div
              key={pipeline.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.15, delay: index * 0.03 }}
            >
              <PipelineRow pipeline={pipeline} />
            </motion.div>
          ))}
        </div>
      </motion.section>
    );
  },
);

PipelineHealthIndicator.displayName = "PipelineHealthIndicator";

export default PipelineHealthIndicator;
