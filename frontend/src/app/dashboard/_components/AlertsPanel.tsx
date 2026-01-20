"use client";

/**
 * AlertsPanel Component
 *
 * Medical-grade alerts panel for clinical dashboard.
 * Displays critical findings, pipeline failures, and system warnings.
 *
 * Design follows hospital monitoring system conventions:
 * - Critical (red): Immediate action required
 * - Elevated (orange): Urgent attention needed
 * - Warning (amber): Monitor closely
 * - Attention (blue): Review when possible
 */

import { memo, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  AlertCircle,
  XCircle,
  Info,
  X,
  ChevronRight,
  Clock,
  RefreshCw,
  Eye,
  Mic,
  Heart,
  Scan,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

// Alert severity levels following medical conventions
type AlertSeverity = "critical" | "elevated" | "warning" | "attention";

interface ClinicalAlert {
  id: string;
  severity: AlertSeverity;
  title: string;
  message: string;
  module?: string;
  timestamp: string;
  actionLabel?: string;
  actionRoute?: string;
  dismissible?: boolean;
}

interface AlertsPanelProps {
  alerts?: ClinicalAlert[];
  onDismiss?: (id: string) => void;
  maxVisible?: number;
}

// Mock alerts for demo - in production, these come from API
const defaultAlerts: ClinicalAlert[] = [
  {
    id: "alert-1",
    severity: "elevated",
    title: "Abnormal Finding Detected",
    message: "Retinal scan shows signs of moderate NPDR requiring review.",
    module: "retinal",
    timestamp: "5 min ago",
    actionLabel: "Review Now",
    actionRoute: "/dashboard/retinal",
    dismissible: false,
  },
  {
    id: "alert-2",
    severity: "warning",
    title: "Voice Biomarker Deviation",
    message: "Jitter values outside normal range in recent speech analysis.",
    module: "speech",
    timestamp: "23 min ago",
    actionLabel: "View Details",
    actionRoute: "/dashboard/speech",
    dismissible: true,
  },
];

const getSeverityConfig = (severity: AlertSeverity) => {
  switch (severity) {
    case "critical":
      return {
        icon: XCircle,
        bgColor: "bg-[#FEF2F2]",
        borderColor: "border-[#FECACA]",
        textColor: "text-[#DC2626]",
        iconColor: "text-[#DC2626]",
        badgeText: "CRITICAL",
      };
    case "elevated":
      return {
        icon: AlertTriangle,
        bgColor: "bg-[#FFF7ED]",
        borderColor: "border-[#FED7AA]",
        textColor: "text-[#EA580C]",
        iconColor: "text-[#EA580C]",
        badgeText: "URGENT",
      };
    case "warning":
      return {
        icon: AlertCircle,
        bgColor: "bg-[#FFFBEB]",
        borderColor: "border-[#FDE68A]",
        textColor: "text-[#D97706]",
        iconColor: "text-[#D97706]",
        badgeText: "WARNING",
      };
    case "attention":
    default:
      return {
        icon: Info,
        bgColor: "bg-[#EFF6FF]",
        borderColor: "border-[#BFDBFE]",
        textColor: "text-[#2563EB]",
        iconColor: "text-[#2563EB]",
        badgeText: "INFO",
      };
  }
};

const getModuleIcon = (module?: string) => {
  switch (module) {
    case "retinal":
      return Eye;
    case "speech":
      return Mic;
    case "cardiology":
      return Heart;
    case "radiology":
      return Scan;
    case "dermatology":
      return Sparkles;
    default:
      return AlertCircle;
  }
};

function AlertItem({
  alert,
  onDismiss,
}: {
  alert: ClinicalAlert;
  onDismiss?: (id: string) => void;
}) {
  const config = getSeverityConfig(alert.severity);
  const SeverityIcon = config.icon;
  const ModuleIcon = getModuleIcon(alert.module);

  return (
    <motion.div
      initial={{ opacity: 0, y: -8, height: 0 }}
      animate={{ opacity: 1, y: 0, height: "auto" }}
      exit={{ opacity: 0, y: -8, height: 0 }}
      transition={{ duration: 0.2 }}
      className={`${config.bgColor} ${config.borderColor} border rounded-lg p-4 relative`}
    >
      <div className="flex items-start gap-3">
        {/* Severity Icon */}
        <div className="flex-shrink-0 mt-0.5">
          <SeverityIcon
            size={18}
            className={config.iconColor}
            strokeWidth={2}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${config.bgColor} ${config.textColor} border ${config.borderColor}`}
            >
              {config.badgeText}
            </span>
            {alert.module && (
              <span className="flex items-center gap-1 text-[11px] text-slate-500">
                <ModuleIcon size={12} strokeWidth={1.5} />
                <span className="capitalize">{alert.module}</span>
              </span>
            )}
            <span className="flex items-center gap-1 text-[11px] text-slate-400">
              <Clock size={10} strokeWidth={1.5} />
              {alert.timestamp}
            </span>
          </div>

          <h4 className="text-[13px] font-semibold text-slate-900 mb-0.5">
            {alert.title}
          </h4>
          <p className="text-[12px] text-slate-600 leading-relaxed">
            {alert.message}
          </p>

          {/* Action Button */}
          {alert.actionRoute && (
            <Link
              href={alert.actionRoute}
              className={`inline-flex items-center gap-1 mt-2 text-[12px] font-medium ${config.textColor} hover:underline`}
            >
              {alert.actionLabel || "View"}
              <ChevronRight size={14} strokeWidth={2} />
            </Link>
          )}
        </div>

        {/* Dismiss Button */}
        {alert.dismissible && onDismiss && (
          <button
            onClick={() => onDismiss(alert.id)}
            className="flex-shrink-0 p-1 rounded hover:bg-black/5 transition-colors"
            aria-label="Dismiss alert"
          >
            <X size={14} className="text-slate-400" strokeWidth={2} />
          </button>
        )}
      </div>
    </motion.div>
  );
}

export const AlertsPanel = memo(
  ({ alerts = defaultAlerts, onDismiss, maxVisible = 3 }: AlertsPanelProps) => {
    const [visibleAlerts, setVisibleAlerts] = useState<ClinicalAlert[]>(alerts);
    const [isExpanded, setIsExpanded] = useState(false);

    useEffect(() => {
      setVisibleAlerts(alerts);
    }, [alerts]);

    const handleDismiss = (id: string) => {
      setVisibleAlerts((prev) => prev.filter((a) => a.id !== id));
      onDismiss?.(id);
    };

    if (visibleAlerts.length === 0) {
      return null; // No alerts, no panel
    }

    const displayAlerts = isExpanded
      ? visibleAlerts
      : visibleAlerts.slice(0, maxVisible);
    const hasMore = visibleAlerts.length > maxVisible;

    // Sort by severity (critical first)
    const sortedAlerts = [...displayAlerts].sort((a, b) => {
      const order = { critical: 0, elevated: 1, warning: 2, attention: 3 };
      return order[a.severity] - order[b.severity];
    });

    return (
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="mb-6"
        aria-label="Clinical Alerts"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <AlertTriangle
              size={16}
              className="text-amber-600"
              strokeWidth={2}
            />
            <h2 className="text-[14px] font-semibold text-slate-900">
              Alerts & Notifications
            </h2>
            <span className="text-[11px] font-medium text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
              {visibleAlerts.length}
            </span>
          </div>
          {hasMore && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-[12px] font-medium text-blue-600 hover:text-blue-700 flex items-center gap-1"
            >
              {isExpanded ? "Show less" : `Show all (${visibleAlerts.length})`}
              <ChevronRight
                size={14}
                className={`transform transition-transform ${isExpanded ? "rotate-90" : ""}`}
              />
            </button>
          )}
        </div>

        {/* Alert Items */}
        <div className="space-y-2">
          <AnimatePresence mode="popLayout">
            {sortedAlerts.map((alert) => (
              <AlertItem
                key={alert.id}
                alert={alert}
                onDismiss={handleDismiss}
              />
            ))}
          </AnimatePresence>
        </div>
      </motion.section>
    );
  },
);

AlertsPanel.displayName = "AlertsPanel";

export default AlertsPanel;
