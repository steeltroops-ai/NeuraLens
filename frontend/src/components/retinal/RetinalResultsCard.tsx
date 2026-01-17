/**
 * RetinalResultsCard Component
 * 
 * Comprehensive display of retinal analysis results.
 * 
 * Features:
 * - Risk score with gauge visualization
 * - Biomarker grid
 * - Segmentation and heatmap visualizations
 * - Report download
 * - Action buttons
 * 
 * Requirements: 1.8, 2.11, 2.12
 * 
 * @module components/retinal/RetinalResultsCard
 */

'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  CheckCircle2, 
  Download, 
  Eye, 
  Activity,
  Brain,
  Disc,
  Droplets,
  FolderOpen,
  ExternalLink,
  Share2,
  Loader2,
  ZoomIn
} from 'lucide-react';
import { RetinalAnalysisResult, RiskCategory, RISK_CATEGORY_COLORS } from '@/types/retinal-analysis';
import { RiskGauge } from './RiskGauge';
import { BiomarkerCard } from './BiomarkerCard';

// ============================================================================
// Types
// ============================================================================

interface RetinalResultsCardProps {
  /** Analysis result to display */
  result: RetinalAnalysisResult;
  /** Whether to show visualizations */
  showVisualizations?: boolean;
  /** Enable report download */
  enableReport?: boolean;
  /** Called when report is downloaded */
  onReportDownload?: (assessmentId: string) => void;
  /** Called when user clicks new analysis */
  onNewAnalysis?: () => void;
  /** Loading state for report */
  isReportLoading?: boolean;
}

// ============================================================================
// Biomarker Reference Ranges
// ============================================================================

const BIOMARKER_REFS: Record<string, { min: number; max: number; unit: string; name: string }> = {
  vessel_density: { min: 4.0, max: 7.0, unit: '%', name: 'Vessel Density' },
  tortuosity_index: { min: 0.8, max: 1.3, unit: '', name: 'Tortuosity Index' },
  avr_ratio: { min: 0.6, max: 0.8, unit: '', name: 'A/V Ratio' },
  cup_to_disc_ratio: { min: 0.3, max: 0.5, unit: '', name: 'Cup-to-Disc Ratio' },
  disc_area: { min: 2.0, max: 3.5, unit: 'mm²', name: 'Disc Area' },
  macular_thickness: { min: 250, max: 320, unit: 'μm', name: 'Macular Thickness' },
  amyloid_presence: { min: 0.0, max: 0.2, unit: '', name: 'Amyloid Score' },
};

// ============================================================================
// Component
// ============================================================================

export function RetinalResultsCard({
  result,
  showVisualizations = true,
  enableReport = true,
  onReportDownload,
  onNewAnalysis,
  isReportLoading = false,
}: RetinalResultsCardProps) {
  const [activeViz, setActiveViz] = useState<'heatmap' | 'segmentation'>('segmentation');
  const [isZoomed, setIsZoomed] = useState(false);

  const categoryColor = RISK_CATEGORY_COLORS[result.risk_assessment.risk_category] || '#6b7280';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-zinc-200 bg-white shadow-sm overflow-hidden"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-cyan-50 via-white to-blue-50 p-6 border-b border-zinc-100">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20">
              <CheckCircle2 className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-zinc-900">Analysis Complete</h2>
              <p className="text-sm text-zinc-600">
                Processed in {result.processing_time_ms}ms • 
                Model v{result.model_version}
              </p>
            </div>
          </div>

          {/* Quality Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-zinc-200">
            <span className="text-xs text-zinc-500">Quality</span>
            <span className="text-sm font-semibold text-zinc-900">
              {result.quality_score.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* Risk Assessment Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Risk Gauge */}
          <div className="lg:col-span-1 flex justify-center">
            <RiskGauge
              score={result.risk_assessment.risk_score}
              category={result.risk_assessment.risk_category}
              confidenceInterval={result.risk_assessment.confidence_interval}
              size="lg"
            />
          </div>

          {/* Contributing Factors */}
          <div className="lg:col-span-2">
            <h3 className="text-sm font-semibold text-zinc-700 mb-3">Contributing Factors</h3>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(result.risk_assessment.contributing_factors).map(([factor, value]) => (
                <div key={factor} className="rounded-lg bg-zinc-50 p-3">
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-xs text-zinc-600 capitalize">
                      {factor.replace(/_/g, ' ')}
                    </span>
                    <span className="text-sm font-semibold text-zinc-900">
                      {value.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-zinc-200 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: categoryColor }}
                      initial={{ width: 0 }}
                      animate={{ width: `${value}%` }}
                      transition={{ duration: 0.5, delay: 0.2 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Biomarkers Grid */}
        <div className="mb-8">
          <h3 className="text-sm font-semibold text-zinc-700 mb-4">Biomarker Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Vessel Biomarkers */}
            <BiomarkerCard
              name="Vessel Density"
              value={result.biomarkers.vessels.density_percentage}
              unit="%"
              normalRange={[4.0, 7.0]}
              confidence={result.biomarkers.vessels.confidence}
              icon={<Droplets className="h-4 w-4 text-cyan-600" />}
            />
            <BiomarkerCard
              name="Tortuosity Index"
              value={result.biomarkers.vessels.tortuosity_index}
              unit=""
              normalRange={[0.8, 1.3]}
              confidence={result.biomarkers.vessels.confidence}
              icon={<Activity className="h-4 w-4 text-blue-600" />}
            />
            
            {/* Optic Disc */}
            <BiomarkerCard
              name="Cup-to-Disc Ratio"
              value={result.biomarkers.optic_disc.cup_to_disc_ratio}
              unit=""
              normalRange={[0.3, 0.5]}
              confidence={result.biomarkers.optic_disc.confidence}
              icon={<Disc className="h-4 w-4 text-purple-600" />}
            />
            
            {/* Amyloid Beta */}
            <BiomarkerCard
              name="Amyloid-β Score"
              value={result.biomarkers.amyloid_beta.presence_score}
              unit=""
              normalRange={[0.0, 0.2]}
              confidence={result.biomarkers.amyloid_beta.confidence}
              icon={<Brain className="h-4 w-4 text-rose-600" />}
            />
          </div>
        </div>

        {/* Visualizations */}
        {showVisualizations && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-zinc-700">Visualizations</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setActiveViz('segmentation')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    activeViz === 'segmentation'
                      ? 'bg-cyan-100 text-cyan-700'
                      : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                  }`}
                >
                  Segmentation
                </button>
                <button
                  onClick={() => setActiveViz('heatmap')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    activeViz === 'heatmap'
                      ? 'bg-cyan-100 text-cyan-700'
                      : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                  }`}
                >
                  Attention Heatmap
                </button>
              </div>
            </div>

            <div className="relative rounded-xl border border-zinc-200 overflow-hidden bg-zinc-50">
              {/* Visualization Image */}
              <div className="aspect-video flex items-center justify-center p-4">
                <div className="relative">
                  <img
                    src={activeViz === 'heatmap' ? result.heatmap_url : result.segmentation_url}
                    alt={`${activeViz} visualization`}
                    className={`rounded-lg object-contain max-h-80 shadow-lg transition-transform ${
                      isZoomed ? 'scale-150 cursor-zoom-out' : 'cursor-zoom-in'
                    }`}
                    onClick={() => setIsZoomed(!isZoomed)}
                  />
                  {/* Zoom hint */}
                  <button
                    onClick={() => setIsZoomed(!isZoomed)}
                    className="absolute top-2 right-2 p-2 rounded-lg bg-black/50 text-white hover:bg-black/70 transition-colors"
                  >
                    <ZoomIn className="h-4 w-4" />
                  </button>
                </div>
              </div>

              {/* Visualization Info */}
              <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-zinc-900/80 to-transparent">
                <div className="flex items-center gap-2 text-white">
                  <Eye className="h-4 w-4" />
                  <span className="text-sm">
                    {activeViz === 'heatmap' 
                      ? 'Attention map highlighting regions of interest'
                      : 'Vessel segmentation with artery (red) and vein (blue) classification'
                    }
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3 pt-4 border-t border-zinc-100">
          {/* Download Report */}
          {enableReport && (
            <button
              onClick={() => onReportDownload?.(result.assessment_id)}
              disabled={isReportLoading}
              className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-medium shadow-md shadow-cyan-500/20 hover:shadow-lg transition-all disabled:opacity-50"
            >
              {isReportLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Download PDF Report
            </button>
          )}

          {/* Save to History */}
          <button className="flex items-center gap-2 px-4 py-2.5 border border-zinc-200 text-zinc-700 rounded-lg font-medium hover:bg-zinc-50 transition-colors">
            <FolderOpen className="h-4 w-4" />
            Save to History
          </button>

          {/* Share */}
          <button className="flex items-center gap-2 px-4 py-2.5 border border-zinc-200 text-zinc-700 rounded-lg font-medium hover:bg-zinc-50 transition-colors">
            <Share2 className="h-4 w-4" />
            Share
          </button>

          {/* New Analysis */}
          {onNewAnalysis && (
            <button
              onClick={onNewAnalysis}
              className="flex items-center gap-2 px-4 py-2.5 border border-zinc-200 text-zinc-700 rounded-lg font-medium hover:bg-zinc-50 transition-colors ml-auto"
            >
              <ExternalLink className="h-4 w-4" />
              New Analysis
            </button>
          )}
        </div>
      </div>

      {/* Footer Meta */}
      <div className="px-6 py-3 bg-zinc-50 border-t border-zinc-100">
        <div className="flex items-center justify-between text-xs text-zinc-500">
          <span>Assessment ID: {result.assessment_id}</span>
          <span>Patient: {result.patient_id}</span>
          <span>Created: {new Date(result.created_at).toLocaleString()}</span>
        </div>
      </div>
    </motion.div>
  );
}

export default RetinalResultsCard;
