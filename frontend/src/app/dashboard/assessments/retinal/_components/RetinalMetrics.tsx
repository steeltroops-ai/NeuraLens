
import React from 'react';
import { RetinalAnalysisResult } from '@/types/retinal-analysis';
import { RiskBadgeWithScore } from '@/components/ui/RiskBadge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';

interface RetinalMetricsProps {
  result: RetinalAnalysisResult;
}

export function RetinalMetrics({ result }: RetinalMetricsProps) {
  const { risk_assessment, biomarkers } = result;

  const MetricRow = ({ label, value, unit = '', confidence }: { label: string, value: number, unit?: string, confidence?: number }) => (
    <div className="flex items-center justify-between py-3 border-b border-zinc-800/50 last:border-0">
      <div className="flex flex-col">
        <span className="text-sm text-zinc-400">{label}</span>
        {confidence && <span className="text-[10px] text-zinc-600">Conf: {(confidence * 100).toFixed(0)}%</span>}
      </div>
      <div className="text-right">
        <span className="text-base font-medium text-zinc-200">{value}</span>
        <span className="text-xs text-zinc-500 ml-1">{unit}</span>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Risk Score Card */}
      <Card className="border-border/5 bg-card/50 backdrop-blur-sm">
        <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
                <CardTitle className="text-sm uppercase tracking-wider font-semibold opacity-90">Neurological Risk</CardTitle>
                <span className="text-xs px-2 py-1 rounded-full bg-secondary font-mono">
                    v{result.model_version}
                </span>
            </div>
        </CardHeader>
        <CardContent>
            <div className="flex flex-col items-center py-4">
                <div className="mb-4">
                     <RiskBadgeWithScore score={risk_assessment.risk_score} size="lg" showScore={false} />
                </div>
                
                <div className="flex items-baseline space-x-2">
                    <span className="text-5xl font-bold tracking-tight">{risk_assessment.risk_score}</span>
                    <span className="text-lg opacity-75">/ 100</span>
                </div>
                
                {/* Confidence Interval Visualization */}
                <div className="mt-6 w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden relative">
                    <div 
                        className="absolute h-full bg-white/20" 
                        style={{ 
                        left: `${risk_assessment.confidence_interval[0]}%`, 
                        right: `${100 - risk_assessment.confidence_interval[1]}%` 
                        }} 
                    />
                    <div 
                        className="absolute h-full w-1 bg-white" 
                        style={{ left: `${risk_assessment.risk_score}%` }} 
                    />
                </div>
                <div className="flex justify-between w-full mt-1 text-[10px] opacity-40">
                    <span>Low Risk</span>
                    <span>High Risk</span>
                </div>
            </div>
        </CardContent>
      </Card>

      {/* Biomarker Details */}
      <div className="space-y-4">
        {/* Vascular */}
        <Card className="bg-zinc-900/30 border-zinc-800">
           <CardHeader className="pb-2">
               <h4 className="text-zinc-100 font-medium flex items-center text-sm">
                 <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-2"></span>
                 Retinal Vasculature
               </h4>
           </CardHeader>
           <CardContent>
             <div className="space-y-1">
               <MetricRow label="Vessel Density" value={biomarkers.vessels.density_percentage} unit="%" confidence={biomarkers.vessels.confidence} />
               <MetricRow label="Tortuosity Index" value={biomarkers.vessels.tortuosity_index} />
               <MetricRow label="AVR Ratio" value={biomarkers.vessels.avr_ratio} />
             </div>
           </CardContent>
        </Card>

        {/* Optic Disc */}
        <Card className="bg-zinc-900/30 border-zinc-800">
           <CardHeader className="pb-2">
               <h4 className="text-zinc-100 font-medium flex items-center text-sm">
                 <span className="w-1.5 h-1.5 bg-amber-500 rounded-full mr-2"></span>
                 Optic Disc
               </h4>
           </CardHeader>
           <CardContent>
             <div className="space-y-1">
               <MetricRow label="Cup-to-Disc" value={biomarkers.optic_disc.cup_to_disc_ratio} confidence={biomarkers.optic_disc.confidence} />
               <MetricRow label="Rim Area" value={biomarkers.optic_disc.rim_area_mm2} unit="mm²" />
             </div>
           </CardContent>
        </Card>

        {/* Amyloid */}
        <Card className="bg-zinc-900/30 border-zinc-800">
           <CardHeader className="pb-2">
               <h4 className="text-zinc-100 font-medium flex items-center text-sm">
                 <span className="w-1.5 h-1.5 bg-purple-500 rounded-full mr-2"></span>
                 Amyloid Indicators
               </h4>
           </CardHeader>
           <CardContent>
               <MetricRow label="Presence Score" value={biomarkers.amyloid_beta.presence_score} confidence={biomarkers.amyloid_beta.confidence} />
               <div className="pt-2 text-xs text-zinc-500">
                 Distribution: {biomarkers.amyloid_beta.distribution_pattern}
               </div>
           </CardContent>
        </Card>
      </div>
    </div>
  );
}
