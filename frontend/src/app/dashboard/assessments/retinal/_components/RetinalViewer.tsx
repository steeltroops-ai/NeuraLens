
import React, { useState } from 'react';
import { RetinalAnalysisResult } from '@/types/retinal-analysis';

interface RetinalViewerProps {
  imageUrl: string;
  result?: RetinalAnalysisResult | null;
}

export function RetinalViewer({ imageUrl, result }: RetinalViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showVessels, setShowVessels] = useState(false);
  const [brightness, setBrightness] = useState(100);

  // Mock overlay URLs if result doesn't have them yet (since backend mocks return empty strings)
  const heatmapOverlay = result?.heatmap_url || ""; 
  const vesselOverlay = result?.segmentation_url || "";

  return (
    <div className="flex flex-col h-full bg-zinc-950 rounded-2xl overflow-hidden shadow-2xl border border-zinc-800">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 bg-zinc-900/50 backdrop-blur-sm border-b border-zinc-800">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Layers</span>
            <button 
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${showHeatmap ? 'bg-red-500/20 text-red-400 border border-red-500/50' : 'bg-zinc-800 text-zinc-400 border border-transparent'}`}
            >
              Heatmap
            </button>
            <button 
              onClick={() => setShowVessels(!showVessels)}
              className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${showVessels ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' : 'bg-zinc-800 text-zinc-400 border border-transparent'}`}
            >
              Vessels
            </button>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button onClick={() => setZoom(z => Math.max(1, z - 0.5))} className="p-1.5 hover:bg-zinc-800 rounded-lg text-zinc-400">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" /></svg>
            </button>
            <span className="text-xs w-12 text-center text-zinc-400">{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(4, z + 0.5))} className="p-1.5 hover:bg-zinc-800 rounded-lg text-zinc-400">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
            </button>
          </div>
        </div>
      </div>

      {/* Viewport */}
      <div className="relative flex-1 bg-black overflow-hidden flex items-center justify-center">
        <div 
          className="relative transition-transform duration-200 ease-out"
          style={{ 
            transform: `scale(${zoom})`,
            filter: `brightness(${brightness}%)`
          }}
        >
          {/* Base Image */}
          <img 
            src={imageUrl} 
            alt="Retinal Scan" 
            className="max-h-[600px] w-auto object-contain"
          />
          
          {/* Overlays (Simulated for Demo) */}
          {showHeatmap && (
             <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-transparent to-red-500/30 mix-blend-overlay pointer-events-none" />
          )}
          
          {showVessels && (
             <div className="absolute inset-0 border-opacity-50 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at center, transparent 0%, rgba(59, 130, 246, 0.1) 100%)' }}>
             </div>
          )}
        </div>
      </div>

      {/* Footer Controls */}
      <div className="p-4 bg-zinc-900 border-t border-zinc-800">
         <div className="flex items-center space-x-4">
            <span className="text-xs text-zinc-500">Brightness</span>
            <input 
              type="range" 
              min="50" 
              max="150" 
              value={brightness} 
              onChange={(e) => setBrightness(Number(e.target.value))}
              className="w-32 h-1 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
            />
         </div>
      </div>
    </div>
  );
}
