'use client';

/**
 * Pipeline Status Bar Component
 * 
 * A VS Code-style status bar at the bottom of the dashboard
 * showing real-time pipeline processing status with step-by-step progress.
 * 
 * Features:
 * - Fixed position at bottom, respects sidebar margin
 * - Same styling as sidebar (black bg, zinc borders)
 * - Shows each processing step with status indicators
 * - Fully responsive for all devices
 */

import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Loader2,
    CheckCircle2,
    XCircle,
    Activity,
    Eye,
    Brain,
    Heart,
    Mic,
    Move3D,
    Zap,
    ChevronRight,
} from 'lucide-react';

// Sidebar collapsed key - same as sidebar
const SIDEBAR_COLLAPSED_KEY = 'medilens-sidebar-collapsed';

// Pipeline types
export type PipelineType = 
    | 'speech' 
    | 'retinal' 
    | 'cardiology' 
    | 'radiology' 
    | 'cognitive' 
    | 'motor' 
    | 'nri'
    | 'voice'
    | 'explain';

export type PipelineStatus = 'idle' | 'processing' | 'success' | 'error';

export interface PipelineStage {
    name: string;
    status: 'pending' | 'active' | 'complete' | 'error';
    duration_ms?: number;
}

export interface PipelineState {
    pipeline: PipelineType;
    status: PipelineStatus;
    currentStage?: string;
    stages?: PipelineStage[];
    message?: string;
    progress?: number; // 0-100
    startTime?: number;
    error?: string;
}

// Context for global pipeline status
interface PipelineStatusContextType {
    pipelines: Record<PipelineType, PipelineState>;
    updatePipeline: (pipeline: PipelineType, state: Partial<PipelineState>) => void;
    startPipeline: (pipeline: PipelineType, stages?: string[]) => void;
    updateStage: (pipeline: PipelineType, stageName: string, status: 'active' | 'complete' | 'error') => void;
    completePipeline: (pipeline: PipelineType, success: boolean, message?: string) => void;
    clearPipeline: (pipeline: PipelineType) => void;
}

const PipelineStatusContext = createContext<PipelineStatusContextType | null>(null);

// Pipeline icons
const PIPELINE_ICONS: Record<PipelineType, React.ElementType> = {
    speech: Mic,
    retinal: Eye,
    cardiology: Heart,
    radiology: Activity,
    cognitive: Brain,
    motor: Move3D,
    nri: Zap,
    voice: Mic,
    explain: Brain,
};

// Pipeline colors
const PIPELINE_COLORS: Record<PipelineType, string> = {
    speech: '#06b6d4',      // cyan
    retinal: '#3b82f6',     // blue
    cardiology: '#ef4444',  // red
    radiology: '#8b5cf6',   // violet
    cognitive: '#f59e0b',   // amber
    motor: '#10b981',       // emerald
    nri: '#ec4899',         // pink
    voice: '#06b6d4',       // cyan
    explain: '#f59e0b',     // amber
};

// Pipeline display names
const PIPELINE_NAMES: Record<PipelineType, string> = {
    speech: 'SpeechMD',
    retinal: 'RetinaScan',
    cardiology: 'CardioPredict',
    radiology: 'ChestXplorer',
    cognitive: 'Cognitive',
    motor: 'Motor',
    nri: 'NRI Fusion',
    voice: 'Voice',
    explain: 'AI Explain',
};

// Initial state for all pipelines
const initialPipelineState = (): Record<PipelineType, PipelineState> => ({
    speech: { pipeline: 'speech', status: 'idle' },
    retinal: { pipeline: 'retinal', status: 'idle' },
    cardiology: { pipeline: 'cardiology', status: 'idle' },
    radiology: { pipeline: 'radiology', status: 'idle' },
    cognitive: { pipeline: 'cognitive', status: 'idle' },
    motor: { pipeline: 'motor', status: 'idle' },
    nri: { pipeline: 'nri', status: 'idle' },
    voice: { pipeline: 'voice', status: 'idle' },
    explain: { pipeline: 'explain', status: 'idle' },
});

// Provider component
export function PipelineStatusProvider({ children }: { children: ReactNode }) {
    const [pipelines, setPipelines] = useState<Record<PipelineType, PipelineState>>(initialPipelineState);

    const updatePipeline = useCallback((pipeline: PipelineType, state: Partial<PipelineState>) => {
        setPipelines(prev => ({
            ...prev,
            [pipeline]: { ...prev[pipeline], ...state },
        }));
    }, []);

    const startPipeline = useCallback((pipeline: PipelineType, stageNames?: string[]) => {
        const stages: PipelineStage[] = stageNames?.map((name, i) => ({
            name,
            status: i === 0 ? 'active' : 'pending',
        })) || [];

        setPipelines(prev => ({
            ...prev,
            [pipeline]: {
                pipeline,
                status: 'processing',
                stages,
                currentStage: stages[0]?.name,
                startTime: Date.now(),
                progress: 0,
            },
        }));
    }, []);

    const updateStage = useCallback((pipeline: PipelineType, stageName: string, status: 'active' | 'complete' | 'error') => {
        setPipelines(prev => {
            const current = prev[pipeline];
            if (!current.stages) return prev;

            const stages = current.stages.map((s, i) => {
                if (s.name === stageName) {
                    return { ...s, status, duration_ms: s.status === 'active' ? Date.now() - (current.startTime || 0) : s.duration_ms };
                }
                // Activate next stage if current completes
                if (status === 'complete' && current.stages![i - 1]?.name === stageName && s.status === 'pending') {
                    return { ...s, status: 'active' };
                }
                return s;
            });

            const completedCount = stages.filter(s => s.status === 'complete').length;
            const progress = Math.round((completedCount / stages.length) * 100);

            return {
                ...prev,
                [pipeline]: {
                    ...current,
                    stages,
                    currentStage: stages.find(s => s.status === 'active')?.name || stageName,
                    progress,
                },
            };
        });
    }, []);

    const completePipeline = useCallback((pipeline: PipelineType, success: boolean, message?: string) => {
        setPipelines(prev => ({
            ...prev,
            [pipeline]: {
                ...prev[pipeline],
                status: success ? 'success' : 'error',
                currentStage: undefined,
                progress: success ? 100 : prev[pipeline].progress,
                message,
                error: success ? undefined : message,
                stages: prev[pipeline].stages?.map(s => ({
                    ...s,
                    status: success ? 'complete' : (s.status === 'active' ? 'error' : s.status),
                })),
            },
        }));

        // Auto-clear after 8 seconds
        setTimeout(() => {
            setPipelines(prev => ({
                ...prev,
                [pipeline]: { pipeline, status: 'idle' },
            }));
        }, 8000);
    }, []);

    const clearPipeline = useCallback((pipeline: PipelineType) => {
        setPipelines(prev => ({
            ...prev,
            [pipeline]: { pipeline, status: 'idle' },
        }));
    }, []);

    return (
        <PipelineStatusContext.Provider value={{
            pipelines,
            updatePipeline,
            startPipeline,
            updateStage,
            completePipeline,
            clearPipeline,
        }}>
            {children}
        </PipelineStatusContext.Provider>
    );
}

// Hook to use pipeline status
export function usePipelineStatus() {
    const context = useContext(PipelineStatusContext);
    if (!context) {
        throw new Error('usePipelineStatus must be used within PipelineStatusProvider');
    }
    return context;
}

// Status Bar Component
export function PipelineStatusBar() {
    const context = useContext(PipelineStatusContext);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [isDesktop, setIsDesktop] = useState(false);
    const [isClient, setIsClient] = useState(false);
    
    // Sync with sidebar state
    useEffect(() => {
        setIsClient(true);
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
            if (saved !== null) {
                setSidebarCollapsed(JSON.parse(saved));
            }
            setIsDesktop(window.innerWidth >= 1024);
        }
    }, []);
    
    // Listen for sidebar changes and window resize
    useEffect(() => {
        if (!isClient) return;
        
        const handleStorageChange = () => {
            const saved = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
            if (saved !== null) {
                setSidebarCollapsed(JSON.parse(saved));
            }
        };
        
        const handleResize = () => {
            setIsDesktop(window.innerWidth >= 1024);
        };
        
        // Check for changes periodically (same-tab updates)
        const interval = setInterval(handleStorageChange, 100);
        
        window.addEventListener('storage', handleStorageChange);
        window.addEventListener('resize', handleResize);
        
        return () => {
            clearInterval(interval);
            window.removeEventListener('storage', handleStorageChange);
            window.removeEventListener('resize', handleResize);
        };
    }, [isClient]);
    
    // Calculate left offset same as header
    const leftOffset = isDesktop ? (sidebarCollapsed ? 60 : 240) : 0;
    
    if (!context) {
        // Render a static placeholder if no provider
        return (
            <div 
                className="fixed bottom-0 right-0 h-8 bg-black border-t border-[#27272a] flex items-center px-4 z-30"
                style={{ 
                    left: `${leftOffset}px`,
                    transition: 'left 350ms cubic-bezier(0.32, 0.72, 0, 1)',
                }}
            >
                <div className="flex items-center gap-4 text-[11px] text-zinc-500 font-mono">
                    <span>MediLens v1.0</span>
                    <span className="text-[#27272a]">|</span>
                    <span className="text-green-500">Ready</span>
                </div>
            </div>
        );
    }

    const { pipelines } = context;
    
    // Get active pipelines
    const activePipelines = Object.values(pipelines).filter(p => p.status !== 'idle');
    const hasActivity = activePipelines.length > 0;

    return (
        <div 
            className="fixed bottom-0 right-0 h-8 bg-black border-t border-[#27272a] flex items-center justify-between px-4 z-30 select-none"
            style={{ 
                left: `${leftOffset}px`,
                transition: 'left 350ms cubic-bezier(0.32, 0.72, 0, 1)',
            }}
        >
            {/* Left section - Pipeline status with steps */}
            <div className="flex items-center gap-4 overflow-hidden flex-1 min-w-0">
                <AnimatePresence mode="popLayout">
                    {activePipelines.length > 0 ? (
                        activePipelines.map(pipeline => (
                            <PipelineIndicator key={pipeline.pipeline} state={pipeline} />
                        ))
                    ) : (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex items-center gap-2 text-[11px] text-zinc-500 font-mono"
                        >
                            <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                            <span className="text-zinc-400 truncate">All pipelines ready</span>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Right section - System info */}
            <div className="flex items-center gap-3 text-[11px] text-zinc-600 font-mono flex-shrink-0">
                {hasActivity && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="hidden sm:flex items-center gap-1.5"
                    >
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                        <span className="text-cyan-400">Processing</span>
                    </motion.div>
                )}
                <span className="text-[#27272a] hidden sm:inline">|</span>
                <span className="text-zinc-500 hidden sm:inline">MediLens v1.0</span>
            </div>
        </div>
    );
}

// Individual pipeline indicator with step-by-step progress
function PipelineIndicator({ state }: { state: PipelineState }) {
    const Icon = PIPELINE_ICONS[state.pipeline];
    const color = PIPELINE_COLORS[state.pipeline];
    const name = PIPELINE_NAMES[state.pipeline];

    // Status icon for the pipeline
    const statusIcon = (() => {
        switch (state.status) {
            case 'processing':
                return <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" style={{ color }} />;
            case 'success':
                return <CheckCircle2 className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />;
            case 'error':
                return <XCircle className="h-3.5 w-3.5 text-red-500 flex-shrink-0" />;
            default:
                return <Icon className="h-3.5 w-3.5 flex-shrink-0" style={{ color }} />;
        }
    })();

    return (
        <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="flex items-center gap-2 text-[11px] font-mono min-w-0"
        >
            {/* Pipeline icon and name */}
            <div className="flex items-center gap-1.5 flex-shrink-0">
                {statusIcon}
                <span className="text-zinc-300 font-medium">{name}</span>
            </div>

            {/* Processing steps - hidden on small screens */}
            {state.status === 'processing' && state.stages && state.stages.length > 0 && (
                <div className="hidden md:flex items-center gap-1">
                    <span className="text-[#27272a]">|</span>
                    <div className="flex items-center gap-0.5">
                        {state.stages.map((stage, idx) => (
                            <React.Fragment key={stage.name}>
                                <StageIndicator stage={stage} />
                                {idx < state.stages!.length - 1 && (
                                    <ChevronRight className="h-2.5 w-2.5 text-zinc-700" />
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            )}

            {/* Current stage label - hidden on small screens */}
            {state.currentStage && state.status === 'processing' && (
                <span className="hidden lg:inline text-zinc-500 truncate max-w-[120px]">
                    {state.currentStage}
                </span>
            )}

            {/* Progress percentage */}
            {typeof state.progress === 'number' && state.status === 'processing' && state.progress > 0 && (
                <span className="text-zinc-600 flex-shrink-0">{state.progress}%</span>
            )}

            {/* Completion message */}
            {state.message && state.status !== 'processing' && (
                <span className={`truncate max-w-[100px] sm:max-w-[150px] ${state.status === 'error' ? 'text-red-400' : 'text-green-400'}`}>
                    {state.message}
                </span>
            )}
        </motion.div>
    );
}

// Individual stage indicator (small dot/icon)
function StageIndicator({ stage }: { stage: PipelineStage }) {
    const getStageStyle = () => {
        switch (stage.status) {
            case 'complete':
                return 'bg-green-500';
            case 'active':
                return 'bg-cyan-500 animate-pulse';
            case 'error':
                return 'bg-red-500';
            default:
                return 'bg-zinc-700';
        }
    };

    return (
        <div 
            className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${getStageStyle()}`}
            title={`${stage.name}: ${stage.status}`}
        />
    );
}

export default PipelineStatusBar;
