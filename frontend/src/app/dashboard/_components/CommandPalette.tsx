'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
    Search,
    Mic,
    Eye,
    Hand,
    Brain,
    Activity,
    Zap,
    BarChart3,
    FileText,
    Settings,
    Home,
    Command
} from 'lucide-react';

interface CommandItem {
    id: string;
    label: string;
    description?: string;
    icon: React.ReactNode;
    action: () => void;
    shortcut?: string;
    category: string;
}

export interface CommandPaletteProps {
    isOpen: boolean;
    onClose: () => void;
}

/**
 * Command Palette Component
 * 
 * Provides quick navigation and search functionality
 * Triggered by Cmd/Ctrl+K keyboard shortcut or search button
 * 
 * Requirements: 3.4
 */
export function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
    const router = useRouter();
    const inputRef = useRef<HTMLInputElement>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);

    // Define command items
    const commands: CommandItem[] = [
        // Quick Actions
        {
            id: 'speech',
            label: 'Start Speech Assessment',
            description: 'Analyze voice biomarkers for Parkinson\'s screening',
            icon: <Mic className="h-4 w-4" />,
            action: () => { router.push('/dashboard/speech'); onClose(); },
            shortcut: '⌘S',
            category: 'Quick Actions',
        },
        {
            id: 'retinal',
            label: 'Start Retinal Scan',
            description: 'Non-invasive Alzheimer\'s detection through retinal imaging',
            icon: <Eye className="h-4 w-4" />,
            action: () => { router.push('/dashboard/retinal'); onClose(); },
            shortcut: '⌘R',
            category: 'Quick Actions',
        },
        {
            id: 'motor',
            label: 'Start Motor Assessment',
            description: 'Movement pattern analysis and tremor detection',
            icon: <Hand className="h-4 w-4" />,
            action: () => { router.push('/dashboard/motor'); onClose(); },
            shortcut: '⌘M',
            category: 'Quick Actions',
        },
        {
            id: 'cognitive',
            label: 'Start Cognitive Testing',
            description: 'Memory and executive function assessment',
            icon: <Brain className="h-4 w-4" />,
            action: () => { router.push('/dashboard/cognitive'); onClose(); },
            shortcut: '⌘C',
            category: 'Quick Actions',
        },
        // Navigation
        {
            id: 'dashboard',
            label: 'Go to Dashboard',
            description: 'Return to main dashboard overview',
            icon: <Home className="h-4 w-4" />,
            action: () => { router.push('/dashboard'); onClose(); },
            category: 'Navigation',
        },
        {
            id: 'multimodal',
            label: 'Multi-Modal Assessment',
            description: 'Comprehensive multi-modality assessment',
            icon: <Activity className="h-4 w-4" />,
            action: () => { router.push('/dashboard/multimodal'); onClose(); },
            category: 'Navigation',
        },
        {
            id: 'nri-fusion',
            label: 'NRI Fusion',
            description: 'Neurological Risk Index fusion analysis',
            icon: <Zap className="h-4 w-4" />,
            action: () => { router.push('/dashboard/nri-fusion'); onClose(); },
            category: 'Navigation',
        },
        {
            id: 'analytics',
            label: 'View Analytics',
            description: 'Population health insights and trends',
            icon: <BarChart3 className="h-4 w-4" />,
            action: () => { router.push('/dashboard/analytics'); onClose(); },
            shortcut: '⌘A',
            category: 'Navigation',
        },
        {
            id: 'reports',
            label: 'View Reports',
            description: 'Clinical reports and documentation',
            icon: <FileText className="h-4 w-4" />,
            action: () => { router.push('/dashboard/reports'); onClose(); },
            category: 'Navigation',
        },
        {
            id: 'settings',
            label: 'Settings',
            description: 'Account and application settings',
            icon: <Settings className="h-4 w-4" />,
            action: () => { router.push('/dashboard/settings'); onClose(); },
            shortcut: '⌘,',
            category: 'Navigation',
        },
    ];

    // Filter commands based on search query
    const filteredCommands = commands.filter(cmd =>
        cmd.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
        cmd.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        cmd.category.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // Group commands by category
    const groupedCommands = filteredCommands.reduce<Record<string, CommandItem[]>>((acc, cmd) => {
        const category = cmd.category;
        if (!acc[category]) {
            acc[category] = [];
        }
        acc[category]!.push(cmd);
        return acc;
    }, {});

    // Focus input when opened
    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
        // Reset state when opened
        if (isOpen) {
            setSearchQuery('');
            setSelectedIndex(0);
        }
    }, [isOpen]);

    // Handle keyboard navigation
    const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < filteredCommands.length - 1 ? prev + 1 : 0
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev > 0 ? prev - 1 : filteredCommands.length - 1
                );
                break;
            case 'Enter':
                e.preventDefault();
                if (filteredCommands[selectedIndex]) {
                    filteredCommands[selectedIndex].action();
                }
                break;
            case 'Escape':
                e.preventDefault();
                onClose();
                break;
        }
    }, [filteredCommands, selectedIndex, onClose]);

    // Reset selected index when search changes
    useEffect(() => {
        setSelectedIndex(0);
    }, [searchQuery]);

    // Close on backdrop click
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-black/40"
            onClick={handleBackdropClick}
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            data-testid="command-palette"
        >
            <div
                className="w-full max-w-lg rounded-lg bg-white border border-zinc-200 shadow-lg overflow-hidden"
                style={{
                    animation: 'commandPaletteIn 150ms ease-out',
                }}
                onKeyDown={handleKeyDown}
            >
                {/* Search Input */}
                <div className="flex items-center border-b border-zinc-100">
                    <div className="flex items-center justify-center w-12 h-12">
                        <Search size={16} strokeWidth={1.5} className="text-zinc-400" aria-hidden="true" />
                    </div>
                    <input
                        ref={inputRef}
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="flex-1 h-12 pr-4 text-[14px] text-zinc-700 bg-transparent border-none outline-none placeholder:text-zinc-400"
                        placeholder="Search commands..."
                        aria-label="Search commands"
                        data-testid="command-palette-input"
                    />
                    <div className="flex items-center gap-1.5 pr-3">
                        <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 bg-zinc-100 rounded">
                            <Command size={10} />K
                        </kbd>
                        <kbd className="px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 bg-zinc-100 rounded">
                            ESC
                        </kbd>
                    </div>
                </div>

                {/* Commands List */}
                <div className="max-h-[360px] overflow-y-auto scrollbar-hide p-2">
                    {Object.entries(groupedCommands).length > 0 ? (
                        Object.entries(groupedCommands).map(([category, items]) => (
                            <div key={category} className="mb-2">
                                <p className="px-2 py-1.5 text-[10px] font-medium uppercase tracking-widest text-zinc-400">
                                    {category}
                                </p>
                                <div className="space-y-0.5">
                                    {items.map((cmd) => {
                                        const globalIndex = filteredCommands.findIndex(c => c.id === cmd.id);
                                        const isSelected = globalIndex === selectedIndex;

                                        return (
                                            <button
                                                key={cmd.id}
                                                className={`
                                                    flex w-full items-center gap-2.5 px-2 py-2 rounded-md text-left
                                                    transition-colors duration-100
                                                    ${isSelected
                                                        ? 'bg-blue-500 text-white'
                                                        : 'hover:bg-zinc-100 text-zinc-700'
                                                    }
                                                `}
                                                onClick={cmd.action}
                                                onMouseEnter={() => setSelectedIndex(globalIndex)}
                                                data-testid={`command-item-${cmd.id}`}
                                            >
                                                <div
                                                    className={`
                                                        flex h-7 w-7 items-center justify-center rounded-md
                                                        ${isSelected ? 'bg-white/20' : 'bg-zinc-100'}
                                                    `}
                                                >
                                                    <span className={isSelected ? 'text-white' : 'text-zinc-500'}>
                                                        {cmd.icon}
                                                    </span>
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <p className={`text-[13px] font-medium truncate ${isSelected ? 'text-white' : ''}`}>
                                                        {cmd.label}
                                                    </p>
                                                    {cmd.description && (
                                                        <p className={`text-[11px] truncate ${isSelected ? 'text-white/70' : 'text-zinc-400'}`}>
                                                            {cmd.description}
                                                        </p>
                                                    )}
                                                </div>
                                                {cmd.shortcut && (
                                                    <kbd
                                                        className={`
                                                            px-1.5 py-0.5 text-[10px] font-medium rounded
                                                            ${isSelected
                                                                ? 'bg-white/20 text-white'
                                                                : 'bg-zinc-100 text-zinc-500'
                                                            }
                                                        `}
                                                    >
                                                        {cmd.shortcut}
                                                    </kbd>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="py-8 text-center">
                            <p className="text-[13px] text-zinc-500">No commands found</p>
                            <p className="text-[11px] text-zinc-400 mt-1">Try a different search term</p>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between px-3 py-2 border-t border-zinc-100 bg-zinc-50">
                    <span className="text-[11px] text-zinc-500">
                        <span className="hidden sm:inline">↑↓ navigate • </span>
                        ↵ select • ESC close
                    </span>
                    <span className="text-[11px] text-zinc-400">
                        {filteredCommands.length} command{filteredCommands.length !== 1 ? 's' : ''}
                    </span>
                </div>
            </div>

            {/* Animation keyframes */}
            <style jsx>{`
                @keyframes commandPaletteIn {
                    from {
                        opacity: 0;
                        transform: scale(0.98) translateY(-8px);
                    }
                    to {
                        opacity: 1;
                        transform: scale(1) translateY(0);
                    }
                }
            `}</style>
        </div>
    );
}

export default CommandPalette;
