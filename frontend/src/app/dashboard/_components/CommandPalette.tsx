"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
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
  Command,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Define command items
  const commands: CommandItem[] = [
    // Quick Actions
    {
      id: "speech",
      label: "Start Speech Assessment",
      description: "Analyze voice biomarkers for Parkinson's screening",
      icon: <Mic className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/speech");
        onClose();
      },
      shortcut: "⌘S",
      category: "Quick Actions",
    },
    {
      id: "retinal",
      label: "Start Retinal Scan",
      description: "Non-invasive Alzheimer's detection through retinal imaging",
      icon: <Eye className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/retinal");
        onClose();
      },
      shortcut: "⌘R",
      category: "Quick Actions",
    },
    {
      id: "motor",
      label: "Start Motor Assessment",
      description: "Movement pattern analysis and tremor detection",
      icon: <Hand className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/motor");
        onClose();
      },
      shortcut: "⌘M",
      category: "Quick Actions",
    },
    {
      id: "cognitive",
      label: "Start Cognitive Testing",
      description: "Memory and executive function assessment",
      icon: <Brain className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/cognitive");
        onClose();
      },
      shortcut: "⌘C",
      category: "Quick Actions",
    },
    // Navigation
    {
      id: "dashboard",
      label: "Go to Dashboard",
      description: "Return to main dashboard overview",
      icon: <Home className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard");
        onClose();
      },
      category: "Navigation",
    },
    {
      id: "multimodal",
      label: "Multi-Modal Assessment",
      description: "Comprehensive multi-modality assessment",
      icon: <Activity className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/multimodal");
        onClose();
      },
      category: "Navigation",
    },
    {
      id: "nri-fusion",
      label: "NRI Fusion",
      description: "Neurological Risk Index fusion analysis",
      icon: <Zap className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/nri-fusion");
        onClose();
      },
      category: "Navigation",
    },
    {
      id: "analytics",
      label: "View Analytics",
      description: "Population health insights and trends",
      icon: <BarChart3 className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/analytics");
        onClose();
      },
      shortcut: "⌘A",
      category: "Navigation",
    },
    {
      id: "reports",
      label: "View Reports",
      description: "Clinical reports and documentation",
      icon: <FileText className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/reports");
        onClose();
      },
      category: "Navigation",
    },
    {
      id: "settings",
      label: "Settings",
      description: "Account and application settings",
      icon: <Settings className="h-4 w-4" />,
      action: () => {
        router.push("/dashboard/settings");
        onClose();
      },
      shortcut: "⌘,",
      category: "Navigation",
    },
  ];

  // Filter commands based on search query
  const filteredCommands = commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      cmd.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      cmd.category.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  // Group commands by category
  const groupedCommands = filteredCommands.reduce<
    Record<string, CommandItem[]>
  >((acc, cmd) => {
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
      // Small timeout ensuring element is visible
      const timer = setTimeout(() => inputRef.current?.focus(), 50);
      return () => clearTimeout(timer);
    }
    // Reset state when opened
    if (isOpen) {
      setSearchQuery("");
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < filteredCommands.length - 1 ? prev + 1 : 0,
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev > 0 ? prev - 1 : filteredCommands.length - 1,
          );
          break;
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filteredCommands, selectedIndex, onClose],
  );

  // Reset selected index when search changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm"
          />

          {/* Modal */}
          <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] p-4 pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -20 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              className="w-full max-w-lg bg-zinc-950/80 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl pointer-events-auto overflow-hidden flex flex-col max-h-[60vh] relative"
              onKeyDown={handleKeyDown}
            >
              {/* Background Effects */}
              <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none z-0" />
              <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-blue-500/[0.05] to-transparent pointer-events-none z-0" />

              {/* Content */}
              <div className="relative z-10 flex flex-col h-full bg-transparent">
                {/* Search Input */}
                <div className="flex items-center border-b border-white/10 shrink-0 bg-white/[0.02]">
                  <div className="flex items-center justify-center w-12 h-12">
                    <Search
                      size={16}
                      strokeWidth={1.5}
                      className="text-zinc-500"
                      aria-hidden="true"
                    />
                  </div>
                  <input
                    ref={inputRef}
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex-1 h-12 pr-4 text-[14px] text-zinc-200 bg-transparent border-none outline-none placeholder:text-zinc-600"
                    placeholder="Type a command or search..."
                    aria-label="Search commands"
                    data-testid="command-palette-input"
                  />
                  <div className="flex items-center gap-1.5 pr-3">
                    <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500 bg-[#18181b] border border-[#27272a] rounded">
                      <span className="text-xs">⌘</span>K
                    </kbd>
                    <button
                      onClick={onClose}
                      className="sm:hidden p-1 text-zinc-500"
                    >
                      <span className="sr-only">Close</span>
                      <span className="text-xs">ESC</span>
                    </button>
                  </div>
                </div>

                {/* Commands List */}
                <div className="flex-1 overflow-y-auto scrollbar-hide p-2 bg-transparent">
                  {Object.entries(groupedCommands).length > 0 ? (
                    Object.entries(groupedCommands).map(([category, items]) => (
                      <div key={category} className="mb-2">
                        <p className="px-2 py-1.5 text-[10px] font-medium uppercase tracking-widest text-zinc-500/80">
                          {category}
                        </p>
                        <div className="space-y-0.5">
                          {items.map((cmd) => {
                            const globalIndex = filteredCommands.findIndex(
                              (c) => c.id === cmd.id,
                            );
                            const isSelected = globalIndex === selectedIndex;

                            return (
                              <button
                                key={cmd.id}
                                className={`
                                                                    flex w-full items-center gap-3 px-3 py-2.5 rounded-lg text-left
                                                                    transition-all duration-150
                                                                    ${
                                                                      isSelected
                                                                        ? "bg-white/10 text-zinc-100 backdrop-blur-sm"
                                                                        : "hover:bg-white/5 text-zinc-400"
                                                                    }
                                                                `}
                                onClick={cmd.action}
                                onMouseEnter={() =>
                                  setSelectedIndex(globalIndex)
                                }
                                data-testid={`command-item-${cmd.id}`}
                              >
                                <div
                                  className={`
                                                                        flex h-5 w-5 items-center justify-center shrink-0
                                                                        ${
                                                                          isSelected
                                                                            ? "text-zinc-100"
                                                                            : "text-zinc-500"
                                                                        }
                                                                    `}
                                >
                                  {cmd.icon}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <p
                                    className={`text-[13px] font-medium truncate ${isSelected ? "text-zinc-100" : "text-zinc-300"}`}
                                  >
                                    {cmd.label}
                                  </p>
                                  {cmd.description && (
                                    <p
                                      className={`text-[11px] truncate mt-0.5 ${isSelected ? "text-zinc-400" : "text-zinc-500"}`}
                                    >
                                      {cmd.description}
                                    </p>
                                  )}
                                </div>
                                {cmd.shortcut && (
                                  <kbd
                                    className={`
                                                                        px-1.5 py-0.5 text-[10px] font-medium rounded border
                                                                        ${
                                                                          isSelected
                                                                            ? "bg-black/40 border-black/20 text-zinc-300"
                                                                            : "bg-[#18181b] border-[#27272a] text-zinc-500"
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
                    <div className="py-12 text-center">
                      <p className="text-[13px] text-zinc-400 font-medium">
                        No results found.
                      </p>
                      <p className="text-[11px] text-zinc-600 mt-1">
                        Try a different search term.
                      </p>
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between px-3 py-2 border-t border-white/10 bg-white/[0.02] shrink-0">
                  <span className="text-[10px] text-zinc-500 flex items-center gap-2">
                    <span className="flex items-center gap-1">
                      <kbd className="bg-[#18181b] border border-[#27272a] px-1 rounded text-[9px] min-w-[16px] text-center">
                        ↑
                      </kbd>
                      <kbd className="bg-[#18181b] border border-[#27272a] px-1 rounded text-[9px] min-w-[16px] text-center">
                        ↓
                      </kbd>
                      <span className="ml-1">Navigate</span>
                    </span>
                    <span className="w-px h-3 bg-[#27272a] mx-1"></span>
                    <span className="flex items-center gap-1">
                      <kbd className="bg-[#18181b] border border-[#27272a] px-1 rounded text-[9px] min-w-[16px] text-center">
                        ↵
                      </kbd>
                      <span className="ml-1">Select</span>
                    </span>
                  </span>
                  <span className="text-[10px] text-zinc-600 font-medium">
                    MediLens
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}

export default CommandPalette;
