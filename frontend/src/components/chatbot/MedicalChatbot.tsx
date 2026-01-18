'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    MessageSquare,
    X,
    Send,
    Minimize2,
    Maximize2,
    Sparkles,
    RefreshCw,
    AlertCircle,
    ChevronDown,
    Bot,
    User,
    ArrowUp
} from 'lucide-react';
import { useChatbot, ChatMessage } from './useChatbot';

interface MedicalChatbotProps {
    context?: string;
}

/**
 * MedicalChatbot Component
 * 
 * Floating chatbot button that expands to a chat window.
 * Follows the MediLens sidebar design philosophy with
 * dark theme, glassmorphism, and smooth animations.
 */
export function MedicalChatbot({ context }: MedicalChatbotProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [inputValue, setInputValue] = useState('');
    const [showScrollButton, setShowScrollButton] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messagesContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const {
        messages,
        isLoading,
        error,
        suggestions,
        sendMessage,
        clearChat,
        loadSuggestions
    } = useChatbot(context);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    // Load suggestions when context changes
    useEffect(() => {
        if (isOpen) {
            loadSuggestions();
        }
    }, [isOpen, context, loadSuggestions]);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen && inputRef.current) {
            setTimeout(() => inputRef.current?.focus(), 300);
        }
    }, [isOpen]);

    // Handle scroll to show/hide scroll button
    const handleScroll = useCallback(() => {
        if (messagesContainerRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
            const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
            setShowScrollButton(!isNearBottom && messages.length > 3);
        }
    }, [messages.length]);

    // Scroll to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Handle send message
    const handleSend = async () => {
        if (!inputValue.trim() || isLoading) return;

        const message = inputValue.trim();
        setInputValue('');
        await sendMessage(message);
    };

    // Handle key press
    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Handle suggestion click
    const handleSuggestionClick = (text: string) => {
        sendMessage(text);
    };

    // Toggle chat open/close
    const toggleChat = () => {
        setIsOpen(prev => !prev);
        if (!isOpen) {
            setIsExpanded(false);
        }
    };

    // Render message content with formatting
    const renderMessageContent = (content: string) => {
        // Simple markdown-like parsing for bold and bullet points
        const lines = content.split('\n');

        return lines.map((line, idx) => {
            // Handle bold text
            let formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            // Handle bullet points
            if (line.trim().startsWith('-') || line.trim().startsWith('*')) {
                const bulletContent = line.trim().slice(1).trim();
                formattedLine = `<span class="flex items-start gap-2"><span class="text-blue-400 mt-1">&#8226;</span><span>${bulletContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</span></span>`;
            }

            return (
                <span
                    key={idx}
                    dangerouslySetInnerHTML={{ __html: formattedLine }}
                    className="block"
                />
            );
        });
    };

    // Auto-resize textarea
    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInputValue(e.target.value);
        e.target.style.height = 'auto';
        e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
    };

    return (
        <>
            {/* Floating Chat Button - Small Rectangle Chat Pill */}
            <button
                onClick={toggleChat}
                className={`
                    fixed bottom-6 right-6 z-[1050]
                    h-[46px] px-5 rounded-xl
                    bg-black border border-zinc-800
                    flex items-center gap-2.5
                    shadow-2xl shadow-black/40
                    transition-all duration-300 cubic-bezier(0.2, 0.8, 0.2, 1)
                    hover:scale-105 hover:shadow-black/60
                    focus:outline-none overflow-hidden
                    ${isOpen ? 'scale-0 opacity-0 translate-y-4 pointer-events-none' : 'scale-100 opacity-100 translate-y-0'}
                `}
                aria-label="Open medical assistant"
                aria-expanded={isOpen}
            >
                <div className="relative flex items-center justify-center">
                    <MessageSquare size={16} className="text-white fill-current" />
                    <span className="absolute -top-0.5 -right-0.5 flex h-1.5 w-1.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500 border border-black"></span>
                    </span>
                </div>
                <span className="text-white font-bold text-[13px] tracking-wide uppercase">Chat</span>
            </button>

            {/* Chat Window */}
            <div
                className={`
                    fixed z-[1050] transition-all duration-300 cubic-bezier(0.2, 0.8, 0.2, 1)
                    ${isExpanded
                        ? 'bottom-6 right-6 w-[calc(100vw-3rem)] h-[calc(100vh-3rem)] md:w-[600px] md:h-[800px] max-h-[calc(100vh-3rem)]'
                        : 'bottom-6 right-6 w-[360px] h-[550px]'
                    }
                    ${isOpen
                        ? 'opacity-100 translate-y-0 pointer-events-auto'
                        : 'opacity-0 translate-y-8 pointer-events-none'
                    }
                    flex flex-col
                    bg-black border border-zinc-900
                    rounded-2xl shadow-2xl shadow-black/50
                    overflow-hidden
                `}
            >
                {/* Header - Minimal Black */}
                <div className="flex items-center justify-between px-4 py-3 bg-black border-b border-zinc-900 shrink-0">
                    <div className="flex items-center gap-2.5">
                        <Sparkles size={14} className="text-white" />
                        <h3 className="text-[14px] font-bold text-white tracking-wide">MediLens AI</h3>
                    </div>

                    <div className="flex items-center gap-1">
                        <button
                            onClick={clearChat}
                            className="p-1.5 rounded-md hover:bg-zinc-900 text-zinc-500 hover:text-white transition-colors"
                            title="Reset"
                        >
                            <RefreshCw size={13} strokeWidth={2} />
                        </button>
                        <button
                            onClick={() => setIsExpanded(prev => !prev)}
                            className="p-1.5 rounded-md hover:bg-zinc-900 text-zinc-500 hover:text-white transition-colors hidden md:block"
                            title={isExpanded ? "Minimize" : "Expand"}
                        >
                            {isExpanded ? <Minimize2 size={13} strokeWidth={2} /> : <Maximize2 size={13} strokeWidth={2} />}
                        </button>
                        <button
                            onClick={toggleChat}
                            className="p-1.5 rounded-md hover:bg-zinc-900 text-zinc-500 hover:text-white transition-colors"
                            title="Close"
                        >
                            <X size={15} strokeWidth={2} />
                        </button>
                    </div>
                </div>

                {/* Body - White */}
                <div className="flex-1 bg-white flex flex-col overflow-hidden relative">
                    <div
                        ref={messagesContainerRef}
                        onScroll={handleScroll}
                        className="flex-1 overflow-y-auto p-4 space-y-4 bg-white scrollbar-thin scrollbar-thumb-zinc-400 scrollbar-track-transparent"
                    >
                        {/* Empty State */}
                        {messages.length === 0 && (
                            <div className="flex flex-col items-center justify-center h-full text-center px-6 animate-in fade-in duration-500">
                                <h4 className="text-[16px] font-bold text-black mb-1.5">
                                    How can I help?
                                </h4>
                                <p className="text-[13px] text-zinc-500 mb-6 max-w-[240px]">
                                    Ask about your health or reports.
                                </p>

                                <div className="grid grid-cols-1 gap-2 w-full max-w-[240px]">
                                    {suggestions.slice(0, 3).map((suggestion, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() => handleSuggestionClick(suggestion.text)}
                                            className="
                                                w-full px-3 py-2.5
                                                text-left text-[12px] font-medium text-zinc-600 
                                                bg-zinc-50 border border-zinc-200
                                                rounded-lg
                                                hover:bg-black hover:text-white hover:border-black
                                                transition-colors duration-200
                                            "
                                        >
                                            {suggestion.text}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {messages.map((message, idx) => (
                            <div
                                key={idx}
                                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2 duration-300`}
                            >
                                <div className={`
                                    max-w-[85%] px-3.5 py-2.5 rounded-xl text-[13px] leading-relaxed
                                    ${message.role === 'user'
                                        ? 'bg-black text-white rounded-tr-sm'
                                        : 'bg-zinc-100 text-zinc-900 rounded-tl-sm'
                                    }
                                `}>
                                    <div className={message.role === 'assistant' ? 'markdown-content' : ''}>
                                        {renderMessageContent(message.content)}
                                    </div>
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2">
                                <div className="bg-zinc-100 px-3 py-2.5 rounded-xl rounded-tl-sm">
                                    <div className="flex items-center gap-1">
                                        <span className="w-1 h-1 rounded-full bg-zinc-500 animate-[bounce_1s_infinite_0ms]" />
                                        <span className="w-1 h-1 rounded-full bg-zinc-500 animate-[bounce_1s_infinite_200ms]" />
                                        <span className="w-1 h-1 rounded-full bg-zinc-500 animate-[bounce_1s_infinite_400ms]" />
                                    </div>
                                </div>
                            </div>
                        )}

                        {error && (
                            <div className="flex justify-center">
                                <div className="flex items-center gap-2 px-3 py-1.5 bg-red-50 border border-red-100 rounded-lg text-[12px] text-red-600">
                                    <AlertCircle size={13} />
                                    <span>{error}</span>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* Footer - Black Input Area */}
                <div className="p-3 bg-black border-t border-zinc-900 shrink-0">
                    <div className="relative flex items-end gap-2 p-1 bg-zinc-900 rounded-xl border border-zinc-800 focus-within:border-zinc-700 transition-colors duration-200">
                        <textarea
                            ref={inputRef as any}
                            value={inputValue}
                            onChange={handleInput}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            placeholder="Type a message..."
                            disabled={isLoading}
                            rows={1}
                            className="
                                flex-1 px-3 py-2.5 min-h-[40px] max-h-32
                                bg-transparent border-none
                                text-[13px] text-zinc-100 placeholder-zinc-500
                                resize-none outline-none
                                font-medium scrollbar-hide
                            "
                        />
                        <button
                            onClick={handleSend}
                            disabled={!inputValue.trim() || isLoading}
                            className={`
                                p-2 rounded-lg
                                flex items-center justify-center
                                transition-all duration-200
                                ${inputValue.trim() && !isLoading
                                    ? 'bg-white text-black hover:scale-105 active:scale-95'
                                    : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                                }
                            `}
                        >
                            <Send size={16} strokeWidth={2.5} className={inputValue.trim() ? 'ml-0.5' : ''} />
                        </button>
                    </div>
                </div>
            </div>
        </>
    );

}

export default MedicalChatbot;
