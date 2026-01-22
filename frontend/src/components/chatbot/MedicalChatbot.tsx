"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  X,
  Send,
  Minimize2,
  Maximize2,
  RefreshCw,
  AlertCircle,
  Mic,
} from "lucide-react";
import { useChatbot, ChatMessage } from "./useChatbot";
import { Logo } from "@/components/common/Logo";

interface MedicalChatbotProps {
  context?: string;
}

/**
 * MedicalChatbot Component
 *
 * Floating chatbot button that expands to a chat window.
 * Features ChatGPT-inspired floating input section with
 * sleek, modern glassmorphism design.
 */
export function MedicalChatbot({ context }: MedicalChatbotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const {
    messages,
    isLoading,
    error,
    suggestions,
    sendMessage,
    clearChat,
    loadSuggestions,
  } = useChatbot(context);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
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
      const { scrollTop, scrollHeight, clientHeight } =
        messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom && messages.length > 3);
      setIsScrolled(scrollTop > 5);
    }
  }, [messages.length]);

  // Scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Handle send message
  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue("");
    // Reset textarea height
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    await sendMessage(message);
  };

  // Handle suggestion click
  const handleSuggestionClick = (text: string) => {
    sendMessage(text);
  };

  // Toggle chat open/close
  const toggleChat = () => {
    setIsOpen((prev) => !prev);
    if (!isOpen) {
      setIsExpanded(false);
    }
  };

  // Toggle recording (placeholder)
  const toggleRecording = () => {
    setIsRecording((prev) => !prev);
    // TODO: Implement actual voice recording
  };

  // Render message content with formatting
  const renderMessageContent = (content: string) => {
    const lines = content.split("\n");

    return lines.map((line, idx) => {
      let formattedLine = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

      if (line.trim().startsWith("-") || line.trim().startsWith("*")) {
        const bulletContent = line.trim().slice(1).trim();
        formattedLine = `<span class="flex items-start gap-2"><span class="text-cyan-400 mt-1">&#8226;</span><span>${bulletContent.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</span></span>`;
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
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 100)}px`;
  };

  return (
    <>
      {/* Floating Chat Button - Small Rectangle Chat Pill (UNCHANGED) */}
      <button
        onClick={toggleChat}
        className={`
                    fixed bottom-12 right-6 z-[1050]
                    h-[46px] px-5 rounded-xl
                    bg-black border border-zinc-800
                    flex items-center gap-2.5
                    shadow-2xl shadow-black/40
                    transition-all duration-300 cubic-bezier(0.2, 0.8, 0.2, 1)
                    hover:scale-105 hover:shadow-black/60
                    focus:outline-none overflow-hidden
                    ${isOpen ? "scale-0 opacity-0 translate-y-4 pointer-events-none" : "scale-100 opacity-100 translate-y-0"}
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
        <span className="text-white font-bold text-[13px] tracking-wide uppercase">
          Chat
        </span>
      </button>

      {/* Chat Window - Redesigned */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2, ease: [0.25, 0.46, 0.45, 0.94] }}
            className={`
              fixed z-[1050]
                ${
                  isExpanded
                    ? "inset-2 w-auto h-auto md:inset-auto md:bottom-4 md:right-4 md:w-[600px] md:h-[700px] md:max-h-[calc(100vh-2rem)]"
                    : "bottom-4 left-4 right-4 w-auto h-[60vh] md:left-auto md:bottom-12 md:right-6 md:w-[380px] md:h-[480px]"
                }
              flex flex-col
              bg-zinc-950
              border border-zinc-800
              rounded-2xl 
              shadow-[0_8px_40px_rgba(0,0,0,0.5),0_2px_8px_rgba(0,0,0,0.3)] hover:shadow-[0_12px_50px_rgba(0,0,0,0.6),0_4px_12px_rgba(0,0,0,0.35)]
              transition-shadow duration-300
              overflow-hidden
            `}
          >
            {/* Subtle grid pattern like homepage */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />
            {/* Subtle gradient accent at top - red like logo */}
            <div className="absolute top-0 left-0 right-0 h-20 bg-gradient-to-b from-red-500/[0.04] to-transparent pointer-events-none" />

            {/* Header - Transparent to Black Rounded on Scroll */}
            <div
              className={`
                relative z-10 flex items-center justify-between px-4 h-14 shrink-0 
                transition-all duration-300 ease-in-out
                ${
                  isScrolled
                    ? "bg-zinc-950/90 backdrop-blur-md border-b border-zinc-800 rounded-b-xl shadow-lg"
                    : "bg-transparent"
                }
              `}
            >
              <div className="flex items-center gap-2">
                <Logo size="sm" showText={false} />
                <h3 className="text-[13px] font-semibold text-zinc-200 tracking-wide uppercase">
                  MediLens AI
                </h3>
              </div>

              <div className="flex items-center gap-[5px]">
                <button
                  onClick={clearChat}
                  className="w-8 h-8 !min-w-0 !min-h-0 flex items-center justify-center text-zinc-500 hover:text-zinc-200 transition-colors"
                  title="New chat"
                >
                  <RefreshCw size={14} strokeWidth={2} />
                </button>
                <button
                  onClick={() => setIsExpanded((prev) => !prev)}
                  className="w-8 h-8 !min-w-0 !min-h-0 hidden md:flex items-center justify-center text-zinc-500 hover:text-zinc-200 transition-colors"
                  title={isExpanded ? "Minimize" : "Expand"}
                >
                  {isExpanded ? (
                    <Minimize2 size={14} strokeWidth={2} />
                  ) : (
                    <Maximize2 size={14} strokeWidth={2} />
                  )}
                </button>
                <button
                  onClick={toggleChat}
                  className="w-8 h-8 !min-w-0 !min-h-0 flex items-center justify-center text-zinc-500 hover:text-red-400 transition-colors"
                  title="Close"
                >
                  <X size={16} strokeWidth={2} />
                </button>
              </div>
            </div>

            {/* Body - Main chat area */}
            <div className="flex-1 flex flex-col overflow-hidden relative z-10">
              <div
                ref={messagesContainerRef}
                onScroll={handleScroll}
                className="flex-1 overflow-y-auto px-3 py-3 pb-20 space-y-2.5 scrollbar-none [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
              >
                {/* Empty State */}
                {messages.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-center px-4 animate-in fade-in duration-500">
                    <h4 className="text-[18px] font-medium text-white mb-2">
                      What's on your mind?
                    </h4>
                    <p className="text-[13px] text-zinc-500 mb-8 max-w-[260px] leading-relaxed">
                      Ask about your health, reports, or get medical guidance.
                    </p>

                    <div className="grid grid-cols-1 gap-2 w-full max-w-[280px]">
                      {suggestions.slice(0, 3).map((suggestion, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSuggestionClick(suggestion.text)}
                          className="
                            w-full px-4 py-3
                            text-left text-[12px] font-medium text-zinc-400 
                            bg-zinc-900 border border-zinc-800
                            rounded-xl
                            hover:bg-zinc-800 hover:text-zinc-200 hover:border-zinc-700
                            active:bg-zinc-700
                            transition-all duration-150
                          "
                        >
                          {suggestion.text}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Messages */}
                {messages.map((message, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`
                        max-w-[85%] px-4 py-2.5 text-[13px] leading-relaxed
                        ${
                          message.role === "user"
                            ? "bg-red-500 text-white rounded-2xl rounded-tr-md shadow-md"
                            : "bg-zinc-900 text-zinc-200 rounded-2xl rounded-tl-md border border-zinc-800 shadow-sm"
                        }
                      `}
                    >
                      <div
                        className={
                          message.role === "assistant" ? "markdown-content" : ""
                        }
                      >
                        {renderMessageContent(message.content)}
                      </div>
                    </div>
                  </motion.div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-zinc-900 px-4 py-3 rounded-2xl rounded-tl-md border border-zinc-800 shadow-sm">
                      <div className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-[bounce_1s_infinite_0ms]" />
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-[bounce_1s_infinite_150ms]" />
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-[bounce_1s_infinite_300ms]" />
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Error message */}
                {error && (
                  <div className="flex justify-center">
                    <div className="flex items-center gap-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-xl text-[12px] text-red-400">
                      <AlertCircle size={14} />
                      <span>{error}</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Floating Input Section - Dark Theme with White Input */}
              <motion.div
                initial={{ y: 6, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1, duration: 0.2 }}
                className="absolute bottom-0 left-0 right-0 px-3 pb-3 pt-4 bg-gradient-to-t from-zinc-950 via-zinc-950/98 to-transparent z-20"
              >
                <div className="relative bg-zinc-900 rounded-full border border-zinc-800 shadow-lg hover:shadow-[0_0_15px_rgba(255,255,255,0.15)] transition-all duration-300">
                  <div className="flex items-center px-3 h-10 gap-0">
                    {/* Input */}
                    <textarea
                      ref={inputRef}
                      value={inputValue}
                      onChange={handleInput}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                        }
                      }}
                      placeholder="Ask anything..."
                      disabled={isLoading}
                      rows={1}
                      className="
                        flex-1 py-0 min-h-[18px] max-h-[50px]
                        bg-transparent border-none
                        text-[12px] text-zinc-200 placeholder-zinc-500
                        resize-none outline-none
                        leading-tight
                        mr-2
                      "
                    />

                    <div className="flex items-center gap-[5px] shrink-0">
                      {/* Mic button */}
                      <button
                        onClick={toggleRecording}
                        className={`
                          w-7 h-7 !min-w-0 !min-h-0 flex items-center justify-center rounded-full transition-colors
                          ${
                            isRecording
                              ? "text-red-500"
                              : "text-zinc-500 hover:text-zinc-300"
                          }
                        `}
                        title={isRecording ? "Stop recording" : "Voice input"}
                      >
                        <Mic size={14} strokeWidth={1.5} />
                      </button>

                      {/* Send button */}
                      <button
                        onClick={handleSend}
                        disabled={!inputValue.trim() || isLoading}
                        className={`
                          w-7 h-7 !min-w-0 !min-h-0 rounded-full
                          flex items-center justify-center
                          transition-all duration-150
                          ${
                            inputValue.trim() && !isLoading
                              ? "text-red-500 hover:text-red-400"
                              : "text-zinc-600 cursor-not-allowed"
                          }
                        `}
                      >
                        <Send
                          size={12}
                          strokeWidth={2.5}
                          className={
                            inputValue.trim() && !isLoading ? "ml-0.5" : ""
                          }
                        />
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

export default MedicalChatbot;
