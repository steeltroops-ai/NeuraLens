'use client';

import { useState, useCallback, useEffect } from 'react';
import { apiClient, ApiResponse } from '@/lib/api/client';

// Types
export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: string;
}

export interface QuickReply {
    text: string;
    icon?: string;
}

interface ChatResponse {
    message: string;
    session_id: string;
    tokens_used?: number;
    processing_time: number;
    confidence?: number;
    sources?: string[];
    disclaimer: string;
}

interface SuggestionsResponse {
    questions: QuickReply[];
    context?: string;
}

// Generate unique session ID
const generateSessionId = (): string => {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
};

// Local storage key for session
const SESSION_KEY = 'medilens-chatbot-session';
const MESSAGES_KEY = 'medilens-chatbot-messages';

/**
 * useChatbot Hook
 * 
 * Custom hook for managing chatbot state, messages, and API communication.
 * Provides persistence through localStorage and handles streaming-like UX.
 */
export function useChatbot(context?: string) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [suggestions, setSuggestions] = useState<QuickReply[]>([]);
    
    // Initialize session on mount
    useEffect(() => {
        // Load session from localStorage
        const savedSession = localStorage.getItem(SESSION_KEY);
        const savedMessages = localStorage.getItem(MESSAGES_KEY);
        
        if (savedSession) {
            setSessionId(savedSession);
        } else {
            const newSession = generateSessionId();
            setSessionId(newSession);
            localStorage.setItem(SESSION_KEY, newSession);
        }
        
        if (savedMessages) {
            try {
                const parsed = JSON.parse(savedMessages);
                if (Array.isArray(parsed)) {
                    setMessages(parsed);
                }
            } catch (e) {
                console.error('Failed to parse saved messages:', e);
            }
        }
    }, []);
    
    // Save messages to localStorage whenever they change
    useEffect(() => {
        if (messages.length > 0) {
            localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));
        }
    }, [messages]);
    
    // Load suggestions
    const loadSuggestions = useCallback(async () => {
        try {
            const response = await apiClient.get<SuggestionsResponse>(
                `/chatbot/suggestions${context ? `?context=${encodeURIComponent(context)}` : ''}`
            );
            
            if (response.success && response.data?.questions) {
                setSuggestions(response.data.questions);
            }
        } catch (e) {
            console.error('Failed to load suggestions:', e);
            // Set default suggestions
            setSuggestions([
                { text: 'What can you help me with?', icon: 'help' },
                { text: 'Explain my results', icon: 'chart' },
                { text: 'Health tips', icon: 'heart' },
                { text: 'When to see a doctor?', icon: 'user-plus' },
            ]);
        }
    }, [context]);
    
    // Send message to chatbot
    const sendMessage = useCallback(async (content: string) => {
        if (!content.trim()) return;
        
        setError(null);
        setIsLoading(true);
        
        // Add user message immediately
        const userMessage: ChatMessage = {
            role: 'user',
            content: content.trim(),
            timestamp: new Date().toISOString(),
        };
        
        setMessages(prev => [...prev, userMessage]);
        
        try {
            // Build conversation history (last 10 messages for context)
            const history = messages.slice(-10).map(msg => ({
                role: msg.role,
                content: msg.content,
            }));
            
            const response = await apiClient.post<ChatResponse>('/chatbot/chat', {
                message: content.trim(),
                conversation_history: history,
                session_id: sessionId,
                context: context,
            });
            
            if (response.success && response.data) {
                const assistantMessage: ChatMessage = {
                    role: 'assistant',
                    content: response.data.message,
                    timestamp: new Date().toISOString(),
                };
                
                setMessages(prev => [...prev, assistantMessage]);
                
                // Update session ID if returned
                if (response.data.session_id && response.data.session_id !== sessionId) {
                    setSessionId(response.data.session_id);
                    localStorage.setItem(SESSION_KEY, response.data.session_id);
                }
            } else {
                throw new Error(response.error?.message || 'Failed to get response');
            }
        } catch (e) {
            const errorMessage = e instanceof Error ? e.message : 'An error occurred';
            setError(errorMessage);
            
            // Add error message as assistant response
            const errorResponse: ChatMessage = {
                role: 'assistant',
                content: "I apologize, but I'm having trouble connecting right now. Please try again in a moment.",
                timestamp: new Date().toISOString(),
            };
            setMessages(prev => [...prev, errorResponse]);
        } finally {
            setIsLoading(false);
        }
    }, [messages, sessionId, context]);
    
    // Clear chat history
    const clearChat = useCallback(() => {
        setMessages([]);
        setError(null);
        localStorage.removeItem(MESSAGES_KEY);
        
        // Generate new session
        const newSession = generateSessionId();
        setSessionId(newSession);
        localStorage.setItem(SESSION_KEY, newSession);
        
        // Reload suggestions
        loadSuggestions();
    }, [loadSuggestions]);
    
    // Get chatbot info
    const getChatbotInfo = useCallback(async () => {
        try {
            const response = await apiClient.get('/chatbot/info');
            return response.data;
        } catch (e) {
            console.error('Failed to get chatbot info:', e);
            return null;
        }
    }, []);
    
    return {
        messages,
        isLoading,
        error,
        sessionId,
        suggestions,
        sendMessage,
        clearChat,
        loadSuggestions,
        getChatbotInfo,
    };
}

export default useChatbot;
