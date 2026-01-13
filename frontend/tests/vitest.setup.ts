// Vitest setup configuration for NeuraLens
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock Next.js router
vi.mock('next/router', () => ({
    useRouter() {
        return {
            route: '/',
            pathname: '/',
            query: {},
            asPath: '/',
            push: vi.fn(),
            pop: vi.fn(),
            reload: vi.fn(),
            back: vi.fn(),
            prefetch: vi.fn().mockResolvedValue(undefined),
            beforePopState: vi.fn(),
            events: {
                on: vi.fn(),
                off: vi.fn(),
                emit: vi.fn(),
            },
            isFallback: false,
            isLocaleDomain: false,
            isReady: true,
            defaultLocale: 'en',
            domainLocales: [],
            isPreview: false,
        };
    },
}));

// Mock Next.js navigation
vi.mock('next/navigation', () => ({
    useRouter() {
        return {
            push: vi.fn(),
            replace: vi.fn(),
            prefetch: vi.fn(),
            back: vi.fn(),
            forward: vi.fn(),
            refresh: vi.fn(),
        };
    },
    useSearchParams() {
        return new URLSearchParams();
    },
    usePathname() {
        return '/';
    },
}));

// Mock Supabase client
vi.mock('@/lib/supabase', () => ({
    supabase: {
        auth: {
            getUser: vi.fn().mockResolvedValue({ data: { user: null }, error: null }),
            signInWithPassword: vi.fn(),
            signUp: vi.fn(),
            signOut: vi.fn(),
            onAuthStateChange: vi.fn(),
        },
        from: vi.fn(() => ({
            select: vi.fn().mockReturnThis(),
            insert: vi.fn().mockReturnThis(),
            update: vi.fn().mockReturnThis(),
            delete: vi.fn().mockReturnThis(),
            eq: vi.fn().mockReturnThis(),
            order: vi.fn().mockReturnThis(),
            limit: vi.fn().mockReturnThis(),
            single: vi.fn().mockResolvedValue({ data: null, error: null }),
        })),
        storage: {
            from: vi.fn(() => ({
                upload: vi.fn().mockResolvedValue({ data: null, error: null }),
                download: vi.fn().mockResolvedValue({ data: null, error: null }),
                remove: vi.fn().mockResolvedValue({ data: null, error: null }),
                list: vi.fn().mockResolvedValue({ data: [], error: null }),
            })),
        },
    },
}));

// Mock MediaRecorder API
global.MediaRecorder = class MediaRecorder {
    static isTypeSupported = vi.fn().mockReturnValue(true);
    state: string = 'inactive';
    ondataavailable: ((event: any) => void) | null = null;
    onstop: (() => void) | null = null;
    onstart: (() => void) | null = null;
    onerror: ((event: any) => void) | null = null;

    start = vi.fn(() => {
        this.state = 'recording';
        if (this.onstart) this.onstart();
    });

    stop = vi.fn(() => {
        this.state = 'inactive';
        if (this.onstop) this.onstop();
    });

    pause = vi.fn(() => {
        this.state = 'paused';
    });

    resume = vi.fn(() => {
        this.state = 'recording';
    });

    requestData = vi.fn();
} as any;

// Mock getUserMedia
Object.defineProperty(navigator, 'mediaDevices', {
    writable: true,
    value: {
        getUserMedia: vi.fn().mockResolvedValue({
            getTracks: () => [
                {
                    stop: vi.fn(),
                    kind: 'audio',
                    enabled: true,
                },
            ],
        }),
    },
});

// Mock AudioContext
global.AudioContext = class AudioContext {
    createAnalyser = vi.fn(() => ({
        fftSize: 256,
        frequencyBinCount: 128,
        getByteFrequencyData: vi.fn(),
    }));

    createMediaStreamSource = vi.fn(() => ({
        connect: vi.fn(),
    }));

    close = vi.fn();
} as any;

// Mock URL.createObjectURL and revokeObjectURL
global.URL.createObjectURL = vi.fn(() => 'mock-url');
global.URL.revokeObjectURL = vi.fn();

// Mock File API
global.File = class File {
    name: string;
    size: number;
    type: string;
    lastModified: number;

    constructor(bits: any[], name: string, options?: any) {
        this.name = name;
        this.size = bits.reduce((acc, bit) => acc + (bit.length || 0), 0);
        this.type = options?.type || '';
        this.lastModified = Date.now();
    }
} as any;

// Mock Blob
global.Blob = class Blob {
    size: number;
    type: string;

    constructor(parts?: any[], options?: any) {
        this.size = parts?.reduce((acc, part) => acc + (part.length || 0), 0) || 0;
        this.type = options?.type || '';
    }
} as any;

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
    constructor() { }
    observe = vi.fn();
    disconnect = vi.fn();
    unobserve = vi.fn();
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
    constructor() { }
    observe = vi.fn();
    disconnect = vi.fn();
    unobserve = vi.fn();
} as any;

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
    })),
});

// Mock scrollTo
global.scrollTo = vi.fn();

// Set up test environment variables
process.env.NODE_ENV = 'test';
process.env.NEXT_PUBLIC_SUPABASE_URL = 'http://localhost:54321';
process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'test-anon-key';

// Global test utilities
export const mockUser = {
    id: '550e8400-e29b-41d4-a716-446655440001',
    email: 'test@example.com',
    username: 'testuser',
    age: 65,
    sex: 'male',
    education_years: 16,
};

export const mockAssessmentSession = {
    id: '660e8400-e29b-41d4-a716-446655440001',
    user_id: mockUser.id,
    session_type: 'screening',
    status: 'completed',
    overall_risk_score: 0.25,
    risk_category: 'low',
    confidence_score: 0.92,
};

export const mockSpeechAssessment = {
    id: '770e8400-e29b-41d4-a716-446655440001',
    session_id: mockAssessmentSession.id,
    fluency_score: 0.85,
    articulation_score: 0.9,
    risk_score: 0.15,
    confidence: 0.92,
    transcription: 'Test transcription',
};

export const mockRetinalAssessment = {
    id: '880e8400-e29b-41d4-a716-446655440001',
    session_id: mockAssessmentSession.id,
    vessel_tortuosity: 0.35,
    av_ratio: 0.72,
    cup_disc_ratio: 0.28,
    vessel_density: 0.65,
    risk_score: 0.25,
    confidence: 0.91,
};

export const mockMotorAssessment = {
    id: '990e8400-e29b-41d4-a716-446655440001',
    session_id: mockAssessmentSession.id,
    tap_frequency: 4.2,
    rhythm_consistency: 0.91,
    tremor_score: 0.08,
    coordination_score: 0.92,
    risk_score: 0.12,
    confidence: 0.95,
};
