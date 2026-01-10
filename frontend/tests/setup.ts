// Test setup configuration for NeuraLens
import '@testing-library/jest-dom';

// Mock Next.js router
const mockJest = typeof jest !== 'undefined' ? jest : { mock: () => {}, fn: () => () => {} };

mockJest.mock('next/router', () => ({
  useRouter() {
    return {
      route: '/',
      pathname: '/',
      query: {},
      asPath: '/',
      push: jest.fn(),
      pop: jest.fn(),
      reload: jest.fn(),
      back: jest.fn(),
      prefetch: jest.fn().mockResolvedValue(undefined),
      beforePopState: jest.fn(),
      events: {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
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
jest.mock('next/navigation', () => ({
  useRouter() {
    return {
      push: jest.fn(),
      replace: jest.fn(),
      prefetch: jest.fn(),
      back: jest.fn(),
      forward: jest.fn(),
      refresh: jest.fn(),
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
jest.mock('@/lib/supabase', () => ({
  supabase: {
    auth: {
      getUser: jest.fn().mockResolvedValue({ data: { user: null }, error: null }),
      signInWithPassword: jest.fn(),
      signUp: jest.fn(),
      signOut: jest.fn(),
      onAuthStateChange: jest.fn(),
    },
    from: jest.fn(() => ({
      select: jest.fn().mockReturnThis(),
      insert: jest.fn().mockReturnThis(),
      update: jest.fn().mockReturnThis(),
      delete: jest.fn().mockReturnThis(),
      eq: jest.fn().mockReturnThis(),
      order: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      single: jest.fn().mockResolvedValue({ data: null, error: null }),
    })),
    storage: {
      from: jest.fn(() => ({
        upload: jest.fn().mockResolvedValue({ data: null, error: null }),
        download: jest.fn().mockResolvedValue({ data: null, error: null }),
        remove: jest.fn().mockResolvedValue({ data: null, error: null }),
        list: jest.fn().mockResolvedValue({ data: [], error: null }),
      })),
    },
  },
}));

// Mock MediaRecorder API
global.MediaRecorder = class MediaRecorder {
  static isTypeSupported = jest.fn().mockReturnValue(true);

  constructor() {
    this.state = 'inactive';
    this.ondataavailable = null;
    this.onstop = null;
    this.onstart = null;
    this.onerror = null;
  }

  start = jest.fn(() => {
    this.state = 'recording';
    if (this.onstart) this.onstart();
  });

  stop = jest.fn(() => {
    this.state = 'inactive';
    if (this.onstop) this.onstop();
  });

  pause = jest.fn(() => {
    this.state = 'paused';
  });

  resume = jest.fn(() => {
    this.state = 'recording';
  });

  requestData = jest.fn();
} as any;

// Mock getUserMedia
Object.defineProperty(navigator, 'mediaDevices', {
  writable: true,
  value: {
    getUserMedia: jest.fn().mockResolvedValue({
      getTracks: () => [
        {
          stop: jest.fn(),
          kind: 'audio',
          enabled: true,
        },
      ],
    }),
  },
});

// Mock AudioContext
global.AudioContext = class AudioContext {
  createAnalyser = jest.fn(() => ({
    fftSize: 256,
    frequencyBinCount: 128,
    getByteFrequencyData: jest.fn(),
  }));

  createMediaStreamSource = jest.fn(() => ({
    connect: jest.fn(),
  }));

  close = jest.fn();
} as any;

// Mock URL.createObjectURL and revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock File API
global.File = class File {
  constructor(bits: any[], name: string, options?: any) {
    this.name = name;
    this.size = bits.reduce((acc, bit) => acc + (bit.length || 0), 0);
    this.type = options?.type || '';
    this.lastModified = Date.now();
  }

  name: string;
  size: number;
  type: string;
  lastModified: number;
} as any;

// Mock Blob
global.Blob = class Blob {
  constructor(parts?: any[], options?: any) {
    this.size = parts?.reduce((acc, part) => acc + (part.length || 0), 0) || 0;
    this.type = options?.type || '';
  }

  size: number;
  type: string;
} as any;

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor() {}
  observe = jest.fn();
  disconnect = jest.fn();
  unobserve = jest.fn();
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  observe = jest.fn();
  disconnect = jest.fn();
  unobserve = jest.fn();
} as any;

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock scrollTo
global.scrollTo = jest.fn();

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  warn: jest.fn(),
  error: jest.fn(),
  log: jest.fn(),
};

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
