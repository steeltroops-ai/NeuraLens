/**
 * Simplified NeuraLens API Client
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// Standard API Response Interface
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

// Simple API Client
class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = `${API_BASE_URL}${API_VERSION}`;
  }

  async post<T>(endpoint: string, data: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const result: ApiResponse<T> = await response.json();

    if (!result.success) {
      throw new Error(result.error?.message || 'API request failed');
    }

    return result.data!;
  }

  async get<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`);

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const result: ApiResponse<T> = await response.json();

    if (!result.success) {
      throw new Error(result.error?.message || 'API request failed');
    }

    return result.data!;
  }

  async uploadFile<T>(endpoint: string, file: File): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const result: ApiResponse<T> = await response.json();

    if (!result.success) {
      throw new Error(result.error?.message || 'API request failed');
    }

    return result.data!;
  }
}

export const apiClient = new ApiClient();

constructor() {
  this.baseUrl = `${API_BASE_URL}${API_VERSION}`;
  this.defaultHeaders = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };
}

  /**
   * Make HTTP request with error handling and retries
   */
  private async makeRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  config: RequestConfig = {},
): Promise < ApiResponse < T >> {
  const { timeout = API_TIMEOUT, retries = 2, headers = {} } = config;

  const url = `${this.baseUrl}${endpoint}`;
  const requestHeaders = { ...this.defaultHeaders, ...headers };

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  const requestOptions: RequestInit = {
    ...options,
    headers: requestHeaders,
    signal: controller.signal,
  };

  let lastError: Error = new Error('Unknown error occurred');

  // Retry logic
  for(let attempt = 0; attempt <= retries; attempt++) {
  try {
    const response = await fetch(url, requestOptions);
    clearTimeout(timeoutId);

    // Handle HTTP errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.error?.code || `HTTP_${response.status}`,
        errorData.error?.message || response.statusText,
        response.status,
        errorData.error?.details,
      );
    }

    // Parse response
    const data = await response.json();

    // Return standardized response
    return {
      success: true,
      data: data.data || data,
      metadata: {
        timestamp: new Date().toISOString(),
        processing_time: data.processing_time,
        request_id: data.request_id,
      },
    };
  } catch (error) {
    lastError = error as Error;

    // Don't retry on client errors (4xx)
    if (error instanceof ApiError && error.status && error.status < 500) {
      break;
    }

    // Wait before retry (exponential backoff)
    if (attempt < retries) {
      await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
    }
  }
}

clearTimeout(timeoutId);

// Return error response
return {
  success: false,
  error: {
    code: lastError instanceof ApiError ? lastError.code : 'NETWORK_ERROR',
    message: lastError.message,
    details: lastError instanceof ApiError ? lastError.details : undefined,
  },
  metadata: {
    timestamp: new Date().toISOString(),
  },
};
  }

  /**
   * GET request
   */
  async get<T>(endpoint: string, config ?: RequestConfig): Promise < ApiResponse < T >> {
  return this.makeRequest<T>(endpoint, { method: 'GET' }, config);
}

  /**
   * POST request
   */
  async post<T>(endpoint: string, data ?: any, config ?: RequestConfig): Promise < ApiResponse < T >> {
  return this.makeRequest<T>(
    endpoint,
    {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    },
    config,
  );
}

  /**
   * POST request with FormData (for file uploads)
   */
  async postFormData<T>(
  endpoint: string,
  formData: FormData,
  config ?: RequestConfig,
): Promise < ApiResponse < T >> {
  const headers = { ...config?.headers };
  delete headers['Content-Type']; // Let browser set multipart boundary

  return this.makeRequest<T>(
    endpoint,
    {
      method: 'POST',
      body: formData,
    },
    { ...config, headers },
  );
}

  /**
   * PUT request
   */
  async put<T>(endpoint: string, data ?: any, config ?: RequestConfig): Promise < ApiResponse < T >> {
  return this.makeRequest<T>(
    endpoint,
    {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    },
    config,
  );
}

  /**
   * DELETE request
   */
  async delete <T>(endpoint: string, config ?: RequestConfig): Promise < ApiResponse < T >> {
  return this.makeRequest<T>(endpoint, { method: 'DELETE' }, config);
}

  /**
   * Health check
   */
  async healthCheck(): Promise < ApiResponse < any >> {
  return this.get('/health');
}

  /**
   * Get API status
   */
  async getStatus(): Promise < ApiResponse < any >> {
  return this.get('/status');
}
}

// Create singleton instance
export const apiClient = new NeuraLensApiClient();

// Export types
export type { RequestConfig };

// Utility functions
export const isApiError = (error: any): error is ApiError => {
  return error instanceof ApiError;
};

export const handleApiError = (error: ApiResponse<any>): string => {
  if (error.error) {
    return error.error.message || 'An unexpected error occurred';
  }
  return 'Network error occurred';
};

// API endpoints constants
export const API_ENDPOINTS = {
  HEALTH: '/health',
  STATUS: '/status',
  SPEECH: {
    ANALYZE: '/speech/analyze',
    INFO: '/speech/info',
  },
  RETINAL: {
    ANALYZE: '/retinal/analyze',
    INFO: '/retinal/info',
  },
  MOTOR: {
    ANALYZE: '/motor/analyze',
    INFO: '/motor/info',
  },
  COGNITIVE: {
    ANALYZE: '/cognitive/analyze',
    INFO: '/cognitive/info',
  },
  NRI: {
    FUSION: '/nri/fusion',
    INFO: '/nri/info',
  },
  VALIDATION: {
    METRICS: '/validation/metrics',
    INFO: '/validation/info',
  },
} as const;
