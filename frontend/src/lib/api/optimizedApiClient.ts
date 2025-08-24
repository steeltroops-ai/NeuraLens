/**
 * Optimized API Client
 * High-performance API client with caching, compression, and monitoring
 */

import { AssessmentResults, AssessmentInput } from '@/lib/assessment/workflow';

// Performance monitoring interface
export interface ApiPerformanceMetrics {
  endpoint: string;
  method: string;
  responseTime: number;
  cacheHit: boolean;
  payloadSize: number;
  timestamp: string;
  statusCode: number;
  error?: string;
}

// API configuration
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  compressionEnabled: boolean;
  cacheTTL: number;
  rateLimitPerMinute: number;
}

// Cache entry interface
interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  etag?: string;
}

// Optimized API client class
export class OptimizedApiClient {
  private config: ApiConfig;
  private cache: Map<string, CacheEntry<any>> = new Map();
  private performanceMetrics: ApiPerformanceMetrics[] = [];
  private requestQueue: Map<string, Promise<any>> = new Map();
  private rateLimitTracker: Map<string, number[]> = new Map();

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = {
      baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001',
      timeout: 30000,
      retryAttempts: 3,
      compressionEnabled: true,
      cacheTTL: 300000, // 5 minutes
      rateLimitPerMinute: 100,
      ...config,
    };

    // Start cache cleanup interval
    setInterval(() => this.cleanupCache(), 60000); // Every minute
  }

  /**
   * Execute assessment with optimized performance
   */
  async executeAssessment(input: AssessmentInput): Promise<AssessmentResults> {
    const startTime = performance.now();
    const endpoint = '/api/assessment/execute';

    try {
      // Check rate limiting
      if (!this.checkRateLimit(endpoint)) {
        throw new Error('Rate limit exceeded');
      }

      // Deduplicate concurrent requests
      const requestKey = this.generateRequestKey(endpoint, input);
      if (this.requestQueue.has(requestKey)) {
        return await this.requestQueue.get(requestKey)!;
      }

      // Create optimized request
      const requestPromise = this.performOptimizedRequest<AssessmentResults>(
        endpoint,
        'POST',
        input,
        { skipCache: true }, // Assessment results should not be cached
      );

      this.requestQueue.set(requestKey, requestPromise);

      const result = await requestPromise;

      // Record performance metrics
      this.recordPerformanceMetric({
        endpoint,
        method: 'POST',
        responseTime: performance.now() - startTime,
        cacheHit: false,
        payloadSize: JSON.stringify(result).length,
        timestamp: new Date().toISOString(),
        statusCode: 200,
      });

      this.requestQueue.delete(requestKey);
      return result;
    } catch (error) {
      this.recordPerformanceMetric({
        endpoint,
        method: 'POST',
        responseTime: performance.now() - startTime,
        cacheHit: false,
        payloadSize: 0,
        timestamp: new Date().toISOString(),
        statusCode: 500,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      this.requestQueue.delete(this.generateRequestKey(endpoint, input));
      throw error;
    }
  }

  /**
   * Get assessment results with caching
   */
  async getAssessmentResults(sessionId: string): Promise<AssessmentResults> {
    const endpoint = `/api/assessment/results/${sessionId}`;
    return this.performOptimizedRequest<AssessmentResults>(endpoint, 'GET');
  }

  /**
   * Get patient assessment history with pagination and caching
   */
  async getAssessmentHistory(
    patientId: string,
    options: {
      page?: number;
      limit?: number;
      dateRange?: { start: string; end: string };
    } = {},
  ): Promise<{
    assessments: AssessmentResults[];
    total: number;
    hasMore: boolean;
  }> {
    const queryParams = new URLSearchParams({
      page: (options.page || 1).toString(),
      limit: (options.limit || 20).toString(),
      ...(options.dateRange && {
        startDate: options.dateRange.start,
        endDate: options.dateRange.end,
      }),
    });

    const endpoint = `/api/assessment/history/${patientId}?${queryParams}`;
    return this.performOptimizedRequest(endpoint, 'GET', null, { cacheTTL: 60000 }); // 1 minute cache
  }

  /**
   * Perform optimized HTTP request with caching and compression
   */
  private async performOptimizedRequest<T>(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    data?: any,
    options: {
      skipCache?: boolean;
      cacheTTL?: number;
    } = {},
  ): Promise<T> {
    const startTime = performance.now();
    const cacheKey = this.generateCacheKey(endpoint, method, data);

    // Check cache for GET requests
    if (method === 'GET' && !options.skipCache) {
      const cachedResult = this.getFromCache<T>(cacheKey);
      if (cachedResult) {
        this.recordPerformanceMetric({
          endpoint,
          method,
          responseTime: performance.now() - startTime,
          cacheHit: true,
          payloadSize: JSON.stringify(cachedResult).length,
          timestamp: new Date().toISOString(),
          statusCode: 200,
        });
        return cachedResult;
      }
    }

    // Prepare request headers
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };

    if (this.config.compressionEnabled) {
      headers['Accept-Encoding'] = 'gzip, deflate, br';
    }

    // Add ETag support for caching
    const cachedEntry = this.cache.get(cacheKey);
    if (cachedEntry?.etag) {
      headers['If-None-Match'] = cachedEntry.etag;
    }

    // Prepare request configuration
    const requestConfig: RequestInit = {
      method,
      headers,
      signal: AbortSignal.timeout(this.config.timeout),
    };

    if (data && (method === 'POST' || method === 'PUT')) {
      requestConfig.body = JSON.stringify(data);
    }

    let attempt = 0;
    let lastError: Error = new Error('Unknown error occurred');

    // Retry logic with exponential backoff
    while (attempt < this.config.retryAttempts) {
      try {
        const response = await fetch(`${this.config.baseUrl}${endpoint}`, requestConfig);

        // Handle 304 Not Modified
        if (response.status === 304 && cachedEntry) {
          this.recordPerformanceMetric({
            endpoint,
            method,
            responseTime: performance.now() - startTime,
            cacheHit: true,
            payloadSize: JSON.stringify(cachedEntry.data).length,
            timestamp: new Date().toISOString(),
            statusCode: 304,
          });
          return cachedEntry.data;
        }

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result: T = await response.json();
        const responseSize = JSON.stringify(result).length;

        // Cache successful GET requests
        if (method === 'GET' && !options.skipCache) {
          const ttl = options.cacheTTL || this.config.cacheTTL;
          const etag = response.headers.get('ETag');

          this.setCache(cacheKey, result, ttl, etag || undefined);
        }

        // Record performance metrics
        this.recordPerformanceMetric({
          endpoint,
          method,
          responseTime: performance.now() - startTime,
          cacheHit: false,
          payloadSize: responseSize,
          timestamp: new Date().toISOString(),
          statusCode: response.status,
        });

        return result;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        attempt++;

        if (attempt < this.config.retryAttempts) {
          // Exponential backoff: 100ms, 200ms, 400ms
          await new Promise(resolve => setTimeout(resolve, 100 * Math.pow(2, attempt - 1)));
        }
      }
    }

    // Record failed request
    this.recordPerformanceMetric({
      endpoint,
      method,
      responseTime: performance.now() - startTime,
      cacheHit: false,
      payloadSize: 0,
      timestamp: new Date().toISOString(),
      statusCode: 500,
      error: lastError.message,
    });

    throw lastError;
  }

  /**
   * Check rate limiting
   */
  private checkRateLimit(endpoint: string): boolean {
    const now = Date.now();
    const windowStart = now - 60000; // 1 minute window

    if (!this.rateLimitTracker.has(endpoint)) {
      this.rateLimitTracker.set(endpoint, []);
    }

    const requests = this.rateLimitTracker.get(endpoint)!;

    // Remove old requests outside the window
    const recentRequests = requests.filter(timestamp => timestamp > windowStart);

    if (recentRequests.length >= this.config.rateLimitPerMinute) {
      return false;
    }

    // Add current request
    recentRequests.push(now);
    this.rateLimitTracker.set(endpoint, recentRequests);

    return true;
  }

  /**
   * Generate cache key
   */
  private generateCacheKey(endpoint: string, method: string, data?: any): string {
    const dataHash = data ? btoa(JSON.stringify(data)).slice(0, 16) : '';
    return `${method}:${endpoint}:${dataHash}`;
  }

  /**
   * Generate request key for deduplication
   */
  private generateRequestKey(endpoint: string, data?: any): string {
    return `${endpoint}:${data ? JSON.stringify(data) : ''}`;
  }

  /**
   * Get from cache
   */
  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    const now = Date.now();
    if (now > entry.timestamp + entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  /**
   * Set cache entry
   */
  private setCache<T>(key: string, data: T, ttl: number, etag?: string): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
      etag,
    });
  }

  /**
   * Cleanup expired cache entries
   */
  private cleanupCache(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.timestamp + entry.ttl) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Record performance metric
   */
  private recordPerformanceMetric(metric: ApiPerformanceMetrics): void {
    this.performanceMetrics.push(metric);

    // Keep only last 1000 metrics
    if (this.performanceMetrics.length > 1000) {
      this.performanceMetrics = this.performanceMetrics.slice(-1000);
    }

    // Log slow requests
    if (metric.responseTime > 1000) {
      console.warn(
        `Slow API request detected: ${metric.endpoint} took ${metric.responseTime.toFixed(2)}ms`,
      );
    }
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): ApiPerformanceMetrics[] {
    return [...this.performanceMetrics];
  }

  /**
   * Get performance summary
   */
  getPerformanceSummary(): {
    averageResponseTime: number;
    cacheHitRate: number;
    errorRate: number;
    slowRequestCount: number;
    totalRequests: number;
  } {
    if (this.performanceMetrics.length === 0) {
      return {
        averageResponseTime: 0,
        cacheHitRate: 0,
        errorRate: 0,
        slowRequestCount: 0,
        totalRequests: 0,
      };
    }

    const totalRequests = this.performanceMetrics.length;
    const totalResponseTime = this.performanceMetrics.reduce((sum, m) => sum + m.responseTime, 0);
    const cacheHits = this.performanceMetrics.filter(m => m.cacheHit).length;
    const errors = this.performanceMetrics.filter(m => m.error).length;
    const slowRequests = this.performanceMetrics.filter(m => m.responseTime > 1000).length;

    return {
      averageResponseTime: totalResponseTime / totalRequests,
      cacheHitRate: (cacheHits / totalRequests) * 100,
      errorRate: (errors / totalRequests) * 100,
      slowRequestCount: slowRequests,
      totalRequests,
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Warm cache with common requests
   */
  async warmCache(
    commonRequests: Array<{ endpoint: string; method: 'GET' | 'POST'; data?: any }>,
  ): Promise<void> {
    const warmupPromises = commonRequests.map(async request => {
      try {
        await this.performOptimizedRequest(request.endpoint, request.method, request.data);
      } catch (error) {
        console.warn(`Cache warmup failed for ${request.endpoint}:`, error);
      }
    });

    await Promise.allSettled(warmupPromises);
  }
}

// Export singleton instance
export const optimizedApiClient = new OptimizedApiClient();
