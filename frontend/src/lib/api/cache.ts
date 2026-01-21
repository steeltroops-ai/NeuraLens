/**
 * Simple Client-Side API Cache
 *
 * Provides lightweight caching for API responses with TTL support.
 * Used to reduce redundant API calls and improve perceived performance.
 */

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class APICache {
  private cache: Map<string, CacheEntry<unknown>> = new Map();
  private defaultTTL: number = 30000; // 30 seconds

  /**
   * Get cached data if valid, otherwise undefined
   */
  get<T>(key: string): T | undefined {
    const entry = this.cache.get(key) as CacheEntry<T> | undefined;
    if (!entry) return undefined;

    const isExpired = Date.now() - entry.timestamp > entry.ttl;
    if (isExpired) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.data;
  }

  /**
   * Set cached data with optional TTL
   */
  set<T>(key: string, data: T, ttl: number = this.defaultTTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  /**
   * Check if key exists and is valid
   */
  has(key: string): boolean {
    return this.get(key) !== undefined;
  }

  /**
   * Delete a specific key
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all cached entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Clear expired entries
   */
  prune(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.cache.delete(key);
      }
    }
  }
}

// Singleton instance
export const apiCache = new APICache();

/**
 * Fetch with caching wrapper
 *
 * @param url - URL to fetch
 * @param options - Fetch options
 * @param cacheTTL - Cache TTL in milliseconds (default: 30s)
 * @returns Cached or fresh response data
 */
export async function fetchWithCache<T>(
  url: string,
  options?: RequestInit,
  cacheTTL: number = 30000,
): Promise<T> {
  const cacheKey = `${options?.method || "GET"}:${url}`;

  // Check cache first
  const cached = apiCache.get<T>(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  // Fetch fresh data
  const response = await fetch(url, options);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = (await response.json()) as T;

  // Cache the result
  apiCache.set(cacheKey, data, cacheTTL);

  return data;
}

/**
 * Stale-While-Revalidate fetch pattern
 *
 * Returns cached data immediately (if available) while
 * fetching fresh data in the background.
 *
 * @param url - URL to fetch
 * @param onUpdate - Callback when fresh data is received
 * @param options - Fetch options
 * @param cacheTTL - Cache TTL in milliseconds
 * @returns Cached or fresh response data
 */
export async function fetchSWR<T>(
  url: string,
  onUpdate?: (data: T) => void,
  options?: RequestInit,
  cacheTTL: number = 30000,
): Promise<T | undefined> {
  const cacheKey = `${options?.method || "GET"}:${url}`;

  // Check for cached data
  const cached = apiCache.get<T>(cacheKey);

  // If we have cached data, return it and revalidate in background
  if (cached !== undefined) {
    // Revalidate in background
    fetch(url, options)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((freshData: T) => {
        apiCache.set(cacheKey, freshData, cacheTTL);
        onUpdate?.(freshData);
      })
      .catch(() => {
        // Silently fail - we have stale data
      });

    return cached;
  }

  // No cache - fetch fresh
  try {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error();

    const data = (await response.json()) as T;
    apiCache.set(cacheKey, data, cacheTTL);
    return data;
  } catch {
    return undefined;
  }
}

export default apiCache;
