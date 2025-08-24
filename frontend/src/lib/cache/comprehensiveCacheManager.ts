/**
 * Comprehensive Cache Manager
 * Multi-layer caching system with intelligent invalidation and monitoring
 */

// Cache configuration interfaces
export interface CacheConfig {
  enableBrowserCache: boolean;
  enableMemoryCache: boolean;
  enableIndexedDBCache: boolean;
  enableServiceWorkerCache: boolean;
  defaultTTL: number;
  maxMemoryCacheSize: number;
  maxIndexedDBSize: number;
  compressionEnabled: boolean;
  encryptionEnabled: boolean;
}

export interface CacheEntry<T> {
  key: string;
  data: T;
  timestamp: number;
  ttl: number;
  accessCount: number;
  lastAccessed: number;
  size: number;
  compressed: boolean;
  encrypted: boolean;
  etag?: string;
  version: string;
}

export interface CacheMetrics {
  hitRate: number;
  missRate: number;
  totalRequests: number;
  totalHits: number;
  totalMisses: number;
  averageResponseTime: number;
  cacheSize: number;
  evictionCount: number;
  compressionRatio: number;
}

export type CacheLayer = 'memory' | 'indexeddb' | 'serviceworker' | 'browser';

// Comprehensive cache manager class
export class ComprehensiveCacheManager {
  private config: CacheConfig;
  private memoryCache: Map<string, CacheEntry<any>> = new Map();
  private indexedDB: IDBDatabase | null = null;
  private metrics: CacheMetrics = {
    hitRate: 0,
    missRate: 0,
    totalRequests: 0,
    totalHits: 0,
    totalMisses: 0,
    averageResponseTime: 0,
    cacheSize: 0,
    evictionCount: 0,
    compressionRatio: 0,
  };
  private responseTimeHistory: number[] = [];

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      enableBrowserCache: true,
      enableMemoryCache: true,
      enableIndexedDBCache: true,
      enableServiceWorkerCache: true,
      defaultTTL: 300000, // 5 minutes
      maxMemoryCacheSize: 50 * 1024 * 1024, // 50MB
      maxIndexedDBSize: 100 * 1024 * 1024, // 100MB
      compressionEnabled: true,
      encryptionEnabled: false,
      ...config,
    };

    this.initializeIndexedDB();
    this.startCleanupInterval();
  }

  /**
   * Initialize IndexedDB for persistent caching
   */
  private async initializeIndexedDB(): Promise<void> {
    if (!this.config.enableIndexedDBCache || typeof window === 'undefined') return;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open('NeuraLensCache', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.indexedDB = request.result;
        resolve();
      };

      request.onupgradeneeded = event => {
        const db = (event.target as IDBOpenDBRequest).result;

        if (!db.objectStoreNames.contains('cache')) {
          const store = db.createObjectStore('cache', { keyPath: 'key' });
          store.createIndex('timestamp', 'timestamp');
          store.createIndex('lastAccessed', 'lastAccessed');
        }
      };
    });
  }

  /**
   * Get data from cache with multi-layer fallback
   */
  async get<T>(key: string): Promise<T | null> {
    const startTime = performance.now();
    this.metrics.totalRequests++;

    try {
      // Layer 1: Memory cache (fastest)
      if (this.config.enableMemoryCache) {
        const memoryResult = this.getFromMemoryCache<T>(key);
        if (memoryResult !== null) {
          this.recordCacheHit(performance.now() - startTime);
          return memoryResult;
        }
      }

      // Layer 2: IndexedDB cache (persistent)
      if (this.config.enableIndexedDBCache && this.indexedDB) {
        const indexedDBResult = await this.getFromIndexedDB<T>(key);
        if (indexedDBResult !== null) {
          // Promote to memory cache
          this.setInMemoryCache(key, indexedDBResult, this.config.defaultTTL);
          this.recordCacheHit(performance.now() - startTime);
          return indexedDBResult;
        }
      }

      // Layer 3: Service Worker cache (network resources)
      if (this.config.enableServiceWorkerCache && 'serviceWorker' in navigator) {
        const swResult = await this.getFromServiceWorkerCache<T>(key);
        if (swResult !== null) {
          // Promote to higher cache layers
          this.setInMemoryCache(key, swResult, this.config.defaultTTL);
          if (this.indexedDB) {
            await this.setInIndexedDB(key, swResult, this.config.defaultTTL);
          }
          this.recordCacheHit(performance.now() - startTime);
          return swResult;
        }
      }

      // Cache miss
      this.recordCacheMiss(performance.now() - startTime);
      return null;
    } catch (error) {
      console.error('Cache get error:', error);
      this.recordCacheMiss(performance.now() - startTime);
      return null;
    }
  }

  /**
   * Set data in cache across all layers
   */
  async set<T>(
    key: string,
    data: T,
    ttl: number = this.config.defaultTTL,
    options: {
      layers?: CacheLayer[];
      compress?: boolean;
      encrypt?: boolean;
    } = {},
  ): Promise<void> {
    const layers = options.layers || ['memory', 'indexeddb'];
    const compress = options.compress ?? this.config.compressionEnabled;
    const encrypt = options.encrypt ?? this.config.encryptionEnabled;

    try {
      // Process data (compression/encryption)
      let processedData = data;
      if (compress) {
        processedData = await this.compressData(data);
      }
      if (encrypt) {
        processedData = await this.encryptData(processedData);
      }

      // Set in requested cache layers
      const promises: Promise<void>[] = [];

      if (layers.includes('memory') && this.config.enableMemoryCache) {
        promises.push(Promise.resolve(this.setInMemoryCache(key, processedData, ttl)));
      }

      if (layers.includes('indexeddb') && this.config.enableIndexedDBCache && this.indexedDB) {
        promises.push(this.setInIndexedDB(key, processedData, ttl));
      }

      if (layers.includes('serviceworker') && this.config.enableServiceWorkerCache) {
        promises.push(this.setInServiceWorkerCache(key, processedData, ttl));
      }

      await Promise.allSettled(promises);
    } catch (error) {
      console.error('Cache set error:', error);
    }
  }

  /**
   * Memory cache operations
   */
  private getFromMemoryCache<T>(key: string): T | null {
    const entry = this.memoryCache.get(key);
    if (!entry) return null;

    const now = Date.now();
    if (now > entry.timestamp + entry.ttl) {
      this.memoryCache.delete(key);
      return null;
    }

    // Update access statistics
    entry.accessCount++;
    entry.lastAccessed = now;

    return entry.data;
  }

  private setInMemoryCache<T>(key: string, data: T, ttl: number): void {
    const now = Date.now();
    const size = this.calculateDataSize(data);

    // Check memory limit and evict if necessary
    this.evictMemoryCacheIfNeeded(size);

    const entry: CacheEntry<T> = {
      key,
      data,
      timestamp: now,
      ttl,
      accessCount: 1,
      lastAccessed: now,
      size,
      compressed: false,
      encrypted: false,
      version: '1.0',
    };

    this.memoryCache.set(key, entry);
    this.updateCacheSize();
  }

  /**
   * IndexedDB cache operations
   */
  private async getFromIndexedDB<T>(key: string): Promise<T | null> {
    if (!this.indexedDB) return null;

    return new Promise(resolve => {
      const transaction = this.indexedDB!.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const request = store.get(key);

      request.onsuccess = () => {
        const entry = request.result as CacheEntry<T>;
        if (!entry) {
          resolve(null);
          return;
        }

        const now = Date.now();
        if (now > entry.timestamp + entry.ttl) {
          // Remove expired entry
          this.deleteFromIndexedDB(key);
          resolve(null);
          return;
        }

        // Update access statistics
        entry.accessCount++;
        entry.lastAccessed = now;
        this.updateIndexedDBEntry(entry);

        resolve(entry.data);
      };

      request.onerror = () => resolve(null);
    });
  }

  private async setInIndexedDB<T>(key: string, data: T, ttl: number): Promise<void> {
    if (!this.indexedDB) return;

    const now = Date.now();
    const size = this.calculateDataSize(data);

    const entry: CacheEntry<T> = {
      key,
      data,
      timestamp: now,
      ttl,
      accessCount: 1,
      lastAccessed: now,
      size,
      compressed: false,
      encrypted: false,
      version: '1.0',
    };

    return new Promise((resolve, reject) => {
      const transaction = this.indexedDB!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Service Worker cache operations
   */
  private async getFromServiceWorkerCache<T>(key: string): Promise<T | null> {
    if (!('caches' in window)) return null;

    try {
      const cache = await caches.open('neuralens-cache-v1');
      const response = await cache.match(key);

      if (response) {
        return await response.json();
      }

      return null;
    } catch (error) {
      console.error('Service Worker cache get error:', error);
      return null;
    }
  }

  private async setInServiceWorkerCache<T>(key: string, data: T, ttl: number): Promise<void> {
    if (!('caches' in window)) return;

    try {
      const cache = await caches.open('neuralens-cache-v1');
      const response = new Response(JSON.stringify(data), {
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': `max-age=${Math.floor(ttl / 1000)}`,
          'X-Cache-Timestamp': Date.now().toString(),
          'X-Cache-TTL': ttl.toString(),
        },
      });

      await cache.put(key, response);
    } catch (error) {
      console.error('Service Worker cache set error:', error);
    }
  }

  /**
   * Cache invalidation
   */
  async invalidate(pattern: string | RegExp): Promise<void> {
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;

    // Invalidate memory cache
    for (const key of this.memoryCache.keys()) {
      if (regex.test(key)) {
        this.memoryCache.delete(key);
      }
    }

    // Invalidate IndexedDB cache
    if (this.indexedDB) {
      await this.invalidateIndexedDB(regex);
    }

    // Invalidate Service Worker cache
    if ('caches' in window) {
      await this.invalidateServiceWorkerCache(regex);
    }

    this.updateCacheSize();
  }

  /**
   * Data compression
   */
  private async compressData<T>(data: T): Promise<T> {
    if (!this.config.compressionEnabled) return data;

    try {
      const jsonString = JSON.stringify(data);
      const compressed = await this.compress(jsonString);
      return compressed as T;
    } catch (error) {
      console.error('Compression error:', error);
      return data;
    }
  }

  private async compress(data: string): Promise<string> {
    // Simple compression using built-in compression
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    const stream = new CompressionStream('gzip');
    const writer = stream.writable.getWriter();
    const reader = stream.readable.getReader();

    writer.write(encoder.encode(data));
    writer.close();

    const chunks: Uint8Array[] = [];
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) chunks.push(value);
    }

    const compressed = new Uint8Array(chunks.reduce((acc, chunk) => acc + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
      compressed.set(chunk, offset);
      offset += chunk.length;
    }

    return btoa(String.fromCharCode(...compressed));
  }

  /**
   * Memory management
   */
  private evictMemoryCacheIfNeeded(newEntrySize: number): void {
    const currentSize = this.getCurrentMemoryCacheSize();

    if (currentSize + newEntrySize > this.config.maxMemoryCacheSize) {
      // Evict least recently used entries
      const entries = Array.from(this.memoryCache.entries()).sort(
        ([, a], [, b]) => a.lastAccessed - b.lastAccessed,
      );

      let freedSize = 0;
      for (const [key, entry] of entries) {
        this.memoryCache.delete(key);
        freedSize += entry.size;
        this.metrics.evictionCount++;

        if (currentSize - freedSize + newEntrySize <= this.config.maxMemoryCacheSize) {
          break;
        }
      }
    }
  }

  /**
   * Utility methods
   */
  private calculateDataSize(data: any): number {
    return new Blob([JSON.stringify(data)]).size;
  }

  private getCurrentMemoryCacheSize(): number {
    return Array.from(this.memoryCache.values()).reduce((total, entry) => total + entry.size, 0);
  }

  private updateCacheSize(): void {
    this.metrics.cacheSize = this.getCurrentMemoryCacheSize();
  }

  private recordCacheHit(responseTime: number): void {
    this.metrics.totalHits++;
    this.recordResponseTime(responseTime);
    this.updateMetrics();
  }

  private recordCacheMiss(responseTime: number): void {
    this.metrics.totalMisses++;
    this.recordResponseTime(responseTime);
    this.updateMetrics();
  }

  private recordResponseTime(time: number): void {
    this.responseTimeHistory.push(time);
    if (this.responseTimeHistory.length > 1000) {
      this.responseTimeHistory = this.responseTimeHistory.slice(-1000);
    }
  }

  private updateMetrics(): void {
    const total = this.metrics.totalHits + this.metrics.totalMisses;
    this.metrics.hitRate = total > 0 ? (this.metrics.totalHits / total) * 100 : 0;
    this.metrics.missRate = total > 0 ? (this.metrics.totalMisses / total) * 100 : 0;

    if (this.responseTimeHistory.length > 0) {
      this.metrics.averageResponseTime =
        this.responseTimeHistory.reduce((sum, time) => sum + time, 0) /
        this.responseTimeHistory.length;
    }
  }

  private startCleanupInterval(): void {
    setInterval(() => {
      this.cleanupExpiredEntries();
    }, 60000); // Every minute
  }

  private cleanupExpiredEntries(): void {
    const now = Date.now();

    // Cleanup memory cache
    for (const [key, entry] of this.memoryCache.entries()) {
      if (now > entry.timestamp + entry.ttl) {
        this.memoryCache.delete(key);
      }
    }

    this.updateCacheSize();
  }

  private async deleteFromIndexedDB(key: string): Promise<void> {
    if (!this.indexedDB) return;

    return new Promise(resolve => {
      const transaction = this.indexedDB!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.delete(key);
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
    });
  }

  private async updateIndexedDBEntry<T>(entry: CacheEntry<T>): Promise<void> {
    if (!this.indexedDB) return;

    return new Promise(resolve => {
      const transaction = this.indexedDB!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.put(entry);
      request.onsuccess = () => resolve();
      request.onerror = () => resolve();
    });
  }

  private async invalidateIndexedDB(regex: RegExp): Promise<void> {
    if (!this.indexedDB) return;

    return new Promise(resolve => {
      const transaction = this.indexedDB!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.openCursor();

      request.onsuccess = event => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          if (regex.test(cursor.key as string)) {
            cursor.delete();
          }
          cursor.continue();
        } else {
          resolve();
        }
      };

      request.onerror = () => resolve();
    });
  }

  private async invalidateServiceWorkerCache(regex: RegExp): Promise<void> {
    try {
      const cache = await caches.open('neuralens-cache-v1');
      const keys = await cache.keys();

      const deletePromises = keys
        .filter(request => regex.test(request.url))
        .map(request => cache.delete(request));

      await Promise.all(deletePromises);
    } catch (error) {
      console.error('Service Worker cache invalidation error:', error);
    }
  }

  private async encryptData<T>(data: T): Promise<T> {
    // Placeholder for encryption implementation
    return data;
  }

  /**
   * Get cache metrics
   */
  getMetrics(): CacheMetrics {
    return { ...this.metrics };
  }

  /**
   * Clear all caches
   */
  async clearAll(): Promise<void> {
    this.memoryCache.clear();

    if (this.indexedDB) {
      const transaction = this.indexedDB.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      store.clear();
    }

    if ('caches' in window) {
      await caches.delete('neuralens-cache-v1');
    }

    this.updateCacheSize();
  }
}

// Export singleton instance
export const comprehensiveCacheManager = new ComprehensiveCacheManager();
