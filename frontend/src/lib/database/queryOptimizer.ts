/**
 * Database Query Optimizer
 * Optimized database queries with indexing, connection pooling, and caching
 */

// Query optimization interfaces
export interface QueryOptimizationConfig {
  enableIndexHints: boolean;
  maxConnectionPoolSize: number;
  connectionTimeout: number;
  queryTimeout: number;
  enableQueryCache: boolean;
  cacheSize: number;
  enableSlowQueryLogging: boolean;
  slowQueryThreshold: number;
}

export interface QueryPerformanceMetrics {
  query: string;
  executionTime: number;
  rowsAffected: number;
  cacheHit: boolean;
  indexesUsed: string[];
  timestamp: string;
}

export interface DatabaseIndex {
  tableName: string;
  indexName: string;
  columns: string[];
  indexType: 'btree' | 'hash' | 'gin' | 'gist';
  isUnique: boolean;
  isPartial: boolean;
  condition?: string;
}

// Optimized database queries for NeuraLens
export class DatabaseQueryOptimizer {
  private config: QueryOptimizationConfig;
  private queryCache: Map<string, { result: any; timestamp: number; ttl: number }> = new Map();
  private performanceMetrics: QueryPerformanceMetrics[] = [];

  constructor(config: Partial<QueryOptimizationConfig> = {}) {
    this.config = {
      enableIndexHints: true,
      maxConnectionPoolSize: 20,
      connectionTimeout: 5000,
      queryTimeout: 30000,
      enableQueryCache: true,
      cacheSize: 1000,
      enableSlowQueryLogging: true,
      slowQueryThreshold: 1000,
      ...config,
    };
  }

  /**
   * Optimized assessment results query
   */
  getOptimizedAssessmentQuery(sessionId: string): {
    query: string;
    params: any[];
    indexes: string[];
  } {
    return {
      query: `
        SELECT 
          ar.session_id,
          ar.completion_time,
          ar.total_processing_time,
          ar.overall_risk_category,
          ar.nri_result,
          ar.speech_result,
          ar.retinal_result,
          ar.motor_result,
          ar.cognitive_result,
          ar.created_at,
          ar.updated_at
        FROM assessment_results ar
        WHERE ar.session_id = $1
          AND ar.deleted_at IS NULL
        LIMIT 1
      `,
      params: [sessionId],
      indexes: ['idx_assessment_results_session_id', 'idx_assessment_results_deleted_at'],
    };
  }

  /**
   * Optimized patient assessment history query with pagination
   */
  getOptimizedHistoryQuery(
    patientId: string,
    page: number = 1,
    limit: number = 20,
    dateRange?: { start: string; end: string },
  ): {
    query: string;
    params: any[];
    indexes: string[];
  } {
    const offset = (page - 1) * limit;
    let whereClause = 'WHERE ar.patient_id = $1 AND ar.deleted_at IS NULL';
    const params: any[] = [patientId];
    let paramIndex = 2;

    if (dateRange) {
      whereClause += ` AND ar.completion_time >= $${paramIndex} AND ar.completion_time <= $${paramIndex + 1}`;
      params.push(dateRange.start, dateRange.end);
      paramIndex += 2;
    }

    return {
      query: `
        WITH assessment_summary AS (
          SELECT 
            ar.session_id,
            ar.completion_time,
            ar.overall_risk_category,
            ar.nri_result->>'nri_score' as nri_score,
            ar.nri_result->>'confidence' as confidence,
            ar.total_processing_time,
            ROW_NUMBER() OVER (ORDER BY ar.completion_time DESC) as row_num,
            COUNT(*) OVER() as total_count
          FROM assessment_results ar
          ${whereClause}
        )
        SELECT 
          session_id,
          completion_time,
          overall_risk_category,
          nri_score::numeric,
          confidence::numeric,
          total_processing_time,
          total_count
        FROM assessment_summary
        WHERE row_num > $${paramIndex} AND row_num <= $${paramIndex + 1}
        ORDER BY completion_time DESC
      `,
      params: [...params, offset, offset + limit],
      indexes: [
        'idx_assessment_results_patient_id',
        'idx_assessment_results_completion_time',
        'idx_assessment_results_deleted_at',
        'idx_assessment_results_composite',
      ],
    };
  }

  /**
   * Optimized patient search query
   */
  getOptimizedPatientSearchQuery(
    searchTerm: string,
    limit: number = 50,
  ): {
    query: string;
    params: any[];
    indexes: string[];
  } {
    return {
      query: `
        SELECT 
          p.patient_id,
          p.first_name,
          p.last_name,
          p.date_of_birth,
          p.email,
          p.phone,
          p.last_assessment_date,
          p.risk_category,
          ts_rank(p.search_vector, plainto_tsquery($1)) as rank
        FROM patients p
        WHERE p.search_vector @@ plainto_tsquery($1)
          AND p.deleted_at IS NULL
        ORDER BY rank DESC, p.last_assessment_date DESC NULLS LAST
        LIMIT $2
      `,
      params: [searchTerm, limit],
      indexes: [
        'idx_patients_search_vector_gin',
        'idx_patients_deleted_at',
        'idx_patients_last_assessment_date',
      ],
    };
  }

  /**
   * Optimized assessment statistics query
   */
  getOptimizedStatsQuery(dateRange: { start: string; end: string }): {
    query: string;
    params: any[];
    indexes: string[];
  } {
    return {
      query: `
        SELECT 
          COUNT(*) as total_assessments,
          COUNT(DISTINCT patient_id) as unique_patients,
          AVG(total_processing_time) as avg_processing_time,
          AVG((nri_result->>'nri_score')::numeric) as avg_nri_score,
          COUNT(*) FILTER (WHERE overall_risk_category = 'low') as low_risk_count,
          COUNT(*) FILTER (WHERE overall_risk_category = 'moderate') as moderate_risk_count,
          COUNT(*) FILTER (WHERE overall_risk_category = 'high') as high_risk_count,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_processing_time) as median_processing_time,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_processing_time) as p95_processing_time
        FROM assessment_results
        WHERE completion_time >= $1 
          AND completion_time <= $2
          AND deleted_at IS NULL
      `,
      params: [dateRange.start, dateRange.end],
      indexes: [
        'idx_assessment_results_completion_time',
        'idx_assessment_results_deleted_at',
        'idx_assessment_results_risk_category',
      ],
    };
  }

  /**
   * Get recommended database indexes
   */
  getRecommendedIndexes(): DatabaseIndex[] {
    return [
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_session_id',
        columns: ['session_id'],
        indexType: 'btree',
        isUnique: true,
        isPartial: false,
      },
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_patient_id',
        columns: ['patient_id'],
        indexType: 'btree',
        isUnique: false,
        isPartial: false,
      },
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_completion_time',
        columns: ['completion_time'],
        indexType: 'btree',
        isUnique: false,
        isPartial: false,
      },
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_deleted_at',
        columns: ['deleted_at'],
        indexType: 'btree',
        isUnique: false,
        isPartial: true,
        condition: 'deleted_at IS NULL',
      },
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_composite',
        columns: ['patient_id', 'completion_time', 'deleted_at'],
        indexType: 'btree',
        isUnique: false,
        isPartial: true,
        condition: 'deleted_at IS NULL',
      },
      {
        tableName: 'assessment_results',
        indexName: 'idx_assessment_results_risk_category',
        columns: ['overall_risk_category'],
        indexType: 'btree',
        isUnique: false,
        isPartial: false,
      },
      {
        tableName: 'patients',
        indexName: 'idx_patients_search_vector_gin',
        columns: ['search_vector'],
        indexType: 'gin',
        isUnique: false,
        isPartial: false,
      },
      {
        tableName: 'patients',
        indexName: 'idx_patients_deleted_at',
        columns: ['deleted_at'],
        indexType: 'btree',
        isUnique: false,
        isPartial: true,
        condition: 'deleted_at IS NULL',
      },
      {
        tableName: 'patients',
        indexName: 'idx_patients_last_assessment_date',
        columns: ['last_assessment_date'],
        indexType: 'btree',
        isUnique: false,
        isPartial: false,
      },
    ];
  }

  /**
   * Generate index creation SQL
   */
  generateIndexCreationSQL(): string[] {
    const indexes = this.getRecommendedIndexes();
    return indexes.map(index => {
      let sql = `CREATE`;

      if (index.isUnique) {
        sql += ` UNIQUE`;
      }

      sql += ` INDEX`;

      if (!index.isUnique) {
        sql += ` CONCURRENTLY`;
      }

      sql += ` ${index.indexName} ON ${index.tableName}`;

      if (index.indexType !== 'btree') {
        sql += ` USING ${index.indexType}`;
      }

      sql += ` (${index.columns.join(', ')})`;

      if (index.condition) {
        sql += ` WHERE ${index.condition}`;
      }

      sql += ';';

      return sql;
    });
  }

  /**
   * Connection pool configuration
   */
  getConnectionPoolConfig(): {
    max: number;
    min: number;
    acquire: number;
    idle: number;
    evict: number;
    handleDisconnects: boolean;
  } {
    return {
      max: this.config.maxConnectionPoolSize,
      min: 2,
      acquire: this.config.connectionTimeout,
      idle: 10000,
      evict: 1000,
      handleDisconnects: true,
    };
  }

  /**
   * Query optimization hints
   */
  getQueryHints(queryType: 'select' | 'insert' | 'update' | 'delete'): string[] {
    if (!this.config.enableIndexHints) return [];

    switch (queryType) {
      case 'select':
        return [
          'SET enable_seqscan = off;',
          'SET random_page_cost = 1.1;',
          "SET effective_cache_size = '4GB';",
        ];
      case 'insert':
        return ['SET synchronous_commit = off;', "SET wal_buffers = '16MB';"];
      case 'update':
        return ["SET work_mem = '256MB';", "SET maintenance_work_mem = '512MB';"];
      default:
        return [];
    }
  }

  /**
   * Record query performance
   */
  recordQueryPerformance(metric: QueryPerformanceMetrics): void {
    this.performanceMetrics.push(metric);

    // Keep only last 1000 metrics
    if (this.performanceMetrics.length > 1000) {
      this.performanceMetrics = this.performanceMetrics.slice(-1000);
    }

    // Log slow queries
    if (
      this.config.enableSlowQueryLogging &&
      metric.executionTime > this.config.slowQueryThreshold
    ) {
      console.warn(
        `Slow query detected: ${metric.query.substring(0, 100)}... took ${metric.executionTime}ms`,
      );
    }
  }

  /**
   * Get query performance summary
   */
  getQueryPerformanceSummary(): {
    averageExecutionTime: number;
    slowQueryCount: number;
    cacheHitRate: number;
    totalQueries: number;
    topSlowQueries: Array<{ query: string; avgTime: number; count: number }>;
  } {
    if (this.performanceMetrics.length === 0) {
      return {
        averageExecutionTime: 0,
        slowQueryCount: 0,
        cacheHitRate: 0,
        totalQueries: 0,
        topSlowQueries: [],
      };
    }

    const totalQueries = this.performanceMetrics.length;
    const totalExecutionTime = this.performanceMetrics.reduce((sum, m) => sum + m.executionTime, 0);
    const slowQueries = this.performanceMetrics.filter(
      m => m.executionTime > this.config.slowQueryThreshold,
    );
    const cacheHits = this.performanceMetrics.filter(m => m.cacheHit).length;

    // Group queries by normalized query string
    const queryGroups = new Map<string, { totalTime: number; count: number }>();
    this.performanceMetrics.forEach(metric => {
      const normalizedQuery = metric.query.replace(/\$\d+/g, '?').substring(0, 100);
      const existing = queryGroups.get(normalizedQuery) || { totalTime: 0, count: 0 };
      queryGroups.set(normalizedQuery, {
        totalTime: existing.totalTime + metric.executionTime,
        count: existing.count + 1,
      });
    });

    const topSlowQueries = Array.from(queryGroups.entries())
      .map(([query, stats]) => ({
        query,
        avgTime: stats.totalTime / stats.count,
        count: stats.count,
      }))
      .sort((a, b) => b.avgTime - a.avgTime)
      .slice(0, 10);

    return {
      averageExecutionTime: totalExecutionTime / totalQueries,
      slowQueryCount: slowQueries.length,
      cacheHitRate: (cacheHits / totalQueries) * 100,
      totalQueries,
      topSlowQueries,
    };
  }

  /**
   * Cache query result
   */
  cacheQueryResult(queryKey: string, result: any, ttl: number = 300000): void {
    if (!this.config.enableQueryCache) return;

    this.queryCache.set(queryKey, {
      result,
      timestamp: Date.now(),
      ttl,
    });

    // Cleanup if cache is too large
    if (this.queryCache.size > this.config.cacheSize) {
      const oldestKey = this.queryCache.keys().next().value;
      this.queryCache.delete(oldestKey);
    }
  }

  /**
   * Get cached query result
   */
  getCachedQueryResult(queryKey: string): any | null {
    if (!this.config.enableQueryCache) return null;

    const cached = this.queryCache.get(queryKey);
    if (!cached) return null;

    const now = Date.now();
    if (now > cached.timestamp + cached.ttl) {
      this.queryCache.delete(queryKey);
      return null;
    }

    return cached.result;
  }

  /**
   * Clear query cache
   */
  clearQueryCache(): void {
    this.queryCache.clear();
  }
}

// Export singleton instance
export const databaseQueryOptimizer = new DatabaseQueryOptimizer();
