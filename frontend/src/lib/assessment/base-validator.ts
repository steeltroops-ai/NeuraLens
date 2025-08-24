/**
 * Base Validation Class for NeuraLens Assessment Data
 * Provides common validation patterns and utilities
 */

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  metadata?: Record<string, any>;
}

export interface FileMetadata {
  size: number;
  type: string;
  name: string;
  duration?: number;
  width?: number;
  height?: number;
  sampleRate?: number;
  channels?: number;
  bitrate?: number;
}

/**
 * Base validator class with common validation patterns
 */
export abstract class BaseValidator {
  protected static readonly MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB default
  protected static readonly SUPPORTED_AUDIO_TYPES = [
    'audio/wav',
    'audio/mp3',
    'audio/mpeg',
    'audio/m4a',
    'audio/webm',
    'audio/ogg',
  ];
  protected static readonly SUPPORTED_IMAGE_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];

  /**
   * Common file validation checks
   */
  protected static validateFileBasics(
    file: File,
    maxSize: number = this.MAX_FILE_SIZE,
    allowedTypes: string[] = [],
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Size validation
    if (file.size === 0) {
      errors.push('File is empty');
    } else if (file.size > maxSize) {
      errors.push(
        `File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (${(maxSize / 1024 / 1024).toFixed(1)}MB)`,
      );
    }

    // Type validation
    if (allowedTypes.length > 0 && !allowedTypes.includes(file.type)) {
      errors.push(
        `Unsupported file type: ${file.type}. Supported types: ${allowedTypes.join(', ')}`,
      );
    }

    // Name validation
    if (!file.name || file.name.trim().length === 0) {
      warnings.push('File has no name');
    }

    return { errors, warnings };
  }

  /**
   * Extract basic file metadata
   */
  protected static async extractFileMetadata(file: File): Promise<FileMetadata> {
    return {
      size: file.size,
      type: file.type,
      name: file.name,
    };
  }

  /**
   * Common data structure validation
   */
  protected static validateDataStructure(
    data: any,
    requiredFields: string[],
    optionalFields: string[] = [],
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!data || typeof data !== 'object') {
      errors.push('Data must be a valid object');
      return { errors, warnings };
    }

    // Check required fields
    for (const field of requiredFields) {
      if (!(field in data) || data[field] === null || data[field] === undefined) {
        errors.push(`Missing required field: ${field}`);
      }
    }

    // Check for unexpected fields
    const allAllowedFields = [...requiredFields, ...optionalFields];
    const unexpectedFields = Object.keys(data).filter(key => !allAllowedFields.includes(key));

    if (unexpectedFields.length > 0) {
      warnings.push(`Unexpected fields found: ${unexpectedFields.join(', ')}`);
    }

    return { errors, warnings };
  }

  /**
   * Validate array data with minimum length requirements
   */
  protected static validateArrayData(
    data: any[],
    fieldName: string,
    minLength: number = 1,
    validator?: (item: any, index: number) => string[],
  ): { errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!Array.isArray(data)) {
      errors.push(`${fieldName} must be an array`);
      return { errors, warnings };
    }

    if (data.length === 0) {
      errors.push(`${fieldName} cannot be empty`);
      return { errors, warnings };
    }

    if (data.length < minLength) {
      warnings.push(
        `${fieldName} has only ${data.length} items, minimum recommended: ${minLength}`,
      );
    }

    // Validate individual items if validator provided
    if (validator) {
      data.forEach((item, index) => {
        const itemErrors = validator(item, index);
        errors.push(...itemErrors.map(err => `${fieldName}[${index}]: ${err}`));
      });
    }

    return { errors, warnings };
  }

  /**
   * Validate numeric ranges
   */
  protected static validateNumericRange(
    value: any,
    fieldName: string,
    min?: number,
    max?: number,
  ): string[] {
    const errors: string[] = [];

    if (typeof value !== 'number' || isNaN(value)) {
      errors.push(`${fieldName} must be a valid number`);
      return errors;
    }

    if (min !== undefined && value < min) {
      errors.push(`${fieldName} must be at least ${min}`);
    }

    if (max !== undefined && value > max) {
      errors.push(`${fieldName} must be at most ${max}`);
    }

    return errors;
  }

  /**
   * Validate timestamp data
   */
  protected static validateTimestamp(
    timestamp: any,
    fieldName: string,
    allowFuture: boolean = false,
  ): string[] {
    const errors: string[] = [];

    if (typeof timestamp !== 'number' && typeof timestamp !== 'string') {
      errors.push(`${fieldName} must be a valid timestamp`);
      return errors;
    }

    const date = new Date(timestamp);
    if (isNaN(date.getTime())) {
      errors.push(`${fieldName} is not a valid date`);
      return errors;
    }

    if (!allowFuture && date > new Date()) {
      errors.push(`${fieldName} cannot be in the future`);
    }

    return errors;
  }

  /**
   * Create standardized validation result
   */
  protected static createValidationResult(
    errors: string[],
    warnings: string[],
    metadata?: Record<string, any>,
  ): ValidationResult {
    return {
      isValid: errors.length === 0,
      errors: [...new Set(errors)], // Remove duplicates
      warnings: [...new Set(warnings)], // Remove duplicates
      metadata,
    };
  }

  /**
   * Merge multiple validation results
   */
  protected static mergeValidationResults(...results: ValidationResult[]): ValidationResult {
    const allErrors: string[] = [];
    const allWarnings: string[] = [];
    const allMetadata: Record<string, any> = {};

    for (const result of results) {
      allErrors.push(...result.errors);
      allWarnings.push(...result.warnings);
      if (result.metadata) {
        Object.assign(allMetadata, result.metadata);
      }
    }

    return this.createValidationResult(allErrors, allWarnings, allMetadata);
  }
}

/**
 * Quality assessment utilities
 */
export class QualityAssessment {
  /**
   * Assess file quality based on size and metadata
   */
  static assessFileQuality(
    file: File,
    metadata: FileMetadata,
    expectedDuration?: number,
  ): { score: number; issues: string[] } {
    const issues: string[] = [];
    let score = 100;

    // Size-based quality assessment
    if (file.size < 50 * 1024) {
      // Less than 50KB
      issues.push('File size is very small, may indicate low quality');
      score -= 30;
    }

    // Duration-based assessment for audio/video
    if (metadata.duration) {
      if (metadata.duration < 1) {
        issues.push('Very short duration, may not provide sufficient data for analysis');
        score -= 20;
      }

      if (
        expectedDuration &&
        Math.abs(metadata.duration - expectedDuration) > expectedDuration * 0.5
      ) {
        issues.push('Duration significantly different from expected');
        score -= 10;
      }
    }

    // Bitrate assessment for audio
    if (metadata.bitrate && metadata.bitrate < 64000) {
      issues.push('Low bitrate detected, may affect analysis quality');
      score -= 15;
    }

    // Resolution assessment for images
    if (metadata.width && metadata.height) {
      const pixelCount = metadata.width * metadata.height;
      if (pixelCount < 100000) {
        // Less than 100k pixels
        issues.push('Low resolution image, may affect analysis accuracy');
        score -= 25;
      }
    }

    return {
      score: Math.max(0, Math.min(100, score)),
      issues,
    };
  }

  /**
   * Assess data completeness
   */
  static assessDataCompleteness(
    data: Record<string, any>,
    requiredFields: string[],
    optionalFields: string[] = [],
  ): { score: number; completeness: number } {
    const totalFields = requiredFields.length + optionalFields.length;
    const presentFields = [...requiredFields, ...optionalFields].filter(
      field => data[field] !== undefined && data[field] !== null,
    );

    const completeness = presentFields.length / totalFields;
    const score = completeness * 100;

    return { score, completeness };
  }
}
