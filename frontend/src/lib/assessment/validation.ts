/**
 * File Validation Utilities
 * Comprehensive validation for audio files and retinal images using base validator
 */

import { BaseValidator, ValidationResult, FileMetadata, QualityAssessment } from './base-validator';

// Re-export for backward compatibility
export { ValidationResult, FileMetadata };

// Audio file validation
export async function validateAudioFile(file: File): Promise<ValidationResult> {
  const errors: string[] = [];
  const warnings: string[] = [];
  const metadata: any = {};

  try {
    // Basic file checks
    if (!file) {
      errors.push('No audio file provided');
      return { isValid: false, errors, warnings };
    }

    // File size validation (max 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      errors.push(`File size too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max: 50MB)`);
    }

    // Minimum file size (1KB)
    const minSize = 1024; // 1KB
    if (file.size < minSize) {
      errors.push('File size too small (minimum: 1KB)');
    }

    metadata.fileSize = file.size;
    metadata.fileName = file.name;

    // File type validation
    const allowedTypes = [
      'audio/wav',
      'audio/wave',
      'audio/x-wav',
      'audio/mpeg',
      'audio/mp3',
      'audio/mp4',
      'audio/m4a',
      'audio/ogg',
      'audio/webm',
      'audio/flac',
    ];

    const allowedExtensions = ['.wav', '.mp3', '.m4a', '.ogg', '.webm', '.flac'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      errors.push(`Unsupported audio format: ${file.type || fileExtension}`);
    }

    metadata.mimeType = file.type;
    metadata.extension = fileExtension;

    // Try to get audio metadata using Web Audio API
    try {
      const audioBuffer = await getAudioMetadata(file);
      if (audioBuffer) {
        metadata.duration = audioBuffer.duration;
        metadata.sampleRate = audioBuffer.sampleRate;
        metadata.channels = audioBuffer.numberOfChannels;

        // Duration validation (minimum 1 second, maximum 5 minutes)
        if (audioBuffer.duration < 1) {
          errors.push('Audio too short (minimum: 1 second)');
        }
        if (audioBuffer.duration > 300) {
          warnings.push('Audio longer than 5 minutes may take longer to process');
        }

        // Sample rate validation
        if (audioBuffer.sampleRate < 8000) {
          warnings.push('Low sample rate detected, may affect analysis quality');
        }
        if (audioBuffer.sampleRate > 48000) {
          warnings.push('High sample rate detected, processing may take longer');
        }

        // Channel validation
        if (audioBuffer.numberOfChannels > 2) {
          warnings.push('Multi-channel audio detected, will be converted to mono');
        }
      }
    } catch (audioError) {
      warnings.push('Could not analyze audio metadata');
    }

    // Quality checks
    if (file.size / (metadata.duration || 1) < 8000) {
      warnings.push('Low bitrate detected, may affect analysis quality');
    }
  } catch (error) {
    errors.push(`Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    metadata,
  };
}

// Image file validation
export async function validateImageFile(file: File): Promise<ValidationResult> {
  const errors: string[] = [];
  const warnings: string[] = [];
  const metadata: any = {};

  try {
    // Basic file checks
    if (!file) {
      errors.push('No image file provided');
      return { isValid: false, errors, warnings };
    }

    // File size validation (max 20MB)
    const maxSize = 20 * 1024 * 1024; // 20MB
    if (file.size > maxSize) {
      errors.push(`File size too large: ${(file.size / 1024 / 1024).toFixed(1)}MB (max: 20MB)`);
    }

    // Minimum file size (10KB)
    const minSize = 10 * 1024; // 10KB
    if (file.size < minSize) {
      errors.push('File size too small (minimum: 10KB)');
    }

    metadata.fileSize = file.size;
    metadata.fileName = file.name;

    // File type validation
    const allowedTypes = [
      'image/jpeg',
      'image/jpg',
      'image/png',
      'image/tiff',
      'image/tif',
      'image/bmp',
      'image/webp',
    ];

    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      errors.push(`Unsupported image format: ${file.type || fileExtension}`);
    }

    metadata.mimeType = file.type;
    metadata.extension = fileExtension;

    // Get image metadata
    try {
      const imageMetadata = await getImageMetadata(file);
      if (imageMetadata) {
        metadata.width = imageMetadata.width;
        metadata.height = imageMetadata.height;
        metadata.aspectRatio = imageMetadata.width / imageMetadata.height;

        // Resolution validation
        const minResolution = 256; // 256x256 minimum
        const maxResolution = 4096; // 4096x4096 maximum

        if (imageMetadata.width < minResolution || imageMetadata.height < minResolution) {
          errors.push(
            `Image resolution too low: ${imageMetadata.width}x${imageMetadata.height} (minimum: ${minResolution}x${minResolution})`,
          );
        }

        if (imageMetadata.width > maxResolution || imageMetadata.height > maxResolution) {
          warnings.push(
            `High resolution image: ${imageMetadata.width}x${imageMetadata.height}, processing may take longer`,
          );
        }

        // Aspect ratio validation for retinal images
        const aspectRatio = imageMetadata.width / imageMetadata.height;
        if (aspectRatio < 0.8 || aspectRatio > 1.25) {
          warnings.push('Unusual aspect ratio detected, ensure image is properly cropped');
        }

        // Recommended resolution for retinal analysis
        const recommendedMin = 512;
        if (imageMetadata.width < recommendedMin || imageMetadata.height < recommendedMin) {
          warnings.push(
            `Resolution below recommended: ${recommendedMin}x${recommendedMin} for optimal analysis`,
          );
        }
      }
    } catch (imageError) {
      warnings.push('Could not analyze image metadata');
    }

    // Quality checks based on file size and resolution
    if (metadata.width && metadata.height) {
      const pixelCount = metadata.width * metadata.height;
      const bytesPerPixel = file.size / pixelCount;

      if (bytesPerPixel < 0.5) {
        warnings.push('Low image quality detected (high compression), may affect analysis');
      }
    }
  } catch (error) {
    errors.push(`Validation error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    metadata,
  };
}

// Get audio metadata using Web Audio API
async function getAudioMetadata(file: File): Promise<AudioBuffer | null> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    audioContext.close();
    return audioBuffer;
  } catch (error) {
    console.warn('Could not decode audio:', error);
    return null;
  }
}

// Get image metadata
async function getImageMetadata(file: File): Promise<{ width: number; height: number } | null> {
  return new Promise(resolve => {
    const img = new Image();

    img.onload = () => {
      resolve({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };

    img.onerror = () => {
      resolve(null);
    };

    img.src = URL.createObjectURL(file);
  });
}

// Validate motor sensor data
export function validateMotorData(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!data) {
    errors.push('No motor data provided');
    return { isValid: false, errors, warnings };
  }

  // Check for required data types
  const hasAccelerometer = data.accelerometer && Array.isArray(data.accelerometer);
  const hasGyroscope = data.gyroscope && Array.isArray(data.gyroscope);
  const hasPosition = data.position && Array.isArray(data.position);

  if (!hasAccelerometer && !hasGyroscope && !hasPosition) {
    errors.push(
      'At least one sensor data type is required (accelerometer, gyroscope, or position)',
    );
  }

  // Validate accelerometer data
  if (hasAccelerometer) {
    if (data.accelerometer.length < 10) {
      warnings.push('Limited accelerometer data points, may affect analysis accuracy');
    }

    // Check data structure
    const invalidPoints = data.accelerometer.filter(
      (point: any) =>
        typeof point.x !== 'number' || typeof point.y !== 'number' || typeof point.z !== 'number',
    );

    if (invalidPoints.length > 0) {
      errors.push('Invalid accelerometer data format (x, y, z coordinates required)');
    }
  }

  // Validate gyroscope data
  if (hasGyroscope) {
    if (data.gyroscope.length < 10) {
      warnings.push('Limited gyroscope data points, may affect analysis accuracy');
    }

    const invalidPoints = data.gyroscope.filter(
      (point: any) =>
        typeof point.x !== 'number' || typeof point.y !== 'number' || typeof point.z !== 'number',
    );

    if (invalidPoints.length > 0) {
      errors.push('Invalid gyroscope data format (x, y, z coordinates required)');
    }
  }

  // Validate position data
  if (hasPosition) {
    if (data.position.length < 5) {
      warnings.push('Limited position data points, may affect analysis accuracy');
    }

    const invalidPoints = data.position.filter(
      (point: any) => typeof point.x !== 'number' || typeof point.y !== 'number',
    );

    if (invalidPoints.length > 0) {
      errors.push('Invalid position data format (x, y coordinates required)');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

// Validate cognitive test data
export function validateCognitiveData(data: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!data) {
    errors.push('No cognitive data provided');
    return { isValid: false, errors, warnings };
  }

  // Check for response times
  if (data.response_times) {
    if (!Array.isArray(data.response_times)) {
      errors.push('Response times must be an array');
    } else if (data.response_times.length < 5) {
      warnings.push('Limited response time data, may affect analysis accuracy');
    } else {
      // Check for reasonable response times (50ms to 10s)
      const invalidTimes = data.response_times.filter((time: number) => time < 50 || time > 10000);
      if (invalidTimes.length > 0) {
        warnings.push('Some response times are outside normal range (50ms - 10s)');
      }
    }
  }

  // Check for accuracy data
  if (data.accuracy) {
    if (!Array.isArray(data.accuracy)) {
      errors.push('Accuracy data must be an array');
    } else if (data.accuracy.length < 5) {
      warnings.push('Limited accuracy data, may affect analysis accuracy');
    }
  }

  // Validate domain-specific data
  const domains = ['memory', 'attention', 'executive'];
  domains.forEach(domain => {
    if (data[domain] && typeof data[domain] !== 'object') {
      errors.push(`${domain} data must be an object`);
    }
  });

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}
