/**
 * Image Capture System for Neuralens Retinal Analysis
 *
 * This utility class handles image input for retinal analysis including
 * file upload, camera capture, and image validation. It provides a unified
 * interface for acquiring retinal fundus images for ML processing.
 *
 * Key Features:
 * - Drag-and-drop file upload with format validation
 * - Camera capture via getUserMedia with preview functionality
 * - Image validation (format, resolution, file size)
 * - Canvas-based image preprocessing and manipulation
 * - Integration with Neuro-Minimalist UI design system
 *
 * Technical Implementation:
 * - Uses HTML5 File API for upload handling
 * - Implements MediaDevices API for camera access
 * - Provides Canvas API integration for image processing
 * - Handles browser compatibility and permissions
 */

import {
  RetinalProcessingState,
  RetinalAnalysisError,
  CameraConfig,
  RETINAL_ANALYSIS_CONSTANTS,
} from '../../../types/retinal-analysis';

export class ImageCaptureSystem {
  private onStateChange: (state: RetinalProcessingState) => void;
  private debug: boolean;
  private currentStream: MediaStream | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private canvasElement: HTMLCanvasElement | null = null;

  constructor(
    onStateChange: (state: RetinalProcessingState) => void,
    debug = false
  ) {
    this.onStateChange = onStateChange;
    this.debug = debug;

    if (this.debug) {
      console.log('[ImageCaptureSystem] Initialized');
    }
  }

  /**
   * Initialize camera capture system
   * Sets up video element and requests camera permissions
   */
  async initializeCamera(config: Partial<CameraConfig> = {}): Promise<void> {
    try {
      if (this.debug) {
        console.log('[ImageCaptureSystem] Initializing camera...');
      }

      // Default camera configuration optimized for retinal imaging
      const cameraConfig: CameraConfig = {
        resolution: { width: 1920, height: 1080 },
        facingMode: 'environment', // Back camera for better quality
        enableFlash: false,
        autoFocus: true,
        captureFormat: 'jpeg',
        jpegQuality: 0.9,
        ...config,
      };

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new RetinalAnalysisError(
          'Camera not supported in this browser',
          'CAMERA_NOT_SUPPORTED'
        );
      }

      // Update state to show camera initialization
      this.updateState({
        status: 'uploading',
        progress: 0.1,
        currentStep: 'Requesting camera access...',
      });

      // Request camera access with constraints
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: cameraConfig.resolution.width },
          height: { ideal: cameraConfig.resolution.height },
          facingMode: cameraConfig.facingMode,
          // Note: focusMode is not a standard MediaTrackConstraints property
          // Advanced camera controls would be handled differently in production
        },
      };

      this.currentStream =
        await navigator.mediaDevices.getUserMedia(constraints);

      // Create video element for preview
      this.videoElement = document.createElement('video');
      this.videoElement.srcObject = this.currentStream;
      this.videoElement.autoplay = true;
      this.videoElement.playsInline = true;

      // Wait for video to be ready
      await new Promise((resolve, reject) => {
        if (!this.videoElement)
          return reject(new Error('Video element not created'));

        this.videoElement.onloadedmetadata = resolve;
        this.videoElement.onerror = reject;
      });

      if (this.debug) {
        console.log('[ImageCaptureSystem] Camera initialized successfully');
      }

      this.updateState({
        status: 'idle',
        progress: 0,
        currentStep: 'Camera ready',
      });
    } catch (error) {
      console.error(
        '[ImageCaptureSystem] Camera initialization failed:',
        error
      );

      let errorMessage = 'Failed to initialize camera';
      let errorCode = 'CAMERA_INIT_ERROR';

      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError') {
          errorMessage =
            'Camera access denied. Please allow camera permissions.';
          errorCode = 'CAMERA_PERMISSION_DENIED';
        } else if (error.name === 'NotFoundError') {
          errorMessage = 'No camera found. Please connect a camera device.';
          errorCode = 'NO_CAMERA_FOUND';
        }
      }

      this.updateState({
        status: 'error',
        progress: 0,
        currentStep: 'Camera initialization failed',
        error: errorMessage,
      });

      throw new RetinalAnalysisError(errorMessage, errorCode, error);
    }
  }

  /**
   * Capture image from camera
   * Takes a snapshot from the video stream and returns as File object
   */
  async captureImage(config: Partial<CameraConfig> = {}): Promise<File> {
    try {
      if (!this.videoElement || !this.currentStream) {
        throw new RetinalAnalysisError(
          'Camera not initialized',
          'CAMERA_NOT_INITIALIZED'
        );
      }

      if (this.debug) {
        console.log('[ImageCaptureSystem] Capturing image...');
      }

      this.updateState({
        status: 'preprocessing',
        progress: 0.3,
        currentStep: 'Capturing image...',
      });

      // Create canvas for image capture
      if (!this.canvasElement) {
        this.canvasElement = document.createElement('canvas');
      }

      const canvas = this.canvasElement;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        throw new Error('Failed to get canvas context');
      }

      // Set canvas dimensions to video dimensions
      canvas.width = this.videoElement.videoWidth;
      canvas.height = this.videoElement.videoHeight;

      // Draw current video frame to canvas
      ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

      // Convert canvas to blob
      const cameraConfig: CameraConfig = {
        resolution: { width: 1920, height: 1080 },
        facingMode: 'environment',
        enableFlash: false,
        autoFocus: true,
        captureFormat: 'jpeg',
        jpegQuality: 0.9,
        ...config,
      };

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (blob) => {
            if (blob) resolve(blob);
            else reject(new Error('Failed to create image blob'));
          },
          `image/${cameraConfig.captureFormat}`,
          cameraConfig.jpegQuality
        );
      });

      // Create File object from blob
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `retinal-capture-${timestamp}.${cameraConfig.captureFormat}`;
      const imageFile = new File([blob], filename, { type: blob.type });

      if (this.debug) {
        console.log('[ImageCaptureSystem] Image captured:', {
          filename,
          size: imageFile.size,
          type: imageFile.type,
          dimensions: { width: canvas.width, height: canvas.height },
        });
      }

      this.updateState({
        status: 'complete',
        progress: 1,
        currentStep: 'Image captured successfully',
        imageDataUrl: URL.createObjectURL(blob),
        imageInfo: {
          name: filename,
          size: imageFile.size,
          type: imageFile.type,
          lastModified: Date.now(),
        },
      });

      return imageFile;
    } catch (error) {
      console.error('[ImageCaptureSystem] Image capture failed:', error);

      this.updateState({
        status: 'error',
        progress: 0,
        currentStep: 'Image capture failed',
        error: `Failed to capture image: ${error}`,
      });

      throw new RetinalAnalysisError(
        `Image capture failed: ${error}`,
        'CAPTURE_ERROR',
        error
      );
    }
  }

  /**
   * Handle file upload with validation
   * Processes uploaded files and validates format, size, and resolution
   */
  async handleFileUpload(file: File): Promise<File> {
    try {
      if (this.debug) {
        console.log('[ImageCaptureSystem] Processing uploaded file:', {
          name: file.name,
          size: file.size,
          type: file.type,
        });
      }

      this.updateState({
        status: 'uploading',
        progress: 0.2,
        currentStep: 'Validating uploaded image...',
        imageInfo: {
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified,
        },
      });

      // Validate file format
      if (
        !RETINAL_ANALYSIS_CONSTANTS.SUPPORTED_FORMATS.includes(file.type as any)
      ) {
        throw new RetinalAnalysisError(
          `Unsupported file format: ${file.type}. Supported formats: ${RETINAL_ANALYSIS_CONSTANTS.SUPPORTED_FORMATS.join(', ')}`,
          'INVALID_FILE_FORMAT'
        );
      }

      // Validate file size
      if (file.size > RETINAL_ANALYSIS_CONSTANTS.MAX_FILE_SIZE) {
        throw new RetinalAnalysisError(
          `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Maximum size: ${(RETINAL_ANALYSIS_CONSTANTS.MAX_FILE_SIZE / 1024 / 1024).toFixed(1)}MB`,
          'FILE_TOO_LARGE'
        );
      }

      if (file.size < 1024) {
        throw new RetinalAnalysisError(
          'File too small. Please select a valid image file.',
          'FILE_TOO_SMALL'
        );
      }

      this.updateState({
        status: 'preprocessing',
        progress: 0.5,
        currentStep: 'Validating image resolution...',
      });

      // Validate image resolution
      await this.validateImageResolution(file);

      // Create data URL for preview
      const imageDataUrl = URL.createObjectURL(file);

      this.updateState({
        status: 'complete',
        progress: 1,
        currentStep: 'Image uploaded successfully',
        imageDataUrl,
        imageInfo: {
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified,
        },
      });

      if (this.debug) {
        console.log('[ImageCaptureSystem] File upload processed successfully');
      }

      return file;
    } catch (error) {
      console.error('[ImageCaptureSystem] File upload failed:', error);

      const errorMessage =
        error instanceof RetinalAnalysisError
          ? error.message
          : `File upload failed: ${error}`;

      this.updateState({
        status: 'error',
        progress: 0,
        currentStep: 'File upload failed',
        error: errorMessage,
      });

      throw error instanceof RetinalAnalysisError
        ? error
        : new RetinalAnalysisError(errorMessage, 'UPLOAD_ERROR', error);
    }
  }

  /**
   * Validate image resolution meets minimum requirements
   */
  private async validateImageResolution(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
      const image = new Image();
      const url = URL.createObjectURL(file);

      image.onload = () => {
        URL.revokeObjectURL(url);

        const { width, height } = image;
        const minRes = RETINAL_ANALYSIS_CONSTANTS.MIN_IMAGE_SIZE;

        if (width < minRes.width || height < minRes.height) {
          reject(
            new RetinalAnalysisError(
              `Image resolution too low: ${width}x${height}. Minimum required: ${minRes.width}x${minRes.height}`,
              'RESOLUTION_TOO_LOW'
            )
          );
        } else {
          resolve();
        }
      };

      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(
          new RetinalAnalysisError(
            'Invalid image file or corrupted data',
            'INVALID_IMAGE_DATA'
          )
        );
      };

      image.src = url;
    });
  }

  /**
   * Handle drag and drop file upload
   */
  handleDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  handleDragEnter(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  handleDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  async handleDrop(event: DragEvent): Promise<File | null> {
    event.preventDefault();
    event.stopPropagation();

    const files = event.dataTransfer?.files;
    if (!files || files.length === 0) {
      return null;
    }

    const file = files[0];
    if (!file) return null;
    return await this.handleFileUpload(file);
  }

  /**
   * Get video element for preview display
   */
  getVideoElement(): HTMLVideoElement | null {
    return this.videoElement;
  }

  /**
   * Update processing state and notify listeners
   */
  private updateState(state: Partial<RetinalProcessingState>): void {
    const currentState: RetinalProcessingState = {
      status: 'idle',
      progress: 0,
      currentStep: '',
      ...state,
    };

    this.onStateChange(currentState);
  }

  /**
   * Stop camera stream and cleanup resources
   */
  async dispose(): Promise<void> {
    try {
      // Stop camera stream
      if (this.currentStream) {
        this.currentStream.getTracks().forEach((track) => track.stop());
        this.currentStream = null;
      }

      // Cleanup video element
      if (this.videoElement) {
        this.videoElement.srcObject = null;
        this.videoElement = null;
      }

      // Cleanup canvas
      this.canvasElement = null;

      if (this.debug) {
        console.log('[ImageCaptureSystem] Resources disposed');
      }
    } catch (error) {
      console.error('[ImageCaptureSystem] Error during disposal:', error);
    }
  }
}
