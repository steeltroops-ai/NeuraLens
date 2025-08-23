/**
 * Audio Recorder for Neuralens Speech Analysis
 * 
 * This utility class handles WebRTC audio recording for speech analysis.
 * It provides real-time audio capture with visual feedback and automatic
 * processing integration with the Speech Processor.
 * 
 * Key Features:
 * - WebRTC MediaRecorder API integration
 * - Real-time audio level monitoring for visual feedback
 * - Automatic 30-second recording duration
 * - Audio quality validation and error handling
 * - Integration with Neuro-Minimalist UI design system
 * 
 * Technical Implementation:
 * - Uses getUserMedia for microphone access
 * - Implements AudioContext for real-time analysis
 * - Provides callback-based state management
 * - Handles browser compatibility and permissions
 */

import { RecordingState, AudioConfig, SpeechAnalysisError } from '../../types/speech-analysis';

export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private microphone: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private recordedChunks: Blob[] = [];
  private recordingTimer: NodeJS.Timeout | null = null;
  private animationFrame: number | null = null;
  
  private config: AudioConfig;
  private onStateChange: (state: RecordingState) => void;
  private debug: boolean;

  constructor(
    config: Partial<AudioConfig> = {},
    onStateChange: (state: RecordingState) => void,
    debug = false
  ) {
    this.config = {
      sampleRate: 16000,
      duration: 30,
      minAudioLevel: 0.01,
      noiseReduction: true,
      ...config
    };
    
    this.onStateChange = onStateChange;
    this.debug = debug;

    if (this.debug) {
      console.log('[AudioRecorder] Initialized with config:', this.config);
    }
  }

  /**
   * Initialize audio recording system
   * Requests microphone permissions and sets up audio context
   */
  async initialize(): Promise<void> {
    try {
      if (this.debug) {
        console.log('[AudioRecorder] Requesting microphone access...');
      }

      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new SpeechAnalysisError(
          'WebRTC not supported in this browser',
          'WEBRTC_NOT_SUPPORTED'
        );
      }

      // Request microphone access with audio constraints
      const constraints: MediaStreamConstraints = {
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: this.config.noiseReduction,
          autoGainControl: true
        }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Create audio context for real-time analysis
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: this.config.sampleRate
      });

      // Set up audio analysis nodes
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      this.microphone = this.audioContext.createMediaStreamSource(this.stream);
      this.microphone.connect(this.analyser);

      // Initialize MediaRecorder
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: this.getSupportedMimeType()
      });

      this.setupMediaRecorderEvents();

      if (this.debug) {
        console.log('[AudioRecorder] Initialization complete');
      }

      // Update state to idle
      this.updateState({
        status: 'idle',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0
      });

    } catch (error) {
      console.error('[AudioRecorder] Initialization failed:', error);
      
      let errorMessage = 'Failed to initialize audio recorder';
      let errorCode = 'INITIALIZATION_ERROR';
      
      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError') {
          errorMessage = 'Microphone access denied. Please allow microphone permissions.';
          errorCode = 'PERMISSION_DENIED';
        } else if (error.name === 'NotFoundError') {
          errorMessage = 'No microphone found. Please connect a microphone.';
          errorCode = 'NO_MICROPHONE';
        }
      }

      this.updateState({
        status: 'error',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0,
        error: errorMessage
      });

      throw new SpeechAnalysisError(errorMessage, errorCode, error);
    }
  }

  /**
   * Start audio recording
   * Begins 30-second recording with real-time audio level monitoring
   */
  async startRecording(): Promise<void> {
    try {
      if (!this.mediaRecorder || !this.audioContext || !this.analyser) {
        throw new SpeechAnalysisError(
          'Audio recorder not initialized',
          'NOT_INITIALIZED'
        );
      }

      if (this.mediaRecorder.state !== 'inactive') {
        throw new SpeechAnalysisError(
          'Recording already in progress',
          'ALREADY_RECORDING'
        );
      }

      if (this.debug) {
        console.log('[AudioRecorder] Starting recording...');
      }

      // Clear previous recording data
      this.recordedChunks = [];

      // Start recording
      this.mediaRecorder.start(100); // Collect data every 100ms

      // Update state to recording
      this.updateState({
        status: 'recording',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0
      });

      // Start audio level monitoring
      this.startAudioLevelMonitoring();

      // Set recording timer for automatic stop
      this.recordingTimer = setTimeout(() => {
        this.stopRecording();
      }, this.config.duration * 1000);

    } catch (error) {
      console.error('[AudioRecorder] Failed to start recording:', error);
      
      this.updateState({
        status: 'error',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0,
        error: `Failed to start recording: ${error}`
      });

      throw error;
    }
  }

  /**
   * Stop audio recording
   * Ends recording and prepares audio data for processing
   */
  stopRecording(): void {
    try {
      if (this.debug) {
        console.log('[AudioRecorder] Stopping recording...');
      }

      // Stop MediaRecorder
      if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.stop();
      }

      // Clear recording timer
      if (this.recordingTimer) {
        clearTimeout(this.recordingTimer);
        this.recordingTimer = null;
      }

      // Stop audio level monitoring
      this.stopAudioLevelMonitoring();

      // Update state to processing
      this.updateState({
        status: 'processing',
        progress: 1,
        audioLevel: 0,
        recordedDuration: this.config.duration
      });

    } catch (error) {
      console.error('[AudioRecorder] Failed to stop recording:', error);
      
      this.updateState({
        status: 'error',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0,
        error: `Failed to stop recording: ${error}`
      });
    }
  }

  /**
   * Get recorded audio as AudioBuffer for processing
   * Converts recorded Blob to AudioBuffer for ML analysis
   */
  async getAudioBuffer(): Promise<AudioBuffer> {
    try {
      if (this.recordedChunks.length === 0) {
        throw new SpeechAnalysisError(
          'No audio data recorded',
          'NO_AUDIO_DATA'
        );
      }

      if (!this.audioContext) {
        throw new SpeechAnalysisError(
          'Audio context not available',
          'NO_AUDIO_CONTEXT'
        );
      }

      // Create blob from recorded chunks
      const audioBlob = new Blob(this.recordedChunks, { 
        type: this.getSupportedMimeType() 
      });

      // Convert blob to array buffer
      const arrayBuffer = await audioBlob.arrayBuffer();

      // Decode audio data
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      if (this.debug) {
        console.log('[AudioRecorder] Audio buffer created:', {
          duration: audioBuffer.duration,
          sampleRate: audioBuffer.sampleRate,
          channels: audioBuffer.numberOfChannels
        });
      }

      return audioBuffer;

    } catch (error) {
      console.error('[AudioRecorder] Failed to create audio buffer:', error);
      throw new SpeechAnalysisError(
        `Failed to process recorded audio: ${error}`,
        'AUDIO_PROCESSING_ERROR',
        error
      );
    }
  }

  /**
   * Start real-time audio level monitoring for visual feedback
   */
  private startAudioLevelMonitoring(): void {
    if (!this.analyser) return;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const startTime = Date.now();

    const updateAudioLevel = () => {
      if (!this.analyser) return;

      this.analyser.getByteFrequencyData(dataArray);
      
      // Calculate RMS audio level
      const sum = dataArray.reduce((acc, value) => acc + value * value, 0);
      const rms = Math.sqrt(sum / bufferLength);
      const audioLevel = rms / 255; // Normalize to 0-1

      // Calculate recording progress
      const elapsed = (Date.now() - startTime) / 1000;
      const progress = Math.min(elapsed / this.config.duration, 1);

      // Update state with current levels
      this.updateState({
        status: 'recording',
        progress,
        audioLevel,
        recordedDuration: elapsed
      });

      // Continue monitoring if still recording
      if (this.mediaRecorder?.state === 'recording') {
        this.animationFrame = requestAnimationFrame(updateAudioLevel);
      }
    };

    this.animationFrame = requestAnimationFrame(updateAudioLevel);
  }

  /**
   * Stop audio level monitoring
   */
  private stopAudioLevelMonitoring(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  /**
   * Set up MediaRecorder event handlers
   */
  private setupMediaRecorderEvents(): void {
    if (!this.mediaRecorder) return;

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.recordedChunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      if (this.debug) {
        console.log('[AudioRecorder] Recording stopped, data chunks:', this.recordedChunks.length);
      }
    };

    this.mediaRecorder.onerror = (event) => {
      console.error('[AudioRecorder] MediaRecorder error:', event);
      
      this.updateState({
        status: 'error',
        progress: 0,
        audioLevel: 0,
        recordedDuration: 0,
        error: 'Recording failed due to media recorder error'
      });
    };
  }

  /**
   * Get supported MIME type for MediaRecorder
   */
  private getSupportedMimeType(): string {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/wav'
    ];

    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return 'audio/webm'; // Fallback
  }

  /**
   * Update recording state and notify listeners
   */
  private updateState(state: RecordingState): void {
    this.onStateChange(state);
  }

  /**
   * Cleanup resources and stop recording
   */
  async dispose(): Promise<void> {
    try {
      // Stop recording if active
      if (this.mediaRecorder?.state === 'recording') {
        this.stopRecording();
      }

      // Stop audio level monitoring
      this.stopAudioLevelMonitoring();

      // Clear timer
      if (this.recordingTimer) {
        clearTimeout(this.recordingTimer);
        this.recordingTimer = null;
      }

      // Disconnect audio nodes
      if (this.microphone) {
        this.microphone.disconnect();
        this.microphone = null;
      }

      // Close audio context
      if (this.audioContext && this.audioContext.state !== 'closed') {
        await this.audioContext.close();
        this.audioContext = null;
      }

      // Stop media stream
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }

      // Clear references
      this.mediaRecorder = null;
      this.analyser = null;
      this.recordedChunks = [];

      if (this.debug) {
        console.log('[AudioRecorder] Resources disposed');
      }

    } catch (error) {
      console.error('[AudioRecorder] Error during disposal:', error);
    }
  }
}
