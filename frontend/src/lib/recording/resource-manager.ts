/**
 * Recording Resource Manager
 * 
 * Manages audio recording resources including MediaRecorder, MediaStream,
 * AudioContext, and timers. Provides cleanup functionality for component
 * unmount and navigation events.
 * 
 * @module recording/resource-manager
 * @validates Requirements 5.1, 5.2
 */

/**
 * Configuration for the resource manager
 */
export interface ResourceManagerConfig {
    targetSampleRate?: number;
    maxRecordingTime?: number;
    minRecordingTime?: number;
    audioConstraints?: MediaTrackConstraints;
}

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: Required<ResourceManagerConfig> = {
    targetSampleRate: 16000,
    maxRecordingTime: 120,
    minRecordingTime: 5,
    audioConstraints: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
    },
};

/**
 * Callback types for resource manager events
 */
export interface ResourceManagerCallbacks {
    onDataAvailable?: (data: Blob) => void;
    onRecordingStop?: (blob: Blob) => void;
    onAudioLevel?: (level: number) => void;
    onError?: (error: Error) => void;
}

/**
 * Recording Resource Manager class
 * 
 * Manages all audio recording resources and provides cleanup functionality
 * to prevent memory leaks and ensure proper resource release.
 */
export class RecordingResourceManager {
    private _config: Required<ResourceManagerConfig>;
    private _callbacks: ResourceManagerCallbacks;

    // Media resources
    private _mediaRecorder: MediaRecorder | null = null;
    private _mediaStream: MediaStream | null = null;
    private _audioContext: AudioContext | null = null;
    private _analyserNode: AnalyserNode | null = null;

    // Collected audio data
    private _audioChunks: Blob[] = [];
    private _recordedBlob: Blob | null = null;

    // Timers and animation frames
    private _recordingTimer: ReturnType<typeof setInterval> | null = null;
    private _animationFrameId: number | null = null;

    // State tracking
    private _isInitialized = false;
    private _isRecording = false;
    private _isCleanedUp = false;

    constructor(
        config: ResourceManagerConfig = {},
        callbacks: ResourceManagerCallbacks = {}
    ) {
        this._config = { ...DEFAULT_CONFIG, ...config };
        this._callbacks = callbacks;
    }

    /**
     * Check if resources are initialized
     */
    get isInitialized(): boolean {
        return this._isInitialized;
    }

    /**
     * Check if currently recording
     */
    get isRecording(): boolean {
        return this._isRecording;
    }

    /**
     * Get the recorded blob
     */
    get recordedBlob(): Blob | null {
        return this._recordedBlob;
    }

    /**
     * Get the analyser node for audio visualization
     */
    get analyserNode(): AnalyserNode | null {
        return this._analyserNode;
    }

    /**
     * Initialize audio recording resources
     * @throws Error if initialization fails
     */
    async initialize(): Promise<void> {
        if (this._isCleanedUp) {
            throw new Error('Resource manager has been cleaned up and cannot be reused');
        }

        if (this._isInitialized) {
            return;
        }

        try {
            // Request microphone permission and get media stream
            this._mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: this._config.audioConstraints,
            });

            // Create audio context for level monitoring
            this._audioContext = new AudioContext({
                sampleRate: this._config.targetSampleRate,
            });

            // Create analyser node for audio visualization
            const source = this._audioContext.createMediaStreamSource(this._mediaStream);
            this._analyserNode = this._audioContext.createAnalyser();
            this._analyserNode.fftSize = 256;
            source.connect(this._analyserNode);

            // Create MediaRecorder with appropriate MIME type
            const mimeType = this._getSupportedMimeType();
            const options: MediaRecorderOptions = {
                mimeType,
            };

            this._mediaRecorder = new MediaRecorder(this._mediaStream, options);
            this._setupMediaRecorderEvents();

            this._isInitialized = true;
        } catch (error) {
            // Clean up any partially initialized resources
            this._cleanupMediaResources();
            throw error;
        }
    }

    /**
     * Get a supported MIME type for MediaRecorder
     */
    private _getSupportedMimeType(): string {
        const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/ogg;codecs=opus',
        ];

        for (const mimeType of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mimeType)) {
                return mimeType;
            }
        }

        // Fallback to default
        return '';
    }

    /**
     * Set up MediaRecorder event handlers
     */
    private _setupMediaRecorderEvents(): void {
        if (!this._mediaRecorder) return;

        this._mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this._audioChunks.push(event.data);
                this._callbacks.onDataAvailable?.(event.data);
            }
        };

        this._mediaRecorder.onstop = () => {
            this._recordedBlob = new Blob(this._audioChunks, { type: 'audio/wav' });
            this._callbacks.onRecordingStop?.(this._recordedBlob);
        };

        this._mediaRecorder.onerror = (event) => {
            const error = new Error(`MediaRecorder error: ${(event as any).error?.message || 'Unknown error'}`);
            this._callbacks.onError?.(error);
        };
    }

    /**
     * Start recording
     * @param timeslice - Time interval for data collection in ms
     */
    startRecording(timeslice = 100): void {
        if (!this._isInitialized || !this._mediaRecorder) {
            throw new Error('Resource manager not initialized');
        }

        if (this._isRecording) {
            return;
        }

        // Reset audio chunks
        this._audioChunks = [];
        this._recordedBlob = null;

        // Start MediaRecorder
        this._mediaRecorder.start(timeslice);
        this._isRecording = true;

        // Start audio level monitoring
        this._startAudioLevelMonitoring();
    }

    /**
     * Stop recording
     */
    stopRecording(): void {
        if (!this._isRecording || !this._mediaRecorder) {
            return;
        }

        // Stop MediaRecorder
        if (this._mediaRecorder.state !== 'inactive') {
            this._mediaRecorder.stop();
        }

        this._isRecording = false;

        // Stop audio level monitoring
        this._stopAudioLevelMonitoring();

        // Stop recording timer
        this._stopRecordingTimer();
    }

    /**
     * Pause recording
     */
    pauseRecording(): void {
        if (!this._isRecording || !this._mediaRecorder) {
            return;
        }

        if (this._mediaRecorder.state === 'recording') {
            this._mediaRecorder.pause();
            this._stopAudioLevelMonitoring();
        }
    }

    /**
     * Resume recording
     */
    resumeRecording(): void {
        if (!this._mediaRecorder) {
            return;
        }

        if (this._mediaRecorder.state === 'paused') {
            this._mediaRecorder.resume();
            this._startAudioLevelMonitoring();
        }
    }

    /**
     * Start audio level monitoring using requestAnimationFrame
     */
    private _startAudioLevelMonitoring(): void {
        if (!this._analyserNode) return;

        const dataArray = new Uint8Array(this._analyserNode.frequencyBinCount);

        const monitor = () => {
            if (!this._isRecording || !this._analyserNode) {
                return;
            }

            this._analyserNode.getByteFrequencyData(dataArray);

            // Calculate average level
            const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
            const normalizedLevel = Math.min(average / 128, 1);

            this._callbacks.onAudioLevel?.(normalizedLevel);

            this._animationFrameId = requestAnimationFrame(monitor);
        };

        this._animationFrameId = requestAnimationFrame(monitor);
    }

    /**
     * Stop audio level monitoring
     */
    private _stopAudioLevelMonitoring(): void {
        if (this._animationFrameId !== null) {
            cancelAnimationFrame(this._animationFrameId);
            this._animationFrameId = null;
        }
    }

    /**
     * Start recording timer
     * @param onTick - Callback for each second
     * @param onMaxTime - Callback when max time is reached
     */
    startRecordingTimer(
        onTick: (seconds: number) => void,
        onMaxTime?: () => void
    ): void {
        let seconds = 0;

        this._recordingTimer = setInterval(() => {
            seconds++;
            onTick(seconds);

            if (seconds >= this._config.maxRecordingTime) {
                onMaxTime?.();
                this.stopRecording();
            }
        }, 1000);
    }

    /**
     * Stop recording timer
     */
    private _stopRecordingTimer(): void {
        if (this._recordingTimer !== null) {
            clearInterval(this._recordingTimer);
            this._recordingTimer = null;
        }
    }

    /**
     * Clean up media resources (stream, context, recorder)
     */
    private _cleanupMediaResources(): void {
        // Stop and release media stream tracks
        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach(track => {
                track.stop();
            });
            this._mediaStream = null;
        }

        // Close audio context
        if (this._audioContext) {
            if (this._audioContext.state !== 'closed') {
                this._audioContext.close().catch(() => {
                    // Ignore errors during cleanup
                });
            }
            this._audioContext = null;
        }

        // Clear analyser node reference
        this._analyserNode = null;

        // Clear media recorder reference
        this._mediaRecorder = null;
    }

    /**
     * Clean up timer resources
     */
    private _cleanupTimers(): void {
        // Stop recording timer
        this._stopRecordingTimer();

        // Cancel animation frame
        this._stopAudioLevelMonitoring();
    }

    /**
     * Full cleanup - releases all resources
     * Call this on component unmount or navigation away
     * 
     * @validates Requirements 5.1, 5.2
     */
    cleanup(): void {
        if (this._isCleanedUp) {
            return;
        }

        // Stop recording if in progress
        if (this._isRecording) {
            this.stopRecording();
        }

        // Clean up timers first
        this._cleanupTimers();

        // Clean up media resources
        this._cleanupMediaResources();

        // Clear audio data
        this._audioChunks = [];
        this._recordedBlob = null;

        // Mark as cleaned up
        this._isInitialized = false;
        this._isCleanedUp = true;
    }

    /**
     * Reset for new recording (without full cleanup)
     */
    reset(): void {
        if (this._isRecording) {
            this.stopRecording();
        }

        this._audioChunks = [];
        this._recordedBlob = null;
    }

    /**
     * Check if minimum recording time has been met
     * @param recordingTime - Current recording time in seconds
     */
    hasMinimumRecordingTime(recordingTime: number): boolean {
        return recordingTime >= this._config.minRecordingTime;
    }

    /**
     * Get configuration
     */
    get config(): Required<ResourceManagerConfig> {
        return { ...this._config };
    }
}

/**
 * Create a new RecordingResourceManager instance
 * @param config - Optional configuration
 * @param callbacks - Optional callbacks
 * @returns New RecordingResourceManager instance
 */
export function createResourceManager(
    config?: ResourceManagerConfig,
    callbacks?: ResourceManagerCallbacks
): RecordingResourceManager {
    return new RecordingResourceManager(config, callbacks);
}
