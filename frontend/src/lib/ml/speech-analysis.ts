// NeuroLens-X Speech Analysis ML Pipeline
// Voice biomarker detection for neurological risk assessment

export interface SpeechFeatures {
  // Temporal Features
  speechRate: number; // Words per minute
  pauseDuration: number; // Average pause duration (ms)
  pauseFrequency: number; // Pauses per minute
  articulationRate: number; // Phonemes per second

  // Acoustic Features
  fundamentalFreq: number; // F0 mean (Hz)
  f0Variability: number; // F0 standard deviation
  jitter: number; // F0 perturbation
  shimmer: number; // Amplitude perturbation

  // Voice Quality
  harmonicNoiseRatio: number; // HNR (dB)
  spectralCentroid: number; // Spectral centroid (Hz)
  spectralBandwidth: number; // Spectral bandwidth (Hz)

  // Tremor Detection
  tremorFrequency: number; // Tremor frequency (Hz)
  tremorAmplitude: number; // Tremor amplitude
  voiceTremor: number; // Voice tremor index (0-1)

  // Prosodic Features
  stressPattern: number[]; // Stress pattern analysis
  intonationRange: number; // Pitch range (semitones)
  rhythmVariability: number; // Rhythm consistency
}

export interface SpeechAnalysisResult {
  features: SpeechFeatures;
  riskScore: number; // 0-100 neurological risk score
  confidence: number; // Confidence interval (±%)
  findings: string[]; // Clinical findings
  processingTime: number; // Analysis time (ms)
  qualityScore: number; // Audio quality score (0-1)
}

export class SpeechAnalyzer {
  private audioContext: AudioContext | null = null;
  private sampleRate: number = 44100;
  private windowSize: number = 2048;
  private hopLength: number = 512;

  constructor() {
    if (typeof window !== 'undefined' && 'AudioContext' in window) {
      this.audioContext = new AudioContext();
    }
  }

  /**
   * Analyze speech from audio file or blob
   */
  async analyzeSpeech(audioData: ArrayBuffer): Promise<SpeechAnalysisResult> {
    const startTime = performance.now();

    try {
      // Decode audio data
      const audioBuffer = await this.decodeAudioData(audioData);

      // Extract features
      const features = await this.extractFeatures(audioBuffer);

      // Calculate risk score
      const riskScore = this.calculateRiskScore(features);

      // Generate clinical findings
      const findings = this.generateFindings(features, riskScore);

      // Calculate confidence
      const confidence = this.calculateConfidence(features);

      // Assess audio quality
      const qualityScore = this.assessAudioQuality(audioBuffer);

      const processingTime = performance.now() - startTime;

      return {
        features,
        riskScore,
        confidence,
        findings,
        processingTime,
        qualityScore,
      };
    } catch (error) {
      console.error('Speech analysis failed:', error);
      throw new Error('Failed to analyze speech: ' + (error as Error).message);
    }
  }

  /**
   * Decode audio data to AudioBuffer
   */
  private async decodeAudioData(audioData: ArrayBuffer): Promise<AudioBuffer> {
    if (!this.audioContext) {
      throw new Error('AudioContext not available');
    }

    return await this.audioContext.decodeAudioData(audioData);
  }

  /**
   * Extract comprehensive speech features
   */
  private async extractFeatures(
    audioBuffer: AudioBuffer
  ): Promise<SpeechFeatures> {
    const channelData = audioBuffer.getChannelData(0);
    const sampleRate = audioBuffer.sampleRate;

    // Temporal features
    const temporalFeatures = this.extractTemporalFeatures(
      channelData,
      sampleRate
    );

    // Acoustic features
    const acousticFeatures = this.extractAcousticFeatures(
      channelData,
      sampleRate
    );

    // Voice quality features
    const voiceQualityFeatures = this.extractVoiceQualityFeatures(
      channelData,
      sampleRate
    );

    // Tremor detection
    const tremorFeatures = this.detectTremor(channelData, sampleRate);

    // Prosodic features
    const prosodicFeatures = this.extractProsodicFeatures(
      channelData,
      sampleRate
    );

    return {
      ...temporalFeatures,
      ...acousticFeatures,
      ...voiceQualityFeatures,
      ...tremorFeatures,
      ...prosodicFeatures,
    };
  }

  /**
   * Extract temporal features (speech rate, pauses, etc.)
   */
  private extractTemporalFeatures(audioData: Float32Array, sampleRate: number) {
    // Voice Activity Detection (VAD)
    const vadResults = this.voiceActivityDetection(audioData, sampleRate);

    // Calculate speech segments and pauses
    const speechSegments = vadResults.speechSegments;
    const pauseSegments = vadResults.pauseSegments;

    // Speech rate calculation (simplified)
    const totalSpeechTime = speechSegments.reduce(
      (sum, seg) => sum + (seg.end - seg.start),
      0
    );
    const estimatedWords = Math.floor(totalSpeechTime * 2.5); // ~2.5 words per second average
    const speechRate = estimatedWords / (totalSpeechTime / 60); // words per minute

    // Pause analysis
    const pauseDurations = pauseSegments.map(
      (seg) => (seg.end - seg.start) * 1000
    ); // ms
    const pauseDuration =
      pauseDurations.length > 0
        ? pauseDurations.reduce((sum, dur) => sum + dur, 0) /
          pauseDurations.length
        : 0;
    const pauseFrequency =
      (pauseSegments.length / (audioData.length / sampleRate)) * 60; // per minute

    // Articulation rate
    const articulationRate = (speechRate / 60) * 4; // approximate phonemes per second

    return {
      speechRate,
      pauseDuration,
      pauseFrequency,
      articulationRate,
    };
  }

  /**
   * Extract acoustic features (F0, jitter, shimmer)
   */
  private extractAcousticFeatures(audioData: Float32Array, sampleRate: number) {
    // Fundamental frequency extraction using autocorrelation
    const f0Values = this.extractF0(audioData, sampleRate);
    const validF0 = f0Values.filter((f0) => f0 > 0);

    const fundamentalFreq =
      validF0.length > 0
        ? validF0.reduce((sum, f0) => sum + f0, 0) / validF0.length
        : 0;

    const f0Variability =
      validF0.length > 1
        ? Math.sqrt(
            validF0.reduce(
              (sum, f0) => sum + Math.pow(f0 - fundamentalFreq, 2),
              0
            ) /
              (validF0.length - 1)
          )
        : 0;

    // Jitter calculation (F0 perturbation)
    const jitter = this.calculateJitter(validF0);

    // Shimmer calculation (amplitude perturbation)
    const shimmer = this.calculateShimmer(audioData, sampleRate);

    return {
      fundamentalFreq,
      f0Variability,
      jitter,
      shimmer,
    };
  }

  /**
   * Extract voice quality features
   */
  private extractVoiceQualityFeatures(
    audioData: Float32Array,
    sampleRate: number
  ) {
    // Harmonic-to-Noise Ratio
    const harmonicNoiseRatio = this.calculateHNR(audioData, sampleRate);

    // Spectral features
    const spectralFeatures = this.extractSpectralFeatures(
      audioData,
      sampleRate
    );

    return {
      harmonicNoiseRatio,
      spectralCentroid: spectralFeatures.centroid,
      spectralBandwidth: spectralFeatures.bandwidth,
    };
  }

  /**
   * Detect voice tremor
   */
  private detectTremor(audioData: Float32Array, sampleRate: number) {
    // Extract amplitude envelope
    const envelope = this.extractAmplitudeEnvelope(audioData, sampleRate);

    // Analyze tremor in 4-12 Hz range (typical for neurological tremor)
    const tremorAnalysis = this.analyzeTremorFrequency(envelope, sampleRate);

    return {
      tremorFrequency: tremorAnalysis.frequency,
      tremorAmplitude: tremorAnalysis.amplitude,
      voiceTremor: tremorAnalysis.tremorIndex,
    };
  }

  /**
   * Extract prosodic features
   */
  private extractProsodicFeatures(audioData: Float32Array, sampleRate: number) {
    // Simplified prosodic analysis
    const f0Values = this.extractF0(audioData, sampleRate);
    const validF0 = f0Values.filter((f0) => f0 > 0);

    // Intonation range (pitch range in semitones)
    const minF0 = Math.min(...validF0);
    const maxF0 = Math.max(...validF0);
    const intonationRange =
      validF0.length > 0 ? 12 * Math.log2(maxF0 / minF0) : 0;

    // Stress pattern (simplified)
    const stressPattern = this.analyzeStressPattern(audioData, sampleRate);

    // Rhythm variability
    const rhythmVariability = this.calculateRhythmVariability(
      audioData,
      sampleRate
    );

    return {
      stressPattern,
      intonationRange,
      rhythmVariability,
    };
  }

  /**
   * Voice Activity Detection
   */
  private voiceActivityDetection(audioData: Float32Array, sampleRate: number) {
    const frameSize = Math.floor(sampleRate * 0.025); // 25ms frames
    const hopSize = Math.floor(sampleRate * 0.01); // 10ms hop

    const speechSegments: Array<{ start: number; end: number }> = [];
    const pauseSegments: Array<{ start: number; end: number }> = [];

    let inSpeech = false;
    let segmentStart = 0;

    for (let i = 0; i < audioData.length - frameSize; i += hopSize) {
      const frame = audioData.slice(i, i + frameSize);
      const energy =
        frame.reduce((sum, sample) => sum + sample * sample, 0) / frame.length;
      const isSpeech = energy > 0.001; // Simple energy threshold

      if (isSpeech && !inSpeech) {
        // Start of speech segment
        if (segmentStart < i / sampleRate) {
          pauseSegments.push({ start: segmentStart, end: i / sampleRate });
        }
        segmentStart = i / sampleRate;
        inSpeech = true;
      } else if (!isSpeech && inSpeech) {
        // End of speech segment
        speechSegments.push({ start: segmentStart, end: i / sampleRate });
        segmentStart = i / sampleRate;
        inSpeech = false;
      }
    }

    return { speechSegments, pauseSegments };
  }

  /**
   * Calculate risk score based on extracted features
   */
  private calculateRiskScore(features: SpeechFeatures): number {
    let riskScore = 0;
    let weightSum = 0;

    // Speech rate (normal: 150-200 wpm)
    const speechRateRisk =
      features.speechRate < 120 || features.speechRate > 220
        ? Math.abs(features.speechRate - 175) / 175
        : 0;
    riskScore += speechRateRisk * 15;
    weightSum += 15;

    // Pause frequency (normal: 2-4 per minute)
    const pauseRisk =
      features.pauseFrequency > 6 ? (features.pauseFrequency - 4) / 10 : 0;
    riskScore += pauseRisk * 20;
    weightSum += 20;

    // Voice tremor
    riskScore += features.voiceTremor * 25;
    weightSum += 25;

    // Jitter (normal: <1%)
    const jitterRisk =
      features.jitter > 0.01 ? Math.min((features.jitter - 0.01) / 0.02, 1) : 0;
    riskScore += jitterRisk * 15;
    weightSum += 15;

    // HNR (normal: >20 dB)
    const hnrRisk =
      features.harmonicNoiseRatio < 15
        ? (20 - features.harmonicNoiseRatio) / 20
        : 0;
    riskScore += hnrRisk * 15;
    weightSum += 15;

    // Rhythm variability
    riskScore += Math.min(features.rhythmVariability, 1) * 10;
    weightSum += 10;

    return Math.min(Math.round((riskScore / weightSum) * 100), 100);
  }

  /**
   * Generate clinical findings based on features
   */
  private generateFindings(
    features: SpeechFeatures,
    riskScore: number
  ): string[] {
    const findings: string[] = [];

    if (features.speechRate < 120) {
      findings.push('Reduced speech rate detected');
    }
    if (features.pauseFrequency > 6) {
      findings.push('Increased pause frequency observed');
    }
    if (features.voiceTremor > 0.3) {
      findings.push('Voice tremor detected');
    }
    if (features.jitter > 0.015) {
      findings.push('Elevated voice jitter');
    }
    if (features.harmonicNoiseRatio < 15) {
      findings.push('Reduced voice quality (low HNR)');
    }
    if (features.rhythmVariability > 0.7) {
      findings.push('Irregular speech rhythm');
    }

    if (findings.length === 0) {
      findings.push('No significant speech abnormalities detected');
    }

    return findings;
  }

  /**
   * Calculate confidence interval
   */
  private calculateConfidence(features: SpeechFeatures): number {
    // Base confidence on feature reliability and audio quality
    let confidence = 95;

    // Reduce confidence for edge cases
    if (features.speechRate < 50 || features.speechRate > 300) confidence -= 20;
    if (features.fundamentalFreq < 80 || features.fundamentalFreq > 400)
      confidence -= 15;

    return Math.max(confidence, 60);
  }

  /**
   * Assess audio quality
   */
  private assessAudioQuality(audioBuffer: AudioBuffer): number {
    const channelData = audioBuffer.getChannelData(0);

    // Check for clipping
    const clippedSamples = channelData.filter(
      (sample) => Math.abs(sample) > 0.95
    ).length;
    const clippingRatio = clippedSamples / channelData.length;

    // Check signal-to-noise ratio (simplified)
    const signalPower =
      channelData.reduce((sum, sample) => sum + sample * sample, 0) /
      channelData.length;
    const snr = 10 * Math.log10(signalPower / 0.001); // Assuming noise floor

    // Quality score (0-1)
    let quality = 1.0;
    quality -= clippingRatio * 0.5; // Penalize clipping
    quality -= Math.max(0, (20 - snr) / 20) * 0.3; // Penalize low SNR

    return Math.max(quality, 0.1);
  }

  // Simplified implementations of complex audio processing functions
  private extractF0(audioData: Float32Array, sampleRate: number): number[] {
    // Simplified F0 extraction using autocorrelation
    const f0Values: number[] = [];
    const frameSize = Math.floor(sampleRate * 0.025);
    const hopSize = Math.floor(sampleRate * 0.01);

    for (let i = 0; i < audioData.length - frameSize; i += hopSize) {
      const frame = audioData.slice(i, i + frameSize);
      const f0 = this.autocorrelationF0(frame, sampleRate);
      f0Values.push(f0);
    }

    return f0Values;
  }

  private autocorrelationF0(frame: Float32Array, sampleRate: number): number {
    // Simplified autocorrelation-based F0 estimation
    const minPeriod = Math.floor(sampleRate / 500); // 500 Hz max
    const maxPeriod = Math.floor(sampleRate / 50); // 50 Hz min

    let maxCorr = 0;
    let bestPeriod = 0;

    for (let period = minPeriod; period <= maxPeriod; period++) {
      let correlation = 0;
      for (let i = 0; i < frame.length - period; i++) {
        const val1 = frame[i] || 0;
        const val2 = frame[i + period] || 0;
        correlation += val1 * val2;
      }

      if (correlation > maxCorr) {
        maxCorr = correlation;
        bestPeriod = period;
      }
    }

    return bestPeriod > 0 ? sampleRate / bestPeriod : 0;
  }

  private calculateJitter(f0Values: number[]): number {
    if (f0Values.length < 2) return 0;

    let jitterSum = 0;
    for (let i = 1; i < f0Values.length; i++) {
      const current = f0Values[i] || 0;
      const previous = f0Values[i - 1] || 0;
      if (current > 0 && previous > 0) {
        jitterSum += Math.abs(current - previous) / previous;
      }
    }

    return jitterSum / (f0Values.length - 1);
  }

  private calculateShimmer(
    audioData: Float32Array,
    sampleRate: number
  ): number {
    // Simplified shimmer calculation
    const frameSize = Math.floor(sampleRate * 0.025);
    const hopSize = Math.floor(sampleRate * 0.01);
    const amplitudes: number[] = [];

    for (let i = 0; i < audioData.length - frameSize; i += hopSize) {
      const frame = audioData.slice(i, i + frameSize);
      const amplitude = Math.sqrt(
        frame.reduce((sum, sample) => sum + sample * sample, 0) / frame.length
      );
      amplitudes.push(amplitude);
    }

    if (amplitudes.length < 2) return 0;

    let shimmerSum = 0;
    for (let i = 1; i < amplitudes.length; i++) {
      const current = amplitudes[i] || 0;
      const previous = amplitudes[i - 1] || 0;
      if (current > 0 && previous > 0) {
        shimmerSum += Math.abs(current - previous) / previous;
      }
    }

    return shimmerSum / (amplitudes.length - 1);
  }

  private calculateHNR(audioData: Float32Array, sampleRate: number): number {
    // Simplified HNR calculation
    // In a real implementation, this would use more sophisticated harmonic analysis
    const frameSize = Math.floor(sampleRate * 0.025);
    const frame = audioData.slice(0, Math.min(frameSize, audioData.length));

    // Calculate signal power
    const signalPower =
      frame.reduce((sum, sample) => sum + sample * sample, 0) / frame.length;

    // Estimate noise power (simplified)
    const noisePower = signalPower * 0.1; // Assume 10% noise

    return 10 * Math.log10(signalPower / noisePower);
  }

  private extractSpectralFeatures(audioData: Float32Array, sampleRate: number) {
    // Simplified spectral analysis
    // In a real implementation, this would use FFT
    return {
      centroid: 1000, // Placeholder
      bandwidth: 2000, // Placeholder
    };
  }

  private extractAmplitudeEnvelope(
    audioData: Float32Array,
    sampleRate: number
  ): Float32Array {
    // Simplified envelope extraction
    const windowSize = Math.floor(sampleRate * 0.01); // 10ms windows
    const envelope = new Float32Array(
      Math.floor(audioData.length / windowSize)
    );

    for (let i = 0; i < envelope.length; i++) {
      const start = i * windowSize;
      const end = Math.min(start + windowSize, audioData.length);
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += Math.abs(audioData[j] || 0);
      }
      envelope[i] = sum / (end - start);
    }

    return envelope;
  }

  private analyzeTremorFrequency(envelope: Float32Array, sampleRate: number) {
    // Simplified tremor analysis
    // In a real implementation, this would use spectral analysis of the envelope
    return {
      frequency: 5.0, // Placeholder: typical tremor frequency
      amplitude: 0.1, // Placeholder
      tremorIndex: 0.2, // Placeholder
    };
  }

  private analyzeStressPattern(
    audioData: Float32Array,
    sampleRate: number
  ): number[] {
    // Simplified stress pattern analysis
    return [0.5, 0.3, 0.7, 0.4]; // Placeholder
  }

  private calculateRhythmVariability(
    audioData: Float32Array,
    sampleRate: number
  ): number {
    // Simplified rhythm analysis
    return 0.3; // Placeholder
  }
}

// Export singleton instance
export const speechAnalyzer = new SpeechAnalyzer();
