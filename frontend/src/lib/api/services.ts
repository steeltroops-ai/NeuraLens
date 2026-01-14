/**
 * API Service Classes for Assessment Modalities
 * Provides typed interfaces for backend API communication
 */

import { apiClient, ApiResponse } from './client';
import {
    SpeechAnalysisRequest,
    SpeechAnalysisResponse,
    RetinalAnalysisRequest,
    RetinalAnalysisResponse,
    MotorAssessmentRequest,
    MotorAssessmentResponse,
    CognitiveAssessmentRequest,
    CognitiveAssessmentResponse,
    NRIFusionRequest,
    NRIFusionResponse,
} from './types';

/**
 * Speech Analysis Service
 * Handles voice biomarker analysis for neurological assessment
 */
export const SpeechAnalysisService = {
    /**
     * Analyze speech audio for biomarkers
     */
    async analyze(request: SpeechAnalysisRequest): Promise<ApiResponse<SpeechAnalysisResponse>> {
        const formData = new FormData();
        formData.append('session_id', request.session_id);
        formData.append('audio_file', request.audio_file);
        if (request.quality_threshold) {
            formData.append('quality_threshold', request.quality_threshold.toString());
        }

        return apiClient.postFormData<SpeechAnalysisResponse>('/speech/analyze', formData);
    },

    /**
     * Get speech analysis results by session ID
     */
    async getResults(sessionId: string): Promise<ApiResponse<SpeechAnalysisResponse>> {
        return apiClient.get<SpeechAnalysisResponse>(`/speech/results/${sessionId}`);
    },

    /**
     * Validate audio file before analysis
     */
    async validate(file: File): Promise<ApiResponse<{ valid: boolean; errors: string[] }>> {
        const formData = new FormData();
        formData.append('audio_file', file);
        return apiClient.postFormData('/speech/validate', formData);
    },
};

/**
 * Retinal Analysis Service
 * Handles retinal image analysis for neurological biomarkers
 */
export const RetinalAnalysisService = {
    /**
     * Analyze retinal image for biomarkers
     */
    async analyze(request: RetinalAnalysisRequest): Promise<ApiResponse<RetinalAnalysisResponse>> {
        const formData = new FormData();
        formData.append('session_id', request.session_id);
        formData.append('image_file', request.image_file);
        if (request.quality_threshold) {
            formData.append('quality_threshold', request.quality_threshold.toString());
        }

        return apiClient.postFormData<RetinalAnalysisResponse>('/retinal/analyze', formData);
    },

    /**
     * Get retinal analysis results by session ID
     */
    async getResults(sessionId: string): Promise<ApiResponse<RetinalAnalysisResponse>> {
        return apiClient.get<RetinalAnalysisResponse>(`/retinal/results/${sessionId}`);
    },

    /**
     * Validate image file before analysis
     */
    async validate(file: File): Promise<ApiResponse<{ valid: boolean; errors: string[] }>> {
        const formData = new FormData();
        formData.append('image_file', file);
        return apiClient.postFormData('/retinal/validate', formData);
    },
};

/**
 * Motor Assessment Service
 * Handles motor function analysis from sensor data
 */
export const MotorAssessmentService = {
    /**
     * Analyze motor sensor data
     */
    async analyze(request: MotorAssessmentRequest): Promise<ApiResponse<MotorAssessmentResponse>> {
        return apiClient.post<MotorAssessmentResponse>('/motor/analyze', request);
    },

    /**
     * Get motor assessment results by session ID
     */
    async getResults(sessionId: string): Promise<ApiResponse<MotorAssessmentResponse>> {
        return apiClient.get<MotorAssessmentResponse>(`/motor/results/${sessionId}`);
    },

    /**
     * Validate sensor data before analysis
     */
    async validate(
        data: MotorAssessmentRequest['sensor_data'],
    ): Promise<ApiResponse<{ valid: boolean; errors: string[] }>> {
        return apiClient.post('/motor/validate', { sensor_data: data });
    },
};

/**
 * Cognitive Assessment Service
 * Handles cognitive test result analysis
 */
export const CognitiveAssessmentService = {
    /**
     * Analyze cognitive test results
     */
    async analyze(
        request: CognitiveAssessmentRequest,
    ): Promise<ApiResponse<CognitiveAssessmentResponse>> {
        return apiClient.post<CognitiveAssessmentResponse>('/cognitive/analyze', request);
    },

    /**
     * Get cognitive assessment results by session ID
     */
    async getResults(sessionId: string): Promise<ApiResponse<CognitiveAssessmentResponse>> {
        return apiClient.get<CognitiveAssessmentResponse>(`/cognitive/results/${sessionId}`);
    },

    /**
     * Validate test results before analysis
     */
    async validate(
        data: CognitiveAssessmentRequest['test_results'],
    ): Promise<ApiResponse<{ valid: boolean; errors: string[] }>> {
        return apiClient.post('/cognitive/validate', { test_results: data });
    },
};

/**
 * NRI Fusion Service
 * Handles multi-modal neurological risk index calculation
 */
export const NRIFusionService = {
    /**
     * Calculate NRI fusion from modality results
     */
    async fusion(request: NRIFusionRequest): Promise<ApiResponse<NRIFusionResponse>> {
        return apiClient.post<NRIFusionResponse>('/nri/fusion', request);
    },

    /**
     * Get NRI fusion results by session ID
     */
    async getResults(sessionId: string): Promise<ApiResponse<NRIFusionResponse>> {
        return apiClient.get<NRIFusionResponse>(`/nri/results/${sessionId}`);
    },

    /**
     * Get NRI calculation breakdown
     */
    async getBreakdown(
        sessionId: string,
    ): Promise<
        ApiResponse<{
            modality_weights: Record<string, number>;
            confidence_factors: Record<string, number>;
            risk_contributions: Record<string, number>;
        }>
    > {
        return apiClient.get(`/nri/breakdown/${sessionId}`);
    },
};
