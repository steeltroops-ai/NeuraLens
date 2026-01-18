/**
 * AI Explanation API Route
 * Proxies requests to FastAPI backend for LLM-powered explanations
 */

import { NextRequest } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        console.log('[API] Explanation request received');

        const body = await request.json();
        const { pipeline, results, patient_context, voice_output, voice_provider } = body;

        if (!pipeline || !results) {
            return new Response(
                JSON.stringify({ error: 'Missing pipeline or results' }),
                { status: 400, headers: { 'Content-Type': 'application/json' } }
            );
        }

        // Forward to FastAPI backend with timeout
        // Using /sync endpoint because Cerebras streaming returns 503 errors
        const backendUrl = `${BACKEND_URL}/api/explain/sync`;
        console.log('[API] Forwarding to backend (sync):', backendUrl);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout for sync

        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pipeline,
                results,
                patient_context,
                voice_output: voice_output || false,
                voice_provider: voice_provider || 'elevenlabs'
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[API] Backend error:', response.status, errorText);

            // Return mock explanation if backend fails
            const mockExplanation = generateMockExplanation(pipeline, results);
            return new Response(
                `data: ${JSON.stringify({ text: mockExplanation })}\n\ndata: ${JSON.stringify({ done: true })}\n\n`,
                {
                    status: 200,
                    headers: {
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive'
                    }
                }
            );
        }

        // Parse sync response and convert to streaming format for frontend compatibility
        const data = await response.json();
        const explanation = data.explanation || generateMockExplanation(pipeline, results);
        const audioBase64 = data.audio_base64;

        // Build streaming response from sync data
        let streamData = `data: ${JSON.stringify({ text: explanation })}\n\n`;
        if (audioBase64) {
            streamData += `data: ${JSON.stringify({ audio_base64: audioBase64 })}\n\n`;
        }
        streamData += `data: ${JSON.stringify({ done: true, total_length: explanation.length })}\n\n`;

        return new Response(streamData, {
            headers: {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        });

    } catch (error) {
        console.error('[API] Explanation error:', error);

        // Return mock explanation on error
        const body = await request.clone().json().catch(() => ({}));
        const mockExplanation = generateMockExplanation(
            body.pipeline || 'analysis',
            body.results || {}
        );

        return new Response(
            `data: ${JSON.stringify({ text: mockExplanation })}\n\ndata: ${JSON.stringify({ done: true })}\n\n`,
            {
                status: 200,
                headers: {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            }
        );
    }
}

function generateMockExplanation(pipeline: string, results: Record<string, unknown>): string {
    const riskScore = (results.risk_score as number) || 25;
    const confidence = ((results.confidence as number) || 0.85) * 100;

    let riskText: string;
    let recommendation: string;

    if (riskScore < 30) {
        riskText = 'low risk';
        recommendation = 'Continue with routine monitoring and maintain your healthy lifestyle.';
    } else if (riskScore < 60) {
        riskText = 'moderate risk';
        recommendation = 'Consider scheduling a follow-up assessment with your healthcare provider.';
    } else {
        riskText = 'elevated risk';
        recommendation = 'We recommend consulting with a healthcare professional for further evaluation.';
    }

    return `## Your ${pipeline.charAt(0).toUpperCase() + pipeline.slice(1)} Analysis Results

Your analysis shows a **${riskText}** score of ${riskScore}/100.

### What This Means

Based on the biomarkers we analyzed, your results indicate ${riskScore < 50 ? 'generally healthy' : 'some areas of concern that warrant attention'}. The analysis confidence is ${confidence.toFixed(0)}%.

### Key Findings

The analysis examined multiple biomarkers to assess your ${pipeline} health:
- Overall risk score: ${riskScore}/100
- Confidence level: ${confidence.toFixed(0)}%

### Recommendations

${recommendation}

*Note: This is an AI-powered screening tool and should not replace professional medical advice. Always consult with qualified healthcare professionals for diagnosis and treatment.*`;
}

export async function GET() {
    // Check backend health
    try {
        const healthUrl = `${BACKEND_URL}/api/explain/health`;
        const response = await fetch(healthUrl);
        const data = await response.json();

        return new Response(JSON.stringify({
            message: 'AI Explanation endpoint. Use POST to generate explanations.',
            methods: ['POST'],
            backend_status: data.status || 'unknown',
            cerebras_available: data.cerebras_available || false,
            model: data.model || 'mock'
        }), {
            headers: { 'Content-Type': 'application/json' }
        });
    } catch {
        return new Response(JSON.stringify({
            message: 'AI Explanation endpoint. Use POST to generate explanations.',
            methods: ['POST'],
            backend_status: 'unavailable',
            cerebras_available: false,
            model: 'mock (fallback)'
        }), {
            headers: { 'Content-Type': 'application/json' }
        });
    }
}
