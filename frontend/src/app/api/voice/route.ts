/**
 * Voice Generation API Route
 * Generates TTS audio using ElevenLabs or fallback providers
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        console.log('[API] Voice generation request received');

        const body = await request.json();
        const { text, voice_provider } = body;

        if (!text) {
            return NextResponse.json(
                { error: 'Missing text for voice generation' },
                { status: 400 }
            );
        }

        // Forward to FastAPI backend
        const backendUrl = `${BACKEND_URL}/api/voice/generate`;
        console.log('[API] Forwarding to backend:', backendUrl);

        const response = await fetch(backendUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                voice_provider: voice_provider || 'elevenlabs'
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[API] Voice generation error:', response.status, errorText);
            return NextResponse.json(
                { error: 'Voice generation failed' },
                { status: response.status }
            );
        }

        const data = await response.json();
        console.log('[API] Voice generated successfully:', {
            hasAudio: !!data.audio_base64,
            provider: data.provider
        });

        return NextResponse.json(data);

    } catch (error) {
        console.error('[API] Voice generation error:', error);
        return NextResponse.json(
            { error: error instanceof Error ? error.message : 'Internal server error' },
            { status: 500 }
        );
    }
}

export async function GET() {
    return NextResponse.json({
        message: 'Voice generation endpoint. Use POST with { text: string, voice_provider?: string }',
        methods: ['POST'],
        providers: ['elevenlabs', 'openai', 'gtts']
    });
}
