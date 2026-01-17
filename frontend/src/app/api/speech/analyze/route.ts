/**
 * Speech Analysis API Route
 * Proxies requests to FastAPI backend with proper form handling
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        console.log('[API] Speech analysis request received');

        // Get form data from request
        const formData = await request.formData();
        
        // Get the audio file - check both 'audio' and 'audio_file' keys
        const audioFile = formData.get('audio') || formData.get('audio_file');
        const sessionId = formData.get('session_id');

        if (!audioFile) {
            return NextResponse.json(
                {
                    status: 'error',
                    error_message: 'No audio file provided',
                },
                { status: 400 }
            );
        }

        // Create new FormData for backend with correct field name
        const backendFormData = new FormData();
        backendFormData.append('audio', audioFile as Blob);
        
        if (sessionId) {
            backendFormData.append('session_id', sessionId.toString());
        }

        // Forward to FastAPI backend
        const backendUrl = `${BACKEND_URL}/api/speech/analyze`;
        console.log('[API] Forwarding to backend:', backendUrl);

        const response = await fetch(backendUrl, {
            method: 'POST',
            body: backendFormData,
            // Don't set Content-Type header - let fetch set it with boundary
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[API] Backend error:', response.status, errorText);

            let errorDetail = 'Speech analysis failed';
            try {
                const errorJson = JSON.parse(errorText);
                errorDetail = errorJson.detail || errorJson.error_message || errorDetail;
            } catch {
                errorDetail = errorText || errorDetail;
            }

            return NextResponse.json(
                {
                    status: 'error',
                    error_message: errorDetail,
                },
                { status: response.status }
            );
        }

        const data = await response.json();
        console.log('[API] Speech analysis successful:', {
            session_id: data.session_id,
            risk_score: data.risk_score,
            status: data.status
        });

        return NextResponse.json(data);

    } catch (error) {
        console.error('[API] Speech analysis error:', error);

        return NextResponse.json(
            {
                status: 'error',
                error_message: error instanceof Error ? error.message : 'Internal server error',
            },
            { status: 500 }
        );
    }
}

export async function GET() {
    // Check backend health
    try {
        const healthUrl = `${BACKEND_URL}/api/speech/health`;
        const response = await fetch(healthUrl);
        const data = await response.json();
        
        return NextResponse.json({
            message: 'Speech analysis endpoint. Use POST to analyze audio.',
            methods: ['POST'],
            backend_status: data.status || 'unknown',
            parselmouth_available: data.parselmouth_available || false,
            audio_libs_available: data.audio_libs_available || false
        });
    } catch {
        return NextResponse.json({
            message: 'Speech analysis endpoint. Use POST to analyze audio.',
            methods: ['POST'],
            backend_status: 'unavailable'
        });
    }
}
