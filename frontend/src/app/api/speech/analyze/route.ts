/**
 * Speech Analysis API Route
 * Proxies requests to FastAPI backend
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        console.log('[API] Speech analysis request received');

        // Get form data from request
        const formData = await request.formData();

        // Forward to FastAPI backend
        const backendUrl = `${BACKEND_URL}/api/v1/speech/analyze`;
        console.log('[API] Forwarding to backend:', backendUrl);

        const response = await fetch(backendUrl, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - let fetch set it with boundary
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[API] Backend error:', response.status, errorText);

            let errorDetail = 'Speech analysis failed';
            try {
                const errorJson = JSON.parse(errorText);
                errorDetail = errorJson.detail || errorDetail;
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
        console.log('[API] Speech analysis successful');

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
    return NextResponse.json(
        {
            message: 'Speech analysis endpoint. Use POST to analyze audio.',
            methods: ['POST'],
        },
        { status: 200 }
    );
}
