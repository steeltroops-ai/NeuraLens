/**
 * Retinal Analysis API Route
 * Proxies image upload requests to FastAPI backend for real ML analysis
 * 
 * This route handles:
 * - Receiving fundus images from the frontend
 * - Forwarding to the FastAPI backend's retinal/analyze endpoint
 * - Returning the complete analysis response
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
    try {
        console.log('[API] Retinal analysis request received');

        // Get form data from request
        const formData = await request.formData();
        
        // Get the image file
        const imageFile = formData.get('image');
        const patientId = formData.get('patient_id') || 'ANONYMOUS';
        const sessionId = formData.get('session_id');
        const patientAge = formData.get('patient_age');

        if (!imageFile) {
            return NextResponse.json(
                {
                    success: false,
                    error: 'No image file provided',
                },
                { status: 400 }
            );
        }

        // Log image file details for debugging
        const file = imageFile as File | Blob;
        const fileSize = file.size;
        const fileType = file.type || 'unknown';
        const fileName = file instanceof File ? file.name : 'blob';
        console.log(`[API] Image file received: ${fileName}, size: ${fileSize} bytes, type: ${fileType}`);

        // Create new FormData for backend
        const backendFormData = new FormData();
        backendFormData.append('image', imageFile as Blob);
        backendFormData.append('patient_id', patientId.toString());
        
        if (sessionId) {
            backendFormData.append('session_id', sessionId.toString());
        }
        if (patientAge) {
            backendFormData.append('patient_age', patientAge.toString());
        }

        // Forward to FastAPI backend
        const backendUrl = `${BACKEND_URL}/api/retinal/analyze`;
        console.log('[API] Forwarding to backend:', backendUrl);

        const response = await fetch(backendUrl, {
            method: 'POST',
            body: backendFormData,
            // Don't set Content-Type header - let fetch set it with boundary
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('[API] Backend error:', response.status, errorText);

            let errorDetail = 'Retinal analysis failed';
            try {
                const errorJson = JSON.parse(errorText);
                errorDetail = errorJson.detail || errorJson.error_message || errorDetail;
            } catch {
                errorDetail = errorText || errorDetail;
            }

            return NextResponse.json(
                {
                    success: false,
                    error: errorDetail,
                },
                { status: response.status }
            );
        }

        const data = await response.json();
        console.log('[API] Retinal analysis successful:', {
            session_id: data.session_id,
            success: data.success,
            dr_grade: data.diabetic_retinopathy?.grade,
            risk_score: data.risk_assessment?.overall_score,
            processing_time: data.total_processing_time_ms
        });

        return NextResponse.json(data);

    } catch (error) {
        console.error('[API] Retinal analysis error:', error);

        return NextResponse.json(
            {
                success: false,
                error: error instanceof Error ? error.message : 'Internal server error',
            },
            { status: 500 }
        );
    }
}

export async function GET() {
    // Check backend health
    try {
        const healthUrl = `${BACKEND_URL}/api/retinal/health`;
        const response = await fetch(healthUrl);
        const data = await response.json();
        
        return NextResponse.json({
            message: 'Retinal analysis endpoint. Use POST to analyze fundus images.',
            methods: ['POST'],
            backend_status: data.status || 'unknown',
            version: data.version || 'unknown',
            architecture: data.architecture || 'unknown'
        });
    } catch {
        return NextResponse.json({
            message: 'Retinal analysis endpoint. Use POST to analyze fundus images.',
            methods: ['POST'],
            backend_status: 'unavailable'
        });
    }
}
