/**
 * Cardiology Analysis API Route
 * Proxies ECG upload requests to FastAPI backend for real analysis
 *
 * This route handles:
 * - Receiving ECG files from the frontend
 * - Forwarding to the FastAPI backend's cardiology/analyze endpoint
 * - Returning the complete analysis response
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    console.log("[API] Cardiology analysis request received");

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const sampleRate = searchParams.get("sample_rate") || "500";
    const includeWaveform = searchParams.get("include_waveform") || "true";

    // Get form data from request
    const formData = await request.formData();

    // Get the ECG file
    const ecgFile = formData.get("file");

    if (!ecgFile) {
      return NextResponse.json(
        {
          success: false,
          error: "No ECG file provided",
        },
        { status: 400 },
      );
    }

    // Log file details for debugging
    const file = ecgFile as File | Blob;
    const fileSize = file.size;
    const fileType = file.type || "unknown";
    const fileName = file instanceof File ? file.name : "data.csv";
    console.log(
      `[API] ECG file received: ${fileName}, size: ${fileSize} bytes, type: ${fileType}`,
    );

    // Create new FormData for backend
    const backendFormData = new FormData();
    backendFormData.append("file", ecgFile as Blob, fileName);

    // Forward to FastAPI backend
    const backendUrl = `${BACKEND_URL}/api/cardiology/analyze?sample_rate=${sampleRate}&include_waveform=${includeWaveform}`;
    console.log("[API] Forwarding to backend:", backendUrl);

    const response = await fetch(backendUrl, {
      method: "POST",
      body: backendFormData,
      // Don't set Content-Type header - let fetch set it with boundary
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[API] Backend error:", response.status, errorText);

      let errorDetail = "Cardiology analysis failed";
      try {
        const errorJson = JSON.parse(errorText);
        errorDetail =
          errorJson.detail || errorJson.error_message || errorDetail;
      } catch {
        errorDetail = errorText || errorDetail;
      }

      return NextResponse.json(
        {
          success: false,
          error: errorDetail,
        },
        { status: response.status },
      );
    }

    const data = await response.json();
    console.log("[API] Cardiology analysis successful:", {
      request_id: data.request_id,
      success: data.success,
      heart_rate: data.ecg_analysis?.rhythm_analysis?.heart_rate_bpm,
      risk_score: data.risk_assessment?.risk_score,
      processing_time: data.processing_time_ms,
    });

    return NextResponse.json(data);
  } catch (error) {
    console.error("[API] Cardiology analysis error:", error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Internal server error",
      },
      { status: 500 },
    );
  }
}

export async function GET() {
  // Check backend health
  try {
    const healthUrl = `${BACKEND_URL}/api/cardiology/health`;
    const response = await fetch(healthUrl);
    const data = await response.json();

    return NextResponse.json({
      message: "Cardiology analysis endpoint. Use POST to analyze ECG files.",
      methods: ["POST"],
      backend_status: data.status || "unknown",
      version: data.version || "unknown",
      supported_formats: ["CSV", "TXT", "JSON"],
    });
  } catch {
    return NextResponse.json({
      message: "Cardiology analysis endpoint. Use POST to analyze ECG files.",
      methods: ["POST"],
      backend_status: "unavailable",
    });
  }
}
