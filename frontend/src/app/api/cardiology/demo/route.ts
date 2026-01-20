/**
 * Cardiology Demo API Route
 * Proxies demo ECG generation requests to FastAPI backend
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    console.log("[API] Cardiology demo request received");

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const heartRate = searchParams.get("heart_rate") || "72";
    const duration = searchParams.get("duration") || "10";
    const addArrhythmia = searchParams.get("add_arrhythmia") || "false";

    // Forward to FastAPI backend
    const backendUrl = `${BACKEND_URL}/api/cardiology/demo?heart_rate=${heartRate}&duration=${duration}&add_arrhythmia=${addArrhythmia}`;
    console.log("[API] Forwarding to backend:", backendUrl);

    const response = await fetch(backendUrl, {
      method: "POST",
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[API] Backend error:", response.status, errorText);

      let errorDetail = "Cardiology demo failed";
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
    console.log("[API] Cardiology demo successful:", {
      request_id: data.request_id,
      success: data.success,
      heart_rate: data.ecg_analysis?.rhythm_analysis?.heart_rate_bpm,
      risk_score: data.risk_assessment?.risk_score,
    });

    return NextResponse.json(data);
  } catch (error) {
    console.error("[API] Cardiology demo error:", error);

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
  return NextResponse.json({
    message:
      "Cardiology demo endpoint. Use POST to generate synthetic ECG analysis.",
    methods: ["POST"],
    params: {
      heart_rate: "Target heart rate (40-200 bpm, default: 72)",
      duration: "Signal duration (5-60 seconds, default: 10)",
      add_arrhythmia: "Add simulated arrhythmia (true/false, default: false)",
    },
  });
}
