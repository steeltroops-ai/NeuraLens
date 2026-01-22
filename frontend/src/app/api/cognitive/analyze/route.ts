/**
 * Cognitive Assessment Analyze API Route
 * Proxies requests to the backend cognitive pipeline
 */

import { NextResponse, type NextRequest } from "next/server";

const API_BASE =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    console.log("[API] Cognitive analyze request:", {
      session_id: body.session_id,
      task_count: body.tasks?.length || 0,
    });

    // Forward to backend
    const response = await fetch(`${API_BASE}/api/cognitive/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("[API] Backend error:", data);
      return NextResponse.json(data, { status: response.status });
    }

    console.log("[API] Cognitive analyze success:", {
      status: data.status,
      risk_level: data.risk_assessment?.risk_level,
      processing_ms: data.processing_time_ms,
    });

    return NextResponse.json(data);
  } catch (error) {
    console.error("[API] Cognitive analyze error:", error);

    return NextResponse.json(
      {
        error_code: "E_PROXY_ERROR",
        error_message: error instanceof Error ? error.message : "Unknown error",
        recoverable: true,
      },
      { status: 502 },
    );
  }
}
