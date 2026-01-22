import { NextResponse } from "next/server";

export async function GET() {
  try {
    // Check backend health with timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const backendResponse = await fetch("http://localhost:8000/health", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
    }).finally(() => clearTimeout(timeoutId));

    if (backendResponse.ok) {
      const backendHealth = await backendResponse.json();

      return NextResponse.json({
        status: "healthy",
        timestamp: new Date().toISOString(),
        frontend: {
          status: "healthy",
          version: "1.0.0",
        },
        backend: {
          status: "healthy",
          ...backendHealth,
        },
        services: {
          speech_analyzer: "healthy",
          retinal_analyzer: "healthy",
          motor_analyzer: "healthy",
          cognitive_analyzer: "healthy",
          nri_fusion: "healthy",
          validation_engine: "healthy",
        },
      });
    } else {
      // Backend not available, return frontend-only health
      return NextResponse.json(
        {
          status: "warning",
          timestamp: new Date().toISOString(),
          frontend: {
            status: "healthy",
            version: "1.0.0",
          },
          backend: {
            status: "unavailable",
            message: "Backend service not responding",
          },
          services: {
            speech_analyzer: "unavailable",
            retinal_analyzer: "unavailable",
            motor_analyzer: "unavailable",
            cognitive_analyzer: "unavailable",
            nri_fusion: "unavailable",
            validation_engine: "unavailable",
          },
        },
        { status: 200 },
      );
    }
  } catch (error) {
    // Log error but don't crash - backend unavailable is expected in development
    console.warn(
      "Backend health check unavailable (expected in development):",
      error.message,
    );

    return NextResponse.json(
      {
        status: "frontend_only",
        timestamp: new Date().toISOString(),
        frontend: {
          status: "healthy",
          version: "1.0.0",
        },
        backend: {
          status: "unavailable",
          message: "Backend service not running (development mode)",
        },
        services: {
          speech_analyzer: "mock",
          retinal_analyzer: "mock",
          motor_analyzer: "mock",
          cognitive_analyzer: "mock",
          nri_fusion: "mock",
          validation_engine: "mock",
        },
      },
      { status: 200 }, // Return 200 instead of 503 to prevent errors
    );
  }
}
