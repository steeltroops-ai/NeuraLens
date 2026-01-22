/**
 * Speech Pipeline API Route - Health Check
 */

import { NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

export async function GET() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${BACKEND_URL}/api/speech/health`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    }).finally(() => clearTimeout(timeoutId));

    if (!response.ok) {
      return NextResponse.json(
        { status: "unhealthy", error: "Backend unavailable" },
        { status: 503 },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { status: "unhealthy", error: "Connection failed" },
      { status: 503 },
    );
  }
}
