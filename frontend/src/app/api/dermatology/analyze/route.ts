/**
 * Dermatology Pipeline API Route - Analyze Endpoint
 *
 * Proxies requests to the FastAPI backend for skin lesion analysis.
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    // Get the form data from request
    const formData = await request.formData();
    const file = formData.get("image") as File;

    if (!file) {
      return NextResponse.json(
        { success: false, error: "No image provided" },
        { status: 400 },
      );
    }

    // Validate file type
    const validTypes = [
      "image/jpeg",
      "image/png",
      "image/jpg",
      "image/heic",
      "image/heif",
    ];
    if (!validTypes.includes(file.type)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid file type: ${file.type}. Supported: JPEG, PNG, HEIC`,
        },
        { status: 400 },
      );
    }

    // Extract optional parameters
    const imageSource = formData.get("image_source") as string | null;
    const bodyLocation = formData.get("body_location") as string | null;
    const sessionId = formData.get("session_id") as string | null;
    const generateExplanation =
      formData.get("generate_explanation") !== "false";
    const includeVisualizations =
      formData.get("include_visualizations") !== "false";

    // Forward to backend
    const backendFormData = new FormData();
    backendFormData.append("image", file);

    if (imageSource) backendFormData.append("image_source", imageSource);
    if (bodyLocation) backendFormData.append("body_location", bodyLocation);
    if (sessionId) backendFormData.append("session_id", sessionId);
    backendFormData.append("generate_explanation", String(generateExplanation));
    backendFormData.append(
      "include_visualizations",
      String(includeVisualizations),
    );

    const response = await fetch(`${BACKEND_URL}/api/dermatology/analyze`, {
      method: "POST",
      body: backendFormData,
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error_message: "Analysis failed" }));
      return NextResponse.json(
        {
          success: false,
          error:
            errorData.error_message ||
            errorData.detail ||
            `Backend error: ${response.status}`,
          ...errorData,
        },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Dermatology API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Analysis failed",
      },
      { status: 500 },
    );
  }
}
