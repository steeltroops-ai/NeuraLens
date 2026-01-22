/**
 * Dermatology Pipeline API Route - Biomarkers Endpoint
 *
 * Returns information about dermatology biomarkers and features.
 */

import { NextResponse } from "next/server";

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/dermatology/biomarkers`);

    if (!response.ok) {
      // Return default biomarkers if backend unavailable
      return NextResponse.json({
        pipeline: "dermatology",
        version: "1.0.0",
        biomarkers: [
          {
            id: "asymmetry",
            name: "Asymmetry",
            category: "ABCDE",
            description:
              "Measures the symmetry of the lesion along multiple axes",
          },
          {
            id: "border",
            name: "Border Irregularity",
            category: "ABCDE",
            description:
              "Analyzes the regularity and definition of the lesion border",
          },
          {
            id: "color",
            name: "Color Variation",
            category: "ABCDE",
            description:
              "Evaluates the number and distribution of colors within the lesion",
          },
          {
            id: "diameter",
            name: "Diameter",
            category: "ABCDE",
            description: "Measures the maximum diameter of the lesion",
          },
          {
            id: "evolution",
            name: "Evolution",
            category: "ABCDE",
            description: "Assesses changes over time",
          },
        ],
        risk_tiers: [
          { tier: 1, name: "CRITICAL", urgency: "24-48 hours" },
          { tier: 2, name: "HIGH", urgency: "1-2 weeks" },
          { tier: 3, name: "MODERATE", urgency: "1-3 months" },
          { tier: 4, name: "LOW", urgency: "Annual check" },
          { tier: 5, name: "BENIGN", urgency: "None" },
        ],
      });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Biomarkers API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to fetch biomarkers",
      },
      { status: 500 },
    );
  }
}
