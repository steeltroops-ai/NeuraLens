import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Retinal Analysis - DR Grading & Biomarkers | MediLens",
  description:
    "AI-powered retinal analysis for diabetic retinopathy grading, vessel segmentation, and biomarker extraction",
};

export default function RetinalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
