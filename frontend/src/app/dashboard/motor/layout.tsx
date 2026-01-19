import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Motor Function - Parkinson's Assessment | MediLens",
  description:
    "AI-powered motor assessment for Parkinson's disease, tremor analysis, and fine motor skills evaluation",
};

export default function MotorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
