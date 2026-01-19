import { Metadata } from "next";

export const metadata: Metadata = {
  title: "CardioPredict AI - Cardiology Diagnostics | MediLens",
  description:
    "AI-powered ECG analysis for arrhythmia, atrial fibrillation, myocardial infarction, and cardiac conditions",
};

export default function CardiologyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
