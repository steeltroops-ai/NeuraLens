import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Cognitive Testing - Neuro Assessment | MediLens",
  description:
    "AI-powered cognitive assessment for detecting early signs of dementia, Alzheimer's, and other neurological conditions",
};

export default function CognitiveLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
