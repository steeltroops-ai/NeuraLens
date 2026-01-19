import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Speech Analysis | NeuraLens",
  description:
    "AI-powered voice biomarker analysis for neurological assessment",
};

export default function SpeechLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
