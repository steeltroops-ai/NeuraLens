import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Multi-Modal Assessment - NRI Fusion | MediLens",
  description:
    "Unified neurological risk analysis combining speech, retinal, motor, and cognitive biomarkers using NRI Fusion Engine",
};

export default function MultiModalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
