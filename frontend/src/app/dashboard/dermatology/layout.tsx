import { Metadata } from "next";

export const metadata: Metadata = {
  title: "SkinSense AI - Dermatology Diagnostics | MediLens",
  description:
    "AI-powered skin lesion analysis for melanoma, skin cancer, and dermatological conditions",
};

export default function DermatologyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
