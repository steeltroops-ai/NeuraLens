import { Metadata } from "next";

export const metadata: Metadata = {
  title: "NRI Fusion Engine - Advanced Analytics | MediLens",
  description:
    "Bayesian Fusion Engine for Neurological Risk Index (NRI) calculation, uncertainty quantification, and predictive modeling",
};

export default function NRIFusionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
