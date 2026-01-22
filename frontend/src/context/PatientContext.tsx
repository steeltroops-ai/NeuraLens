"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

// Types (should ideally match backend)
export interface Patient {
  id: string;
  full_name: string;
  phone_number: string;
  age?: number;
  gender?: string;
  medical_notes?: string;
  created_at: string;
}

interface PatientContextType {
  activePatient: Patient | null;
  setActivePatient: (patient: Patient | null) => void;
  isLoading: boolean;
}

const PatientContext = createContext<PatientContextType | undefined>(undefined);

export function PatientProvider({ children }: { children: ReactNode }) {
  const [activePatient, setActivePatient] = useState<Patient | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load from local storage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem("medilens_active_patient");
      if (saved) {
        setActivePatient(JSON.parse(saved));
      }
    } catch (e) {
      console.error("Failed to parse saved patient", e);
      localStorage.removeItem("medilens_active_patient");
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save to local storage on change
  const handleSetActivePatient = (patient: Patient | null) => {
    setActivePatient(patient);
    if (patient) {
      localStorage.setItem("medilens_active_patient", JSON.stringify(patient));
    } else {
      localStorage.removeItem("medilens_active_patient");
    }
  };

  return (
    <PatientContext.Provider
      value={{
        activePatient,
        setActivePatient: handleSetActivePatient,
        isLoading,
      }}
    >
      {children}
    </PatientContext.Provider>
  );
}

export function usePatient() {
  const context = useContext(PatientContext);
  if (context === undefined) {
    throw new Error("usePatient must be used within a PatientProvider");
  }
  return context;
}
