import React, { useState, useEffect, useRef } from "react";
import { usePatient } from "@/context/PatientContext";
import { X, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface NewPatientModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const NewPatientModal: React.FC<NewPatientModalProps> = ({
  isOpen,
  onClose,
}) => {
  const { setActivePatient } = usePatient();

  const [formData, setFormData] = useState<{
    full_name: string;
    phone_number: string;
    date_of_birth: string;
    gender: string;
  }>({
    full_name: "",
    phone_number: "", // ID
    date_of_birth: "",
    gender: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const firstInputRef = useRef<HTMLInputElement>(null);

  // Focus management
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => firstInputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Basic validation
    if (!formData.full_name.trim() || !formData.phone_number.trim()) {
      setError("Name and ID are required");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/patients`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...formData,
            // remove empty string date
            date_of_birth: formData.date_of_birth || null,
          }),
        },
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to create patient");
      }

      const newPatient = await response.json();
      setActivePatient(newPatient);
      onClose();
      resetForm();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      full_name: "",
      phone_number: "",
      date_of_birth: "",
      gender: "",
    });
  };

  const handleChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    if (error) setError(null);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          />

          {/* Modal */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              className="w-full max-w-[440px] bg-[#09090b] border border-[#27272a] rounded-xl shadow-2xl pointer-events-auto overflow-hidden"
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-[#27272a] bg-[#09090b]">
                <h2 className="text-sm font-medium text-zinc-100">
                  New Patient
                </h2>
                <button
                  onClick={onClose}
                  className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors rounded-md hover:bg-white/5"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Body */}
              <form onSubmit={handleSubmit} className="p-4 space-y-4">
                {error && (
                  <div className="p-2.5 bg-red-500/10 border border-red-500/20 rounded-md">
                    <p className="text-xs text-red-500 font-medium">{error}</p>
                  </div>
                )}

                <div className="space-y-4">
                  {/* Name */}
                  <div className="space-y-1.5">
                    <label className="text-[13px] font-medium text-zinc-400">
                      Full Name
                    </label>
                    <input
                      ref={firstInputRef}
                      type="text"
                      required
                      value={formData.full_name}
                      onChange={(e) =>
                        handleChange("full_name", e.target.value)
                      }
                      placeholder="e.g. Alex Doe"
                      className="w-full h-9 px-3 bg-[#18181b] border border-[#27272a] rounded-lg text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all font-medium"
                    />
                  </div>

                  {/* Phone / ID */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center">
                      <label className="text-[13px] font-medium text-zinc-400">
                        Patient ID / Phone
                      </label>
                      <span className="text-[11px] text-zinc-600 uppercase tracking-wider font-medium">
                        Unique
                      </span>
                    </div>
                    <input
                      type="tel"
                      required
                      value={formData.phone_number}
                      onChange={(e) =>
                        handleChange("phone_number", e.target.value)
                      }
                      placeholder="e.g. 555-0123"
                      className="w-full h-9 px-3 bg-[#18181b] border border-[#27272a] rounded-lg text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all font-mono"
                    />
                  </div>

                  {/* Split Row */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-[13px] font-medium text-zinc-400">
                        Date of Birth
                      </label>
                      <input
                        type="date"
                        required
                        value={formData.date_of_birth}
                        onChange={(e) =>
                          handleChange("date_of_birth", e.target.value)
                        }
                        className="w-full h-9 px-3 bg-[#18181b] border border-[#27272a] rounded-lg text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-[13px] font-medium text-zinc-400">
                        Gender
                      </label>
                      <div className="relative">
                        <select
                          value={formData.gender}
                          onChange={(e) =>
                            handleChange("gender", e.target.value)
                          }
                          className="w-full h-9 px-3 bg-[#18181b] border border-[#27272a] rounded-lg text-[13px] text-zinc-100 focus:outline-none focus:ring-1 focus:ring-zinc-600 focus:border-zinc-600 transition-all appearance-none cursor-pointer"
                        >
                          <option value="" className="text-zinc-500">
                            Select...
                          </option>
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                          <option value="Other">Other</option>
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                          <svg
                            width="10"
                            height="10"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="m6 9 6 6 6-6" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div className="pt-2 flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={onClose}
                    className="h-8 px-3 text-[13px] font-medium text-zinc-400 hover:text-zinc-200 transition-colors rounded-lg hover:bg-[#27272a]"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="h-8 px-4 text-[13px] font-medium text-[#09090b] bg-white hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-lg flex items-center gap-2"
                  >
                    {loading && <Loader2 size={12} className="animate-spin" />}
                    {loading ? "Created" : "Create Patient"}
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
};
