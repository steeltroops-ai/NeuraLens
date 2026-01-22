import React, { useState, useEffect, useRef } from "react";
import { usePatient, Patient } from "@/context/PatientContext";
import { Search, X, User } from "lucide-react";

export const PatientSelector: React.FC = () => {
  const { activePatient, setActivePatient } = usePatient();
  const [searchTerm, setSearchTerm] = useState("");
  const [results, setResults] = useState<Patient[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Debounced search
  useEffect(() => {
    if (!searchTerm || searchTerm.length < 2) {
      setResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/patients?q=${encodeURIComponent(searchTerm)}`,
        );
        if (response.ok) {
          const data = await response.json();
          setResults(data);
          setIsOpen(true);
        }
      } catch (error) {
        console.error("Search failed", error);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm]);

  const handleSelect = (patient: Patient) => {
    setActivePatient(patient);
    setSearchTerm("");
    setIsOpen(false);
  };

  const handleClear = () => {
    setActivePatient(null);
    setSearchTerm("");
    // Focus input after clearing to allow immediate typing
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  if (activePatient) {
    return (
      <div className="flex items-center gap-2 bg-[#18181b] border border-[#27272a] rounded-md shadow-sm px-2.5 h-[30px] !min-h-[30px] group hover:border-zinc-700 transition-all select-none">
        <div className="flex items-center justify-center w-4 h-4 rounded overflow-hidden bg-blue-500/10 text-blue-400">
          <User size={12} strokeWidth={2.5} />
        </div>
        <div className="flex flex-col justify-center">
          <span className="text-[9px] font-medium text-zinc-500 uppercase tracking-wider leading-none">
            Active
          </span>
          <span className="text-[11px] font-medium text-zinc-200 leading-none mt-0.5 max-w-[95px] truncate">
            {activePatient.full_name}
          </span>
        </div>
        <div className="h-3 w-[1px] bg-zinc-800 mx-1" />
        <button
          onClick={handleClear}
          className="text-zinc-500 hover:text-red-400 transition-colors p-0.5 rounded hover:bg-white/5"
          title="Clear active patient"
        >
          <X size={13} />
        </button>
      </div>
    );
  }

  return (
    <div className="relative w-44 lg:w-64 group" ref={wrapperRef}>
      <div className="relative flex items-center">
        <div className="absolute inset-y-0 left-0 pl-2.5 flex items-center pointer-events-none">
          <Search
            size={13}
            className="text-zinc-500 group-hover:text-zinc-400 transition-colors"
          />
        </div>
        <input
          ref={inputRef}
          type="text"
          className="block w-full pl-8 pr-2.5 h-[30px] !min-h-[30px] text-[12px] bg-[#09090b] border border-[#27272a] rounded-md text-zinc-200 placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-700 focus:border-zinc-700 transition-all hover:bg-[#18181b] hover:border-zinc-700 font-medium shadow-sm leading-none"
          placeholder="Search patient..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onFocus={() => {
            if (results.length > 0) setIsOpen(true);
          }}
        />
        {loading && (
          <div className="absolute inset-y-0 right-0 pr-2 flex items-center">
            <div className="animate-spin h-2 w-2 border-2 border-zinc-600 rounded-full border-t-transparent"></div>
          </div>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-8 left-0 w-[260px] bg-[#09090b] border border-[#27272a] rounded-lg shadow-xl overflow-hidden z-50">
          <div className="px-2.5 py-1.5 text-[9px] font-semibold text-zinc-500 uppercase tracking-wider border-b border-[#27272a] bg-[#18181b]/50">
            Results
          </div>
          <ul className="max-h-56 overflow-y-auto py-1 custom-scrollbar">
            {results.map((patient) => (
              <li key={patient.id}>
                <button
                  onClick={() => handleSelect(patient)}
                  className="w-full text-left px-2.5 py-1.5 hover:bg-[#18181b] transition-colors flex items-center gap-2.5 group/item"
                >
                  <div className="h-6 w-6 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-500 group-hover/item:text-zinc-300 group-hover/item:border-zinc-700 transition-colors">
                    <span className="text-[10px] font-bold">
                      {patient.full_name.charAt(0)}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[11px] font-medium text-zinc-300 group-hover/item:text-zinc-100 transition-colors">
                      {patient.full_name}
                    </span>
                    <span className="text-[9px] text-zinc-600 group-hover/item:text-zinc-500 font-mono">
                      {patient.phone_number}
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #27272a;
          border-radius: 2px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #3f3f46;
        }
      `}</style>
    </div>
  );
};
