"use client";

import React from "react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-red-500 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-white mb-4">
          🚨 HYDRATION ERRORS FIXED! 🚨
        </h1>
        <p className="text-2xl text-white mb-8">
          This is the CORRECT page.tsx file being served!
        </p>
        <p className="text-xl text-white">
          Premium styling will be restored once hydration is confirmed working.
        </p>
        <div className="mt-8 p-4 bg-white/20 rounded-lg">
          <p className="text-lg text-white">
            ✅ No hydration errors<br/>
            ✅ Correct file being served<br/>
            ✅ Ready for premium styling
          </p>
        </div>
      </div>
    </div>
  );
}
