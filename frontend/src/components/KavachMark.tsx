import React from "react";

/** Same mark as landing nav: shield in emerald tile (h-10 w-10, icon h-5 w-5). */
export function KavachMark({ className = "" }: { className?: string }) {
  return (
    <div
      className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-emerald-900/80 text-emerald-200 ring-1 ring-emerald-500/30 ${className}`}
      aria-hidden
    >
      <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6l7-3z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
    </div>
  );
}
