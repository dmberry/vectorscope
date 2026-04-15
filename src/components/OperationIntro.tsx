"use client";

import { useState, useEffect, type ReactNode } from "react";
import { ChevronDown, ChevronRight, X } from "lucide-react";

interface OperationIntroProps {
  /** Operation name shown as bold label (e.g. "Embedding Table"). */
  name: string;
  /** Short 2-3 sentence summary — the always-available quick orientation. */
  summary: ReactNode;
  /** Deep explanation shown in a modal when the user clicks "Learn more". */
  details: ReactNode;
  /** Stable key used to remember fold state across reloads. Defaults to name. */
  storageKey?: string;
}

export default function OperationIntro({
  name,
  summary,
  details,
  storageKey,
}: OperationIntroProps) {
  const key = `vectorscope.intro.${storageKey ?? name}.collapsed`;
  const [collapsed, setCollapsed] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  // Rehydrate fold state from localStorage after mount.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(key);
      if (stored === "1") setCollapsed(true);
    } catch {
      // ignore
    }
  }, [key]);

  const toggle = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(key, next ? "1" : "0");
      } catch {
        // ignore
      }
      return next;
    });
  };

  // Close modal on Escape.
  useEffect(() => {
    if (!modalOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setModalOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  return (
    <>
      <div className="bg-cream/40 border border-parchment/30 rounded-sm text-caption text-muted-foreground">
        {/* Header row — always visible. Click to toggle fold. */}
        <button
          onClick={toggle}
          className="w-full flex items-center gap-1.5 px-3 py-1.5 text-left hover:bg-cream/60 transition-colors"
          aria-expanded={!collapsed}
        >
          {collapsed ? (
            <ChevronRight className="w-3 h-3 text-muted-foreground flex-shrink-0" />
          ) : (
            <ChevronDown className="w-3 h-3 text-muted-foreground flex-shrink-0" />
          )}
          <strong className="text-foreground">{name}</strong>
          {collapsed && (
            <span className="text-muted-foreground/70 ml-1 truncate">
              · click to show description
            </span>
          )}
        </button>

        {/* Expanded summary. */}
        {!collapsed && (
          <div className="px-3 pb-2 pl-[26px] leading-relaxed">
            {summary}
            <button
              onClick={() => setModalOpen(true)}
              className="ml-1 text-burgundy hover:text-burgundy-600 underline-offset-2 hover:underline transition-colors"
            >
              Learn more →
            </button>
          </div>
        )}
      </div>

      {/* Modal overlay. */}
      {modalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink-900/40 backdrop-blur-sm animate-fade-in"
          onClick={() => setModalOpen(false)}
        >
          <div
            className="relative max-w-2xl w-full max-h-[80vh] overflow-y-auto bg-ivory border border-parchment rounded-sm shadow-editorial-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 flex items-center justify-between px-5 py-3 bg-ivory border-b border-parchment/60">
              <h2 className="font-display text-display-md text-foreground">{name}</h2>
              <button
                onClick={() => setModalOpen(false)}
                className="p-1 text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="px-5 py-4 space-y-2.5 text-caption text-foreground leading-relaxed">
              {details}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
