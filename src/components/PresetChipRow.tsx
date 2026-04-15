"use client";

import type { ReactNode } from "react";

export interface ChipItem {
  /** Short label shown on the chip. Truncated at 45 chars with ellipsis. */
  label: string;
  /** Click handler. Should fill the operation's input state. */
  onClick: () => void;
  /** Tooltip shown on hover. Falls back to the full label if omitted. */
  title?: string;
  /** Per-item disabled override. The row-level `disabled` takes precedence. */
  disabled?: boolean;
}

interface PresetChipRowProps {
  /** Chips to render. The row renders nothing when the array is empty. */
  items: ChipItem[];
  /** Disables every chip at once (e.g. while a run is in flight). */
  disabled?: boolean;
  /** Leading label. Defaults to "Try:". Pass null to omit. */
  prefix?: ReactNode;
  /** Extra classes appended to the row. */
  className?: string;
}

/**
 * Thin preset-chip row modelled on LLMbench's `DefaultPromptChips`. Clicks
 * fill the operation's inputs but do NOT auto-run — research operations in
 * Vectorscope can be expensive (7B-model forward passes, full-vocab
 * extractions), and the user should see what got filled before hitting Run.
 */
export default function PresetChipRow({
  items,
  disabled = false,
  prefix = "Try:",
  className = "",
}: PresetChipRowProps) {
  if (items.length === 0) return null;
  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      {prefix !== null && (
        <span className="font-sans text-[11px] text-slate/70 shrink-0">{prefix}</span>
      )}
      {items.map((item, i) => (
        <button
          key={`${item.label}-${i}`}
          type="button"
          onClick={item.onClick}
          disabled={disabled || item.disabled}
          title={item.title ?? item.label}
          className="font-sans text-[11px] bg-cream hover:bg-parchment border border-parchment/60 px-2 py-0.5 rounded-sm text-slate hover:text-ink transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-left"
        >
          {item.label.length > 45 ? item.label.slice(0, 45) + "…" : item.label}
        </button>
      ))}
    </div>
  );
}
