"use client";

import { APP_NAME } from "@/lib/version";
import { useModel } from "@/context/ModelContext";
import { TAB_GROUPS } from "@/types/model";
import { ChevronDown, Cpu, Loader2 } from "lucide-react";
import AboutDialog from "./AboutDialog";
import BackendInfoDialog from "./BackendInfoDialog";
import HelpDialog from "./HelpDialog";
import SettingsDialog from "./SettingsDialog";

interface HeaderProps {
  onOpenModelPicker?: () => void;
  /** Current operation tab id; used to render the "Vector Lab | Tool | Section" crumb. */
  activeTab?: string;
}

// Flatten the tab registry into an id → label map so the header can show the
// current operation name next to the wordmark, matching LLMbench's header.
const TAB_LABELS: Record<string, string> = Object.values(TAB_GROUPS).reduce(
  (acc, group) => {
    for (const tab of group.tabs) acc[tab.id] = tab.label;
    return acc;
  },
  {} as Record<string, string>
);

export default function Header({ onOpenModelPicker, activeTab }: HeaderProps) {
  const { backendStatus } = useModel();
  const currentLabel = activeTab ? TAB_LABELS[activeTab] : undefined;

  return (
    <header className="px-4 py-2 border-b border-parchment-dark bg-cream/30 flex items-center gap-3">
      {/* Vector Lab family mark — far-left "VECTOR LAB" wordmark, the
          LLMbench pattern. Tool identity sits to the right of the separator. */}
      <a
        href="https://vector-lab-tools.github.io"
        target="_blank"
        rel="noopener noreferrer"
        title="Vector Lab — research instruments for critical vector theory"
        className="flex items-center gap-1.5 opacity-70 hover:opacity-100 transition-opacity shrink-0"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/icons/vector-lab-mark.svg"
          alt="Vector Lab"
          width={20}
          height={20}
          className="w-5 h-5"
        />
        <span className="text-[10px] font-medium tracking-widest uppercase text-slate hidden sm:inline">
          Vector Lab
        </span>
      </a>
      <div className="h-4 w-px bg-parchment-dark/60" />

      {/* Vectorscope identity */}
      <div className="flex items-center gap-2 shrink-0">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/icons/vector-lab.svg"
          alt=""
          width={18}
          height={18}
          className="w-[18px] h-[18px]"
        />
        <h1 className="font-display text-body-sm font-bold text-ink">{APP_NAME}</h1>
      </div>

      {/* Current operation — shown only when a tab is active and a model is loaded */}
      {currentLabel && backendStatus.model && (
        <>
          <div className="h-4 w-px bg-parchment-dark/60" />
          <span className="text-caption text-slate">{currentLabel}</span>
        </>
      )}

      <div className="flex-1" />

      {/* Model status — clickable when a model is loaded, to swap */}
      <div className="flex items-center gap-2 font-sans text-caption">
        <Cpu className="w-3.5 h-3.5 text-slate" />
        {backendStatus.model ? (
          <button
            onClick={onOpenModelPicker}
            disabled={!onOpenModelPicker}
            className="group flex items-center gap-1 text-ink hover:text-burgundy transition-colors disabled:cursor-default disabled:hover:text-ink"
            title="Change model"
          >
            <span className="underline decoration-dotted decoration-slate/40 group-hover:decoration-burgundy underline-offset-4">
              {backendStatus.model.name}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-slate group-hover:text-burgundy" />
          </button>
        ) : backendStatus.status === "loading" ? (
          <span className="text-slate flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            Loading...
          </span>
        ) : (
          <span className="text-slate">No model loaded</span>
        )}
      </div>

      {/* Backend indicator — click for model-server technical details */}
      <BackendInfoDialog />

      {/* Toolbar */}
      <div className="flex items-center gap-2 ml-1 border-l border-parchment-dark pl-3">
        <AboutDialog />
        <HelpDialog />
        <SettingsDialog />
      </div>
    </header>
  );
}
