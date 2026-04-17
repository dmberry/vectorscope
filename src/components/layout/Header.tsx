"use client";

import { APP_NAME, APP_TAGLINE, VERSION } from "@/lib/version";
import { useModel } from "@/context/ModelContext";
import { ChevronDown, Cpu, Loader2 } from "lucide-react";
import AboutDialog from "./AboutDialog";
import BackendInfoDialog from "./BackendInfoDialog";
import HelpDialog from "./HelpDialog";
import SettingsDialog from "./SettingsDialog";

interface HeaderProps {
  onOpenModelPicker?: () => void;
}

export default function Header({ onOpenModelPicker }: HeaderProps) {
  const { backendStatus } = useModel();

  return (
    <header className="border-b border-parchment-dark bg-card px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <h1 className="font-display text-display-md text-ink tracking-tight">
              {APP_NAME}
            </h1>
            <p className="font-sans text-caption text-slate mt-0.5">
              {APP_TAGLINE}
            </p>
          </div>
          <span className="font-sans text-caption text-slate-light bg-cream px-2 py-0.5 rounded-sm">
            v{VERSION}
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Model status — clickable when a model is loaded, to swap */}
          <div className="flex items-center gap-2 font-sans text-body-sm">
            <Cpu className="w-4 h-4 text-slate" />
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
          <div className="flex items-center gap-2 ml-2 border-l border-parchment-dark pl-3">
            <AboutDialog />
            <HelpDialog />
            <SettingsDialog />
          </div>
        </div>
      </div>
    </header>
  );
}
