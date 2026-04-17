"use client";

import { useState } from "react";
import { Settings, X } from "lucide-react";
import { useModel } from "@/context/ModelContext";

export default function SettingsDialog() {
  const [open, setOpen] = useState(false);
  const { backendStatus } = useModel();

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="text-slate hover:text-burgundy transition-colors"
        title="Settings"
      >
        <Settings className="w-4 h-4" />
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setOpen(false)}>
          <div
            className="bg-ivory border border-parchment-dark rounded-lg shadow-editorial-lg max-w-sm w-full mx-4 p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <h2 className="font-display text-display-md">Settings</h2>
              <button onClick={() => setOpen(false)} className="text-slate hover:text-burgundy">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-6">
              {/* Backend status */}
              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-3">Backend</h3>
                <div className="space-y-2 font-sans text-caption text-slate">
                  <div className="flex justify-between">
                    <span>Status</span>
                    <span className={backendStatus.status === "connected" ? "text-green-700" : "text-red-600"}>
                      {backendStatus.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Device</span>
                    <span className="font-mono">{backendStatus.device}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Available Memory</span>
                    <span className="font-mono">{Math.round(backendStatus.availableMemoryMb / 1024 * 10) / 10} GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Backend URL</span>
                    <span className="font-mono text-xs">http://localhost:8000</span>
                  </div>
                </div>
              </section>

              {/* Model info */}
              {backendStatus.model && (
                <section>
                  <h3 className="font-display text-heading-sm font-semibold mb-3">Loaded Model</h3>
                  <div className="space-y-2 font-sans text-caption text-slate">
                    <div className="flex justify-between">
                      <span>Model</span>
                      <span className="font-mono text-xs">{backendStatus.model.modelId}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Architecture</span>
                      <span>{backendStatus.model.architecture}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Size in memory</span>
                      <span>{(backendStatus.model.sizeBytes / 1e6).toFixed(0)} MB</span>
                    </div>
                    <div
                      className="flex justify-between"
                      title="Precision at which the weights are currently held in memory. May differ from the model's native precision: fp32 models (e.g. GPT-2) are cast to fp16 on MPS/CUDA to save memory; bf16 models are kept at bf16 when the device supports it."
                    >
                      <span>Loaded precision</span>
                      <span className="font-mono text-xs">
                        {backendStatus.model.nativeDtype &&
                        backendStatus.model.nativeDtype !== backendStatus.model.dtype &&
                        backendStatus.model.nativeDtype !== "unknown"
                          ? `downcast ${backendStatus.model.nativeDtype}\u2192${backendStatus.model.dtype}`
                          : backendStatus.model.dtype}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Hidden Size</span>
                      <span>{backendStatus.model.hiddenSize}d</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Layers</span>
                      <span>{backendStatus.model.numLayers}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Attention Heads</span>
                      <span>{backendStatus.model.numAttentionHeads}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Vocabulary</span>
                      <span>{backendStatus.model.vocabSize.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Weight Tying</span>
                      <span>{backendStatus.model.weightTied ? "Yes" : "No"}</span>
                    </div>
                  </div>
                </section>
              )}

              {/* Placeholder for future settings */}
              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-3">Display</h3>
                <p className="font-sans text-caption text-slate/60">
                  Visualisation preferences coming in a future version.
                </p>
              </section>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
