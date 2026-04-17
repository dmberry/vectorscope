"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { useModel } from "@/context/ModelContext";

/**
 * Clickable model-server indicator.
 *
 * Shows "Model server: <device>" or "Model server: disconnected" and opens a
 * dialog with technical information about what the local FastAPI+PyTorch
 * backend can provide — its compute device, memory budget, precision policy,
 * and the set of capabilities exposed to the UI.
 */
export default function BackendInfoDialog() {
  const { backendStatus } = useModel();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const connected = backendStatus.status === "connected";
  const statusColor = connected
    ? "bg-success-500"
    : backendStatus.status === "loading"
    ? "bg-warning-500 animate-pulse-subtle"
    : "bg-error-500";

  const deviceLabel = connected ? backendStatus.device : "disconnected";
  const deviceMeta: Record<string, { name: string; blurb: string }> = {
    mps: {
      name: "Apple Metal Performance Shaders",
      blurb:
        "Metal-backed GPU acceleration for Apple Silicon. Supports bfloat16, float16, and float32.",
    },
    cuda: {
      name: "NVIDIA CUDA",
      blurb:
        "NVIDIA GPU acceleration. Supports bfloat16, float16, and float32; TF32 where available.",
    },
    cpu: {
      name: "CPU",
      blurb:
        "No accelerator detected. PyTorch runs on the CPU in float32. bfloat16 works but is slow.",
    },
  };
  const meta =
    deviceMeta[backendStatus.device] ?? {
      name: "Unknown device",
      blurb: "The backend did not report a recognised compute device.",
    };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-1.5 font-sans text-caption text-slate hover:text-burgundy transition-colors"
        title="Model server details"
      >
        <span className={`w-2 h-2 rounded-full ${statusColor}`} />
        <span>
          Model server:{" "}
          <span className="underline decoration-dotted decoration-slate/40 underline-offset-4 font-mono">
            {deviceLabel}
          </span>
        </span>
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
          onClick={() => setOpen(false)}
        >
          <div
            className="bg-ivory border border-parchment-dark rounded-lg shadow-editorial-lg max-w-lg w-full mx-4 p-6 max-h-[85vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="font-display text-heading-lg">Model server</h2>
                <p className="font-sans text-caption text-slate mt-0.5">
                  Local FastAPI + PyTorch process on <span className="font-mono">:8000</span>
                </p>
              </div>
              <button onClick={() => setOpen(false)} className="text-slate hover:text-burgundy">
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Live state */}
            <section className="mb-4">
              <h3 className="font-display text-heading-sm font-semibold mb-2">State</h3>
              <div className="space-y-1 font-sans text-caption text-slate">
                <div className="flex justify-between">
                  <span>Status</span>
                  <span className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${statusColor}`} />
                    <span className="font-mono">{backendStatus.status}</span>
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Compute device</span>
                  <span className="font-mono">{backendStatus.device}</span>
                </div>
                <div className="flex justify-between">
                  <span>Available memory</span>
                  <span className="font-mono">
                    {(backendStatus.availableMemoryMb / 1024).toFixed(1)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Model loaded</span>
                  <span className="font-mono">
                    {backendStatus.model ? backendStatus.model.name : "none"}
                  </span>
                </div>
              </div>
            </section>

            <hr className="my-4 border-parchment-dark" />

            {/* Device explanation */}
            <section className="mb-4">
              <h3 className="font-display text-heading-sm font-semibold mb-1">
                What <span className="font-mono">{deviceLabel}</span> means
              </h3>
              <p className="font-sans text-caption text-slate">
                <strong>{meta.name}.</strong> {meta.blurb}
              </p>
            </section>

            <hr className="my-4 border-parchment-dark" />

            {/* Capabilities */}
            <section className="mb-4">
              <h3 className="font-display text-heading-sm font-semibold mb-2">
                What the model server provides
              </h3>
              <ul className="list-disc pl-5 space-y-1 font-sans text-caption text-slate">
                <li>
                  Loads a single open-weight causal language model at a time and keeps it resident
                  in memory.
                </li>
                <li>
                  Forward passes with <em>full hidden-state capture</em> at every transformer
                  layer, plus attention weights per head.
                </li>
                <li>
                  Direct access to the embedding table, unembedding head, and logits (with top-K
                  predictions).
                </li>
                <li>
                  Geometry primitives: norms, cosine similarities, layer-wise isotropy and rank
                  estimates.
                </li>
                <li>
                  HuggingFace cache management — list, download, and delete cached model weights
                  without leaving the app.
                </li>
                <li>
                  Precision resolution: honours each model's native dtype (bf16 / fp16 / fp32)
                  where the device supports it, down-casts only when necessary.
                </li>
              </ul>
            </section>

            <hr className="my-4 border-parchment-dark" />

            {/* Limits */}
            <section>
              <h3 className="font-display text-heading-sm font-semibold mb-2">Limits</h3>
              <ul className="list-disc pl-5 space-y-1 font-sans text-caption text-slate">
                <li>
                  Only <em>open-weight</em>{" "}models that expose hidden states through
                  HuggingFace&nbsp;Transformers. GGUF / Ollama-style releases hide the internals
                  and are not supported.
                </li>
                <li>
                  One model at a time. Loading a new model unloads the current one.
                </li>
                <li>
                  Runs locally only — no inference is sent to any cloud provider.
                </li>
              </ul>
            </section>
          </div>
        </div>
      )}
    </>
  );
}
