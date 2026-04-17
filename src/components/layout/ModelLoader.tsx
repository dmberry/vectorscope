"use client";

import { useEffect, useState } from "react";
import { useModel } from "@/context/ModelContext";
import { Check, Download, HardDrive, Loader2, Trash2, X } from "lucide-react";
import type { CacheInfo, CachedRepo } from "@/types/model";

// Preset model catalogue. `size` is the on-disk weights footprint after
// download. `minRam` is the minimum system RAM recommended to run the model in
// its native precision without paging to swap (Apple Silicon / modern Linux
// numbers; GPUs with sufficient VRAM may allow lower). The remaining fields
// are the canonical architecture specs published by the model's authors —
// shown in the selected-model detail panel so users can compare models
// without loading them. These will soon move to a user-editable markdown file
// so new releases can be added without rebuilding the frontend.
interface PresetModel {
  id: string;
  name: string;
  size: string;
  minRam: string;
  architecture: string;
  params: string;
  nativeDtype: string;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  vocabSize: number;
  contextLength: number;
  organisation: string;
  releaseYear: number;
  description: string;
}

const PRESET_MODELS: PresetModel[] = [
  {
    id: "openai-community/gpt2",
    name: "GPT-2 (124M)",
    size: "~500 MB",
    minRam: "8 GB",
    architecture: "GPT-2",
    params: "124 M",
    nativeDtype: "float32",
    hiddenSize: 768,
    numLayers: 12,
    numHeads: 12,
    vocabSize: 50257,
    contextLength: 1024,
    organisation: "OpenAI",
    releaseYear: 2019,
    description:
      "The original small GPT-2. Fast, well-understood, the de facto reference model for mechanistic interpretability work. Tied input/output embeddings. BPE tokenizer.",
  },
  {
    id: "Qwen/Qwen3-0.6B",
    name: "Qwen3 0.6B",
    size: "~1.2 GB",
    minRam: "16 GB",
    architecture: "Qwen3",
    params: "0.6 B",
    nativeDtype: "bfloat16",
    hiddenSize: 1024,
    numLayers: 28,
    numHeads: 16,
    vocabSize: 151936,
    contextLength: 32768,
    organisation: "Alibaba",
    releaseYear: 2025,
    description:
      "Alibaba's smallest Qwen 3 dense model. Multilingual, strong reasoning for its size, native bf16. Good stress test for anisotropy / isotropy work at small scale.",
  },
  {
    id: "meta-llama/Llama-3.2-1B",
    name: "Llama 3.2 1B",
    size: "~2.4 GB",
    minRam: "16 GB",
    architecture: "Llama",
    params: "1.2 B",
    nativeDtype: "bfloat16",
    hiddenSize: 2048,
    numLayers: 16,
    numHeads: 32,
    vocabSize: 128256,
    contextLength: 131072,
    organisation: "Meta",
    releaseYear: 2024,
    description:
      "Meta's small Llama 3.2 base. Long context window (128k), grouped-query attention, native bf16. Useful for tracing how a modern production architecture builds up geometry.",
  },
  {
    id: "meta-llama/Llama-3.2-3B",
    name: "Llama 3.2 3B",
    size: "~6.4 GB",
    minRam: "16 GB",
    architecture: "Llama",
    params: "3.2 B",
    nativeDtype: "bfloat16",
    hiddenSize: 3072,
    numLayers: 28,
    numHeads: 24,
    vocabSize: 128256,
    contextLength: 131072,
    organisation: "Meta",
    releaseYear: 2024,
    description:
      "Larger Llama 3.2. Same architecture family as the 1B, deeper and wider. A useful point of comparison for layer-count effects on representation geometry.",
  },
  {
    id: "mistralai/Mistral-7B-v0.3",
    name: "Mistral 7B",
    size: "~14 GB",
    minRam: "24 GB",
    architecture: "Mistral",
    params: "7.2 B",
    nativeDtype: "bfloat16",
    hiddenSize: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 32768,
    contextLength: 32768,
    organisation: "Mistral AI",
    releaseYear: 2024,
    description:
      "Mistral's flagship 7B base. Sliding-window attention, grouped-query attention, native bf16. Tight fit on a 24 GB Mac; prefer to run with nothing else open.",
  },
];

const BACKEND_URL = "http://localhost:8000";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseCacheInfo(raw: any): CacheInfo {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const repos: CachedRepo[] = (raw.repos || []).map((r: any) => ({
    repoId: r.repo_id,
    sizeBytes: r.size_bytes,
    lastAccessed: r.last_accessed,
    lastModified: r.last_modified,
    nbFiles: r.nb_files,
    refs: r.refs || [],
  }));
  return {
    cachePath: raw.cache_path,
    totalSizeBytes: raw.total_size_bytes,
    repoCount: raw.repo_count,
    repos,
  };
}

interface ModelLoaderProps {
  /**
   * Called after a successful load. The page uses this to close the
   * "change model" overlay once the new model is live.
   */
  onLoaded?: () => void;
  /**
   * Called when the user hits the close button on the overlay. Only
   * rendered when provided, so the inline "no model loaded" view stays
   * unadorned.
   */
  onClose?: () => void;
}

export default function ModelLoader({ onLoaded, onClose }: ModelLoaderProps = {}) {
  const { backendStatus, loadModel, unloadModel } = useModel();
  const [customModelId, setCustomModelId] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cache, setCache] = useState<CacheInfo | null>(null);
  // Select-then-confirm: clicking a preset row highlights it and shows the
  // detail panel; loading only happens when the user clicks the explicit
  // Load button. Prevents accidental multi-GB downloads from stray clicks.
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Pull the cache on mount and whenever the backend reconnects, so the "Cached"
  // badges refresh after a download completes.
  useEffect(() => {
    if (backendStatus.status !== "connected") return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/cache`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setCache(parseCacheInfo(data));
      } catch {
        // backend might be mid-restart; cache panel stays absent
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [backendStatus.status, backendStatus.model?.modelId]);

  const cachedSet = new Set((cache?.repos ?? []).map(r => r.repoId));

  const refreshCache = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/cache`);
      if (!res.ok) return;
      const data = await res.json();
      setCache(parseCacheInfo(data));
    } catch {
      // ignore
    }
  };

  const handleDownload = async (modelId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    const preset = PRESET_MODELS.find(p => p.id === modelId);
    const sizeStr = preset ? ` (${preset.size})` : "";
    const ok = window.confirm(
      `Download ${modelId}${sizeStr} from HuggingFace Hub into ~/.cache/huggingface/hub/?\n\n` +
      `This only fetches the weights to disk — the model is not loaded into memory. ` +
      `You can load it later with one click. Download time depends on your connection; ` +
      `multi-gigabyte models may take several minutes.`
    );
    if (!ok) return;

    setDownloadingId(modelId);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/cache/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: modelId }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Download failed");
      }
      await refreshCache();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloadingId(null);
    }
  };

  const handleDelete = async (modelId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const repo = cache?.repos.find(r => r.repoId === modelId);
    const sizeStr = repo ? ` (${(repo.sizeBytes / 1e9).toFixed(2)} GB)` : "";
    const ok = window.confirm(
      `Delete ${modelId}${sizeStr} from the HuggingFace cache?\n\n` +
      `This removes the downloaded weights from ~/.cache/huggingface/hub/ ` +
      `to reclaim disk space. You can re-download later by loading the model again.`
    );
    if (!ok) return;

    try {
      const res = await fetch(`${BACKEND_URL}/cache/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: modelId }),
      });
      if (!res.ok) {
        const err = await res.json();
        setError(err.detail || "Failed to delete model");
        return;
      }
      setError(null);
      await refreshCache();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete model");
    }
  };

  const handleLoad = async (modelId: string) => {
    // Download warning for the custom-ID path (which has no detail panel).
    // Preset downloads are already explicit in the selected-model card,
    // so we skip the extra modal there.
    const isCached = cachedSet.has(modelId);
    const isPreset = PRESET_MODELS.some(p => p.id === modelId);
    if (!isCached && !isPreset) {
      const ok = window.confirm(
        `${modelId} is not in your local cache.\n\n` +
        `Loading will download the weights from HuggingFace Hub into ~/.cache/huggingface/hub/. ` +
        `Depending on your connection and model size this can take seconds to many minutes.\n\n` +
        `Continue?`
      );
      if (!ok) return;
    }

    setLoading(true);
    setLoadingId(modelId);
    setError(null);
    try {
      await loadModel(modelId);
      setSelectedId(null);
      onLoaded?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load model");
    } finally {
      setLoading(false);
      setLoadingId(null);
    }
  };

  if (backendStatus.status === "disconnected") {
    return (
      <div className="card-editorial p-6 max-w-2xl mx-auto mt-12">
        <h2 className="font-display text-display-md text-ink mb-3">Backend not connected</h2>
        <p className="font-body text-body-md text-slate mb-4">
          Vectorscope requires a Python backend running locally. Start it with:
        </p>
        <pre className="bg-cream px-4 py-3 rounded-sm font-sans text-body-sm text-ink overflow-x-auto">
          cd backend && python -m venv .venv && source .venv/bin/activate{"\n"}
          pip install -r requirements.txt{"\n"}
          python main.py
        </pre>
      </div>
    );
  }

  // The full chooser is used both inline (when no model is loaded) and as a
  // modal (when the user clicks the model name in the Header to swap). The
  // "Currently loaded" hint and the unload button below only render in the
  // latter case.

  return (
    <div className="card-editorial p-6 max-w-2xl mx-auto mt-12 relative">
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate hover:text-burgundy transition-colors"
          aria-label="Close model picker"
          title="Close"
        >
          <X className="w-5 h-5" />
        </button>
      )}
      <h2 className="font-display text-display-md text-ink mb-1">
        {backendStatus.model ? "Change model" : "Load a model"}
      </h2>
      {backendStatus.model && (
        <div className="flex items-start justify-between gap-3 mb-2">
          <p className="font-sans text-caption text-slate flex-1">
            Currently loaded: <span className="font-mono text-[11px]">{backendStatus.model.modelId}</span>.
            Loading a new model automatically unloads the current one.
          </p>
          <button
            onClick={async () => {
              await unloadModel();
              onLoaded?.();
            }}
            className="btn-editorial-ghost px-2 py-1 text-[11px] shrink-0"
            title="Unload without loading a different model"
          >
            <X className="w-3 h-3 mr-1 inline-block" />
            Unload
          </button>
        </div>
      )}
      {cache && (
        <p className="font-sans text-caption text-slate/70 mb-4">
          <HardDrive className="w-3 h-3 inline-block mr-1 -mt-0.5" />
          Cache: {cache.repoCount} model{cache.repoCount === 1 ? "" : "s"},{" "}
          {(cache.totalSizeBytes / 1e9).toFixed(2)} GB on disk at{" "}
          <code className="text-[10px] bg-cream px-1 rounded-sm">{cache.cachePath}</code>
        </p>
      )}
      {!cache && <div className="mb-4" />}

      {error && (
        <div className="bg-error-50 border border-error-500/20 text-error-600 px-4 py-2 rounded-sm mb-4 font-sans text-body-sm">
          {error}
        </div>
      )}

      {/* Column headers — make the abbreviations legible at a glance */}
      <div className="flex items-center justify-between px-4 pb-1.5 font-sans text-[10px] uppercase tracking-wider text-slate/60">
        <span>Model</span>
        <div className="flex items-center gap-3">
          <span title="Approximate download / on-disk size of the weights">Download size</span>
          <span className="text-parchment-dark">|</span>
          <span title="Minimum system RAM recommended to run the model in fp16/bf16 precision">
            Min RAM
          </span>
          <span className="text-parchment-dark">|</span>
          <span title="Load into memory (enabled once the model is cached)">Load</span>
          <span className="w-6 text-right" title="Download / delete">·</span>
        </div>
      </div>

      <div className="space-y-2 mb-4">
        {PRESET_MODELS.map(model => {
          const isCached = cachedSet.has(model.id);
          const isThisLoading = loadingId === model.id;
          const isThisDownloading = downloadingId === model.id;
          const isSelected = selectedId === model.id;
          const anyBusy = loading || !!downloadingId;
          return (
            <div
              key={model.id}
              className={`w-full flex items-center justify-between px-4 py-3 border rounded-sm transition-colors ${
                isSelected
                  ? "bg-burgundy/5 border-burgundy/40 ring-1 ring-burgundy/20"
                  : "bg-cream/50 hover:bg-cream border-parchment"
              }`}
            >
              <button
                onClick={() => setSelectedId(isSelected ? null : model.id)}
                disabled={anyBusy}
                className="flex items-center gap-3 flex-1 text-left disabled:opacity-50"
                title={
                  isSelected
                    ? "Click again to deselect"
                    : isCached
                    ? "Cached locally — click to select, then Load"
                    : "Click to select and view details"
                }
              >
                {isThisLoading ? (
                  <Loader2 className="w-4 h-4 text-burgundy animate-spin" />
                ) : isThisDownloading ? (
                  <Loader2 className="w-4 h-4 text-slate animate-spin" />
                ) : isCached ? (
                  <Check className="w-4 h-4 text-green-700" />
                ) : (
                  <Download className="w-4 h-4 text-slate" />
                )}
                <span className="font-sans text-body-sm font-medium text-ink">{model.name}</span>
                {isThisDownloading ? (
                  <span className="font-sans text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-burgundy/10 text-burgundy">
                    Downloading…
                  </span>
                ) : isCached ? (
                  <span className="font-sans text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-green-700/10 text-green-800">
                    Cached
                  </span>
                ) : (
                  <span className="font-sans text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-parchment text-slate">
                    Not downloaded
                  </span>
                )}
                <span className="font-sans text-[10px] uppercase tracking-wider text-slate/60">
                  {model.params} · {model.nativeDtype}
                </span>
              </button>
              <div className="flex items-center gap-3 font-sans text-caption text-slate">
                <span title="Approximate download / on-disk size of the weights">{model.size}</span>
                <span className="text-parchment-dark">|</span>
                <span title="Minimum system RAM recommended to run this model in fp16/bf16 precision without swap thrash">
                  {model.minRam} RAM
                </span>
                <span className="text-parchment-dark">|</span>
                {/* Per-row Load — the primary action when the weights are
                    cached. Shortcut path that skips the detail panel. */}
                <button
                  onClick={e => {
                    e.stopPropagation();
                    handleLoad(model.id);
                  }}
                  disabled={!isCached || anyBusy}
                  className={
                    isCached
                      ? "btn-editorial-primary px-3 py-1 text-[11px] min-w-[72px]"
                      : "px-3 py-1 text-[11px] min-w-[72px] rounded-sm border border-parchment-dark bg-parchment/40 text-slate/50 cursor-not-allowed"
                  }
                  title={
                    !isCached
                      ? "Download the model first to enable Load"
                      : `Load ${model.name} into memory`
                  }
                  aria-label={`Load ${model.id}`}
                >
                  {isThisLoading ? (
                    <span className="flex items-center gap-1 justify-center">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Loading…
                    </span>
                  ) : (
                    "Load"
                  )}
                </button>
                {isCached ? (
                  <button
                    onClick={e => handleDelete(model.id, e)}
                    disabled={anyBusy}
                    className="w-6 h-6 flex items-center justify-center text-slate/60 hover:text-red-700 hover:bg-red-50 rounded-sm transition-colors disabled:opacity-30"
                    title={`Delete ${model.name} from cache to reclaim disk`}
                    aria-label={`Delete ${model.id} from cache`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                ) : (
                  <button
                    onClick={e => handleDownload(model.id, e)}
                    disabled={anyBusy}
                    className="w-6 h-6 flex items-center justify-center text-slate/60 hover:text-burgundy hover:bg-burgundy/5 rounded-sm transition-colors disabled:opacity-30"
                    title={`Download ${model.name} (${model.size}) to cache, without loading into memory`}
                    aria-label={`Download ${model.id} to cache`}
                  >
                    {isThisDownloading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Download className="w-3.5 h-3.5" />
                    )}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected-model detail panel — tech specs + explicit Load button */}
      {selectedId && (() => {
        const m = PRESET_MODELS.find(p => p.id === selectedId);
        if (!m) return null;
        const isCached = cachedSet.has(m.id);
        const isThisLoading = loadingId === m.id;
        return (
          <div className="mb-6 border border-burgundy/30 bg-burgundy/5 rounded-sm p-4">
            <div className="flex items-start justify-between gap-4 mb-3">
              <div>
                <div className="font-sans text-body-sm font-semibold text-ink">{m.name}</div>
                <div className="font-mono text-[11px] text-slate">{m.id}</div>
              </div>
              <button
                onClick={() => setSelectedId(null)}
                className="text-slate/60 hover:text-slate"
                aria-label="Clear selection"
                title="Clear selection"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="font-sans text-caption text-slate mb-3 leading-relaxed">{m.description}</p>

            {/* Two-column spec grid */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-sans text-caption text-slate mb-3">
              <div className="flex justify-between"><span>Architecture</span><span className="text-ink">{m.architecture}</span></div>
              <div className="flex justify-between"><span>Parameters</span><span className="text-ink">{m.params}</span></div>
              <div className="flex justify-between" title="Precision declared by the model's authors. Vectorscope honours this on load, except for fp32 models on MPS/CUDA where it down-casts to fp16 to save memory.">
                <span>Native precision</span>
                <span className="font-mono text-[11px] text-ink">{m.nativeDtype}</span>
              </div>
              <div className="flex justify-between"><span>Hidden size</span><span className="text-ink">{m.hiddenSize.toLocaleString()}</span></div>
              <div className="flex justify-between"><span>Layers</span><span className="text-ink">{m.numLayers}</span></div>
              <div className="flex justify-between"><span>Attention heads</span><span className="text-ink">{m.numHeads}</span></div>
              <div className="flex justify-between"><span>Vocabulary</span><span className="text-ink">{m.vocabSize.toLocaleString()}</span></div>
              <div className="flex justify-between"><span>Context length</span><span className="text-ink">{m.contextLength.toLocaleString()}</span></div>
              <div className="flex justify-between"><span>Released</span><span className="text-ink">{m.organisation}, {m.releaseYear}</span></div>
              <div className="flex justify-between"><span>On disk</span><span className="text-ink">{m.size}</span></div>
            </div>

            {m.nativeDtype === "float32" && (
              <p className="font-sans text-caption text-slate/70 mb-3">
                <strong>Note on precision:</strong> {m.name} ships in float32. On MPS/CUDA devices
                Vectorscope down-casts it to float16 on load to halve the memory footprint, so the
                Settings panel will show the loaded dtype as <code className="font-mono text-[11px] bg-cream px-1 rounded-sm">float16</code>{" "}
                even though the native precision is <code className="font-mono text-[11px] bg-cream px-1 rounded-sm">float32</code>.
                bf16 models (Qwen, Llama, Mistral) are loaded at native precision.
              </p>
            )}

            {!isCached && (
              <p className="font-sans text-caption text-burgundy/80 mb-3">
                This model is not yet downloaded. You can either <strong>download</strong> it now to
                cache for later use, or go straight to <strong>Download and load</strong> to fetch
                the weights and load them into memory in one step (<strong>{m.size}</strong> from
                HuggingFace Hub).
              </p>
            )}

            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => handleLoad(m.id)}
                disabled={loading || !!downloadingId}
                className="btn-editorial-primary text-body-sm"
              >
                {isThisLoading ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Loading…
                  </span>
                ) : isCached ? (
                  "Load model"
                ) : (
                  `Download and load (${m.size})`
                )}
              </button>
              {!isCached && (
                <button
                  onClick={() => handleDownload(m.id)}
                  disabled={loading || !!downloadingId}
                  className="btn-editorial-ghost text-body-sm"
                  title="Download weights to the cache without loading into memory"
                >
                  {downloadingId === m.id ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Downloading…
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Download className="w-3.5 h-3.5" />
                      Download only
                    </span>
                  )}
                </button>
              )}
              <button
                onClick={() => setSelectedId(null)}
                disabled={loading || !!downloadingId}
                className="btn-editorial-ghost text-body-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        );
      })()}

      <p className="font-sans text-caption text-slate/70 mb-3">
        Models marked <em>Will download</em> are fetched from HuggingFace Hub the first time you
        load them and cached to disk. Depending on connection speed and model size, first load can
        take anywhere from a few seconds to several minutes. After that the model lives in your
        cache and loads instantly.
      </p>

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="HuggingFace model ID (e.g. Qwen/Qwen3-0.6B)..."
          value={customModelId}
          onChange={e => setCustomModelId(e.target.value)}
          className="input-editorial flex-1"
        />
        <button
          onClick={() => customModelId.trim() && handleLoad(customModelId.trim())}
          disabled={!customModelId.trim() || loading}
          className="btn-editorial-ghost disabled:opacity-40 disabled:cursor-not-allowed"
          title={
            customModelId.trim()
              ? `Load ${customModelId.trim()} (may trigger download)`
              : "Type a HuggingFace model ID above to enable"
          }
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Load custom"}
        </button>
      </div>

      {/* Roadmap note — locally-trained models */}
      <div className="mt-5 pt-4 border-t border-parchment/60">
        <p className="font-sans text-caption text-slate/70">
          <span className="font-sans text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-burgundy/10 text-burgundy mr-1.5">
            Coming soon
          </span>
          Dedicated UI for loading <strong>locally-trained or fine-tuned models</strong> — pick a
          directory on disk that contains a <code className="text-[10px] bg-cream px-1 rounded-sm">config.json</code>{" "}
          and <code className="text-[10px] bg-cream px-1 rounded-sm">.safetensors</code> weights, no HuggingFace
          round-trip required. This will make Vectorscope usable for inspecting your own checkpoints,
          LoRA adapters, and institutional fine-tunes. Basic support exists today via the custom
          input above (paste an absolute path), but a proper picker and validation are on the Phase 4
          roadmap.
        </p>
        <p className="font-sans text-caption text-slate/70 mt-2">
          <span className="font-sans text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-slate/10 text-slate mr-1.5">
            Also planned
          </span>
          Alternative model sources (ModelScope, institutional mirrors via <code className="text-[10px] bg-cream px-1 rounded-sm">HF_ENDPOINT</code>),
          and an editable preset list so you can curate your own default models without rebuilding.
        </p>
      </div>
    </div>
  );
}
