"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useModel } from "@/context/ModelContext";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { LAYER_PROBE_PRESETS, resolveLayer } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

interface LayerProbeResult {
  inputText: string;
  layer: number;
  numLayers: number;
  tokens: string[];
  tokenIds: number[];
  norms: number[];
  tokenSimilarities: number[][];
  vectors: number[][];
}

export default function LayerProbe() {
  const { backendStatus } = useModel();
  const [text, setText] = useState("The cat sat on the mat");
  const [layer, setLayer] = useState(6);
  const [result, setResult] = useState<LayerProbeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const numLayers = backendStatus.model?.numLayers ?? 12;

  const run = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/layer-probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), layer }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to probe layer");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text, layer]);

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Layer Probe"
        summary="Freezes a forward pass at a chosen layer and extracts the hidden-state vector for every token. Reports per-token norms and the pairwise cosine similarity matrix across the sequence. Lets you inspect how much each token has been pulled toward its neighbours at a specific depth in the model."
        details={
          <>
            <p>
              Layer Probe takes a forward pass and halts it conceptually at a single depth. You choose the layer, it returns the hidden state of every token at that exact point in the computation. Think of it as a geometric x-ray at a single slice of the model.
            </p>
            <p>
              Two outputs matter. The first is the norm of each token: how much geometric weight it carries at this depth. The second is the pairwise cosine similarity matrix — a token-by-token grid showing how close each token&apos;s vector is to every other. Diagonal entries are always 1.0; off-diagonal entries reveal local geometric gravity. Two tokens with a high cosine at layer 6 have been pulled into alignment by the first six layers of attention.
            </p>
            <p>
              Shallow layers tend to keep tokens distinct. Middle layers start to merge neighbours into shared subspaces. Later layers often show the manifold collapsing into fewer, thicker clusters. Layer Probe is the right tool when you want to ask: at what depth did these words start to look alike?
            </p>
          </>
        }
      />
      {/* Controls */}
      <div className="card-editorial p-4 space-y-2">
        <div className="flex items-center gap-4 flex-wrap">
          <label className="font-sans text-[11px] text-slate">
            Input
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              className="input-editorial ml-2 w-64"
              placeholder="Enter text..."
            />
          </label>
          <label className="font-sans text-[11px] text-slate">
            Layer
            <input
              type="range"
              min={0}
              max={numLayers}
              value={layer}
              onChange={(e) => setLayer(Number(e.target.value))}
              className="ml-2 w-32"
            />
            <span className="ml-1 font-mono text-xs">{layer}</span>
          </label>
          <button onClick={run} disabled={loading} className="btn-editorial-primary">
            {loading ? "Probing..." : "Probe"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={LAYER_PROBE_PRESETS.map((p) => ({
            label: p.label,
            title: p.title,
            onClick: () => {
              setText(p.text);
              setLayer(resolveLayer(p.layer, numLayers));
            },
          }))}
        />
      </div>

      {result && (
        <>
          {/* Token info */}
          <div className="card-editorial p-3">
            <div className="flex items-center gap-2 flex-wrap font-sans text-[11px]">
              <span className="text-slate">Layer {result.layer} / {result.numLayers} |</span>
              {result.tokens.map((t, i) => (
                <span key={i} className="bg-cream px-1.5 py-0.5 rounded-sm font-mono text-[10px]">
                  {t}
                  <span className="text-slate ml-0.5">({result.norms[i].toFixed(1)})</span>
                </span>
              ))}
            </div>
          </div>

          {/* Token similarity heatmap */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Token Position Cosine Similarities (Layer {result.layer})
            </h3>
            <Plot
              data={[{
                type: "heatmap",
                z: result.tokenSimilarities,
                x: result.tokens,
                y: result.tokens,
                colorscale: [[0, "#F5F0E8"], [0.5, "#A67F6F"], [1, "#5B2333"]],
                hovertemplate: "%{x} vs %{y}<br>cosine: %{z:.3f}<extra></extra>",
                colorbar: { title: "cos", thickness: 12 },
              }]}
              layout={{
                height: 400,
                margin: { l: 80, r: 15, t: 5, b: 80 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
                xaxis: { side: "bottom" },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Norm bar chart */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Hidden State Norms by Token Position
            </h3>
            <Plot
              data={[{
                type: "bar",
                x: result.tokens,
                y: result.norms,
                marker: { color: "#5B2333", opacity: 0.8 },
              }]}
              layout={{
                height: 200,
                margin: { l: 40, r: 15, t: 5, b: 60 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                yaxis: { title: "Norm", gridcolor: "#E8E0D4" },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Deep dive */}
          <div className="card-editorial">
            <button
              onClick={() => setDeepDiveOpen(!deepDiveOpen)}
              className="w-full flex items-center justify-between px-4 py-2 font-sans text-xs font-medium text-slate hover:text-ink transition-colors"
            >
              <span>Deep Dive</span>
              {deepDiveOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {deepDiveOpen && (
              <div className="px-4 pb-4 border-t border-parchment">
                <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-3">
                  Full Similarity Matrix
                </h4>
                <div className="overflow-x-auto">
                  <table className="font-mono text-[10px]">
                    <thead>
                      <tr>
                        <th className="p-1"></th>
                        {result.tokens.map((t, i) => (
                          <th key={i} className="p-1 text-slate font-normal">{t}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.tokens.map((t, i) => (
                        <tr key={i}>
                          <td className="p-1 text-slate">{t}</td>
                          {result.tokenSimilarities[i].map((sim, j) => (
                            <td key={j} className="p-1 text-right" style={{
                              backgroundColor: `rgba(91, 35, 51, ${Math.max(0, sim) * 0.3})`,
                            }}>
                              {sim.toFixed(3)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
