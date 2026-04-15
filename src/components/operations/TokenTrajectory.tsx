"use client";

import { useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { TokenTrajectoryResult } from "@/types/model";
import { projectPCA3D } from "@/lib/geometry/pca";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import Plot3DWrapper from "@/components/Plot3DWrapper";
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { TOKEN_TRAJECTORY_PRESETS } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

export default function TokenTrajectory() {
  const [text, setText] = useState("justice");
  const [result, setResult] = useState<TokenTrajectoryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const run = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/token-trajectory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to trace token trajectory");
      }
      const data: TokenTrajectoryResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text]);

  // PCA project the first token's vector at each layer into 3D
  const trajectoryCoords = useMemo(() => {
    if (!result) return null;
    // Collect the first token's hidden state at each layer
    const vectors = result.layers.map((l) => l.vectors[0]);
    if (vectors.length < 3) return null;
    return projectPCA3D(vectors);
  }, [result]);

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Token Trajectory"
        summary="Runs a piece of text through every layer of the model and records the hidden-state vector for each token at each depth. The first token's trajectory is projected to 3D via PCA so you can watch how a single word moves through the model's representational space as context accumulates."
        details={
          <>
            <p>
              A transformer does not transform a token once. It transforms it dozens of times, layer after layer, each layer rewriting the vector in light of its neighbours. Token Trajectory makes that journey visible. You give it a short text, and it runs a full forward pass while recording the hidden state of every token at every layer.
            </p>
            <p>
              The 3D view shows the path of the first token through the layers. PCA is fitted across all the layers&apos; positions so the axes reflect the directions of greatest variation in the trajectory itself. A token that barely moves is one the model considers stable. A token that swings through the space is one whose meaning is being aggressively rewritten by context.
            </p>
            <p>
              The per-layer norms show how much geometric energy the vector carries as it travels. Rising norms typically mean the model is accumulating information; falling norms mean it is compressing.
            </p>
            <p>
              This operation is the simplest answer to the question: what does a word look like inside a language model? It looks like a path.
            </p>
          </>
        }
      />
      {/* Controls */}
      <div className="card-editorial p-4 space-y-2">
        <div className="flex items-center gap-4">
          <label className="font-sans text-[11px] text-slate">
            Input text
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              className="input-editorial ml-2 w-64"
              placeholder="Enter a word or phrase..."
            />
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="btn-editorial-primary"
          >
            {loading ? "Tracing..." : "Trace"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={TOKEN_TRAJECTORY_PRESETS.map((p) => ({
            label: p.label,
            title: p.title,
            onClick: () => setText(p.text),
          }))}
        />
      </div>

      {result && (
        <>
          {/* Token info */}
          <div className="card-editorial p-3">
            <div className="flex items-center gap-2 flex-wrap font-sans text-[11px]">
              <span className="text-slate">Tokens:</span>
              {result.tokens.map((t, i) => (
                <span key={i} className="bg-cream px-1.5 py-0.5 rounded-sm font-mono text-[10px]">
                  {t}
                  <span className="text-slate ml-0.5">({result.tokenIds[i]})</span>
                </span>
              ))}
              <span className="text-slate ml-2">{result.layers.length} layers</span>
            </div>
          </div>

          {/* 3D trajectory */}
          {trajectoryCoords && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Token Trajectory through Layers (PCA → 3D)
              </h3>
              <Plot3DWrapper
                data={[
                  // Trajectory line
                  {
                    type: "scatter3d",
                    mode: "lines+markers",
                    x: trajectoryCoords.map((c) => c[0]),
                    y: trajectoryCoords.map((c) => c[1]),
                    z: trajectoryCoords.map((c) => c[2]),
                    text: result.layers.map((l) => `Layer ${l.layer}`),
                    hovertemplate: "<b>%{text}</b><br>norm: %{marker.color:.2f}<extra></extra>",
                    marker: {
                      size: 5,
                      color: result.layers.map((l) => l.norms[0]),
                      colorscale: [
                        [0, "#F5F0E8"],
                        [1, "#5B2333"],
                      ],
                      colorbar: { title: "Norm", thickness: 15 },
                    },
                    line: {
                      color: "#A67F6F",
                      width: 3,
                    },
                  },
                  // Start marker
                  {
                    type: "scatter3d",
                    mode: "markers+text",
                    x: [trajectoryCoords[0][0]],
                    y: [trajectoryCoords[0][1]],
                    z: [trajectoryCoords[0][2]],
                    text: ["Input"],
                    textposition: "top center",
                    marker: { size: 10, color: "#5B8C5A", symbol: "diamond" },
                    showlegend: false,
                  },
                  // End marker
                  {
                    type: "scatter3d",
                    mode: "markers+text",
                    x: [trajectoryCoords[trajectoryCoords.length - 1][0]],
                    y: [trajectoryCoords[trajectoryCoords.length - 1][1]],
                    z: [trajectoryCoords[trajectoryCoords.length - 1][2]],
                    text: ["Output"],
                    textposition: "top center",
                    marker: { size: 10, color: "#5B2333", symbol: "diamond" },
                    showlegend: false,
                  },
                ]}
                layout={{
                  height: 550,
                  margin: { l: 0, r: 0, t: 0, b: 0 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  scene: {
                    xaxis: { title: "PC1", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                    yaxis: { title: "PC2", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                    zaxis: { title: "PC3", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                    bgcolor: "transparent",
                  },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A" },
                  showlegend: false,
                }}
                config={{ displayModeBar: true, displaylogo: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Layer similarity chart */}
          {result.layerSimilarities.length > 0 && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Layer-to-Layer Cosine Similarity
              </h3>
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: result.layerSimilarities.map((_, i) => `${i} \u2192 ${i + 1}`),
                    y: result.layerSimilarities,
                    marker: { color: "#5B2333", size: 6 },
                    line: { color: "#A67F6F", width: 2 },
                  },
                ]}
                layout={{
                  height: 250,
                  margin: { l: 50, r: 20, t: 10, b: 60 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Layer Transition", gridcolor: "#E8E0D4" },
                  yaxis: { title: "Cosine Similarity", gridcolor: "#E8E0D4", range: [0, 1.05] },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A" },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Norm profile */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">Norm Profile across Layers</h3>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: result.layers.map((l) => l.layer),
                  y: result.layers.map((l) => l.norms[0]),
                  marker: { color: "#5B2333", size: 6 },
                  line: { color: "#A67F6F", width: 2 },
                },
              ]}
              layout={{
                height: 250,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Layer", gridcolor: "#E8E0D4" },
                yaxis: { title: "Norm", gridcolor: "#E8E0D4" },
                font: { family: "Inter, sans-serif", color: "#4A4A4A" },
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
                  Per-Layer Statistics (Token 0: &ldquo;{result.tokens[0]}&rdquo;)
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-parchment">
                        <th className="p-1 text-left text-slate font-normal">Layer</th>
                        <th className="p-1 text-right text-slate font-normal">Norm</th>
                        <th className="p-1 text-right text-slate font-normal">Cos to Next</th>
                        <th className="p-1 text-right text-slate font-normal">Cos to Input</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.layers.map((l, i) => (
                        <tr key={i} className="border-b border-parchment">
                          <td className="p-1 text-slate">{l.layer}</td>
                          <td className="p-1 text-right">{l.norms[0].toFixed(2)}</td>
                          <td className="p-1 text-right">
                            {i < result.layerSimilarities.length
                              ? result.layerSimilarities[i].toFixed(4)
                              : "—"}
                          </td>
                          <td className="p-1 text-right">
                            {i === 0 ? "1.0000" : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-3">
                  <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                    Summary
                  </h4>
                  <table className="font-mono text-[11px]">
                    <tbody>
                      <tr className="border-b border-parchment"><td className="py-1 pr-4 text-slate">Layers</td><td className="py-1 text-right">{result.layers.length}</td></tr>
                      <tr className="border-b border-parchment"><td className="py-1 pr-4 text-slate">Tokens</td><td className="py-1 text-right">{result.tokens.length}</td></tr>
                      <tr className="border-b border-parchment"><td className="py-1 pr-4 text-slate">Norm range</td><td className="py-1 text-right">{Math.min(...result.layers.map(l => l.norms[0])).toFixed(2)} — {Math.max(...result.layers.map(l => l.norms[0])).toFixed(2)}</td></tr>
                      <tr className="border-b border-parchment"><td className="py-1 pr-4 text-slate">Mean layer similarity</td><td className="py-1 text-right">{result.layerSimilarities.length > 0 ? (result.layerSimilarities.reduce((a, b) => a + b, 0) / result.layerSimilarities.length).toFixed(4) : "—"}</td></tr>
                      <tr className="border-b border-parchment"><td className="py-1 pr-4 text-slate">Min layer similarity</td><td className="py-1 text-right">{result.layerSimilarities.length > 0 ? Math.min(...result.layerSimilarities).toFixed(4) : "—"}</td></tr>
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
