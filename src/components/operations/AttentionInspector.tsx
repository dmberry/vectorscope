"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useModel } from "@/context/ModelContext";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { ATTENTION_PRESETS, resolveLayer } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

interface HeadStat {
  head: number;
  meanEntropy: number;
  maxAttention: number;
  diagonalMean: number;
}

interface AttentionResult {
  inputText: string;
  tokens: string[];
  tokenIds: number[];
  layer: number;
  numHeads: number;
  seqLen: number;
  attentionWeights: number[][][]; // [heads, seq, seq]
  meanAttention: number[][];
  headStats: HeadStat[];
}

export default function AttentionInspector() {
  const { backendStatus } = useModel();
  const [text, setText] = useState("The cat sat on the mat");
  const [layer, setLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState<number | "mean">("mean");
  const [result, setResult] = useState<AttentionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const numLayers = backendStatus.model?.numLayers ?? 12;
  const numHeads = backendStatus.model?.numAttentionHeads ?? 12;

  const run = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/attention-layer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim(), layer }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to extract attention");
      }
      const data: AttentionResult = await res.json();
      setResult(data);
      setSelectedHead("mean");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text, layer]);

  const currentAttn =
    result && selectedHead === "mean"
      ? result.meanAttention
      : result && typeof selectedHead === "number"
      ? result.attentionWeights[selectedHead]
      : null;

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Attention Inspector"
        summary="Runs a forward pass and extracts the raw attention weights for every head at a chosen layer. Visualises each head's attention matrix as a heatmap and reports per-head statistics (entropy, diagonal mass, peak weight) so you can see which heads spread their attention widely and which lock onto single tokens."
        details={
          <>
            <p>
              Every transformer layer contains several attention heads working in parallel. Each head learns its own strategy for deciding which tokens in the context matter to which. Attention Inspector pulls those learned strategies out of the model and lets you look at them directly.
            </p>
            <p>
              For a chosen layer, you get a heatmap per head. Rows are query tokens, columns are key tokens, the colour shows the weight the head assigned. A head that mostly follows the diagonal is paying attention to each token&apos;s own position. A head with vertical stripes is treating specific tokens as landmarks the whole sequence pays attention to. A head with diffuse colour is hedging.
            </p>
            <p>
              The per-head statistics quantify these shapes. <strong>Entropy</strong> measures how spread out a head&apos;s attention is — low entropy means it locks onto one or two tokens, high entropy means it smears its attention across the sequence. <strong>Diagonal mean</strong> indicates how self-focused the head is. <strong>Max attention</strong> shows the peak weight it assigns to any single pairing.
            </p>
            <p>
              Heads are not interpretable on their own, but they often cluster into recognisable types: positional heads, copying heads, induction heads, null heads that have learned to do very little. This operation is the first step in sorting them.
            </p>
          </>
        }
      />
      {/* Controls */}
      <div className="card-editorial p-4 space-y-2">
        <div className="flex items-center gap-4 flex-wrap">
          <label className="font-sans text-[11px] text-slate">
            Input text
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
              max={numLayers - 1}
              value={layer}
              onChange={(e) => setLayer(Number(e.target.value))}
              className="ml-2 w-24 align-middle"
            />
            <span className="ml-1 font-mono text-[11px]">{layer}</span>
          </label>
          <button onClick={run} disabled={loading} className="btn-editorial-primary">
            {loading ? "Extracting..." : "Inspect"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={ATTENTION_PRESETS.map((p) => ({
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
          {/* Head selector */}
          <div className="card-editorial p-3">
            <div className="flex items-center gap-1 flex-wrap">
              <span className="font-sans text-[10px] text-slate uppercase tracking-wider mr-2">
                Head
              </span>
              <button
                onClick={() => setSelectedHead("mean")}
                className={`px-2 py-0.5 rounded-sm font-mono text-[10px] transition-colors ${
                  selectedHead === "mean"
                    ? "bg-burgundy text-white"
                    : "bg-cream text-slate hover:bg-parchment"
                }`}
              >
                Mean
              </button>
              {Array.from({ length: result.numHeads }, (_, h) => (
                <button
                  key={h}
                  onClick={() => setSelectedHead(h)}
                  className={`px-2 py-0.5 rounded-sm font-mono text-[10px] transition-colors ${
                    selectedHead === h
                      ? "bg-burgundy text-white"
                      : "bg-cream text-slate hover:bg-parchment"
                  }`}
                >
                  {h}
                </button>
              ))}
            </div>
          </div>

          {/* Attention heatmap */}
          {currentAttn && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Attention Pattern — Layer {result.layer}
                {selectedHead === "mean" ? " (mean across heads)" : `, Head ${selectedHead}`}
              </h3>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    z: currentAttn,
                    x: result.tokens,
                    y: result.tokens,
                    colorscale: [
                      [0, "#F5F0E8"],
                      [0.2, "#D4C9B8"],
                      [0.5, "#A67F6F"],
                      [0.8, "#8B2252"],
                      [1, "#5B2333"],
                    ],
                    colorbar: { title: "Attn", thickness: 12 },
                    hovertemplate:
                      "From: %{y}<br>To: %{x}<br>Weight: %{z:.4f}<extra></extra>",
                  },
                ]}
                layout={{
                  height: Math.max(300, result.seqLen * 35 + 80),
                  width: Math.max(400, result.seqLen * 50 + 120),
                  margin: { l: 80, r: 15, t: 5, b: 60 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Key (attending to)", side: "bottom" as const },
                  yaxis: { title: "Query (from)", autorange: "reversed" as const },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
                }}
                config={{ displayModeBar: true, displaylogo: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Head entropy comparison */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Head Statistics — Layer {result.layer}
            </h3>
            <Plot
              data={[
                {
                  type: "bar",
                  x: result.headStats.map((s) => `H${s.head}`),
                  y: result.headStats.map((s) => s.meanEntropy),
                  marker: {
                    color: result.headStats.map((s) =>
                      selectedHead === s.head ? "#5B2333" : "#A67F6F"
                    ),
                    opacity: 0.8,
                  },
                  name: "Mean Entropy",
                },
              ]}
              layout={{
                height: 180,
                margin: { l: 40, r: 15, t: 5, b: 30 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { gridcolor: "#E8E0D4" },
                yaxis: { title: "Entropy (bits)", gridcolor: "#E8E0D4" },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
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
              {deepDiveOpen ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </button>
            {deepDiveOpen && (
              <div className="px-4 pb-4 border-t border-parchment">
                <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-3">
                  Per-Head Statistics
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-parchment">
                        <th className="p-1 text-left text-slate font-normal">Head</th>
                        <th className="p-1 text-right text-slate font-normal">Mean Entropy</th>
                        <th className="p-1 text-right text-slate font-normal">Max Attention</th>
                        <th className="p-1 text-right text-slate font-normal">Diagonal Mean</th>
                        <th className="p-1 text-left text-slate font-normal">Pattern</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.headStats.map((s) => (
                        <tr
                          key={s.head}
                          className={`border-b border-parchment cursor-pointer hover:bg-cream/30 ${
                            selectedHead === s.head ? "bg-cream/50" : ""
                          }`}
                          onClick={() => setSelectedHead(s.head)}
                        >
                          <td className="p-1">{s.head}</td>
                          <td className="p-1 text-right">{s.meanEntropy.toFixed(3)}</td>
                          <td className="p-1 text-right">{s.maxAttention.toFixed(4)}</td>
                          <td className="p-1 text-right">{s.diagonalMean.toFixed(4)}</td>
                          <td className="p-1 text-[9px] text-slate">
                            {s.meanEntropy < 1.0
                              ? "focused"
                              : s.diagonalMean > 0.3
                              ? "diagonal"
                              : s.meanEntropy > 3.0
                              ? "diffuse"
                              : "mixed"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Attention matrix for current selection */}
                {currentAttn && (
                  <>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-4">
                      Attention Matrix ({selectedHead === "mean" ? "Mean" : `Head ${selectedHead}`})
                    </h4>
                    <div className="overflow-x-auto max-h-64 overflow-y-auto">
                      <table className="font-mono text-[9px]">
                        <thead>
                          <tr className="border-b border-parchment">
                            <th className="p-0.5 text-left text-slate font-normal">From \ To</th>
                            {result.tokens.map((t, i) => (
                              <th key={i} className="p-0.5 text-right text-slate font-normal">
                                {t.trim() || "\\n"}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {currentAttn.map((row, i) => (
                            <tr key={i} className="border-b border-parchment">
                              <td className="p-0.5 text-slate">{result.tokens[i].trim() || "\\n"}</td>
                              {row.map((v, j) => (
                                <td
                                  key={j}
                                  className="p-0.5 text-right"
                                  style={{
                                    backgroundColor: `rgba(91, 35, 51, ${Math.min(v * 2, 1) * 0.4})`,
                                  }}
                                >
                                  {v.toFixed(3)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
