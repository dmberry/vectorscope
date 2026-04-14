"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { EmbeddingTableResult } from "@/types/model";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const BACKEND_URL = "http://localhost:8000";

export default function EmbeddingTable() {
  const [result, setResult] = useState<EmbeddingTableResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleSize, setSampleSize] = useState(3000);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);
  const [showTokens, setShowTokens] = useState(false);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/embedding-table`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_size: sampleSize }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to extract embedding table");
      }
      const data: EmbeddingTableResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [sampleSize]);

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      {/* Controls */}
      <div className="card-editorial p-4">
        <div className="flex items-center gap-4">
          <label className="font-sans text-[11px] text-slate">
            Sample size
            <input
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="input-editorial ml-2 w-20"
              min={100}
              max={50000}
            />
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="btn-editorial-primary"
          >
            {loading ? "Extracting..." : "Extract Embedding Table"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
      </div>

      {result && (
        <>
          {/* Stats cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <StatCard label="Vocabulary" value={result.shape[0].toLocaleString()} unit="tokens" onClick={() => setShowTokens(!showTokens)} />
            <StatCard label="Dimension" value={result.shape[1].toString()} unit="d" />
            <StatCard label="Mean Norm" value={result.stats.meanNorm.toFixed(2)} />
            <StatCard label="Effective Rank" value={result.stats.effectiveRank.toFixed(1)} unit={`/ ${result.shape[1]}`} />
            <StatCard label="Isotropy" value={result.stats.isotropyScore.toFixed(3)} />
            <StatCard label="Norm Range" value={`${result.stats.minNorm.toFixed(1)} \u2013 ${result.stats.maxNorm.toFixed(1)}`} />
          </div>

          {/* Token list (toggled from Vocabulary card) */}
          {showTokens && (
            <div className="card-editorial p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-sans text-xs font-semibold text-slate">
                  Sampled Vocabulary Tokens ({result.sampleTokens.length.toLocaleString()})
                </h3>
                <button onClick={() => setShowTokens(false)} className="font-sans text-[11px] text-slate hover:text-ink">
                  Close
                </button>
              </div>
              <div className="max-h-48 overflow-y-auto border border-parchment rounded-sm p-2">
                <div className="flex flex-wrap gap-1">
                  {result.sampleTokens.map((token, i) => (
                    <span
                      key={i}
                      className="inline-block bg-cream px-1.5 py-0.5 rounded-sm font-mono text-[10px] text-slate hover:bg-parchment hover:text-ink transition-colors cursor-default"
                      title={`norm: ${result.sampleNorms[i].toFixed(3)}`}
                    >
                      {token}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 3D Scatter */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Vocabulary Embedding Space (PCA → 3D, {result.sampleTokens.length.toLocaleString()} tokens sampled)
            </h3>
            <Plot
              data={[
                {
                  type: "scatter3d",
                  mode: "markers",
                  x: result.sampleCoords3d.map((c) => c[0]),
                  y: result.sampleCoords3d.map((c) => c[1]),
                  z: result.sampleCoords3d.map((c) => c[2]),
                  text: result.sampleTokens,
                  hovertemplate: "<b>%{text}</b><br>norm: %{marker.color:.2f}<extra></extra>",
                  marker: {
                    size: 2,
                    color: result.sampleNorms,
                    colorscale: [
                      [0, "#5B2333"],
                      [0.5, "#A67F6F"],
                      [1, "#F5F0E8"],
                    ],
                    colorbar: {
                      title: "Norm",
                      thickness: 15,
                    },
                    opacity: 0.7,
                  },
                },
              ]}
              layout={{
                height: 500,
                margin: { l: 0, r: 0, t: 0, b: 0 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                scene: {
                  xaxis: { title: "PC1", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                  yaxis: { title: "PC2", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                  zaxis: { title: "PC3", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                  bgcolor: "transparent",
                },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: true, displaylogo: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Norm histogram */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">Norm Distribution</h3>
            <Plot
              data={[
                {
                  type: "histogram",
                  x: result.sampleNorms,
                  nbinsx: 50,
                  marker: {
                    color: "#5B2333",
                    opacity: 0.8,
                  },
                },
              ]}
              layout={{
                height: 200,
                margin: { l: 40, r: 15, t: 5, b: 30 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Norm", gridcolor: "#E8E0D4" },
                yaxis: { title: "Count", gridcolor: "#E8E0D4" },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Deep dive panel */}
          <div className="card-editorial">
            <button
              onClick={() => setDeepDiveOpen(!deepDiveOpen)}
              className="w-full flex items-center justify-between px-4 py-2 font-sans text-xs font-medium text-slate hover:text-ink transition-colors"
            >
              <span>Deep Dive</span>
              {deepDiveOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {deepDiveOpen && (
              <div className="px-4 pb-4 border-t border-parchment space-y-4">
                <div className="grid grid-cols-2 gap-6 mt-3">
                  {/* Left column: detailed stats */}
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">Embedding Matrix</h4>
                    <table className="w-full font-mono text-[11px]">
                      <tbody>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Shape</td>
                          <td className="py-1 text-right">{result.shape[0].toLocaleString()} × {result.shape[1]}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Total parameters</td>
                          <td className="py-1 text-right">{(result.shape[0] * result.shape[1]).toLocaleString()}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Effective rank</td>
                          <td className="py-1 text-right">{result.stats.effectiveRank.toFixed(2)} / {result.shape[1]}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Rank utilisation</td>
                          <td className="py-1 text-right">{(result.stats.effectiveRank / result.shape[1] * 100).toFixed(1)}%</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Isotropy score</td>
                          <td className="py-1 text-right">{result.stats.isotropyScore.toFixed(4)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Right column: norm stats */}
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">Norm Statistics</h4>
                    <table className="w-full font-mono text-[11px]">
                      <tbody>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Mean</td>
                          <td className="py-1 text-right">{result.stats.meanNorm.toFixed(4)}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Std dev</td>
                          <td className="py-1 text-right">{result.stats.stdNorm.toFixed(4)}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Min</td>
                          <td className="py-1 text-right">{result.stats.minNorm.toFixed(4)}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Max</td>
                          <td className="py-1 text-right">{result.stats.maxNorm.toFixed(4)}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Range</td>
                          <td className="py-1 text-right">{(result.stats.maxNorm - result.stats.minNorm).toFixed(4)}</td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">CV (σ/μ)</td>
                          <td className="py-1 text-right">{(result.stats.stdNorm / result.stats.meanNorm).toFixed(4)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Interpretation notes */}
                <div className="bg-cream/50 rounded-sm p-3">
                  <p className="font-sans text-[11px] text-slate leading-relaxed">
                    <strong>Effective rank</strong> measures how many dimensions carry significant variance (via Shannon entropy of singular values).
                    A rank of {result.stats.effectiveRank.toFixed(0)} / {result.shape[1]} means {(result.stats.effectiveRank / result.shape[1] * 100).toFixed(0)}% of
                    available dimensions are utilised. <strong>Isotropy</strong> measures directional uniformity: 1.0 = perfectly uniform,
                    0.0 = all vectors aligned. Low isotropy indicates anisotropic geometry with privileged directions.
                  </p>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, unit, onClick }: { label: string; value: string; unit?: string; onClick?: () => void }) {
  return (
    <div
      className={`card-editorial p-3 text-center ${onClick ? "cursor-pointer hover:border-burgundy/30" : ""}`}
      onClick={onClick}
    >
      <div className="font-sans text-[10px] text-slate uppercase tracking-wider">
        {label}
        {onClick && <span className="ml-1 text-burgundy/50">▸</span>}
      </div>
      <div className="font-sans text-sm font-semibold mt-0.5">
        {value}
        {unit && <span className="font-sans text-[10px] text-slate font-normal ml-1">{unit}</span>}
      </div>
    </div>
  );
}
