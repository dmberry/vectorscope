"use client";

import { useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { EmbeddingTableResult } from "@/types/model";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import Plot3DWrapper from "@/components/Plot3DWrapper";
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { VOCABULARY_MAP_PRESETS } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

export default function VocabularyMap() {
  const [result, setResult] = useState<EmbeddingTableResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleSize, setSampleSize] = useState(3000);
  const [searchTerm, setSearchTerm] = useState("");
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

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
        throw new Error(err.detail || "Failed to extract vocabulary");
      }
      const data: EmbeddingTableResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [sampleSize]);

  // Filter tokens for highlighting
  const highlightIndices = result && searchTerm.trim()
    ? result.sampleTokens
        .map((t, i) => (t.toLowerCase().includes(searchTerm.toLowerCase()) ? i : -1))
        .filter((i) => i >= 0)
    : [];

  // Compute token length categories for colour mapping
  const tokenLengths = useMemo(() => {
    if (!result) return [];
    return result.sampleTokens.map((t) => t.trim().length);
  }, [result]);

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Vocabulary Map"
        summary="Samples the input embedding matrix and projects tokens into a navigable 3D map of the vocabulary. A search field highlights tokens matching a query; colour encodes token length. A way to see where morphemes, punctuation, whitespace tokens, and rare words sit relative to one another in the model's raw geometry."
        details={
          <>
            <p>
              Vocabulary Map is Embedding Table turned into something you can walk around in. The same underlying matrix is loaded and projected to 3D, but here the focus is navigation rather than statistics. You can search for a substring and see every matching token highlighted in place. You can colour by token length to see whether long tokens and short tokens occupy different regions of the space.
            </p>
            <p>
              Because modern tokenisers are byte-pair encoders, the vocabulary is not a list of words. It is a mix of whole words, word fragments, morphemes, punctuation sequences, whitespace-prefixed variants, and the occasional piece of accidental unicode. Each of these categories tends to cluster in its own part of the embedding space, and the map makes those clusters visible.
            </p>
            <p>
              Use this operation when you want to develop intuition for how the tokeniser sees the world before you start asking theoretical questions about what the model does with that vocabulary. It is the geography of the model&apos;s raw ontology.
            </p>
          </>
        }
      />
      {/* Controls */}
      <div className="card-editorial p-4 space-y-2">
        <div className="flex items-center gap-4 flex-wrap">
          <label className="font-sans text-[11px] text-slate">
            Sample
            <input
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="input-editorial ml-2 w-24"
              min={1000}
              max={50000}
            />
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="btn-editorial-primary"
          >
            {loading ? "Mapping..." : "Map Vocabulary"}
          </button>
          {result && (
            <label className="font-sans text-[11px] text-slate ml-4">
              Highlight
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-editorial ml-2 w-48"
                placeholder="Search tokens..."
              />
            </label>
          )}
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={VOCABULARY_MAP_PRESETS.map((p) => ({
            label: p.label,
            title: p.title,
            onClick: () => setSampleSize(p.value),
          }))}
        />
      </div>

      {result && (
        <>
          {/* Info bar */}
          <div className="card-editorial p-3">
            <div className="font-sans text-[11px] text-slate flex items-center gap-4 flex-wrap">
              <span>{result.sampleTokens.length.toLocaleString()} tokens sampled from {result.shape[0].toLocaleString()}</span>
              <span>Effective rank: {result.stats.effectiveRank.toFixed(1)} / {result.shape[1]}</span>
              <span>Isotropy: {result.stats.isotropyScore.toFixed(3)}</span>
              {highlightIndices && highlightIndices.length > 0 && (
                <span className="font-semibold" style={{ color: "#5B8C5A" }}>
                  {highlightIndices.length} matches for &ldquo;{searchTerm}&rdquo;
                </span>
              )}
            </div>
          </div>

          {/* 3D Map */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Vocabulary Topology (PCA → 3D, coloured by embedding norm)
            </h3>
            <Plot3DWrapper
              data={[
                // All tokens (background) — rich colour by norm
                {
                  type: "scatter3d",
                  mode: "markers",
                  x: result.sampleCoords3d.map((c) => c[0]),
                  y: result.sampleCoords3d.map((c) => c[1]),
                  z: result.sampleCoords3d.map((c) => c[2]),
                  text: result.sampleTokens,
                  hovertemplate: "<b>%{text}</b><br>norm: %{marker.color:.2f}<extra></extra>",
                  marker: {
                    size: 1.5,
                    color: result.sampleNorms,
                    colorscale: [[0, "#4A6FA5"], [0.5, "#A67F6F"], [1, "#5B2333"]],
                    colorbar: { title: "Norm", thickness: 12 },
                    opacity: highlightIndices && highlightIndices.length > 0 ? 0.15 : 0.5,
                  },
                  name: "Vocabulary",
                },
                // Highlighted tokens
                ...(highlightIndices && highlightIndices.length > 0
                  ? [
                      {
                        type: "scatter3d" as const,
                        mode: "markers+text" as const,
                        x: highlightIndices.map((i) => result.sampleCoords3d[i][0]),
                        y: highlightIndices.map((i) => result.sampleCoords3d[i][1]),
                        z: highlightIndices.map((i) => result.sampleCoords3d[i][2]),
                        text: highlightIndices.map((i) => result.sampleTokens[i]),
                        textposition: "top center" as const,
                        textfont: { size: 9, color: "#2D3A4E" },
                        hovertemplate: "<b>%{text}</b><br>norm: %{customdata:.2f}<extra></extra>",
                        customdata: highlightIndices.map((i) => result.sampleNorms[i]),
                        marker: {
                          size: 5,
                          color: "#5B8C5A",
                          opacity: 1,
                          line: { width: 0.5, color: "#3A5C3A" },
                        },
                        name: "Matches",
                      },
                    ]
                  : []),
              ]}
              layout={{
                height: 600,
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
                showlegend: true,
                legend: { x: 0.02, y: 0.98, font: { size: 9 } },
              }}
              config={{ displayModeBar: true, displaylogo: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Norm distribution */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Norm Distribution
            </h3>
            <Plot
              data={[{
                type: "histogram",
                x: result.sampleNorms,
                nbinsx: 50,
                marker: { color: "#5B2333", opacity: 0.8 },
              }]}
              layout={{
                height: 180,
                margin: { l: 40, r: 15, t: 5, b: 30 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Embedding Norm", gridcolor: "#E8E0D4" },
                yaxis: { title: "Count", gridcolor: "#E8E0D4" },
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
                <div className="grid grid-cols-2 gap-6 mt-3">
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Embedding Space Statistics
                    </h4>
                    <table className="w-full font-mono text-[11px]">
                      <tbody>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Shape</td><td className="py-1 text-right">{result.shape[0].toLocaleString()} x {result.shape[1]}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Sampled</td><td className="py-1 text-right">{result.sampleTokens.length.toLocaleString()} tokens</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Effective rank</td><td className="py-1 text-right">{result.stats.effectiveRank.toFixed(2)} / {result.shape[1]}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Rank utilisation</td><td className="py-1 text-right">{(result.stats.effectiveRank / result.shape[1] * 100).toFixed(1)}%</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Isotropy</td><td className="py-1 text-right">{result.stats.isotropyScore.toFixed(4)}</td></tr>
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Norm Statistics
                    </h4>
                    <table className="w-full font-mono text-[11px]">
                      <tbody>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Mean norm</td><td className="py-1 text-right">{result.stats.meanNorm.toFixed(4)}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Std norm</td><td className="py-1 text-right">{result.stats.stdNorm.toFixed(4)}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Min norm</td><td className="py-1 text-right">{result.stats.minNorm.toFixed(4)}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Max norm</td><td className="py-1 text-right">{result.stats.maxNorm.toFixed(4)}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">Range</td><td className="py-1 text-right">{(result.stats.maxNorm - result.stats.minNorm).toFixed(4)}</td></tr>
                        <tr className="border-b border-parchment"><td className="py-1 text-slate">CV</td><td className="py-1 text-right">{(result.stats.stdNorm / result.stats.meanNorm).toFixed(4)}</td></tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                {/* Token length distribution */}
                <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-4">
                  Token Length Distribution
                </h4>
                <Plot
                  data={[{
                    type: "histogram",
                    x: tokenLengths,
                    nbinsx: 20,
                    marker: { color: "#4A6FA5", opacity: 0.7 },
                  }]}
                  layout={{
                    height: 150,
                    margin: { l: 40, r: 15, t: 5, b: 30 },
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    xaxis: { title: "Token length (chars)", gridcolor: "#E8E0D4" },
                    yaxis: { title: "Count", gridcolor: "#E8E0D4" },
                    font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
