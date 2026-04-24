"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { GrammarSteeringResult } from "@/types/model";
import OperationIntro from "@/components/OperationIntro";
import ExportMenu from "@/components/ExportMenu";
import Plot3DWrapper from "@/components/Plot3DWrapper";
import { useModel } from "@/context/ModelContext";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const BACKEND_URL = "http://localhost:8000";

// Twelve hand-written contrastive pairs for the "Not X but Y" rhetorical
// antithesis. Positive carries the pattern; negative states the concept
// plainly. Matched for topic and register (both are dictionary-register
// prose). Users should curate their own pair set for serious work — this is
// a demo starter that exercises the operation end to end.
const NOT_X_BUT_Y_PRESET = `+ Democracy is not just a system of government, but a living practice of collective self-rule.
- Democracy is a system in which citizens elect representatives who make and enforce laws.

+ Science is not merely the accumulation of facts, but a disciplined method of inquiry.
- Science investigates natural phenomena using hypothesis, experiment, and observation.

+ Art is not mere decoration, but an encounter with meaning.
- Art consists of works created for aesthetic or expressive purposes, including paintings, sculpture, and music.

+ Education is not just the transfer of information, but the cultivation of judgement.
- Education is the process of acquiring knowledge and skills through study and teaching.

+ Freedom is not simply the absence of constraint, but the capacity for self-direction.
- Freedom refers to the state of being able to act without external restrictions.

+ Justice is not only about punishment, but about restoring what has been broken.
- Justice is the legal and moral principle of treating people fairly under the law.

+ Friendship is not just familiarity, but the slow building of mutual trust.
- Friendship is a voluntary relationship between people based on shared interests and affection.

+ Writing is not merely recording thoughts, but discovering them.
- Writing is the act of producing text using letters or symbols to communicate meaning.

+ A city is not just a collection of buildings, but a network of shared lives.
- A city is a large, densely populated settlement with administrative and commercial functions.

+ Memory is not simply the retrieval of the past, but the ongoing construction of identity.
- Memory is the faculty by which information is encoded, stored, and recalled.

+ Reading is not only decoding words, but inhabiting another mind.
- Reading is the process of understanding written text through visual perception.

+ A good teacher is not one who gives answers, but one who sharpens questions.
- A good teacher conveys information clearly and helps students learn effectively.
`;

// Parser: blank-line-separated blocks. Each block = one line starting "+" and
// one line starting "-". Returns the list of pairs and any parse warnings so
// users see what they did wrong when the format slips.
interface ParsedPairs {
  pairs: { positive: string; negative: string }[];
  warnings: string[];
}
function parsePairs(src: string): ParsedPairs {
  const blocks = src
    .split(/\n\s*\n+/)
    .map((b) => b.trim())
    .filter((b) => b.length > 0);
  const pairs: { positive: string; negative: string }[] = [];
  const warnings: string[] = [];
  blocks.forEach((block, i) => {
    const lines = block.split(/\n+/).map((l) => l.trim()).filter(Boolean);
    const pos = lines.find((l) => l.startsWith("+"));
    const neg = lines.find((l) => l.startsWith("-"));
    if (!pos || !neg) {
      warnings.push(`Block ${i + 1}: expected one "+" line and one "-" line`);
      return;
    }
    pairs.push({
      positive: pos.replace(/^\+\s*/, "").trim(),
      negative: neg.replace(/^-\s*/, "").trim(),
    });
  });
  return { pairs, warnings };
}

export default function GrammarSteering() {
  const { backendStatus } = useModel();
  const containerRef = useRef<HTMLDivElement>(null);
  const [src, setSrc] = useState(NOT_X_BUT_Y_PRESET);
  const [result, setResult] = useState<GrammarSteeringResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const parsed = useMemo(() => parsePairs(src), [src]);

  const run = useCallback(async () => {
    if (parsed.pairs.length < 2) {
      setError("Need at least 2 valid pairs. Check the format (+ line, - line, blank line between pairs).");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/grammar-steering`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pairs: parsed.pairs,
          pca_layer: selectedLayer,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Grammar Steering failed");
      }
      const data: GrammarSteeringResult = await res.json();
      setResult(data);
      if (selectedLayer === null) setSelectedLayer(data.pcaLayer);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [parsed.pairs, selectedLayer]);

  // Re-run when the layer slider moves only if the backend hasn't already
  // produced that layer's PCA (it always computes ALL layers' stats, but PCA
  // is for one layer). To keep the UI snappy we re-run lazily on explicit
  // user action; the chart below the slider always shows per-layer norms.
  const rerunForLayer = useCallback(
    async (layer: number) => {
      if (!result) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${BACKEND_URL}/grammar-steering`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pairs: parsed.pairs, pca_layer: layer }),
        });
        if (!res.ok) throw new Error("Re-run failed");
        setResult(await res.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Re-run failed");
      } finally {
        setLoading(false);
      }
    },
    [parsed.pairs, result]
  );

  // ---- Plot data ----
  const layerIdx = result ? result.layers.map((l) => l.layer) : [];
  const normSeries = result ? result.layers.map((l) => l.steeringNorm) : [];
  const sepSeries = result ? result.layers.map((l) => l.separability) : [];

  return (
    <div className="max-w-7xl mx-auto space-y-4" ref={containerRef}>
      <OperationIntro
        name="Grammar Steering"
        summary={
          "Extracts the activation signature of a rhetorical pattern using Contrastive Activation Addition (CAA). " +
          "Takes a list of matched (positive, negative) prompt pairs — positive carries the pattern, negative is " +
          "the same topic without it — runs both populations through the loaded model, and computes a per-layer " +
          "steering vector: the mean-activation difference between the two sets. Per-layer separability and norm " +
          "trajectory show at which depth the pattern lives. The 3D PCA scatter makes the separation visible. " +
          "This is Phase 1: extraction and geometry. Phase 2 (forthcoming) uses the extracted vector to steer " +
          "generation via forward-pass intervention."
        }
        details={
          <>
            <p>
              Contrastive Activation Addition (Turner et al. 2023, Rimsky et al. 2024). The
              procedure starts from a simple claim: if a model has learned a stylistic
              regularity — say the &quot;Not X but Y&quot; rhetorical antithesis pattern — that
              regularity should show up as a distinguishable direction in activation space.
              Extract it by contrast: subtract the mean activation of examples that lack the
              pattern from the mean activation of examples that carry it. What remains is a
              per-layer <em>steering vector</em>.
            </p>
            <p>
              Each prompt is run through the model with <code className="bg-parchment px-1 rounded">output_hidden_states=True</code>. The last-token
              hidden state at each layer serves as the summary representation of that prompt —
              standard practice in the CAA literature. Across the set, the positive centroid minus the
              negative centroid gives the steering vector. Its norm across layers reveals at
              which depth the pattern is inscribed; its <em>leave-one-out separability</em>
              (nearest-centroid classification accuracy along the direction) tells you how
              cleanly the pattern carves the space.
            </p>
            <p>
              A peak at middle layers with a drop at the final layer is the classic CAA
              signature: the pattern is a mid-depth abstraction that the output layers
              re-absorb into logit space. A monotonically rising norm is the optimiser
              sharpening the distinction through depth. A flat or noisy trajectory means the
              contrast isn&apos;t really about the pattern (check your pairs — are they
              controlled for topic and register?).
            </p>
            <p>
              For robust results, aim for <strong>≥20 matched pairs</strong>. The vector is only as
              good as the pair matching: positive and negative must be the same topic, register,
              length, and speaker role, differing only in whether the pattern is used. The
              preset below (12 &quot;Not X but Y&quot; pairs) is a demonstration starter, not a
              research-grade set.
            </p>
            <p>
              Phase 2 will use the extracted vector to intervene during generation: a PyTorch
              forward hook adds <code className="bg-parchment px-1 rounded">scale × steering_vector</code> to the
              residual stream at the chosen layer every forward step. Positive scaling amplifies
              the pattern; negative scaling suppresses it. If the model can be steered away from
              the pattern by subtracting a single vector, that vector <em>is</em> the pattern&apos;s
              internal representation.
            </p>
          </>
        }
      />

      <div className="card-editorial p-4 space-y-3">
        <div>
          <label className="font-sans text-[11px] text-slate block mb-1">
            Contrastive pairs · format: line starting <code className="bg-parchment px-1 rounded text-[10px]">+</code> for positive,
            line starting <code className="bg-parchment px-1 rounded text-[10px]">-</code> for negative, blank line between pairs
          </label>
          <textarea
            value={src}
            onChange={(e) => setSrc(e.target.value)}
            rows={10}
            className="input-editorial w-full font-mono text-[11px] whitespace-pre"
          />
          <div className="flex items-center justify-between mt-1 text-[10px] text-slate/70">
            <span>
              Parsed <strong>{parsed.pairs.length}</strong> pair{parsed.pairs.length === 1 ? "" : "s"}
              {parsed.warnings.length > 0 && (
                <span className="text-amber-700"> · {parsed.warnings.length} warning(s)</span>
              )}
            </span>
            <button
              onClick={() => setSrc(NOT_X_BUT_Y_PRESET)}
              className="underline decoration-dotted hover:text-burgundy"
            >
              Reload &ldquo;Not X but Y&rdquo; preset (12 pairs)
            </button>
          </div>
          {parsed.warnings.length > 0 && (
            <ul className="list-disc pl-5 mt-1 text-[10px] text-amber-700">
              {parsed.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={run}
            disabled={loading || parsed.pairs.length < 2}
            className="btn-editorial-primary"
          >
            {loading ? "Extracting…" : "Extract steering vectors"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
      </div>

      {result && (
        <>
          <div className="flex justify-end">
            <ExportMenu
              operationName="Grammar Steering"
              modelName={backendStatus.model?.name}
              getBundle={() => ({
                json: result,
                csvTables: [
                  {
                    title: "Per-layer steering diagnostics",
                    headers: [
                      "layer",
                      "steering_norm",
                      "separability",
                      "positive_mean_norm",
                      "negative_mean_norm",
                    ],
                    rows: result.layers.map((l) => [
                      l.layer,
                      l.steeringNorm,
                      l.separability,
                      l.positiveMeanNorm,
                      l.negativeMeanNorm,
                    ]),
                  },
                  {
                    title: `Steering vector at layer ${result.pcaLayer}`,
                    headers: ["dim", "value"],
                    rows: result.layers[result.pcaLayer].steeringVector.map((v, i) => [i, v]),
                  },
                ],
                plotContainer: containerRef.current,
                pdfTitle: "Grammar Steering",
                pdfSubtitle: `${result.numPairs} contrastive pairs · baseline layer ${result.pcaLayer}`,
                pdfMetadata: [
                  { label: "Pairs", value: String(result.numPairs) },
                  { label: "Layers", value: String(result.numLayers) },
                  { label: "Hidden size", value: String(result.hiddenSize) },
                  { label: "PCA layer", value: String(result.pcaLayer) },
                  {
                    label: "PCA explained var (3D)",
                    value: result.pca3d.explainedVariance
                      .map((v) => (v * 100).toFixed(1) + "%")
                      .join(" / "),
                  },
                ],
              })}
            />
          </div>

          {/* Summary cards */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
            <SummaryCard label="Pairs" value={String(result.numPairs)} />
            <SummaryCard label="Layers" value={String(result.numLayers)} />
            <SummaryCard label="Hidden dim" value={result.hiddenSize.toLocaleString()} />
            <SummaryCard
              label="Peak separability"
              value={Math.max(...sepSeries).toFixed(2)}
              sub={`layer ${
                sepSeries.indexOf(Math.max(...sepSeries))
              }`}
            />
            <SummaryCard
              label="Peak ‖steering‖"
              value={Math.max(...normSeries).toFixed(2)}
              sub={`layer ${
                normSeries.indexOf(Math.max(...normSeries))
              }`}
            />
          </div>

          {/* Per-layer norm plot */}
          <div className="card-editorial p-4">
            <div className="font-display text-heading-sm font-semibold mb-1">
              Steering vector norm, per layer
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-2">
              ‖mean(positive) − mean(negative)‖ at each layer. Rising with depth means the
              optimiser is sharpening the contrast; a peak at middle layers with a drop at the
              output is the classic CAA signature of a mid-depth abstraction absorbed into
              logit space near the end.
            </p>
            <Plot
              data={[
                {
                  x: layerIdx,
                  y: normSeries,
                  type: "scatter",
                  mode: "lines+markers",
                  name: "‖steering‖",
                  line: { color: "#862e9c", width: 2 },
                  marker: { size: 6 },
                },
              ]}
              layout={{
                autosize: true,
                height: 280,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: { text: "Layer" }, dtick: 1 },
                yaxis: { title: { text: "‖steering vector‖" }, rangemode: "tozero" },
                showlegend: false,
                font: { family: "ui-sans-serif, system-ui", size: 11 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
              }}
              style={{ width: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>

          {/* Per-layer separability plot */}
          <div className="card-editorial p-4">
            <div className="font-display text-heading-sm font-semibold mb-1">
              Separability, per layer
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-2">
              Leave-one-out accuracy of a nearest-centroid classifier along the steering
              direction. 1.0 means the pattern&apos;s positive and negative populations are
              perfectly separated at that layer; 0.5 is chance. Look for where separability
              first crosses 0.8 — that&apos;s where the abstraction becomes legible.
            </p>
            <Plot
              data={[
                {
                  x: layerIdx,
                  y: sepSeries,
                  type: "scatter",
                  mode: "lines+markers",
                  name: "separability",
                  line: { color: "#0b7285", width: 2 },
                  marker: { size: 6 },
                },
                {
                  x: [layerIdx[0], layerIdx[layerIdx.length - 1]],
                  y: [0.5, 0.5],
                  type: "scatter",
                  mode: "lines",
                  name: "chance",
                  line: { color: "#9aa0a6", width: 1, dash: "dot" },
                  showlegend: false,
                },
              ]}
              layout={{
                autosize: true,
                height: 280,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: { text: "Layer" }, dtick: 1 },
                yaxis: { title: { text: "Separability (LOO)" }, range: [0.4, 1.05] },
                showlegend: false,
                font: { family: "ui-sans-serif, system-ui", size: 11 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
              }}
              style={{ width: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>

          {/* 3D PCA */}
          <div className="card-editorial p-4">
            <div className="flex items-center justify-between mb-1">
              <div className="font-display text-heading-sm font-semibold">
                Positive vs negative populations at layer {result.pca3d.layer}
              </div>
              <div className="flex items-center gap-2 text-[11px] text-slate">
                <label htmlFor="layer-picker">Layer</label>
                <input
                  id="layer-picker"
                  type="number"
                  min={0}
                  max={result.numLayers - 1}
                  value={selectedLayer ?? result.pca3d.layer}
                  onChange={(e) => setSelectedLayer(parseInt(e.target.value, 10))}
                  className="input-editorial w-16 py-0.5 px-2 text-[11px]"
                />
                <button
                  onClick={() => selectedLayer !== null && rerunForLayer(selectedLayer)}
                  disabled={loading || selectedLayer === null || selectedLayer === result.pca3d.layer}
                  className="btn-editorial-ghost text-[11px] px-2 py-0.5"
                >
                  Reproject
                </button>
              </div>
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-2">
              Last-token hidden states at layer {result.pca3d.layer}, projected to 3D via SVD.
              Explained variance: {result.pca3d.explainedVariance.map((v) => (v * 100).toFixed(1) + "%").join(" / ")}.
              Visible clustering confirms the pattern is geometrically legible at this depth.
            </p>
            <Plot3DWrapper
              data={[
                {
                  x: result.pca3d.positive.map((p) => p[0]),
                  y: result.pca3d.positive.map((p) => p[1]),
                  z: result.pca3d.positive.map((p) => p[2]),
                  type: "scatter3d",
                  mode: "markers",
                  name: "positive (pattern)",
                  marker: { size: 6, color: "#862e9c", opacity: 0.9 },
                  text: result.pairs.map((p) => truncate(p.positive, 80)),
                  hovertemplate: "%{text}<extra>positive</extra>",
                },
                {
                  x: result.pca3d.negative.map((p) => p[0]),
                  y: result.pca3d.negative.map((p) => p[1]),
                  z: result.pca3d.negative.map((p) => p[2]),
                  type: "scatter3d",
                  mode: "markers",
                  name: "negative (plain)",
                  marker: { size: 6, color: "#0b7285", opacity: 0.9 },
                  text: result.pairs.map((p) => truncate(p.negative, 80)),
                  hovertemplate: "%{text}<extra>negative</extra>",
                },
              ]}
              layout={{
                autosize: true,
                height: 420,
                margin: { l: 0, r: 0, t: 0, b: 0 },
                scene: {
                  xaxis: { title: { text: "PC1" } },
                  yaxis: { title: { text: "PC2" } },
                  zaxis: { title: { text: "PC3" } },
                },
                legend: { orientation: "h", y: -0.05 },
                font: { family: "ui-sans-serif, system-ui", size: 11 },
                paper_bgcolor: "transparent",
              }}
              style={{ width: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>

          {/* Deep Dive */}
          <div className="card-editorial p-4">
            <button
              onClick={() => setDeepDiveOpen((o) => !o)}
              className="flex items-center gap-2 font-sans text-[11px] text-slate hover:text-burgundy"
            >
              {deepDiveOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              Deep Dive ({result.numLayers} layers, {result.numPairs} pairs)
            </button>
            {deepDiveOpen && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full font-mono text-[11px]">
                  <thead>
                    <tr className="border-b border-parchment-dark">
                      <th className="text-right py-1 pr-3 font-sans">Layer</th>
                      <th className="text-right py-1 pr-3 font-sans">‖steering‖</th>
                      <th className="text-right py-1 pr-3 font-sans">Separability</th>
                      <th className="text-right py-1 pr-3 font-sans">Mean ‖pos‖</th>
                      <th className="text-right py-1 pr-3 font-sans">Mean ‖neg‖</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.layers.map((l) => (
                      <tr key={l.layer} className="border-b border-parchment/60">
                        <td className="py-1 pr-3 text-right">{l.layer}</td>
                        <td className="py-1 pr-3 text-right">{l.steeringNorm.toFixed(3)}</td>
                        <td className="py-1 pr-3 text-right">{l.separability.toFixed(3)}</td>
                        <td className="py-1 pr-3 text-right">{l.positiveMeanNorm.toFixed(2)}</td>
                        <td className="py-1 pr-3 text-right">{l.negativeMeanNorm.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="font-sans text-[10px] text-slate/60 mt-2">
                  Full steering vectors (hidden_size × num_layers floats) are in the JSON export.
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SummaryCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border border-parchment rounded-sm p-3 bg-cream/30">
      <div className="font-sans text-[10px] uppercase tracking-wider text-slate/70">{label}</div>
      <div className="font-mono text-[16px] text-ink">{value}</div>
      {sub && <div className="font-sans text-[10px] text-slate/60">{sub}</div>}
    </div>
  );
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  return s.slice(0, n - 1) + "…";
}
