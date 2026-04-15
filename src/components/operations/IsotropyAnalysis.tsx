"use client";

import { useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { IsotropyResult } from "@/types/model";
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { ISOTROPY_PRESETS } from "@/lib/presets/defaults";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const BACKEND_URL = "http://localhost:8000";

const DEFAULT_TEXT =
  "The cat sat on the mat. The dog barked at the door. Light filtered through the window and fell across the floor. Outside, traffic hummed and the city went about its business without noticing.";

export default function IsotropyAnalysis() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [result, setResult] = useState<IsotropyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/isotropy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to compute isotropy");
      }
      const data: IsotropyResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text]);

  // Prepare data for plots if we have a result
  const layerIdx = result ? result.layers.map((l) => l.layer) : [];
  const isotropySeries = result ? result.layers.map((l) => l.isotropyScore) : [];
  const meanAbsCosSeries = result ? result.layers.map((l) => l.meanAbsCos) : [];
  const top1Series = result ? result.layers.map((l) => l.top1VarianceRatio) : [];
  const top3Series = result ? result.layers.map((l) => l.top3VarianceRatio) : [];
  const top10Series = result ? result.layers.map((l) => l.top10VarianceRatio) : [];
  const meanNormSeries = result ? result.layers.map((l) => l.meanNorm) : [];

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Isotropy Analysis"
        summary="Traces the anisotropy of hidden states through every layer of the model. Runs a prompt forward, pools each layer's token hidden states, and computes the isotropy score, mean pairwise cosine similarity, and top principal component variance ratios at each depth. Makes visible the phenomenon Ethayarajh (2019) identified: contextual representations collapse onto a common direction as you climb the stack."
        details={
          <>
            <p>
              Mu and Viswanath (2018) showed that pretrained word embeddings are
              concentrated in a narrow cone rather than spread uniformly over the
              unit sphere. Ethayarajh (2019) extended the result to contextual
              representations inside a transformer and showed that the problem
              gets worse with depth: by the final layer of BERT or GPT-2, any two
              random token representations have an expected cosine similarity
              close to 1.0. The representation space looks high-dimensional on
              paper but is functionally one-dimensional in large parts of it.
            </p>
            <p>
              This operation computes four things at every layer. The <strong>isotropy score</strong> is
              1 minus the absolute mean pairwise cosine similarity; 1.0 means the
              vectors are spread evenly, 0.0 means they are all aligned. The <strong>mean absolute
              cosine similarity</strong> is Ethayarajh&apos;s raw statistic and tends to
              rise monotonically with layer depth. The <strong>top principal component variance ratios</strong> tell
              you how much of the layer&apos;s variance lives in its first 1/3/10
              directions — when PC1 eats half the variance, the manifold has
              collapsed onto a common direction. And the <strong>norm distribution</strong> tells
              you whether that collapse is accompanied by inflation or deflation
              of the vectors themselves.
            </p>
            <p>
              Theoretically, anisotropy is the geometric signature of the
              training corpus and of the optimiser that fit it. It is not a bug
              to be fixed; it is the shape the model learns to carry meaning in.
              Whether the shape is ideologically innocent is the critical
              question that motivates the whole Critique tab.
            </p>
          </>
        }
      />

      {/* Controls */}
      <div className="card-editorial p-4 space-y-3">
        <div>
          <label htmlFor="isotropy-text" className="font-sans text-[11px] text-slate block mb-1">
            Text (multi-sentence paragraphs give the most stable stats)
          </label>
          <textarea
            id="isotropy-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            className="input-editorial w-full font-mono text-[11px]"
          />
        </div>
        <div className="flex items-center gap-3">
          <button onClick={run} disabled={loading} className="btn-editorial-primary">
            {loading ? "Analysing..." : "Analyse Isotropy"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={ISOTROPY_PRESETS.map((p) => ({
            label: p.label,
            title: p.title,
            onClick: () => setText(p.text),
          }))}
        />
      </div>

      {result && (
        <>
          {/* Top-line numbers */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-2">
            <SummaryCard
              label="Layers"
              value={result.numLayers.toString()}
            />
            <SummaryCard
              label="Tokens analysed"
              value={result.tokens.length.toString()}
            />
            <SummaryCard
              label="Layer 0 isotropy"
              value={result.layers[0].isotropyScore.toFixed(3)}
            />
            <SummaryCard
              label="Final-layer isotropy"
              value={result.layers[result.layers.length - 1].isotropyScore.toFixed(3)}
            />
            <SummaryCard
              label="Collapse (Δ isotropy)"
              value={(
                result.layers[0].isotropyScore -
                result.layers[result.layers.length - 1].isotropyScore
              ).toFixed(3)}
            />
          </div>

          {/* Isotropy vs layer */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-1">
              Isotropy vs Layer
            </h3>
            <p className="font-sans text-[11px] text-slate mb-2">
              Isotropy score (1 = perfectly uniform, 0 = collapsed) and mean
              absolute pairwise cosine similarity across the layer stack. The
              canonical Ethayarajh signature is isotropy falling and mean cosine
              rising as you climb the layers.
            </p>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: isotropySeries,
                  name: "Isotropy score",
                  line: { color: "#5B2333", width: 2 },
                  marker: { size: 5, color: "#5B2333" },
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: meanAbsCosSeries,
                  name: "Mean |cos|",
                  line: { color: "#4A6FA5", width: 2, dash: "dot" },
                  marker: { size: 5, color: "#4A6FA5" },
                  yaxis: "y2",
                },
              ]}
              layout={{
                height: 260,
                margin: { l: 50, r: 50, t: 10, b: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Layer", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                yaxis: {
                  title: "Isotropy",
                  gridcolor: "#E8E0D4",
                  zerolinecolor: "#D4C9B8",
                  range: [0, 1],
                },
                yaxis2: {
                  title: "Mean |cos|",
                  overlaying: "y",
                  side: "right",
                  range: [0, 1],
                },
                legend: { orientation: "h", y: -0.25 },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Top PC variance ratios */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-1">
              Top Principal Component Dominance
            </h3>
            <p className="font-sans text-[11px] text-slate mb-2">
              Fraction of per-layer variance captured by the first 1, 3, and 10
              principal components. When the top-1 ratio climbs above 0.5 the
              layer has collapsed onto a common direction — a single axis is
              doing most of the geometric work. Mu &amp; Viswanath (2018) call
              this the &ldquo;rogue dimension&rdquo; effect.
            </p>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: top1Series,
                  name: "Top 1 PC",
                  line: { color: "#5B2333", width: 2 },
                  marker: { size: 5 },
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: top3Series,
                  name: "Top 3 PCs",
                  line: { color: "#A67F6F", width: 2 },
                  marker: { size: 5 },
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: top10Series,
                  name: "Top 10 PCs",
                  line: { color: "#4A6FA5", width: 2 },
                  marker: { size: 5 },
                },
              ]}
              layout={{
                height: 260,
                margin: { l: 50, r: 15, t: 10, b: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Layer", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                yaxis: {
                  title: "Variance fraction",
                  gridcolor: "#E8E0D4",
                  zerolinecolor: "#D4C9B8",
                  range: [0, 1],
                },
                legend: { orientation: "h", y: -0.25 },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Norm trajectory */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-1">
              Mean Hidden State Norm by Layer
            </h3>
            <p className="font-sans text-[11px] text-slate mb-2">
              Average Euclidean length of a token hidden state at each layer.
              Watch for growth, collapse, or sudden jumps — they often line up
              with the layers where the representation reshapes most violently.
            </p>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: layerIdx,
                  y: meanNormSeries,
                  line: { color: "#5B8C5A", width: 2 },
                  marker: { size: 5, color: "#5B8C5A" },
                  name: "Mean norm",
                },
              ]}
              layout={{
                height: 200,
                margin: { l: 50, r: 15, t: 10, b: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "Layer", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                yaxis: { title: "‖h‖", gridcolor: "#E8E0D4", zerolinecolor: "#D4C9B8" },
                showlegend: false,
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>

          {/* Cosine histograms: first / middle / last */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-1">
              Pairwise Cosine Distribution (first / middle / last layer)
            </h3>
            <p className="font-sans text-[11px] text-slate mb-2">
              Histogram of pairwise cosine similarities between tokens at three
              depths. A distribution that shifts right and narrows with depth is
              the visual signature of a collapsing manifold.
            </p>
            <Plot
              data={[
                {
                  type: "bar",
                  x: histogramCentres(result.cosineHistograms.first.edges),
                  y: result.cosineHistograms.first.counts,
                  name: `Layer ${result.cosineHistograms.first.layer}`,
                  marker: { color: "#5B8C5A", opacity: 0.55 },
                },
                {
                  type: "bar",
                  x: histogramCentres(result.cosineHistograms.middle.edges),
                  y: result.cosineHistograms.middle.counts,
                  name: `Layer ${result.cosineHistograms.middle.layer}`,
                  marker: { color: "#A67F6F", opacity: 0.55 },
                },
                {
                  type: "bar",
                  x: histogramCentres(result.cosineHistograms.last.edges),
                  y: result.cosineHistograms.last.counts,
                  name: `Layer ${result.cosineHistograms.last.layer}`,
                  marker: { color: "#5B2333", opacity: 0.55 },
                },
              ]}
              layout={{
                height: 260,
                margin: { l: 50, r: 15, t: 10, b: 40 },
                barmode: "overlay",
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: {
                  title: "Pairwise cosine",
                  gridcolor: "#E8E0D4",
                  zerolinecolor: "#D4C9B8",
                  range: [-1, 1],
                },
                yaxis: { title: "Count", gridcolor: "#E8E0D4" },
                legend: { orientation: "h", y: -0.25 },
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
              <span>Deep Dive — per-layer statistics table</span>
              {deepDiveOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {deepDiveOpen && (
              <div className="px-4 pb-4 border-t border-parchment">
                <div className="overflow-x-auto mt-3">
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="border-b border-parchment">
                        <th className="text-left py-1 px-2 text-slate">Layer</th>
                        <th className="text-right py-1 px-2 text-slate">Isotropy</th>
                        <th className="text-right py-1 px-2 text-slate">Mean cos</th>
                        <th className="text-right py-1 px-2 text-slate">Mean |cos|</th>
                        <th className="text-right py-1 px-2 text-slate">Top 1</th>
                        <th className="text-right py-1 px-2 text-slate">Top 3</th>
                        <th className="text-right py-1 px-2 text-slate">Top 10</th>
                        <th className="text-right py-1 px-2 text-slate">‖h‖ mean</th>
                        <th className="text-right py-1 px-2 text-slate">‖h‖ std</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.layers.map((l) => (
                        <tr key={l.layer} className="border-b border-parchment/60">
                          <td className="text-left py-1 px-2">{l.layer}</td>
                          <td className="text-right py-1 px-2">{l.isotropyScore.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.meanCos.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.meanAbsCos.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.top1VarianceRatio.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.top3VarianceRatio.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.top10VarianceRatio.toFixed(4)}</td>
                          <td className="text-right py-1 px-2">{l.meanNorm.toFixed(3)}</td>
                          <td className="text-right py-1 px-2">{l.stdNorm.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="font-sans text-[11px] text-slate mt-3 leading-relaxed">
                  Isotropy = 1 − |mean pairwise cos|. Top-k variance ratios are
                  the fraction of total variance captured by the first k
                  principal components at that layer. Sample size per layer
                  equals the number of tokens in the prompt.
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card-editorial p-2 text-center">
      <div className="font-sans text-[10px] text-slate uppercase tracking-wider">{label}</div>
      <div className="font-sans text-xs font-semibold mt-0.5">{value}</div>
    </div>
  );
}

function histogramCentres(edges: number[]): number[] {
  const centres: number[] = [];
  for (let i = 0; i < edges.length - 1; i++) {
    centres.push((edges[i] + edges[i + 1]) / 2);
  }
  return centres;
}
