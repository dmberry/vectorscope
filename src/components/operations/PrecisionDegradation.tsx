"use client";

import { useCallback, useState } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import type {
  PrecisionDegradationResult,
  PrecisionSweepEntry,
} from "@/types/model";
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const BACKEND_URL = "http://localhost:8000";

// All supported precisions shown in the ladder. The baseline (loaded dtype)
// is always implicit; selecting a precision above or equal to the baseline
// still works — the round-trip is approximately identity.
const ALL_PRECISIONS = [
  { id: "bfloat16", label: "bf16", kind: "float", bits: 16 },
  { id: "float16", label: "fp16", kind: "float", bits: 16 },
  { id: "int8", label: "int8", kind: "int", bits: 8 },
  { id: "int4", label: "int4", kind: "int", bits: 4 },
  { id: "int2", label: "int2", kind: "int", bits: 2 },
] as const;

const DEFAULT_TEXT = "The capital of France is";

// Theoretically-motivated prompts. The interesting thing about Precision
// Degradation is watching the prediction flip: factual completions, ambiguous
// references, and contested concepts all fail at different precisions.
const PRECISION_PRESETS = [
  {
    label: "France capital",
    title: "Well-known factual completion — monitor the top-1 token.",
    text: "The capital of France is",
  },
  {
    label: "Euler identity",
    title: "Mathematical identity — does the model preserve it under compression?",
    text: "The value of e raised to the power of i times pi is",
  },
  {
    label: "Gender bias",
    title: "Stereotype probe — bias may survive at lower precisions, or amplify.",
    text: "The nurse said that she",
  },
  {
    label: "Contested concept",
    title: "Ideologically-loaded term — precision loss compresses the manifold onto dominant readings.",
    text: "True freedom means",
  },
  {
    label: "Agreement trap",
    title: "Subject-verb agreement test with an attractor noun.",
    text: "The keys to the cabinet that contained the manuscripts",
  },
  {
    label: "Berry on capital",
    title: "Author quote — how does the model complete the characteristic idiom?",
    text: "Capital is no longer disciplining the signal into bits, it is",
  },
];

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(1)} kB`;
}

// Deterministic per-precision colour assignment.
const PRECISION_COLOR: Record<string, string> = {
  bfloat16: "#0b7285",   // teal
  float16: "#1864ab",    // deep blue
  int8: "#d9480f",       // rust
  int4: "#862e9c",       // plum
  int2: "#6a1b1b",       // dried blood (it earned it)
};

export default function PrecisionDegradation() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [selected, setSelected] = useState<string[]>(["float16", "int8", "int4"]);
  const [result, setResult] = useState<PrecisionDegradationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);

  const toggle = (id: string) => {
    setSelected(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const run = useCallback(async () => {
    if (selected.length === 0) {
      setError("Pick at least one target precision.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/precision-degradation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, precisions: selected, top_k: 10 }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Precision Degradation failed");
      }
      const data: PrecisionDegradationResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text, selected]);

  // ---- Plot data ---------------------------------------------------------

  const cosinePlotData =
    result?.precisions.map(p => ({
      x: p.layers.map(l => l.layer),
      y: p.layers.map(l => l.meanCosine),
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: p.label,
      line: { color: PRECISION_COLOR[p.precision] ?? "#444", width: 2 },
      marker: { size: 5 },
    })) ?? [];

  const relErrPlotData =
    result?.precisions.map(p => ({
      x: p.layers.map(l => l.layer),
      y: p.layers.map(l => l.relError),
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: p.label,
      line: { color: PRECISION_COLOR[p.precision] ?? "#444", width: 2 },
      marker: { size: 5 },
    })) ?? [];

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Precision Degradation"
        summary="The Signal Degradation Laboratory in miniature. Runs the same prompt through the loaded model at baseline precision, then at each selected target precision (bf16 / fp16 / int8 / int4 / int2), comparing hidden states layer-by-layer and final predictions at the output. Quantisation is applied in-process via round-to-nearest fake-quant, so all observed differences come from the weights losing grain rather than from a different model."
        details={
          <>
            <p>
              Industry narrates low-precision inference as an engineering
              optimisation: faster, cheaper, same behaviour. A critical account
              asks a different question. What signal does a compressed medium
              no longer carry? Which distinctions does the geometry stop being
              able to make? This operation makes those questions empirical.
            </p>
            <p>
              The procedure is in-process fake-quantisation. A baseline forward
              pass captures hidden states and logits at the model&apos;s loaded
              precision. For each selected target precision every parameter
              tensor is round-tripped through that precision (cast down, cast
              back), a second forward pass captures the degraded outputs, and
              the parameters are restored from a CPU snapshot. Integer
              precisions use per-tensor symmetric quantisation with round-to-
              nearest; no calibration, no per-channel scales. The point is a
              visible simple baseline, not a match for GPTQ or AWQ.
            </p>
            <p>
              <strong>How to read this.</strong> <em>Mean cosine</em> per layer
              tells you where the geometry first starts to diverge from the
              baseline; it stays near 1.0 while compression is tolerable and
              falls once the precision floor removes meaningful information.{" "}
              <em>Relative error</em> shows the growing magnitude of that
              divergence. The <em>output metrics</em> tell you whether the
              difference has crossed into a different prediction: argmax match,
              top-K overlap with the baseline distribution, and KL divergence.
              The <em>prediction table</em> shows the top tokens at each
              precision, side by side with the baseline.
            </p>
            <p>
              Implements the Signal Degradation Laboratory method described in
              the Leverhulme Centre for Vector Media proposal. Part of a family
              of critical-geometric tools that treat quantisation as a
              diagnostic rather than a deliverable.
            </p>
          </>
        }
      />

      {/* Controls */}
      <div className="card-editorial p-4 space-y-3">
        <div>
          <label htmlFor="precision-text" className="font-sans text-[11px] text-slate block mb-1">
            Prompt (the last token&apos;s logits are compared across precisions)
          </label>
          <textarea
            id="precision-text"
            value={text}
            onChange={e => setText(e.target.value)}
            rows={2}
            className="input-editorial w-full font-mono text-[11px]"
          />
        </div>

        <div>
          <div className="font-sans text-[11px] text-slate mb-1">Target precisions</div>
          <div className="flex flex-wrap gap-2">
            {ALL_PRECISIONS.map(p => {
              const checked = selected.includes(p.id);
              return (
                <button
                  key={p.id}
                  onClick={() => toggle(p.id)}
                  disabled={loading}
                  className={
                    "px-3 py-1 rounded-sm font-mono text-[11px] border transition-colors " +
                    (checked
                      ? "bg-burgundy text-white border-burgundy"
                      : "bg-cream/50 text-slate border-parchment hover:bg-cream")
                  }
                  title={`${p.label} — ${p.kind}, ${p.bits} bits`}
                >
                  {checked ? "✓ " : ""}
                  {p.label}
                </button>
              );
            })}
          </div>
          <p className="font-sans text-[10px] text-slate/60 mt-1">
            Baseline is the model&apos;s currently-loaded dtype. Selecting a precision above the
            baseline is effectively a near-identity check.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button onClick={run} disabled={loading} className="btn-editorial-primary">
            {loading ? "Degrading…" : "Run precision sweep"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>

        <PresetChipRow
          disabled={loading}
          items={PRECISION_PRESETS.map(p => ({
            label: p.label,
            title: p.title,
            onClick: () => setText(p.text),
          }))}
        />
      </div>

      {/* Results */}
      {result && (
        <>
          {result.memoryWarning && (
            <div className="bg-warning-50 border border-warning-500/30 text-warning-700 px-4 py-2 rounded-sm font-sans text-caption">
              <strong>Memory warning.</strong> This model is{" "}
              {fmtBytes(result.modelSizeBytes)}. Precision Degradation snapshots parameters to
              CPU, temporarily doubling memory use. If the operation fails, pick a smaller model.
            </div>
          )}

          {/* Summary cards — one per precision */}
          <div className="card-editorial p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="font-display text-heading-sm font-semibold">Degradation summary</div>
                <div className="font-sans text-[11px] text-slate/70">
                  Baseline: <span className="font-mono">{result.baselineDtype}</span> · last-token
                  argmax: <span className="font-mono">{result.baselineTopPredictions[0]?.token ?? "–"}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {result.precisions.map(p => (
                <PrecisionCard key={p.precision} entry={p} />
              ))}
            </div>
          </div>

          {/* Per-layer cosine plot */}
          <div className="card-editorial p-4">
            <div className="font-display text-heading-sm font-semibold mb-1">
              Mean cosine similarity vs baseline, per layer
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-2">
              How closely does each layer&apos;s hidden-state geometry track the baseline as
              precision drops? Values near 1.0 mean the layer survived compression; values
              significantly below 1.0 are where meaningful geometric information was lost.
            </p>
            <Plot
              data={cosinePlotData}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: { text: "Layer" }, dtick: 1 },
                yaxis: { title: { text: "Mean cosine" }, range: [Math.min(0.5, ...cosinePlotData.flatMap(d => d.y)) - 0.05, 1.01] },
                legend: { orientation: "h", y: -0.2 },
                font: { family: "ui-sans-serif, system-ui", size: 11 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
              }}
              style={{ width: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>

          {/* Per-layer relative error plot */}
          <div className="card-editorial p-4">
            <div className="font-display text-heading-sm font-semibold mb-1">
              Relative error vs baseline, per layer
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-2">
              ‖Δh‖ / ‖h‖ at each layer, where Δh is the difference between degraded and baseline
              hidden states. Reads as a fraction: 0.05 means a 5% deviation in the layer&apos;s
              geometry.
            </p>
            <Plot
              data={relErrPlotData}
              layout={{
                autosize: true,
                height: 320,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: { text: "Layer" }, dtick: 1 },
                yaxis: { title: { text: "Relative error" }, rangemode: "tozero" },
                legend: { orientation: "h", y: -0.2 },
                font: { family: "ui-sans-serif, system-ui", size: 11 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
              }}
              style={{ width: "100%" }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>

          {/* Top-5 prediction comparison */}
          <div className="card-editorial p-4 overflow-x-auto">
            <div className="font-display text-heading-sm font-semibold mb-2">
              Top-5 next-token predictions
            </div>
            <p className="font-sans text-[11px] text-slate/70 mb-3">
              What does the model want to say next, at each precision? Baseline column is the
              ground truth for this model; each precision column shows the top-5 tokens and their
              probabilities after fake-quant.
            </p>
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="border-b border-parchment-dark">
                  <th className="text-left py-1 pr-4 font-sans font-semibold">Rank</th>
                  <th className="text-left py-1 pr-4 font-sans font-semibold">
                    baseline <span className="text-slate/60 font-normal">({result.baselineDtype})</span>
                  </th>
                  {result.precisions.map(p => (
                    <th key={p.precision} className="text-left py-1 pr-4 font-sans font-semibold">
                      {p.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3, 4].map(rank => (
                  <tr key={rank} className="border-b border-parchment/60">
                    <td className="py-1 pr-4 text-slate/60">{rank + 1}</td>
                    <td className="py-1 pr-4">
                      <PredCell p={result.baselineTopPredictions[rank]} />
                    </td>
                    {result.precisions.map(p => (
                      <td key={p.precision} className="py-1 pr-4">
                        <PredCell
                          p={p.topPredictions[rank]}
                          highlight={rank === 0 && !p.output.argmaxMatch}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Deep Dive */}
          <div className="card-editorial p-4">
            <button
              onClick={() => setDeepDiveOpen(o => !o)}
              className="flex items-center gap-2 font-sans text-[11px] text-slate hover:text-burgundy"
            >
              {deepDiveOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              Deep Dive ({result.numLayers} layers × {result.precisions.length} precisions)
            </button>

            {deepDiveOpen && (
              <div className="mt-3 space-y-4">
                {/* Per-precision output metrics */}
                <div>
                  <div className="font-sans text-[11px] text-slate font-semibold mb-1">
                    Output-level metrics
                  </div>
                  <table className="w-full font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-parchment-dark">
                        <th className="text-left py-1 pr-3 font-sans">Precision</th>
                        <th className="text-right py-1 pr-3 font-sans">Argmax match</th>
                        <th className="text-right py-1 pr-3 font-sans">Top-K overlap</th>
                        <th className="text-right py-1 pr-3 font-sans">KL (nats)</th>
                        <th className="text-right py-1 pr-3 font-sans">H_ref (bits)</th>
                        <th className="text-right py-1 pr-3 font-sans">H_quant (bits)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.precisions.map(p => (
                        <tr key={p.precision} className="border-b border-parchment/60">
                          <td className="py-1 pr-3">{p.label}</td>
                          <td className="py-1 pr-3 text-right">{p.output.argmaxMatch ? "✓" : "✗"}</td>
                          <td className="py-1 pr-3 text-right">{p.output.topKOverlap.toFixed(3)}</td>
                          <td className="py-1 pr-3 text-right">{p.output.klDivergence.toFixed(4)}</td>
                          <td className="py-1 pr-3 text-right">{p.output.entropyRef.toFixed(3)}</td>
                          <td className="py-1 pr-3 text-right">{p.output.entropyQuant.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Per-layer stats per precision */}
                <div className="space-y-3">
                  {result.precisions.map(p => (
                    <details key={p.precision} className="border border-parchment rounded-sm">
                      <summary className="cursor-pointer px-3 py-1.5 font-sans text-[11px] font-semibold bg-cream/40">
                        <span className="font-mono">{p.label}</span> — per-layer MSE, rel err, mean cos, min cos
                      </summary>
                      <div className="px-3 py-2 overflow-x-auto">
                        <table className="w-full font-mono text-[10px]">
                          <thead>
                            <tr className="border-b border-parchment-dark">
                              <th className="text-right py-0.5 pr-3 font-sans">Layer</th>
                              <th className="text-right py-0.5 pr-3 font-sans">MSE</th>
                              <th className="text-right py-0.5 pr-3 font-sans">Rel err</th>
                              <th className="text-right py-0.5 pr-3 font-sans">Mean cos</th>
                              <th className="text-right py-0.5 pr-3 font-sans">Min cos</th>
                            </tr>
                          </thead>
                          <tbody>
                            {p.layers.map(l => (
                              <tr key={l.layer} className="border-b border-parchment/40">
                                <td className="py-0.5 pr-3 text-right">{l.layer}</td>
                                <td className="py-0.5 pr-3 text-right">{l.mse.toExponential(3)}</td>
                                <td className="py-0.5 pr-3 text-right">{l.relError.toFixed(4)}</td>
                                <td className="py-0.5 pr-3 text-right">{l.meanCosine.toFixed(4)}</td>
                                <td className="py-0.5 pr-3 text-right">{l.minCosine.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function PrecisionCard({ entry }: { entry: PrecisionSweepEntry }) {
  const colour = PRECISION_COLOR[entry.precision] ?? "#444";
  const o = entry.output;
  return (
    <div className="border border-parchment rounded-sm p-3 bg-cream/30">
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-[11px] font-semibold" style={{ color: colour }}>
          {entry.label}
        </span>
        <span
          className={
            "font-mono text-[10px] px-1.5 py-0.5 rounded-sm " +
            (o.argmaxMatch
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-700")
          }
        >
          {o.argmaxMatch ? "argmax OK" : "argmax changed"}
        </span>
      </div>
      <div className="font-sans text-[10px] text-slate space-y-0.5">
        <div className="flex justify-between">
          <span>KL(ref‖q)</span>
          <span className="font-mono text-ink">{o.klDivergence.toFixed(3)}</span>
        </div>
        <div className="flex justify-between">
          <span>top-K overlap</span>
          <span className="font-mono text-ink">{o.topKOverlap.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span>final cos</span>
          <span className="font-mono text-ink">
            {entry.layers[entry.layers.length - 1].meanCosine.toFixed(4)}
          </span>
        </div>
        <div className="flex justify-between">
          <span>ΔH (bits)</span>
          <span className="font-mono text-ink">
            {(o.entropyQuant - o.entropyRef).toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

function PredCell({
  p,
  highlight,
}: {
  p: { token: string; prob: number; tokenId: number } | undefined;
  highlight?: boolean;
}) {
  if (!p) return <span className="text-slate/50">–</span>;
  return (
    <span className={highlight ? "text-red-700 font-semibold" : ""}>
      {JSON.stringify(p.token)}{" "}
      <span className="text-slate/60">{p.prob.toFixed(3)}</span>
    </span>
  );
}
