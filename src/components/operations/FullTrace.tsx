"use client";

import { useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useModel } from "@/context/ModelContext";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { FULL_TRACE_PRESETS } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

/** Shape of each NDJSON event from the backend. */
interface TokenizeEvent {
  stage: "tokenize";
  tokens: string[];
  tokenIds: number[];
  numTokens: number;
}
interface EmbeddingEvent {
  stage: "embedding";
  norms: number[];
  meanNorm: number;
  shape: [number, number];
}
interface LayerEvent {
  stage: "layer";
  layer: number;
  norms: number[];
  meanNorm: number;
  cosineToPrev: number | null;
}
interface OutputEvent {
  stage: "output";
  topPredictions: Array<{
    token: string;
    tokenId: number;
    probability: number;
    logit: number;
  }>;
  entropyBits: number;
  topProbability: number;
}
interface CompleteEvent {
  stage: "complete";
  numLayers: number;
  numTokens: number;
  normRange: [number, number];
  inputText: string;
}
interface ErrorEvent {
  stage: "error";
  message: string;
}

type TraceEvent =
  | TokenizeEvent
  | EmbeddingEvent
  | LayerEvent
  | OutputEvent
  | CompleteEvent
  | ErrorEvent;

interface TraceState {
  tokenize: TokenizeEvent | null;
  embedding: EmbeddingEvent | null;
  layers: LayerEvent[];
  output: OutputEvent | null;
  complete: CompleteEvent | null;
}

export default function FullTrace() {
  const { backendStatus } = useModel();
  const [text, setText] = useState("The cat sat on the mat");
  const [trace, setTrace] = useState<TraceState | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);
  const [stagesReceived, setStagesReceived] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const numLayers = backendStatus.model?.numLayers ?? 12;

  const run = useCallback(async () => {
    if (!text.trim() || running) return;
    setRunning(true);
    setError(null);
    setStagesReceived(0);
    setTrace({ tokenize: null, embedding: null, layers: [], output: null, complete: null });

    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${BACKEND_URL}/full-trace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Full trace failed");
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";
      let count = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const event: TraceEvent = JSON.parse(line);
          count++;
          setStagesReceived(count);

          if (event.stage === "error") {
            setError(event.message);
            break;
          }

          setTrace((prev) => {
            if (!prev) return prev;
            switch (event.stage) {
              case "tokenize":
                return { ...prev, tokenize: event };
              case "embedding":
                return { ...prev, embedding: event };
              case "layer":
                return { ...prev, layers: [...prev.layers, event] };
              case "output":
                return { ...prev, output: event };
              case "complete":
                return { ...prev, complete: event };
              default:
                return prev;
            }
          });
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setRunning(false);
    }
  }, [text, running]);

  const totalStages = 2 + numLayers + 1 + 1 + 1; // tokenize + embed + layers + output + complete
  const progressPct = totalStages > 0 ? Math.min((stagesReceived / totalStages) * 100, 100) : 0;

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Full Trace"
        summary="Streams a complete forward pass stage by stage over NDJSON: tokenisation, input embedding, every transformer layer, and the final output distribution. Reports per-layer norm statistics, cosine similarity between successive layers, and the top predictions with their entropy. A live geometric chronicle of one prompt moving through the entire model."
        details={
          <>
            <p>
              Full Trace is the most complete single view Vectorscope offers of a forward pass. Instead of stopping at one layer or following one token, it streams every stage of the computation back to the browser as it happens, using NDJSON so you can watch the model think in roughly real time.
            </p>
            <p>
              The stages are fixed. First tokenisation, showing exactly how the text was broken into subword pieces. Then the input embedding with its norm distribution. Then every transformer layer in sequence, each reporting its mean norm and the cosine similarity between its output and the previous layer&apos;s output — a measure of how much the representation has changed. Finally the output head, with the top predicted tokens, their probabilities, and the entropy of the full distribution.
            </p>
            <p>
              Low cosine between adjacent layers means a layer has done substantial rewriting. High cosine means the layer mostly passed the representation through unchanged. Reading this sequence is the closest thing Vectorscope offers to watching the model work through a prompt.
            </p>
            <p>
              Low output entropy means the model is confident; high entropy means it sees several viable continuations. Use this operation when you want the whole story of a single forward pass rather than a slice of it.
            </p>
          </>
        }
      />
      {/* Controls */}
      <div className="card-editorial p-4">
        <div className="flex items-center gap-4">
          <label className="font-sans text-[11px] text-slate flex-1">
            Input text
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
              className="input-editorial ml-2 w-full max-w-md"
              placeholder="Enter text to trace through the model..."
            />
          </label>
          <button onClick={run} disabled={running} className="btn-editorial-primary">
            {running ? "Tracing..." : "Full Trace"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        {running && (
          <div className="mt-2">
            <div className="w-full bg-parchment rounded-full h-1.5">
              <div
                className="bg-burgundy h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <p className="font-sans text-[10px] text-slate mt-1">
              Stage {stagesReceived} / {totalStages}
            </p>
          </div>
        )}
        <div className="mt-2">
          <PresetChipRow
            disabled={running}
            items={FULL_TRACE_PRESETS.map((p) => ({
              label: p.label,
              title: p.title,
              onClick: () => setText(p.text),
            }))}
          />
        </div>
      </div>

      {trace?.tokenize && (
        <>
          {/* Pipeline stages visualisation */}
          <div className="card-editorial p-3">
            <h3 className="font-sans text-xs font-semibold text-slate mb-2">
              Pipeline: tokens → embeddings → layers → predictions
            </h3>

            {/* Token chips */}
            <div className="flex items-center gap-1 flex-wrap mb-3">
              <span className="font-sans text-[10px] text-slate uppercase tracking-wider mr-1">Tokens</span>
              {trace.tokenize.tokens.map((t, i) => (
                <span key={i} className="bg-cream px-1.5 py-0.5 rounded-sm font-mono text-[10px]">
                  {t}
                  <span className="text-slate/50 ml-0.5">({trace.tokenize!.tokenIds[i]})</span>
                </span>
              ))}
            </div>

            {/* Embedding norms */}
            {trace.embedding && (
              <div className="flex items-center gap-1 flex-wrap mb-3">
                <span className="font-sans text-[10px] text-slate uppercase tracking-wider mr-1">Embed norms</span>
                {trace.embedding.norms.map((n, i) => (
                  <span key={i} className="font-mono text-[10px] px-1.5 py-0.5 rounded-sm"
                    style={{
                      backgroundColor: `rgba(91, 35, 51, ${Math.min(n / 5, 1) * 0.3})`,
                    }}>
                    {n.toFixed(2)}
                  </span>
                ))}
                <span className="font-sans text-[10px] text-slate ml-2">
                  mean: {trace.embedding.meanNorm.toFixed(3)}
                </span>
              </div>
            )}
          </div>

          {/* Norm evolution across layers */}
          {trace.layers.length > 0 && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Norm Evolution across Layers
              </h3>
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: trace.layers.map((l) => l.layer),
                    y: trace.layers.map((l) => l.meanNorm),
                    marker: { color: "#5B2333", size: 4 },
                    line: { color: "#A67F6F", width: 2 },
                    name: "Mean norm",
                  },
                ]}
                layout={{
                  height: 200,
                  margin: { l: 50, r: 15, t: 5, b: 30 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Layer", gridcolor: "#E8E0D4" },
                  yaxis: { title: "Mean Norm", gridcolor: "#E8E0D4" },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Layer-to-layer cosine similarity */}
          {trace.layers.filter((l) => l.cosineToPrev !== null).length > 0 && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Layer-to-Layer Cosine Similarity
              </h3>
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: trace.layers
                      .filter((l) => l.cosineToPrev !== null)
                      .map((l) => `${l.layer - 1} → ${l.layer}`),
                    y: trace.layers
                      .filter((l) => l.cosineToPrev !== null)
                      .map((l) => l.cosineToPrev!),
                    marker: { color: "#5B2333", size: 4 },
                    line: { color: "#A67F6F", width: 2 },
                  },
                ]}
                layout={{
                  height: 200,
                  margin: { l: 50, r: 15, t: 5, b: 50 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Layer Transition", gridcolor: "#E8E0D4" },
                  yaxis: { title: "Cosine Sim", gridcolor: "#E8E0D4", range: [0, 1.05] },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Output predictions */}
          {trace.output && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate">
                Top Predictions (next token)
              </h3>
              <p className="font-mono text-[10px] text-slate/80 mb-2 italic truncate">
                after: &ldquo;{trace.complete?.inputText ?? trace.tokenize?.tokens.join("") ?? ""}&rdquo;
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        x: trace.output.topPredictions.slice(0, 10).map((p) => p.probability),
                        y: trace.output.topPredictions.slice(0, 10).map((p) => `"${p.token.trim() || "\\n"}"`),
                        orientation: "h" as const,
                        marker: { color: "#5B2333", opacity: 0.8 },
                      },
                    ]}
                    layout={{
                      height: 250,
                      margin: { l: 80, r: 15, t: 5, b: 30 },
                      paper_bgcolor: "transparent",
                      plot_bgcolor: "transparent",
                      xaxis: { title: "Probability", gridcolor: "#E8E0D4" },
                      yaxis: { autorange: "reversed" as const, gridcolor: "#E8E0D4" },
                      font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
                    }}
                    config={{ displayModeBar: false }}
                    style={{ width: "100%" }}
                  />
                </div>
                <div className="font-sans text-[11px] text-slate space-y-1 pt-1">
                  <p>
                    <span className="font-medium">Entropy:</span>{" "}
                    <span className="font-mono">{trace.output.entropyBits.toFixed(2)} bits</span>
                  </p>
                  <p>
                    <span className="font-medium">Top probability:</span>{" "}
                    <span className="font-mono">{(trace.output.topProbability * 100).toFixed(1)}%</span>
                  </p>
                  <p>
                    <span className="font-medium">Top token:</span>{" "}
                    <span className="font-mono">"{trace.output.topPredictions[0]?.token.trim()}"</span>
                  </p>
                  {trace.complete && (
                    <>
                      <p className="mt-2">
                        <span className="font-medium">Layers traversed:</span>{" "}
                        <span className="font-mono">{trace.complete.numLayers}</span>
                      </p>
                      <p>
                        <span className="font-medium">Norm range:</span>{" "}
                        <span className="font-mono">
                          {trace.complete.normRange[0].toFixed(1)} — {trace.complete.normRange[1].toFixed(1)}
                        </span>
                      </p>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Per-token norm heatmap across layers */}
          {trace.layers.length > 0 && trace.tokenize && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Per-Token Norms across Layers
              </h3>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    z: trace.layers.map((l) => l.norms),
                    x: trace.tokenize.tokens,
                    y: trace.layers.map((l) => `Layer ${l.layer}`),
                    colorscale: [
                      [0, "#F5F0E8"],
                      [0.3, "#A67F6F"],
                      [0.7, "#8B2252"],
                      [1, "#5B2333"],
                    ],
                    colorbar: { title: "Norm", thickness: 12 },
                    hovertemplate: "Token: %{x}<br>%{y}<br>Norm: %{z:.2f}<extra></extra>",
                  },
                ]}
                layout={{
                  height: Math.max(200, trace.layers.length * 18 + 60),
                  margin: { l: 70, r: 15, t: 5, b: 40 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

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
                {/* Per-layer table */}
                <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-3">
                  Layer-by-Layer Statistics
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-[10px]">
                    <thead>
                      <tr className="border-b border-parchment">
                        <th className="p-1 text-left text-slate font-normal">Layer</th>
                        <th className="p-1 text-right text-slate font-normal">Mean Norm</th>
                        <th className="p-1 text-right text-slate font-normal">Cos to Prev</th>
                        {trace.tokenize.tokens.map((t, i) => (
                          <th key={i} className="p-1 text-right text-slate font-normal">
                            {t.trim() || "\\n"}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {trace.embedding && (
                        <tr className="border-b border-parchment bg-cream/30">
                          <td className="p-1 text-slate">Embed</td>
                          <td className="p-1 text-right">{trace.embedding.meanNorm.toFixed(2)}</td>
                          <td className="p-1 text-right text-slate">—</td>
                          {trace.embedding.norms.map((n, i) => (
                            <td key={i} className="p-1 text-right">
                              {n.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      )}
                      {trace.layers.map((l) => (
                        <tr key={l.layer} className="border-b border-parchment">
                          <td className="p-1 text-slate">{l.layer}</td>
                          <td className="p-1 text-right">{l.meanNorm.toFixed(2)}</td>
                          <td className="p-1 text-right">
                            {l.cosineToPrev !== null ? l.cosineToPrev.toFixed(4) : "—"}
                          </td>
                          {l.norms.map((n, i) => (
                            <td key={i} className="p-1 text-right">
                              {n.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Output predictions table */}
                {trace.output && (
                  <>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2 mt-4">
                      Full Prediction Table (Top {trace.output.topPredictions.length})
                    </h4>
                    <div className="overflow-x-auto max-h-48 overflow-y-auto">
                      <table className="w-full font-mono text-[10px]">
                        <thead>
                          <tr className="border-b border-parchment">
                            <th className="p-1 text-left text-slate font-normal">Rank</th>
                            <th className="p-1 text-left text-slate font-normal">Token</th>
                            <th className="p-1 text-right text-slate font-normal">ID</th>
                            <th className="p-1 text-right text-slate font-normal">Probability</th>
                            <th className="p-1 text-right text-slate font-normal">Logit</th>
                          </tr>
                        </thead>
                        <tbody>
                          {trace.output.topPredictions.map((p, i) => (
                            <tr key={i} className="border-b border-parchment">
                              <td className="p-1 text-slate">{i + 1}</td>
                              <td className="p-1 font-semibold">"{p.token.trim() || "\\n"}"</td>
                              <td className="p-1 text-right text-slate">{p.tokenId}</td>
                              <td className="p-1 text-right">{(p.probability * 100).toFixed(2)}%</td>
                              <td className="p-1 text-right">{p.logit.toFixed(2)}</td>
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
