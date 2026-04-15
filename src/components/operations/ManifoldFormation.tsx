"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import { ChevronDown, ChevronUp, Play, Pause, SkipBack, SkipForward } from "lucide-react";
import Plot3DWrapper from "@/components/Plot3DWrapper";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { MANIFOLD_FORMATION_PRESETS } from "@/lib/presets/defaults";

const BACKEND_URL = "http://localhost:8000";

interface LayerData {
  layer: number;
  coords3d: [number, number, number][];
  norms: number[];
  meanNorm: number;
  meanCosSim: number;
}

interface ManifoldResult {
  inputText: string;
  tokens: string[];
  tokenIds: number[];
  numLayers: number;
  seqLen: number;
  layers: LayerData[];
  pcaVarianceExplained: number[];
}

export default function ManifoldFormation() {
  const [text, setText] = useState("The cat sat on the mat");
  const [result, setResult] = useState<ManifoldResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const run = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/manifold-formation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to compute manifold formation");
      }
      const data: ManifoldResult = await res.json();
      setResult(data);
      setCurrentLayer(0);
      setPlaying(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [text]);

  // Animation playback
  useEffect(() => {
    if (playing && result) {
      intervalRef.current = setInterval(() => {
        setCurrentLayer((prev) => {
          if (prev >= result.numLayers - 1) {
            setPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 600);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, result]);

  const layerData = result?.layers[currentLayer];

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Manifold Formation"
        summary="Runs a forward pass and projects every token at every layer into a shared 3D space via PCA. An animation control lets you scrub through the depths of the model and watch the cloud of tokens deform layer by layer, showing how local context pulls words into clusters and how the manifold takes shape."
        details={
          <>
            <p>
              Manifold Formation answers a different question from Token Trajectory. Instead of following a single token through the layers, it shows the entire sequence at once, layer by layer, and lets you scrub through the depths of the model like frames of an animation.
            </p>
            <p>
              PCA is fitted once across all layers so the axes stay consistent as you scroll through. This means the animation shows real geometric motion, not the artefact of a moving coordinate frame. Tokens that start scattered often drift toward each other as context accumulates; tokens that started close may split apart when the model decides they mean different things in this sentence.
            </p>
            <p>
              The per-layer mean cosine similarity is reported alongside the view: a quantitative summary of how collapsed or spread out the token cloud is at that depth. Watching this number rise is watching the manifold tighten.
            </p>
            <p>
              This is the operation to use when you want to see the manifold as a shape rather than as a trajectory. It is the closest Vectorscope comes to visualising what people mean when they call a language model&apos;s internal space a &quot;geometry of meaning.&quot;
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
              placeholder="Enter text..."
            />
          </label>
          <button onClick={run} disabled={loading} className="btn-editorial-primary">
            {loading ? "Computing..." : "Compute"}
          </button>
          {error && <span className="text-red-600 font-sans text-[11px]">{error}</span>}
        </div>
        <PresetChipRow
          disabled={loading}
          items={MANIFOLD_FORMATION_PRESETS.map((p) => ({
            label: p.label,
            title: p.title,
            onClick: () => setText(p.text),
          }))}
        />
      </div>

      {result && (
        <>
          {/* Playback controls */}
          <div className="card-editorial p-3">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setCurrentLayer(0)}
                className="p-1 text-slate hover:text-ink"
                title="First layer"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setPlaying(!playing)}
                className="p-1 text-slate hover:text-ink"
                title={playing ? "Pause" : "Play"}
              >
                {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              </button>
              <button
                onClick={() => setCurrentLayer(result.numLayers - 1)}
                className="p-1 text-slate hover:text-ink"
                title="Last layer"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
              <input
                type="range"
                min={0}
                max={result.numLayers - 1}
                value={currentLayer}
                onChange={(e) => {
                  setPlaying(false);
                  setCurrentLayer(Number(e.target.value));
                }}
                className="flex-1"
              />
              <span className="font-mono text-[11px] text-slate min-w-[80px]">
                Layer {currentLayer} / {result.numLayers - 1}
              </span>
            </div>
          </div>

          {/* 3D scatter for current layer */}
          {layerData && (
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Token Geometry at Layer {currentLayer}
                <span className="font-normal text-[10px] ml-2">
                  mean norm: {layerData.meanNorm.toFixed(1)} | mean cos sim: {layerData.meanCosSim.toFixed(3)}
                </span>
              </h3>
              <Plot3DWrapper
                data={[
                  {
                    type: "scatter3d",
                    mode: "markers+text",
                    x: layerData.coords3d.map((c) => c[0]),
                    y: layerData.coords3d.map((c) => c[1]),
                    z: layerData.coords3d.map((c) => c[2]),
                    text: result.tokens,
                    textposition: "top center" as const,
                    textfont: { size: 9, color: "#4A4A4A" },
                    hovertemplate:
                      "<b>%{text}</b><br>norm: %{marker.color:.2f}<extra></extra>",
                    marker: {
                      size: 6,
                      color: layerData.norms,
                      colorscale: [
                        [0, "#4A6FA5"],
                        [0.5, "#A67F6F"],
                        [1, "#5B2333"],
                      ],
                      colorbar: { title: "Norm", thickness: 12 },
                      opacity: 0.9,
                    },
                  },
                ]}
                layout={{
                  height: 500,
                  margin: { l: 0, r: 0, t: 0, b: 0 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  scene: {
                    xaxis: {
                      title: "PC1",
                      gridcolor: "#E8E0D4",
                      zerolinecolor: "#D4C9B8",
                    },
                    yaxis: {
                      title: "PC2",
                      gridcolor: "#E8E0D4",
                      zerolinecolor: "#D4C9B8",
                    },
                    zaxis: {
                      title: "PC3",
                      gridcolor: "#E8E0D4",
                      zerolinecolor: "#D4C9B8",
                    },
                    bgcolor: "transparent",
                  },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
                }}
                config={{ displayModeBar: true, displaylogo: false }}
                style={{ width: "100%" }}
              />
            </div>
          )}

          {/* Norm and similarity evolution */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Mean Norm across Layers
              </h3>
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: result.layers.map((l) => l.layer),
                    y: result.layers.map((l) => l.meanNorm),
                    marker: {
                      color: result.layers.map((l) =>
                        l.layer === currentLayer ? "#5B2333" : "#A67F6F"
                      ),
                      size: result.layers.map((l) =>
                        l.layer === currentLayer ? 8 : 4
                      ),
                    },
                    line: { color: "#A67F6F", width: 2 },
                  },
                ]}
                layout={{
                  height: 180,
                  margin: { l: 50, r: 15, t: 5, b: 30 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Layer", gridcolor: "#E8E0D4" },
                  yaxis: { title: "Mean Norm", gridcolor: "#E8E0D4" },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
            <div className="card-editorial p-3">
              <h3 className="font-sans text-xs font-semibold text-slate mb-2">
                Mean Token Similarity across Layers
              </h3>
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: result.layers.map((l) => l.layer),
                    y: result.layers.map((l) => l.meanCosSim),
                    marker: {
                      color: result.layers.map((l) =>
                        l.layer === currentLayer ? "#5B2333" : "#A67F6F"
                      ),
                      size: result.layers.map((l) =>
                        l.layer === currentLayer ? 8 : 4
                      ),
                    },
                    line: { color: "#A67F6F", width: 2 },
                  },
                ]}
                layout={{
                  height: 180,
                  margin: { l: 50, r: 15, t: 5, b: 30 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  xaxis: { title: "Layer", gridcolor: "#E8E0D4" },
                  yaxis: { title: "Mean Cos Sim", gridcolor: "#E8E0D4" },
                  font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 9 },
                }}
                config={{ displayModeBar: false }}
                style={{ width: "100%" }}
              />
            </div>
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
                <div className="grid grid-cols-2 gap-6 mt-3">
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      PCA Info
                    </h4>
                    <table className="w-full font-mono text-[11px]">
                      <tbody>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">PC1 variance</td>
                          <td className="py-1 text-right">
                            {(result.pcaVarianceExplained[0] * 100).toFixed(1)}%
                          </td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">PC2 variance</td>
                          <td className="py-1 text-right">
                            {(result.pcaVarianceExplained[1] * 100).toFixed(1)}%
                          </td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">PC3 variance</td>
                          <td className="py-1 text-right">
                            {(result.pcaVarianceExplained[2] * 100).toFixed(1)}%
                          </td>
                        </tr>
                        <tr className="border-b border-parchment">
                          <td className="py-1 text-slate">Total explained</td>
                          <td className="py-1 text-right">
                            {(result.pcaVarianceExplained.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Per-Layer Table
                    </h4>
                    <div className="overflow-y-auto max-h-48">
                      <table className="w-full font-mono text-[10px]">
                        <thead>
                          <tr className="border-b border-parchment">
                            <th className="p-1 text-left text-slate font-normal">Layer</th>
                            <th className="p-1 text-right text-slate font-normal">Mean Norm</th>
                            <th className="p-1 text-right text-slate font-normal">Mean Cos</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.layers.map((l) => (
                            <tr
                              key={l.layer}
                              className={`border-b border-parchment cursor-pointer hover:bg-cream/30 ${
                                l.layer === currentLayer ? "bg-cream/50" : ""
                              }`}
                              onClick={() => {
                                setCurrentLayer(l.layer);
                                setPlaying(false);
                              }}
                            >
                              <td className="p-1">{l.layer}</td>
                              <td className="p-1 text-right">{l.meanNorm.toFixed(2)}</td>
                              <td className="p-1 text-right">{l.meanCosSim.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
