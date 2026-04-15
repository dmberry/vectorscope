"use client";

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import dynamic from "next/dynamic";
import { Play, Pause, SkipBack, SkipForward, ArrowLeftRight, ChevronDown, ChevronUp } from "lucide-react";
import OperationIntro from "@/components/OperationIntro";
import PresetChipRow from "@/components/PresetChipRow";
import { GENERATION_VECTOR_PRESETS } from "@/lib/presets/defaults";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const BACKEND_URL = "http://localhost:8000";

/* ------------------------------------------------------------------ */
/*  NDJSON event shapes                                                */
/* ------------------------------------------------------------------ */

interface StartEvent {
  stage: "start";
  maxNewTokens: number;
  maxNewTokensCap: number;
  temperature: number;
  topP: number;
  topK: number;
  prompt: string;
}

interface PromptEvent {
  stage: "prompt";
  tokens: string[];
  tokenIds: number[];
  promptLen: number;
  inputEmbedNorms: number[];
}

interface StepEvent {
  stage: "step";
  stepIndex: number;
  absolutePos: number;
  token: string;
  tokenId: number;
  hiddenNorms: number[];
  layerDeltas: (number | null)[];
  attentionByLayer: number[][]; // [numLayers][contextLen]
  topPredictions: { token: string; tokenId: number; probability: number }[];
  entropyBits: number;
  contextLen: number;
}

interface GeometryEvent {
  stage: "geometry";
  numTokens: number;
  numLayers: number;
  hiddenDim: number;
  promptLen: number;
  coords: [number, number, number][][]; // [token][layer]
}

interface TokenisationSeries {
  perTokenNorms: number[];
  cumulativeSumNorms: number[];
  cumulativeMeanNorms: number[];
  independenceBaseline: number[];
  previewDims: number;
  vectorPreview: number[][];
}

interface CompleteEvent {
  stage: "complete";
  numLayers: number;
  promptLen: number;
  numGenerated: number;
  generatedTokens: string[];
  generatedText: string;
  fullText: string;
  entropySeries: number[];
  meanNormSeries: number[];
  tokenisationSeries?: TokenisationSeries;
}

interface ErrorEvent {
  stage: "error";
  message: string;
}

type Event =
  | StartEvent
  | PromptEvent
  | StepEvent
  | GeometryEvent
  | CompleteEvent
  | ErrorEvent;

interface GenerationState {
  start: StartEvent | null;
  prompt: PromptEvent | null;
  steps: StepEvent[];
  geometry: GeometryEvent | null;
  complete: CompleteEvent | null;
}

/* ------------------------------------------------------------------ */
/*  Panel definitions                                                  */
/* ------------------------------------------------------------------ */

type PanelId =
  | "tokenisation"
  | "embedding"
  | "attention"
  | "layers"
  | "output"
  | "text";

const PANELS: { id: PanelId; label: string }[] = [
  { id: "tokenisation", label: "Tokenisation" },
  { id: "embedding", label: "Input Embedding" },
  { id: "attention", label: "Attention" },
  { id: "layers", label: "Layer Progression" },
  { id: "output", label: "Output Distribution" },
  { id: "text", label: "Decoded Text" },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function GenerationVector() {
  const [prompt, setPrompt] = useState("The nature of consciousness is");
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [maxCap, setMaxCap] = useState(150);
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);

  const [state, setState] = useState<GenerationState | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [focusIdx, setFocusIdx] = useState<number | null>(null); // absolute position in final sequence
  const [panelIdx, setPanelIdx] = useState(0);
  const [reverseOrder, setReverseOrder] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [scrubberIdx, setScrubberIdx] = useState(0); // step index for playback
  const [deepDiveOpen, setDeepDiveOpen] = useState(false);
  const playTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  /* ----- Fetch hard cap from backend ----- */
  useEffect(() => {
    fetch(`${BACKEND_URL}/generation-vector/config`)
      .then((r) => r.json())
      .then((d) => {
        if (typeof d.maxNewTokensCap === "number") {
          setMaxCap(d.maxNewTokensCap);
          setMaxNewTokens((prev) => Math.min(prev, d.maxNewTokensCap));
        }
      })
      .catch(() => {});
  }, []);

  /* ----- Run generation ----- */
  const run = useCallback(async () => {
    if (!prompt.trim()) return;
    setRunning(true);
    setError(null);
    setState({ start: null, prompt: null, steps: [], geometry: null, complete: null });
    setFocusIdx(null);
    setScrubberIdx(0);
    setPlaying(false);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch(`${BACKEND_URL}/generation-vector`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          max_new_tokens: maxNewTokens,
          temperature,
          top_p: topP,
          top_k: topK,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const raw of lines) {
          const line = raw.trim();
          if (!line) continue;
          try {
            const event = JSON.parse(line) as Event;
            setState((prev) => {
              if (!prev) return prev;
              switch (event.stage) {
                case "start":
                  return { ...prev, start: event };
                case "prompt":
                  return { ...prev, prompt: event };
                case "step":
                  return { ...prev, steps: [...prev.steps, event] };
                case "geometry":
                  return { ...prev, geometry: event };
                case "complete":
                  return { ...prev, complete: event };
                case "error":
                  setError(event.message);
                  return prev;
                default:
                  return prev;
              }
            });
          } catch {
            // skip malformed line
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    } finally {
      setRunning(false);
    }
  }, [prompt, maxNewTokens, temperature, topP, topK]);

  /* ----- Playback scrubber ----- */
  useEffect(() => {
    if (!playing) {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
      playTimerRef.current = null;
      return;
    }
    const numSteps = state?.steps.length ?? 0;
    if (numSteps === 0) return;
    playTimerRef.current = setInterval(() => {
      setScrubberIdx((i) => {
        const next = i + 1;
        if (next >= numSteps) {
          setPlaying(false);
          return i;
        }
        return next;
      });
    }, 400);
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [playing, state?.steps.length]);

  /* ----- Focus follows scrubber ----- */
  useEffect(() => {
    if (!state?.prompt || !state?.steps.length) return;
    const absPos = state.prompt.promptLen + scrubberIdx;
    setFocusIdx(absPos);
  }, [scrubberIdx, state?.prompt, state?.steps.length]);

  /* ----- Derived: full token list ----- */
  const allTokens = useMemo(() => {
    if (!state?.prompt) return [];
    const promptTokens = state.prompt.tokens.map((t, i) => ({
      str: t,
      absPos: i,
      kind: "prompt" as const,
      stepIdx: null as number | null,
    }));
    const genTokens = state.steps.map((s, i) => ({
      str: s.token,
      absPos: state.prompt!.promptLen + i,
      kind: "generated" as const,
      stepIdx: i,
    }));
    return [...promptTokens, ...genTokens];
  }, [state?.prompt, state?.steps]);

  const focusedToken = useMemo(() => {
    if (focusIdx === null) return null;
    return allTokens.find((t) => t.absPos === focusIdx) ?? null;
  }, [focusIdx, allTokens]);

  const focusedStep = useMemo(() => {
    if (!focusedToken || focusedToken.kind !== "generated") return null;
    return state?.steps[focusedToken.stepIdx!] ?? null;
  }, [focusedToken, state?.steps]);

  const displayPanels = useMemo(
    () => (reverseOrder ? [...PANELS].reverse() : PANELS),
    [reverseOrder]
  );

  /* ----- Deep dive statistics ----- */
  const deepStats = useMemo(() => {
    if (!state?.complete || state.steps.length === 0) return null;
    const entropy = state.complete.entropySeries;
    const norms = state.complete.meanNormSeries;
    const numLayers = state.complete.numLayers;

    const mean = (arr: number[]) =>
      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    const std = (arr: number[]) => {
      if (!arr.length) return 0;
      const m = mean(arr);
      return Math.sqrt(mean(arr.map((v) => (v - m) ** 2)));
    };

    let eMinV = Infinity,
      eMinI = -1,
      eMaxV = -Infinity,
      eMaxI = -1;
    entropy.forEach((v, i) => {
      if (v < eMinV) {
        eMinV = v;
        eMinI = i;
      }
      if (v > eMaxV) {
        eMaxV = v;
        eMaxI = i;
      }
    });

    // Per-layer mean cosine delta across all generated steps.
    const perLayerDeltaSum = Array(numLayers).fill(0) as number[];
    const perLayerDeltaCount = Array(numLayers).fill(0) as number[];
    state.steps.forEach((s) => {
      s.layerDeltas.forEach((d, l) => {
        if (d !== null) {
          perLayerDeltaSum[l] += d;
          perLayerDeltaCount[l] += 1;
        }
      });
    });
    const perLayerMeanDelta = perLayerDeltaSum.map((sum, i) =>
      perLayerDeltaCount[i] > 0 ? sum / perLayerDeltaCount[i] : null
    );

    // Per-layer mean norm across generated steps.
    const perLayerNormSum = Array(numLayers).fill(0) as number[];
    state.steps.forEach((s) => {
      s.hiddenNorms.forEach((n, l) => {
        perLayerNormSum[l] += n;
      });
    });
    const perLayerMeanNorm = perLayerNormSum.map((sum) => sum / state.steps.length);

    return {
      entropyMean: mean(entropy),
      entropyStd: std(entropy),
      entropyMin: { v: eMinV, i: eMinI, token: state.steps[eMinI]?.token ?? "" },
      entropyMax: { v: eMaxV, i: eMaxI, token: state.steps[eMaxI]?.token ?? "" },
      normMean: mean(norms),
      normMin: Math.min(...norms),
      normMax: Math.max(...norms),
      perLayerMeanDelta,
      perLayerMeanNorm,
      numLayers,
    };
  }, [state]);

  const activePanel = displayPanels[panelIdx]?.id ?? "tokenisation";

  /* ----- Keyboard: arrow keys for panels (or tokens on Tokenisation) ----- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const isArrow =
        e.key === "ArrowRight" ||
        e.key === "ArrowLeft" ||
        e.key === "ArrowUp" ||
        e.key === "ArrowDown";
      if (!isArrow) return;

      // Total token count (prompt + generated) for token stepping
      const totalTokens = (state?.prompt?.promptLen ?? 0) + (state?.steps.length ?? 0);

      // On the Tokenisation panel, arrow keys navigate the chip grid.
      if (activePanel === "tokenisation" && totalTokens > 0) {
        e.preventDefault();

        // Left/Right: step by 1
        if (e.key === "ArrowRight") {
          setFocusIdx((cur) => (cur === null ? 0 : Math.min(cur + 1, totalTokens - 1)));
          return;
        }
        if (e.key === "ArrowLeft") {
          setFocusIdx((cur) => (cur === null ? totalTokens - 1 : Math.max(cur - 1, 0)));
          return;
        }

        // Up/Down: jump between rows using DOM layout measurement.
        const chips = Array.from(
          document.querySelectorAll<HTMLElement>("[data-gv-chip]")
        );
        if (!chips.length) return;

        setFocusIdx((cur) => {
          if (cur === null) return e.key === "ArrowDown" ? 0 : totalTokens - 1;
          const curEl = chips.find(
            (el) => parseInt(el.dataset.gvChip ?? "-1", 10) === cur
          );
          if (!curEl) return cur;
          const curTop = curEl.offsetTop;
          const curCenter = curEl.offsetLeft + curEl.offsetWidth / 2;

          // Find the next row's offsetTop in the chosen direction.
          let targetTop: number | null = null;
          chips.forEach((el) => {
            const t = el.offsetTop;
            if (e.key === "ArrowDown" && t > curTop) {
              if (targetTop === null || t < targetTop) targetTop = t;
            } else if (e.key === "ArrowUp" && t < curTop) {
              if (targetTop === null || t > targetTop) targetTop = t;
            }
          });
          if (targetTop === null) return cur; // already at first/last row

          // Among chips in the target row, pick the closest by horizontal centre.
          let best = cur;
          let bestDist = Infinity;
          chips.forEach((el) => {
            if (el.offsetTop !== targetTop) return;
            const c = el.offsetLeft + el.offsetWidth / 2;
            const d = Math.abs(c - curCenter);
            if (d < bestDist) {
              bestDist = d;
              best = parseInt(el.dataset.gvChip ?? String(cur), 10);
            }
          });
          return best;
        });
        return;
      }

      // Otherwise, Left/Right step panels (Up/Down are no-ops off the tokenisation panel).
      if (e.key === "ArrowRight") {
        setPanelIdx((i) => Math.min(i + 1, displayPanels.length - 1));
      } else if (e.key === "ArrowLeft") {
        setPanelIdx((i) => Math.max(i - 1, 0));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [displayPanels.length, activePanel, state?.prompt, state?.steps]);

  /* ================================================================ */
  /*  Render                                                           */
  /* ================================================================ */

  return (
    <div className="max-w-7xl mx-auto space-y-4">
      <OperationIntro
        name="Generation Vector"
        summary="Runs a full autoregressive generation from your prompt and instruments every step. A sticky header shows the prompt and the generated text as clickable token chips; horizontal panels below walk through tokenisation, embedding, attention, layer progression, the output distribution, and the final text. Click any word to focus all panels on it. Use the scrubber to replay generation token by token."
        details={
          <>
            <p>
              Generation Vector is Full Trace for an entire generation. Instead of running a single forward pass and showing you what happened inside it, it runs the autoregressive loop and captures every step: every time the model decides what word to write next, the operation records the hidden-state norms, layer-to-layer cosines, attention pattern, and top-k predictions that produced that decision.
            </p>
            <p>
              The name is also a joke. &quot;Generation Vector&quot; reads as a demographic category, the cohort that grew up inside vector media and whose language has been geometrically compressed. The instrument gestures at the subject it studies.
            </p>
            <p>
              <strong>Controls.</strong> Temperature, top-p, and top-k govern sampling. Temperature = 0 is greedy (deterministic). Temperature = 0.8 with top-p 0.9 gives naturalistic generation. The length slider caps at {maxCap} tokens, which is a constant on the backend — easy to raise later if we want longer runs.
            </p>
            <p>
              <strong>Panels.</strong> By default they run in pipeline order: Tokenisation → Input Embedding → Attention → Layer Progression → Output Distribution → Decoded Text. The reverse toggle flips the order so you can start at the finished text and peel back through the layers that produced it.
            </p>
            <p>
              <strong>Focus.</strong> Click any word in the sticky header to focus all panels on that token. Prompt tokens show their input geometry and what the model eventually paid attention to when it was being read. Generated tokens show everything: the exact distribution the model sampled from, the attention pattern the final layer used, and the token&apos;s trajectory through the depth of the model.
            </p>
            <p>
              <strong>Scrubber.</strong> Press play to animate through the generation token by token. The focused token advances with the scrubber, so you can watch the model&apos;s confidence, attention, and geometry evolve as it writes.
            </p>
          </>
        }
      />

      {/* Controls */}
      <div className="card-editorial p-4 space-y-3">
        <div>
          <label className="block font-sans text-[11px] text-slate mb-1">Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="input-editorial w-full font-mono text-[12px]"
            rows={2}
            placeholder="Enter a prompt to generate from..."
          />
          <div className="mt-2">
            <PresetChipRow
              disabled={running}
              items={GENERATION_VECTOR_PRESETS.map((p) => ({
                label: p.label,
                title: p.title,
                onClick: () => {
                  setPrompt(p.prompt);
                  if (p.maxNewTokens !== undefined) setMaxNewTokens(p.maxNewTokens);
                  if (p.temperature !== undefined) setTemperature(p.temperature);
                  if (p.topP !== undefined) setTopP(p.topP);
                  if (p.topK !== undefined) setTopK(p.topK);
                },
              }))}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <SliderField
            label="Max new tokens"
            min={1}
            max={maxCap}
            step={1}
            value={maxNewTokens}
            onChange={setMaxNewTokens}
            suffix={` / ${maxCap}`}
          />
          <SliderField
            label="Temperature"
            min={0}
            max={2}
            step={0.05}
            value={temperature}
            onChange={setTemperature}
            suffix={temperature === 0 ? " (greedy)" : ""}
          />
          <SliderField
            label="Top-p"
            min={0}
            max={1}
            step={0.01}
            value={topP}
            onChange={setTopP}
          />
          <SliderField
            label="Top-k"
            min={1}
            max={200}
            step={1}
            value={topK}
            onChange={setTopK}
          />
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={run}
            disabled={running}
            className="btn-editorial-primary"
          >
            {running ? "Generating..." : "Generate"}
          </button>
          {running && (
            <button
              onClick={() => abortRef.current?.abort()}
              className="btn-editorial-secondary"
            >
              Stop
            </button>
          )}
          {error && (
            <span className="text-red-600 font-sans text-[11px]">{error}</span>
          )}
          {state?.complete && !error && (
            <span className="font-sans text-[11px] text-slate">
              {state.complete.numGenerated} tokens generated · entropy{" "}
              {(
                state.complete.entropySeries.reduce((a, b) => a + b, 0) /
                Math.max(state.complete.entropySeries.length, 1)
              ).toFixed(2)}{" "}
              bits avg
            </span>
          )}
        </div>
      </div>

      {state?.prompt && (
        <>
          {/* Sticky header: prompt + generation token strip */}
          <div className="sticky top-0 z-10 card-editorial p-3 bg-ivory/95 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
                Prompt → Completion
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setReverseOrder((v) => !v)}
                  className="flex items-center gap-1 font-sans text-[10px] text-slate hover:text-ink transition-colors px-2 py-1 border border-parchment rounded-sm"
                  title="Swap panel order"
                >
                  <ArrowLeftRight className="w-3 h-3" />
                  {reverseOrder ? "Reverse order" : "Pipeline order"}
                </button>
              </div>
            </div>
            <TokenStrip
              allTokens={allTokens}
              promptLen={state.prompt.promptLen}
              focusIdx={focusIdx}
              onFocus={setFocusIdx}
              entropySeries={state.complete?.entropySeries}
            />
          </div>

          {/* Scrubber */}
          {state.steps.length > 0 && (
            <div className="card-editorial p-3">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => {
                    setScrubberIdx(0);
                    setPlaying(false);
                  }}
                  className="text-slate hover:text-ink transition-colors"
                >
                  <SkipBack className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setPlaying((p) => !p)}
                  className="text-slate hover:text-ink transition-colors"
                >
                  {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => {
                    setScrubberIdx(state.steps.length - 1);
                    setPlaying(false);
                  }}
                  className="text-slate hover:text-ink transition-colors"
                >
                  <SkipForward className="w-4 h-4" />
                </button>
                <input
                  type="range"
                  min={0}
                  max={Math.max(state.steps.length - 1, 0)}
                  value={scrubberIdx}
                  onChange={(e) => {
                    setScrubberIdx(Number(e.target.value));
                    setPlaying(false);
                  }}
                  className="flex-1"
                />
                <span className="font-mono text-[11px] text-slate w-20 text-right">
                  step {scrubberIdx + 1} / {state.steps.length}
                </span>
              </div>
            </div>
          )}

          {/* Panel tabs */}
          <div className="card-editorial">
            <div className="flex border-b border-parchment overflow-x-auto">
              {displayPanels.map((p, i) => (
                <button
                  key={p.id}
                  onClick={() => setPanelIdx(i)}
                  className={`px-3 py-2 font-sans text-[11px] whitespace-nowrap border-b-2 transition-colors ${
                    i === panelIdx
                      ? "border-burgundy text-ink"
                      : "border-transparent text-slate hover:text-ink"
                  }`}
                >
                  {i + 1}. {p.label}
                </button>
              ))}
            </div>

            <div className="p-4 min-h-[360px]">
              {activePanel === "tokenisation" && (
                <PanelTokenisation
                  state={state}
                  focusedToken={focusedToken}
                  focusIdx={focusIdx}
                  onFocus={setFocusIdx}
                />
              )}
              {activePanel === "embedding" && (
                <PanelEmbedding
                  state={state}
                  focusedToken={focusedToken}
                />
              )}
              {activePanel === "attention" && (
                <PanelAttention
                  state={state}
                  focusedStep={focusedStep}
                />
              )}
              {activePanel === "layers" && (
                <PanelLayers
                  state={state}
                  focusIdx={focusIdx}
                />
              )}
              {activePanel === "output" && (
                <PanelOutput
                  state={state}
                  focusedStep={focusedStep}
                />
              )}
              {activePanel === "text" && (
                <PanelText
                  state={state}
                  focusIdx={focusIdx}
                  onFocus={setFocusIdx}
                />
              )}
            </div>
          </div>

          {/* Deep dive panel */}
          {state.complete && deepStats && (
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
                <div className="px-4 pb-4 border-t border-parchment space-y-4">
                  <div className="grid grid-cols-2 gap-6 mt-3">
                    {/* Left column: generation summary */}
                    <div>
                      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                        Generation
                      </h4>
                      <table className="w-full font-mono text-[11px]">
                        <tbody>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Prompt length</td>
                            <td className="py-1 text-right">{state.complete.promptLen} tokens</td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Generated length</td>
                            <td className="py-1 text-right">{state.complete.numGenerated} tokens</td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Total context</td>
                            <td className="py-1 text-right">
                              {state.complete.promptLen + state.complete.numGenerated} tokens
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Layers</td>
                            <td className="py-1 text-right">{state.complete.numLayers}</td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Temperature</td>
                            <td className="py-1 text-right">
                              {temperature.toFixed(2)}
                              {temperature === 0 && " (greedy)"}
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Top-p / Top-k</td>
                            <td className="py-1 text-right">
                              {topP.toFixed(2)} / {topK}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    {/* Right column: entropy + norm statistics */}
                    <div>
                      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                        Entropy &amp; Norms
                      </h4>
                      <table className="w-full font-mono text-[11px]">
                        <tbody>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Mean entropy</td>
                            <td className="py-1 text-right">
                              {deepStats.entropyMean.toFixed(3)} bits
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Std entropy</td>
                            <td className="py-1 text-right">
                              {deepStats.entropyStd.toFixed(3)} bits
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Most confident</td>
                            <td className="py-1 text-right">
                              <span className="text-burgundy">
                                {JSON.stringify(deepStats.entropyMin.token)}
                              </span>{" "}
                              · {deepStats.entropyMin.v.toFixed(2)} bits
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Least confident</td>
                            <td className="py-1 text-right">
                              <span className="text-burgundy">
                                {JSON.stringify(deepStats.entropyMax.token)}
                              </span>{" "}
                              · {deepStats.entropyMax.v.toFixed(2)} bits
                            </td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Mean norm (μ layers)</td>
                            <td className="py-1 text-right">{deepStats.normMean.toFixed(3)}</td>
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Norm range</td>
                            <td className="py-1 text-right">
                              {deepStats.normMin.toFixed(2)} – {deepStats.normMax.toFixed(2)}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Per-layer table */}
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Per-layer dynamics (averaged across generated tokens)
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full font-mono text-[10px]">
                        <thead>
                          <tr className="border-b border-parchment text-slate">
                            <th className="py-1 text-left">Layer</th>
                            {Array.from({ length: deepStats.numLayers }, (_, i) => (
                              <th key={i} className="py-1 px-1 text-right">
                                L{i}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Mean norm</td>
                            {deepStats.perLayerMeanNorm.map((n, i) => (
                              <td key={i} className="py-1 px-1 text-right">
                                {n.toFixed(1)}
                              </td>
                            ))}
                          </tr>
                          <tr className="border-b border-parchment">
                            <td className="py-1 text-slate">Cos(L, L−1)</td>
                            {deepStats.perLayerMeanDelta.map((d, i) => (
                              <td key={i} className="py-1 px-1 text-right">
                                {d === null ? "—" : d.toFixed(3)}
                              </td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Per-step table (scrollable) */}
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Per-step log
                    </h4>
                    <div className="max-h-64 overflow-y-auto border border-parchment rounded-sm">
                      <table className="w-full font-mono text-[10px]">
                        <thead className="sticky top-0 bg-cream/90 backdrop-blur-sm">
                          <tr className="border-b border-parchment text-slate">
                            <th className="py-1 px-2 text-left">Step</th>
                            <th className="py-1 px-2 text-left">Token</th>
                            <th className="py-1 px-2 text-right">Entropy</th>
                            <th className="py-1 px-2 text-right">Top-1 p</th>
                            <th className="py-1 px-2 text-right">Mean norm</th>
                            <th className="py-1 px-2 text-right">Final cos</th>
                          </tr>
                        </thead>
                        <tbody>
                          {state.steps.map((s, i) => {
                            const finalCos = s.layerDeltas[s.layerDeltas.length - 1];
                            const meanN =
                              s.hiddenNorms.reduce((a, b) => a + b, 0) / s.hiddenNorms.length;
                            return (
                              <tr
                                key={i}
                                className="border-b border-parchment/50 hover:bg-cream/40 cursor-pointer"
                                onClick={() =>
                                  setFocusIdx(state.prompt!.promptLen + i)
                                }
                              >
                                <td className="py-1 px-2 text-slate">{i + 1}</td>
                                <td className="py-1 px-2 text-ink">
                                  {JSON.stringify(s.token)}
                                </td>
                                <td className="py-1 px-2 text-right">
                                  {s.entropyBits.toFixed(2)}
                                </td>
                                <td className="py-1 px-2 text-right">
                                  {s.topPredictions[0]?.probability.toFixed(3)}
                                </td>
                                <td className="py-1 px-2 text-right">{meanN.toFixed(2)}</td>
                                <td className="py-1 px-2 text-right">
                                  {finalCos === null ? "—" : finalCos.toFixed(3)}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <p className="font-sans text-[10px] text-slate mt-1">
                      Click a row to focus all panels on that step.
                    </p>
                  </div>

                  {/* Interpretation notes */}
                  <div className="bg-cream/50 rounded-sm p-3">
                    <p className="font-sans text-[11px] text-slate leading-relaxed">
                      <strong>Entropy</strong> (in bits) measures how much uncertainty the model
                      had at each generation step — 0 bits means one token was certain, higher
                      values mean it saw many viable continuations. <strong>Mean norm</strong>{" "}
                      tracks the geometric energy of the last hidden state: rising norms through
                      generation often indicate accumulating context.{" "}
                      <strong>Layer-to-layer cosine</strong> shows how much each layer rewrote the
                      representation: values near 1.0 mean the layer passed the token through
                      mostly unchanged, lower values mean the layer substantially reshaped it.
                      The final layer&apos;s cosine is the most telling — it reveals how much the
                      projection head relied on late-stage rewriting to produce the next token.
                    </p>
                  </div>

                  {/* Tokenisation numeric table */}
                  {state.complete.tokenisationSeries && (
                    <div>
                      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                        Tokenisation — full vector numerics
                      </h4>
                      <p className="font-sans text-[10px] text-slate mb-2">
                        Per-token input-embedding norm, running cumulative sum norm, running
                        mean norm, independence baseline (mean-norm × √t), and the first{" "}
                        {state.complete.tokenisationSeries.previewDims} components of each
                        token&apos;s input-embedding vector. Click a row to focus.
                      </p>
                      <div className="max-h-72 overflow-auto border border-parchment rounded-sm">
                        <table className="w-full font-mono text-[10px]">
                          <thead className="sticky top-0 bg-cream/90 backdrop-blur-sm">
                            <tr className="border-b border-parchment text-slate">
                              <th className="py-1 px-2 text-left">Pos</th>
                              <th className="py-1 px-2 text-left">Token</th>
                              <th className="py-1 px-2 text-right">‖v‖</th>
                              <th className="py-1 px-2 text-right">‖Σv‖</th>
                              <th className="py-1 px-2 text-right">‖Σv‖/t</th>
                              <th className="py-1 px-2 text-right">√t·n̄</th>
                              <th className="py-1 px-2 text-left">
                                v[0..{state.complete.tokenisationSeries.previewDims - 1}]
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {state.complete.tokenisationSeries.perTokenNorms.map((n, i) => {
                              const ts = state.complete!.tokenisationSeries!;
                              const promptL = state.complete!.promptLen;
                              const isPrompt = i < promptL;
                              const tokStr = isPrompt
                                ? state.prompt!.tokens[i]
                                : state.steps[i - promptL]?.token ?? "";
                              const displayTok =
                                tokStr === "\n" ? "↵" : tokStr === " " ? "·" : tokStr;
                              return (
                                <tr
                                  key={i}
                                  className="border-b border-parchment/50 hover:bg-cream/40 cursor-pointer"
                                  onClick={() => setFocusIdx(i)}
                                >
                                  <td className="py-1 px-2 text-slate">{i}</td>
                                  <td className="py-1 px-2 text-ink">{displayTok}</td>
                                  <td className="py-1 px-2 text-right">{n.toFixed(3)}</td>
                                  <td className="py-1 px-2 text-right">
                                    {ts.cumulativeSumNorms[i].toFixed(3)}
                                  </td>
                                  <td className="py-1 px-2 text-right">
                                    {ts.cumulativeMeanNorms[i].toFixed(3)}
                                  </td>
                                  <td className="py-1 px-2 text-right text-slate">
                                    {ts.independenceBaseline[i].toFixed(3)}
                                  </td>
                                  <td className="py-1 px-2 text-slate whitespace-nowrap">
                                    [{ts.vectorPreview[i]
                                      .map((v) => (v >= 0 ? " " : "") + v.toFixed(3))
                                      .join(", ")}
                                    ]
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Full generated text */}
                  <div>
                    <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider mb-2">
                      Full completion
                    </h4>
                    <div className="bg-cream/50 rounded-sm p-3 font-mono text-[11px] text-ink whitespace-pre-wrap leading-relaxed">
                      {state.complete.fullText}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Subcomponents                                                      */
/* ------------------------------------------------------------------ */

function SliderField({
  label,
  min,
  max,
  step,
  value,
  onChange,
  suffix,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  suffix?: string;
}) {
  return (
    <label className="block font-sans text-[10px] text-slate">
      <div className="flex justify-between mb-1">
        <span>{label}</span>
        <span className="font-mono text-ink">
          {value}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </label>
  );
}

function TokenStrip({
  allTokens,
  promptLen,
  focusIdx,
  onFocus,
  entropySeries,
}: {
  allTokens: { str: string; absPos: number; kind: "prompt" | "generated"; stepIdx: number | null }[];
  promptLen: number;
  focusIdx: number | null;
  onFocus: (idx: number | null) => void;
  entropySeries?: number[];
}) {
  const maxEntropy = useMemo(
    () => (entropySeries && entropySeries.length ? Math.max(...entropySeries) : 1),
    [entropySeries]
  );

  return (
    <div className="flex flex-wrap gap-1 items-center">
      {allTokens.map((t) => {
        const isFocus = t.absPos === focusIdx;
        const isPrompt = t.kind === "prompt";
        const entropy =
          !isPrompt && t.stepIdx !== null && entropySeries?.[t.stepIdx] !== undefined
            ? entropySeries[t.stepIdx]
            : null;
        const entropyOpacity = entropy !== null ? 0.3 + 0.7 * (entropy / maxEntropy) : 1;

        return (
          <span key={t.absPos}>
            {t.absPos === promptLen && (
              <span className="mx-2 text-burgundy font-mono text-[11px]">▸</span>
            )}
            <button
              onClick={() => onFocus(isFocus ? null : t.absPos)}
              className={`inline-block font-mono text-[11px] px-1.5 py-0.5 rounded-sm transition-colors ${
                isFocus
                  ? "bg-burgundy text-ivory"
                  : isPrompt
                  ? "bg-parchment text-ink hover:bg-parchment/80"
                  : "bg-gold/20 text-ink hover:bg-gold/40"
              }`}
              style={!isFocus && !isPrompt ? { opacity: entropyOpacity } : undefined}
              title={
                isPrompt
                  ? `prompt token ${t.absPos}`
                  : `generated token ${t.stepIdx! + 1}${entropy !== null ? ` · entropy ${entropy.toFixed(2)} bits` : ""}`
              }
            >
              {t.str === "\n" ? "↵" : t.str === " " ? "·" : t.str}
            </button>
          </span>
        );
      })}
    </div>
  );
}

function PanelTokenisation({
  state,
  focusedToken,
  focusIdx,
  onFocus,
}: {
  state: GenerationState;
  focusedToken: { str: string; absPos: number; kind: "prompt" | "generated"; stepIdx: number | null } | null;
  focusIdx: number | null;
  onFocus: (idx: number | null) => void;
}) {
  if (!state.prompt) return null;

  const promptLen = state.prompt.promptLen;

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Tokenisation
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          Before the model sees anything, the tokeniser breaks the prompt into subword units drawn
          from a fixed vocabulary. Each chip below is one token; the number underneath is its
          vocabulary id, the integer index into the embedding table. Prompt tokens are cream;
          generated tokens are gold.
        </p>
        <p>
          Watch for how common words stay whole while rarer forms fragment into pieces
          (&ldquo;ing&rdquo;, &ldquo;tion&rdquo;, byte-level leading spaces). Leading spaces appear
          as &ldquo;·&rdquo; and newlines as &ldquo;↵&rdquo;. Every downstream panel indexes tokens
          by the absolute position shown here, so click a chip to lock focus across all panels —
          or use{" "}
          <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">←</kbd>{" "}
          <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">→</kbd>{" "}
          to step chip by chip and{" "}
          <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">↑</kbd>{" "}
          <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">↓</kbd>{" "}
          to jump between rows.
        </p>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {state.prompt.tokens.map((tok, i) => (
          <TokenIdChip
            key={`p${i}`}
            token={tok}
            id={state.prompt!.tokenIds[i]}
            kind="prompt"
            absPos={i}
            focused={focusedToken?.absPos === i}
            onClick={() => onFocus(focusIdx === i ? null : i)}
          />
        ))}
        {state.steps.map((s, i) => {
          const absPos = promptLen + i;
          return (
            <TokenIdChip
              key={`g${i}`}
              token={s.token}
              id={s.tokenId}
              kind="generated"
              absPos={absPos}
              focused={focusedToken?.absPos === absPos}
              onClick={() => onFocus(focusIdx === absPos ? null : absPos)}
            />
          );
        })}
      </div>
      {focusedToken && (
        <FocusedTokenDetail state={state} focusedToken={focusedToken} />
      )}

      {state.complete?.tokenisationSeries && (
        <TokenisationVectorChart
          series={state.complete.tokenisationSeries}
          labels={[
            ...state.prompt.tokens.map((t) => (t === "\n" ? "↵" : t === " " ? "·" : t)),
            ...state.steps.map((s) => (s.token === "\n" ? "↵" : s.token === " " ? "·" : s.token)),
          ]}
          promptLen={promptLen}
          focusX={focusedToken?.absPos ?? null}
        />
      )}
    </div>
  );
}

function FocusedTokenDetail({
  state,
  focusedToken,
}: {
  state: GenerationState;
  focusedToken: { str: string; absPos: number; kind: "prompt" | "generated"; stepIdx: number | null };
}) {
  if (!state.prompt) return null;
  const promptLen = state.prompt.promptLen;
  const absPos = focusedToken.absPos;
  const isPrompt = focusedToken.kind === "prompt";
  const tokenId = isPrompt
    ? state.prompt.tokenIds[absPos]
    : state.steps[focusedToken.stepIdx!]?.tokenId ?? -1;

  const tokStr = focusedToken.str;
  const bytes = Array.from(new TextEncoder().encode(tokStr));
  const codePoints = Array.from(tokStr).map((c) => c.codePointAt(0) ?? 0);
  const hasLeadingSpace = tokStr.startsWith(" ");
  const isNewline = tokStr === "\n";
  const isWhitespaceOnly = /^\s+$/.test(tokStr) && !isNewline;

  // Tokenisation series stats (if generation completed)
  const ts = state.complete?.tokenisationSeries;
  const inputNorm = ts?.perTokenNorms[absPos];
  const cumNorm = ts?.cumulativeSumNorms[absPos];
  const cumMean = ts?.cumulativeMeanNorms[absPos];
  const baseline = ts?.independenceBaseline[absPos];
  const preview = ts?.vectorPreview[absPos];

  // Per-layer geometry from Geometry event
  const geom = state.geometry?.coords[absPos];
  const numLayers = state.geometry?.numLayers ?? 0;

  // If it's a generated token, pull the step event for richer data.
  const step =
    !isPrompt && focusedToken.stepIdx !== null ? state.steps[focusedToken.stepIdx] : null;
  const hiddenNorms = step?.hiddenNorms;
  const layerDeltas = step?.layerDeltas;
  const topPred = step?.topPredictions?.slice(0, 5);
  const entropy = step?.entropyBits;
  const topP1 = topPred?.[0]?.probability;
  const rank = step
    ? step.topPredictions.findIndex((p) => p.tokenId === step.tokenId)
    : -1;

  const displayTok = isNewline
    ? "↵ (newline)"
    : isWhitespaceOnly
    ? `· (${tokStr.length}× space)`
    : JSON.stringify(tokStr);

  return (
    <div className="mt-4 p-3 bg-cream/50 rounded-sm space-y-2">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <div className="font-sans text-[9px] text-slate uppercase tracking-wider">
            Focused token
          </div>
          <div className="font-mono text-[14px] text-ink mt-0.5">{displayTok}</div>
        </div>
        <div className="font-mono text-[9px] text-slate text-right leading-relaxed">
          pos {absPos} of {(state.complete?.numGenerated ?? state.steps.length) + promptLen} ·{" "}
          {isPrompt ? "PROMPT" : `GENERATED · step ${(focusedToken.stepIdx ?? 0) + 1}`}
          <br />
          id {tokenId} · len {tokStr.length} char{tokStr.length === 1 ? "" : "s"} ·{" "}
          {bytes.length} byte{bytes.length === 1 ? "" : "s"}
        </div>
      </div>

      {/* Lexical / byte detail */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[9px] text-slate">
        <div>
          <span className="text-slate/70">bytes:</span>{" "}
          <span className="text-ink">
            {bytes.map((b) => b.toString(16).padStart(2, "0")).join(" ") || "∅"}
          </span>
        </div>
        <div>
          <span className="text-slate/70">code pts:</span>{" "}
          <span className="text-ink">
            {codePoints.map((c) => "U+" + c.toString(16).toUpperCase().padStart(4, "0")).join(" ") || "∅"}
          </span>
        </div>
        <div>
          <span className="text-slate/70">leading space:</span>{" "}
          <span className="text-ink">{hasLeadingSpace ? "yes" : "no"}</span>
        </div>
        <div>
          <span className="text-slate/70">class:</span>{" "}
          <span className="text-ink">
            {isNewline
              ? "newline"
              : isWhitespaceOnly
              ? "whitespace"
              : /^[A-Za-z]+$/.test(tokStr.trim())
              ? "alpha"
              : /^[0-9]+$/.test(tokStr.trim())
              ? "digit"
              : /^[\p{P}]+$/u.test(tokStr.trim())
              ? "punct"
              : "mixed"}
          </span>
        </div>
      </div>

      {/* Geometry detail — visible once tokenisation series is available */}
      {ts && inputNorm !== undefined && (
        <div className="pt-1.5 border-t border-parchment/60">
          <div className="font-sans text-[9px] text-slate uppercase tracking-wider mb-1">
            Input embedding (layer 0)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-0.5 font-mono text-[9px]">
            <div>
              <span className="text-slate/70">‖v‖:</span>{" "}
              <span className="text-ink">{inputNorm.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate/70">‖Σv‖:</span>{" "}
              <span className="text-ink">{cumNorm?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate/70">‖Σv‖/t:</span>{" "}
              <span className="text-ink">{cumMean?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate/70">√t·n̄:</span>{" "}
              <span className="text-ink">{baseline?.toFixed(4)}</span>
            </div>
          </div>
          {preview && (
            <div className="mt-1 font-mono text-[9px] text-slate whitespace-nowrap overflow-x-auto">
              <span className="text-slate/70">v[0..{preview.length - 1}]:</span>{" "}
              <span className="text-ink">
                [
                {preview
                  .map((v) => (v >= 0 ? " " : "") + v.toFixed(3))
                  .join(", ")}
                ]
              </span>
            </div>
          )}
        </div>
      )}

      {/* Per-layer summary (PCA coords + layer cosines if generated) */}
      {geom && (
        <div className="pt-1.5 border-t border-parchment/60">
          <div className="flex items-baseline justify-between mb-1">
            <div className="font-sans text-[9px] text-slate uppercase tracking-wider">
              Layer geometry (PCA3) — all {numLayers} layers
            </div>
            <div className="font-sans text-[8px] text-slate/70">
              L0 = post-embedding · L{numLayers - 1} = final hidden
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-[10px]">
              <thead>
                <tr className="text-slate/70 border-b border-parchment/60">
                  <th className="text-left py-0.5 px-1.5">Layer</th>
                  <th className="text-right py-0.5 px-1.5">PC1</th>
                  <th className="text-right py-0.5 px-1.5">PC2</th>
                  <th className="text-right py-0.5 px-1.5">PC3</th>
                  {hiddenNorms && <th className="text-right py-0.5 px-1.5">‖h‖</th>}
                  {layerDeltas && <th className="text-right py-0.5 px-1.5">cos(L, L−1)</th>}
                  {layerDeltas && <th className="text-left py-0.5 px-1.5">rewrite</th>}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: numLayers }).map((_, li) => {
                  const pt = geom[li];
                  const delta = layerDeltas?.[li] ?? null;
                  // Bar showing (1 - cos) magnitude: near 1.0 = passthrough, low = strong rewrite
                  const rewrite =
                    delta === null
                      ? null
                      : Math.max(0, Math.min(1, 1 - delta));
                  return (
                    <tr
                      key={li}
                      className="text-ink border-b border-parchment/30 hover:bg-cream/40"
                    >
                      <td className="py-0.5 px-1.5 text-slate whitespace-nowrap">
                        {li === 0
                          ? `L0 embed`
                          : li === numLayers - 1
                          ? `L${li} final`
                          : `L${li}`}
                      </td>
                      <td className="py-0.5 px-1.5 text-right">{pt[0].toFixed(3)}</td>
                      <td className="py-0.5 px-1.5 text-right">{pt[1].toFixed(3)}</td>
                      <td className="py-0.5 px-1.5 text-right">{pt[2].toFixed(3)}</td>
                      {hiddenNorms && (
                        <td className="py-0.5 px-1.5 text-right">
                          {hiddenNorms[li]?.toFixed(2)}
                        </td>
                      )}
                      {layerDeltas && (
                        <td className="py-0.5 px-1.5 text-right">
                          {delta === null ? "—" : delta.toFixed(3)}
                        </td>
                      )}
                      {layerDeltas && (
                        <td className="py-0.5 px-1.5">
                          {rewrite === null ? (
                            <span className="text-slate/40">—</span>
                          ) : (
                            <div className="w-16 h-1.5 bg-parchment/60 rounded-sm overflow-hidden inline-block align-middle">
                              <div
                                className="h-full bg-burgundy"
                                style={{ width: `${Math.round(rewrite * 100)}%` }}
                              />
                            </div>
                          )}
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="font-sans text-[9px] text-slate/70 mt-1">
            PC1/PC2/PC3 are the token&apos;s coordinates in the global PCA basis fitted
            across the whole sequence. ‖h‖ is the hidden-state norm at this layer.
            cos(L, L−1) is the cosine similarity between this layer&apos;s hidden state
            and the previous — near 1.0 means the layer left the representation nearly
            untouched, lower values mean stronger rewriting. The burgundy bar shows the
            rewrite magnitude (1 − cos) on a fixed scale.
          </p>
        </div>
      )}

      {/* Generation-time detail only for generated tokens */}
      {step && (
        <div className="pt-1.5 border-t border-parchment/60">
          <div className="font-sans text-[9px] text-slate uppercase tracking-wider mb-1">
            Sampling at this step
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-0.5 font-mono text-[9px]">
            <div>
              <span className="text-slate/70">entropy:</span>{" "}
              <span className="text-ink">{entropy?.toFixed(3)} bits</span>
            </div>
            <div>
              <span className="text-slate/70">top-1 p:</span>{" "}
              <span className="text-ink">{topP1?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-slate/70">rank in top-k:</span>{" "}
              <span className="text-ink">{rank >= 0 ? `#${rank + 1}` : "off-list"}</span>
            </div>
            <div>
              <span className="text-slate/70">ctx len:</span>{" "}
              <span className="text-ink">{step.contextLen}</span>
            </div>
          </div>
          {topPred && topPred.length > 0 && (
            <div className="mt-1 font-mono text-[9px] text-slate">
              <span className="text-slate/70">top 5:</span>{" "}
              {topPred.map((p, i) => (
                <span key={i} className="text-ink">
                  {i > 0 ? " · " : ""}
                  {JSON.stringify(p.token)}:{p.probability.toFixed(3)}
                  {p.tokenId === step.tokenId ? "✓" : ""}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {isPrompt && state.prompt.inputEmbedNorms && (
        <div className="pt-1.5 border-t border-parchment/60 font-mono text-[9px] text-slate">
          <span className="text-slate/70">prompt-stage ‖v‖:</span>{" "}
          <span className="text-ink">{state.prompt.inputEmbedNorms[absPos]?.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}

function TokenisationVectorChart({
  series,
  labels,
  promptLen,
  focusX,
}: {
  series: TokenisationSeries;
  labels: string[];
  promptLen: number;
  focusX: number | null;
}) {
  const x = labels.map((_, i) => i);
  const maxVal = Math.max(
    ...series.perTokenNorms,
    ...series.cumulativeSumNorms,
    ...series.independenceBaseline,
  );

  return (
    <div className="mt-4 space-y-2">
      <h5 className="font-sans text-[10px] font-semibold text-slate uppercase tracking-wider">
        Vector calculation across tokens
      </h5>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          Each token&apos;s input embedding is an <em>independent</em> lookup: token at position
          t does not see token at position t&minus;1. At layer 0 the computation is not
          cumulative. What becomes cumulative further up the stack is the hidden state,
          because causal attention from layer 1 onwards lets each position read the prefix.
        </p>
        <p>
          The bars below are per-token L2 norms of the input embedding. The burgundy line is
          the L2 norm of the running vector sum Σ<sub>i≤t</sub> v<sub>i</sub>. The clay dashed
          line is the independence baseline — what that running-sum norm would be on average
          if the v<sub>i</sub> were independent vectors of the mean observed norm, growing
          like √t. When the burgundy line tracks the clay line, consecutive tokens cancel as
          randomness expects; when it climbs above, tokens are pulling geometrically in the
          same direction; when it drops below, they are cancelling.
        </p>
      </div>
      <Plot
        data={[
          {
            type: "bar",
            x,
            y: series.perTokenNorms,
            name: "per-token ‖v‖",
            marker: {
              color: labels.map((_, i) =>
                i < promptLen ? "#A67F6F" : "#C9A227"
              ),
            },
            hovertemplate: "%{text}<br>‖v‖ = %{y:.3f}<extra></extra>",
            text: labels,
          },
          {
            type: "scatter",
            mode: "lines+markers",
            x,
            y: series.cumulativeSumNorms,
            name: "‖Σ v‖ cumulative",
            line: { color: "#5B2333", width: 2 },
            marker: { size: 4, color: "#5B2333" },
            hovertemplate: "pos %{x}<br>‖Σ v‖ = %{y:.3f}<extra></extra>",
          },
          {
            type: "scatter",
            mode: "lines",
            x,
            y: series.independenceBaseline,
            name: "independence baseline (√t)",
            line: { color: "#A67F6F", width: 1.5, dash: "dash" },
            hovertemplate: "pos %{x}<br>baseline %{y:.3f}<extra></extra>",
          },
        ]}
        layout={{
          height: 260,
          margin: { l: 40, r: 15, t: 10, b: 40 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: {
            title: "token position",
            tickmode: "array",
            tickvals: x.filter((_, i) => i % Math.max(1, Math.floor(x.length / 18)) === 0),
            ticktext: labels.filter((_, i) => i % Math.max(1, Math.floor(x.length / 18)) === 0),
            tickangle: -45,
            tickfont: { size: 9, family: "monospace" },
          },
          yaxis: { title: "L2 norm", gridcolor: "#E8E0D4" },
          font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
          legend: { orientation: "h", y: -0.28, font: { size: 9 } },
          shapes:
            focusX !== null
              ? [
                  {
                    type: "line",
                    x0: focusX,
                    x1: focusX,
                    y0: 0,
                    y1: maxVal,
                    line: { color: "#8B1A3B", width: 1, dash: "dot" },
                  },
                ]
              : [
                  {
                    type: "line",
                    x0: promptLen - 0.5,
                    x1: promptLen - 0.5,
                    y0: 0,
                    y1: maxVal,
                    line: { color: "#A67F6F", width: 1 },
                  },
                ],
        }}
        config={{ displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function TokenIdChip({
  token,
  id,
  kind,
  focused,
  absPos,
  onClick,
}: {
  token: string;
  id: number;
  kind: "prompt" | "generated";
  focused: boolean;
  absPos?: number;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      data-gv-chip={absPos}
      onClick={onClick}
      className={`inline-flex flex-col items-center px-2 py-1 rounded-sm border cursor-pointer transition-colors ${
        focused
          ? "bg-burgundy text-ivory border-burgundy"
          : kind === "prompt"
          ? "bg-parchment/60 border-parchment hover:bg-parchment hover:border-burgundy/40"
          : "bg-gold/20 border-gold/40 hover:bg-gold/30 hover:border-burgundy/40"
      }`}
    >
      <span className="font-mono text-[11px]">{token === "\n" ? "↵" : token === " " ? "·" : token}</span>
      <span className={`font-mono text-[8px] ${focused ? "text-ivory/80" : "text-slate"}`}>{id}</span>
    </button>
  );
}

function PanelEmbedding({
  state,
  focusedToken,
}: {
  state: GenerationState;
  focusedToken: { str: string; absPos: number; kind: "prompt" | "generated"; stepIdx: number | null } | null;
}) {
  if (!state.prompt) return null;

  // Build norm series across all tokens: prompt uses inputEmbedNorms,
  // generated tokens use the mean of their layer 0 hidden state norm.
  const promptNorms = state.prompt.inputEmbedNorms;
  const generatedLayer0Norms = state.steps.map((s) => s.hiddenNorms[0] ?? 0);
  const allNorms = [...promptNorms, ...generatedLayer0Norms];
  const labels = [
    ...state.prompt.tokens.map((t) => (t === "\n" ? "↵" : t)),
    ...state.steps.map((s) => (s.token === "\n" ? "↵" : s.token)),
  ];

  const focusX = focusedToken ? focusedToken.absPos : null;

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Input Embedding — Norms
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          Each token id is looked up in the embedding table and becomes a vector in the model&apos;s
          hidden space — the raw material every later layer will rewrite. The bar height is the
          L2 norm of that vector, a blunt measure of how much geometric &ldquo;weight&rdquo; the
          model has pre-assigned to the token before any context is applied.
        </p>
        <p>
          Function words and punctuation often sit low; rare or semantically loaded tokens often
          sit high, though the relationship is far from clean. Because this is context-free,
          every occurrence of a token gets the same bar. The burgundy dotted line marks the
          currently focused token.
        </p>
      </div>
      <Plot
        data={[
          {
            type: "bar",
            x: labels,
            y: allNorms,
            marker: {
              color: labels.map((_, i) =>
                i === focusX
                  ? "#8B1A3B"
                  : i < state.prompt!.promptLen
                  ? "#A67F6F"
                  : "#C9A227"
              ),
            },
            hovertemplate: "%{x}<br>norm: %{y:.3f}<extra></extra>",
          },
        ]}
        layout={{
          height: 280,
          margin: { l: 40, r: 15, t: 10, b: 60 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { tickangle: -45, tickfont: { size: 9 } },
          yaxis: { title: "norm", gridcolor: "#E8E0D4" },
          font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
          shapes:
            focusX !== null
              ? [
                  {
                    type: "line",
                    x0: focusX,
                    x1: focusX,
                    y0: 0,
                    y1: Math.max(...allNorms),
                    line: { color: "#8B1A3B", width: 1, dash: "dot" },
                  },
                ]
              : [],
        }}
        config={{ displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function PanelAttention({
  state,
  focusedStep,
}: {
  state: GenerationState;
  focusedStep: StepEvent | null;
}) {
  if (!state.prompt) return null;

  if (!focusedStep) {
    return (
      <div className="space-y-3">
        <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
          Attention
        </h4>
        <div className="font-sans text-[11px] text-slate space-y-1.5">
          <p>
            Attention is the mechanism by which a position at depth L pulls information from
            earlier positions. For each generated token we record, at every layer, the
            head-averaged weight the query position placed on every token in its context. That
            gives one heatmap row per layer.
          </p>
          <p>
            Click a generated token above to load its heatmap. Dark cells mean the model was
            leaning heavily on that context token when it produced the current one. Early layers
            tend to look broad and local; later layers often sharpen onto a few tokens carrying
            the meaning the model needs.
          </p>
        </div>
      </div>
    );
  }

  const numLayers = focusedStep.attentionByLayer.length;
  const contextLen = focusedStep.contextLen;

  // Build token labels for the x axis: prompt + previously-generated tokens
  const ctxTokens: string[] = [];
  state.prompt.tokens.forEach((t) => ctxTokens.push(t === "\n" ? "↵" : t));
  for (let i = 0; i < focusedStep.stepIndex; i++) {
    ctxTokens.push(state.steps[i].token === "\n" ? "↵" : state.steps[i].token);
  }
  // Trim to contextLen — might already match.
  const xLabels = ctxTokens.slice(0, contextLen);

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Attention — Generated Token: {JSON.stringify(focusedStep.token)}
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          Each row is a layer (L0 at the top), each column a position in the context that was
          available when &ldquo;{focusedStep.token}&rdquo; was produced. The cell is the mean
          attention weight over all heads at that layer from the query position back to that
          context token. Dark = high attention, light = near zero.
        </p>
        <p>
          Averaging over heads flattens out specialist circuits, so this is a coarse summary
          rather than a mechanistic read. Even so, vertical stripes reveal tokens the whole
          stack keeps returning to, and horizontal bands reveal layers where the model&apos;s
          focus widens or collapses.
        </p>
      </div>
      <Plot
        data={[
          {
            type: "heatmap",
            z: focusedStep.attentionByLayer,
            x: xLabels,
            y: Array.from({ length: numLayers }, (_, i) => `L${i}`),
            colorscale: [
              [0, "#F5F0E8"],
              [0.5, "#C9A227"],
              [1, "#5B2333"],
            ],
            hovertemplate: "layer %{y}<br>token: %{x}<br>weight: %{z:.4f}<extra></extra>",
          },
        ]}
        layout={{
          height: 300,
          margin: { l: 40, r: 15, t: 10, b: 60 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { tickangle: -45, tickfont: { size: 9 } },
          yaxis: { autorange: "reversed" },
          font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function PanelLayers({
  state,
  focusIdx,
}: {
  state: GenerationState;
  focusIdx: number | null;
}) {
  if (!state.geometry) {
    return (
      <div className="space-y-3">
        <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
          Layer Progression
        </h4>
        <p className="font-sans text-[11px] text-slate">
          Computing global PCA. Once generation completes, the backend fits a single principal
          component projection across every (token, layer) hidden state in the whole sequence and
          returns consistent 3D coordinates. The trajectory through the layers will appear here.
        </p>
      </div>
    );
  }

  const { coords, numLayers } = state.geometry;

  // If a token is focused, plot just that token's layer trajectory.
  // Otherwise plot every token as a faint line.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [];

  if (focusIdx !== null && coords[focusIdx]) {
    const t = coords[focusIdx];
    const layerLabels = Array.from({ length: numLayers }, (_, i) =>
      i === 0 ? "L0 (embed)" : i === numLayers - 1 ? `L${i} (final)` : `L${i}`
    );
    traces.push({
      type: "scatter3d",
      mode: "lines+markers+text",
      x: t.map((p) => p[0]),
      y: t.map((p) => p[1]),
      z: t.map((p) => p[2]),
      text: layerLabels,
      textposition: "top center",
      textfont: { family: "Inter, sans-serif", size: 10, color: "#5B2333" },
      hovertemplate: "%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
      line: { color: "#5B2333", width: 4 },
      marker: { size: 5, color: "#5B2333" },
      name: "focused token",
    });
  } else {
    // Aggregate: all tokens as faint lines.
    coords.forEach((t, idx) => {
      const isPrompt = idx < state.geometry!.promptLen;
      traces.push({
        type: "scatter3d",
        mode: "lines",
        x: t.map((p) => p[0]),
        y: t.map((p) => p[1]),
        z: t.map((p) => p[2]),
        line: { color: isPrompt ? "#A67F6F" : "#C9A227", width: 1 },
        opacity: 0.4,
        showlegend: false,
      });
    });
  }

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Layer Progression {focusIdx !== null ? "— Focused Token" : "— All Tokens"}
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          Every token&apos;s hidden state, at every layer, is projected into the same 3D frame via
          a single PCA fit across the whole sequence. This shared basis is what makes the paths
          comparable: a token&apos;s journey from embedding (L0) to final representation is one
          polyline, and different tokens can be read against each other.
        </p>
        <p>
          {focusIdx !== null
            ? "The highlighted polyline is the focused token. Numbered points are layer indices (L0 = post-embedding, final = last transformer block before the unembedding). A long jump between two layers means that block rewrote the representation substantially; a short jump means the residual stream was left largely intact."
            : "With no token focused, every token is shown as a faint line — prompt tokens in clay, generated tokens in gold. Click a token in the sticky header or in the decoded-text panel to isolate its trajectory with layer labels."}
        </p>
      </div>
      <Plot
        data={traces}
        layout={{
          height: 400,
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
          showlegend: false,
        }}
        config={{ displayModeBar: true, displaylogo: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function PanelOutput({
  state,
  focusedStep,
}: {
  state: GenerationState;
  focusedStep: StepEvent | null;
}) {
  if (!focusedStep) {
    return (
      <div className="space-y-3">
        <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
          Output Distribution
        </h4>
        <div className="font-sans text-[11px] text-slate space-y-1.5">
          <p>
            At the final layer each query position is multiplied by the unembedding matrix to
            produce a logit for every token in the vocabulary, which softmax turns into a
            probability distribution. Generation samples the next token from exactly that
            distribution.
          </p>
          <p>
            Click a generated token to see its top-10 candidates and the entropy (in bits) of the
            full softmax. Low entropy means the model was near-certain; high entropy means many
            tokens were plausible and the choice was largely down to the sampling rule. The line
            chart below tracks that entropy across the whole generation.
          </p>
        </div>
        {state.complete && state.complete.entropySeries.length > 0 && (
          <>
            <h5 className="font-sans text-[10px] font-semibold text-slate uppercase tracking-wider mt-4">
              Entropy across generation
            </h5>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: state.complete.entropySeries.map((_, i) => i + 1),
                  y: state.complete.entropySeries,
                  line: { color: "#5B2333", width: 2 },
                  marker: { size: 4, color: "#5B2333" },
                },
              ]}
              layout={{
                height: 200,
                margin: { l: 40, r: 15, t: 10, b: 30 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                xaxis: { title: "step", gridcolor: "#E8E0D4" },
                yaxis: { title: "bits", gridcolor: "#E8E0D4" },
                font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </>
        )}
      </div>
    );
  }

  const top = focusedStep.topPredictions.slice(0, 10);
  const selectedIdx = top.findIndex((p) => p.tokenId === focusedStep.tokenId);

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Output Distribution — Step {focusedStep.stepIndex + 1}
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          The bars are the ten most probable next-token candidates from the full softmax at this
          step. The burgundy bar is the one actually emitted —{" "}
          <span className="font-mono text-burgundy">{JSON.stringify(focusedStep.token)}</span> —
          which may or may not be the top candidate depending on temperature, top-p and top-k.
        </p>
        <p>
          Entropy of the full distribution was{" "}
          <strong>{focusedStep.entropyBits.toFixed(2)} bits</strong>. A top-1 probability close to
          1 with low entropy means the model was locked in; a flat distribution with high entropy
          means this was a genuine fork in the path. The near-synonyms or grammatical variants
          clustered in the top-k are often more revealing than the winner itself.
        </p>
      </div>
      <Plot
        data={[
          {
            type: "bar",
            orientation: "h",
            x: top.map((p) => p.probability),
            y: top.map((p) => (p.token === "\n" ? "↵" : p.token === " " ? "·" : p.token)),
            marker: {
              color: top.map((_, i) => (i === selectedIdx ? "#5B2333" : "#C9A227")),
            },
            hovertemplate: "%{y}<br>p = %{x:.4f}<extra></extra>",
          },
        ]}
        layout={{
          height: 280,
          margin: { l: 80, r: 15, t: 10, b: 30 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: { title: "probability", gridcolor: "#E8E0D4" },
          yaxis: { autorange: "reversed", tickfont: { family: "monospace" } },
          font: { family: "Inter, sans-serif", color: "#4A4A4A", size: 10 },
        }}
        config={{ displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function PanelText({
  state,
  focusIdx,
  onFocus,
}: {
  state: GenerationState;
  focusIdx: number | null;
  onFocus: (idx: number | null) => void;
}) {
  if (!state.prompt) return null;

  return (
    <div className="space-y-3">
      <h4 className="font-sans text-[11px] font-semibold text-slate uppercase tracking-wider">
        Decoded Text
      </h4>
      <div className="font-sans text-[11px] text-slate space-y-1.5">
        <p>
          The pipeline collapses back into prose. Token ids, vector arithmetic, attention weights,
          layer rewrites and softmax sampling all terminate here: a linear string the reader
          recognises as text. The prompt is in slate, the model&apos;s completion in ink, separated
          by the burgundy ▸ marker.
        </p>
        <p>
          This is the same sequence the other panels are indexing into. Click any token — prompt
          or generated — to lock focus across the whole operation; click it again to clear. The
          scrubber in the header replays the generation in order so you can watch the completion
          unfold token by token.
        </p>
      </div>
      <div className="bg-cream/30 rounded-sm p-4 font-body text-body-md leading-loose">
        {state.prompt.tokens.map((t, i) => (
          <span
            key={`p${i}`}
            onClick={() => onFocus(focusIdx === i ? null : i)}
            className={`cursor-pointer transition-colors ${
              focusIdx === i ? "bg-burgundy text-ivory px-0.5" : "text-slate hover:text-ink"
            }`}
          >
            {t}
          </span>
        ))}
        <span className="text-burgundy mx-1">▸</span>
        {state.steps.map((s, i) => {
          const absPos = state.prompt!.promptLen + i;
          return (
            <span
              key={`g${i}`}
              onClick={() => onFocus(focusIdx === absPos ? null : absPos)}
              className={`cursor-pointer transition-colors ${
                focusIdx === absPos
                  ? "bg-burgundy text-ivory px-0.5"
                  : "text-ink hover:text-burgundy"
              }`}
            >
              {s.token}
            </span>
          );
        })}
      </div>
    </div>
  );
}
