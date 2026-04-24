"use client";

import { useState } from "react";
import { HelpCircle, X } from "lucide-react";

export default function HelpDialog() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="text-slate hover:text-burgundy transition-colors"
        title="Help"
      >
        <HelpCircle className="w-4 h-4" />
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setOpen(false)}>
          <div
            className="bg-ivory border border-parchment-dark rounded-lg shadow-editorial-lg max-w-md w-full mx-4 p-6 max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start gap-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/icons/vector-lab.svg"
                  alt=""
                  width={40}
                  height={40}
                  className="mt-1 shrink-0"
                />
                <div>
                  <h2 className="font-display text-display-md">Help</h2>
                  <p className="font-sans text-caption text-slate/70 mt-0.5">
                    Part of the{" "}
                    <a
                      href="https://vector-lab-tools.github.io"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-burgundy hover:underline"
                    >
                      Vector Lab
                    </a>
                  </p>
                </div>
              </div>
              <button onClick={() => setOpen(false)} className="text-slate hover:text-burgundy">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 font-sans text-caption text-slate">
              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">Getting Started</h3>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Start the Python backend (<code className="bg-parchment px-1 rounded">cd backend && python main.py</code>)</li>
                  <li>Load a model from the model selector in the header</li>
                  <li>Pick an operation from the tab groups (Inspect / Trace / Critique)</li>
                  <li>Most operations now have a <strong>preset chip row</strong> under the input controls. Click a chip to fill the inputs with a theoretically-motivated example (contested concepts, bias probes, subject-verb traps, sampling-config bundles); then click <em>Run</em>. Chips do not auto-run, because some operations are expensive</li>
                  <li>Every operation has a collapsible introduction at the top with a <em>Learn more</em> modal; every operation also has a <em>Deep Dive</em> panel at the bottom with the full quantitative data</li>
                </ol>
              </section>

              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">Tab Groups</h3>
                <dl className="space-y-1">
                  <div>
                    <dt className="font-semibold">Inspect</dt>
                    <dd className="ml-4">Component-level weight examination. Embedding Table, Projection Head, Weight Comparison, Attention Inspector.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Trace</dt>
                    <dd className="ml-4">Follow data through the pipeline. Token Trajectory, Layer Probe, Full Trace, Generation Vector, Manifold Formation.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Critique</dt>
                    <dd className="ml-4">Theoretical instruments. Vocabulary Map, Isotropy Analysis, and Precision Degradation.</dd>
                  </div>
                </dl>
              </section>

              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">Operations</h3>
                <dl className="space-y-1.5">
                  <div>
                    <dt className="font-semibold">Embedding Table</dt>
                    <dd className="ml-4">Input embedding matrix. Vocab size, hidden dim, norm stats, effective rank, isotropy score, PCA 3D scatter of a sampled subset.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Projection Head</dt>
                    <dd className="ml-4">The lm_head / unembedding matrix. 3D scatter of output-side token directions, weight-tying detection, comparative effective rank.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Weight Comparison</dt>
                    <dd className="ml-4">Input embeddings vs output unembedding side by side. Per-token cosine distribution and norm scatter; shows how input and output representations diverge when weights are not tied.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Attention Inspector</dt>
                    <dd className="ml-4">Multi-head attention heatmaps at any layer. Per-head entropy and a focused/diagonal/diffuse pattern classifier.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Token Trajectory</dt>
                    <dd className="ml-4">One token through every layer. 3D PCA path, layer-to-layer cosine similarity, norm profile.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Layer Probe</dt>
                    <dd className="ml-4">Hidden states at a specific layer. Token-position similarity heatmap, norm bars, full similarity matrix.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Full Trace</dt>
                    <dd className="ml-4">Complete tokens → embeddings → layers → predictions pipeline, streamed progressively via NDJSON. Per-token norm heatmap, top-K prediction chart, entropy series.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Generation Vector</dt>
                    <dd className="ml-4">Instrumented autoregressive generation. Sticky token-chip header with playback scrubber, six horizontal panels (Tokenisation, Input Embedding, Attention, Layer Progression, Output Distribution, Decoded Text), click-to-focus across all panels, global PCA 3D trajectory per token. The Tokenisation panel shows per-token ‖v‖, running ‖Σv‖, and a √t independence baseline; the focused-token card exposes bytes, code points, per-layer PCA coordinates, layer-to-layer cosines, and sampling detail for generated tokens.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Manifold Formation</dt>
                    <dd className="ml-4">Animated layer-by-layer PCA geometry with play/pause. Shows how the manifold forms through depth.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Vocabulary Map</dt>
                    <dd className="ml-4">Global vocabulary topology. Searchable 3D scatter with token highlight, norm distribution.</dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Isotropy Analysis</dt>
                    <dd className="ml-4">Per-layer isotropy, mean pairwise cosine, top principal component variance ratios, and pairwise-cosine histograms at first/middle/last layer. Makes Ethayarajh&apos;s anisotropy collapse visible through depth.</dd>
                    <dt className="font-semibold">Precision Degradation</dt>
                    <dd className="ml-4">The Signal Degradation Laboratory in miniature. Runs a prompt at baseline precision and at each selected target (bf16 / fp16 / int8 / int4 / int2), compares hidden states layer-by-layer, and reports prediction changes. In-process fake-quant via round-to-nearest, not a load of a pre-quantised variant.</dd>
                    <dt className="font-semibold">Grammar Steering</dt>
                    <dd className="ml-4">Contrastive Activation Addition (Turner et al. 2023, Rimsky et al. 2024). Takes matched (positive, negative) prompt pairs and extracts a per-layer steering vector for the pattern under study — &ldquo;Not X but Y&rdquo; rhetorical antithesis ships as the default preset. Per-layer norm trajectory, leave-one-out separability, and a 3D PCA scatter make the pattern&apos;s activation signature visible. Phase 1 of the planned Grammar Steering work; Phase 2 (forward-pass intervention) is a separate session.</dd>
                  </div>
                </dl>
              </section>

              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">Keyboard Shortcuts</h3>
                <dl className="space-y-1">
                  <div>
                    <dt className="font-semibold">Generation Vector — panel navigation</dt>
                    <dd className="ml-4">
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">←</kbd>{" "}
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">→</kbd>{" "}
                      step between the six panels when no token is focused on Tokenisation.
                    </dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Generation Vector — Tokenisation chip navigation</dt>
                    <dd className="ml-4">
                      On the Tokenisation panel:{" "}
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">←</kbd>{" "}
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">→</kbd>{" "}
                      step chip by chip;{" "}
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">↑</kbd>{" "}
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">↓</kbd>{" "}
                      jump between rows of chips. Focus syncs across every panel.
                    </dd>
                  </div>
                  <div>
                    <dt className="font-semibold">3D plots (all operations)</dt>
                    <dd className="ml-4">
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">Shift</kbd>{" "}+ scroll for fast zoom on any Plotly 3D scatter.
                    </dd>
                  </div>
                  <div>
                    <dt className="font-semibold">Dialogs</dt>
                    <dd className="ml-4">
                      <kbd className="font-mono text-[9px] px-1 border border-parchment bg-cream rounded-sm">Esc</kbd>{" "}
                      closes Help, About, and any operation <em>Learn more</em> modal.
                    </dd>
                  </div>
                </dl>
              </section>

              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">How precision works</h3>
                <p className="mb-2">
                  Vectorscope tries to load every model at its{" "}
                  <em>native precision</em> — the numeric format its authors trained and published
                  it in — so the geometry you inspect is faithful to the weights on disk. Most
                  modern open-weight models (Qwen, Llama, Mistral) ship in{" "}
                  <details className="inline-block">
                    <summary className="inline cursor-pointer underline decoration-dotted decoration-slate/40 hover:decoration-burgundy hover:text-burgundy">
                      true bfloat16
                    </summary>
                    <div className="mt-2 p-3 bg-cream border border-parchment rounded-sm text-caption text-slate space-y-2">
                      <p>
                        <strong>bfloat16</strong> (brain floating point, 16-bit) is a numeric format
                        invented at Google Brain and now the default precision for training and
                        releasing most modern transformers. It uses 1 bit for the sign, 8 bits for
                        the exponent, and 7 bits for the mantissa.
                      </p>
                      <p>
                        The trick: bf16 has the <em>same exponent range</em> as float32 (8 bits,
                        representing numbers from roughly 10⁻³⁸ to 10³⁸) but only 7 bits of
                        precision. Regular float16 has 5 exponent bits and 10 mantissa bits —{" "}
                        <em>more precision near zero but a much narrower range</em>. That narrow
                        range is what causes activations to overflow to infinity or underflow to
                        zero during training.
                      </p>
                      <p>
                        bf16 trades precision for range. For neural networks that turns out to be
                        the right trade: gradients can take a very wide range of magnitudes, and a
                        little rounding error in the mantissa matters less than the ability to
                        represent very small and very large numbers without overflow. This is why
                        Qwen, Llama, Mistral, GPT-4-class models all use bf16.
                      </p>
                      <p>
                        Vectorscope keeps bf16 weights in bf16 on Apple Silicon (MPS), NVIDIA
                        (CUDA), and CPU, so isotropy and precision-degradation analyses run on the
                        same numerical ground the authors trained on.
                      </p>
                    </div>
                  </details>
                  ; legacy models like GPT-2 ship in float32.
                </p>
                <p className="mb-2">
                  <strong>Automatic down-cast for non-bf16 models.</strong> When a model's native
                  precision is not bf16 — GPT-2 is the main case, it's float32 — Vectorscope loads
                  it as{" "}
                  <code className="bg-parchment px-1 rounded">float16</code>{" "}
                  on MPS and CUDA to halve the memory footprint. The Settings panel marks this
                  with the form{" "}
                  <code className="bg-parchment px-1 rounded">downcast float32→float16</code>{" "}
                  so the transformation is visible at a glance. This is a mild, well-understood
                  precision loss (fp32 → fp16 rounds a few decimal places but preserves the
                  structure of the geometry). On CPU, float32 is kept as-is.
                </p>
                <p>
                  The Settings panel shows the actual <em>loaded precision</em>, so you'll see{" "}
                  <code className="bg-parchment px-1 rounded">downcast float32→float16</code>{" "}
                  for GPT-2 and <code className="bg-parchment px-1 rounded">bfloat16</code>{" "}
                  for Qwen / Llama / Mistral (kept native). The model-picker modal shows the{" "}
                  <em>native precision</em>, which is a property of the release, not of your
                  current session. This distinction matters for the upcoming Precision Degradation
                  operation, which applies in-process quantisation starting from the loaded
                  precision as ground truth.
                </p>
              </section>

              <section>
                <h3 className="font-display text-heading-sm font-semibold mb-2">Easter Eggs</h3>
                <p>
                  Clippy drifts in from the margins every few minutes with a short aphorism on
                  vector media and critical theory. Type <code className="bg-parchment px-1 rounded">hacker</code>{" "}
                  anywhere to summon Hackerman; type <code className="bg-parchment px-1 rounded">hermes</code>{" "}
                  to summon Hermes Trismegistus. Click the character to cycle through messages.
                </p>
              </section>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
