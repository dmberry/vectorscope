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
              <h2 className="font-display text-display-md">Help</h2>
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
                    <dd className="ml-4">Theoretical instruments. Vocabulary Map now; Isotropy Analysis, Cross-Model Anatomy, Precision Degradation are Phase 4.</dd>
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
