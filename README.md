> Part of the [Vector Lab](https://github.com/vector-lab-tools) —
> vector methods for vector theory.
> [Overview and map](https://vector-lab-tools.github.io) ·
> [Org profile](https://github.com/vector-lab-tools)
>
> **Tier:** single-model scope. **Object:** a single open-weight model.
>
> **Sibling instruments:**
> [Manifoldscope](https://github.com/vector-lab-tools/manifoldscope) ·
> [Theoryscope](https://github.com/vector-lab-tools/theoryscope) ·
> [Manifold Atlas](https://github.com/vector-lab-tools/manifold-atlas) ·
> [LLMbench](https://github.com/vector-lab-tools/LLMbench)

# Vectorscope

**Internal geometry of open-weight language models.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.5.0
**Date:** 24 April 2026
**Licence:** MIT



Vectorscope is a web-based research instrument that enables scholars and researchers to examine the internal geometry of open-weight language models. It loads a model locally, inspects its embedding tables and projection heads, traces token representations through transformer layers, visualises attention patterns, and analyses the material grain of learned geometry. Twelve operations are grouped under three analytical postures: inspect, trace, and critique, including contrastive activation extraction for identifying rhetorical patterns as directions in representation space.

The tool is designed for critical and empirical inquiry into the internal structure of language models, not engineering evaluation. Where existing model-inspection tools (BertViz, TransformerLens, Baukit) focus on mechanistic interpretability for alignment research, Vectorscope treats the vector space as a *medium* to be examined: its grain, its geometry, its ideological affordances.

## Scholarly Context

Vectorscope emerges from the convergence of three research programmes.

**Critical theory of vector space.** Berry (2026a, 2026b) develops an account of vector space as a material instantiation shaped by economic constraints rather than as a mathematical *a priori*. The design choices that determine its structure, the number of dimensions, the precision of each coordinate, the training corpus that shapes the manifold, are economic decisions, not neutral engineering parameters. As Berry argues, "capital is no longer disciplining the signal into bits, it is disciplining the bits into vectors." Vectorscope is the empirical instrument that makes this geometry inspectable rather than only theoretically asserted.

**Methodological critique of commercial APIs.** Commercial embedding APIs do not give researchers what they think they give them. They return sentence-level composite vectors from a separately trained embedding model, not the internal representations of the generative model itself. The outputs are doubly mediated: first by the embedding model's own training objective, then by the API's post-processing. Open-weight models with accessible architectures are the methodologically stronger route for studying how language models organise meaning. Vectorscope is the practical implementation of that critique.

**Signal degradation as critical method.** This includes systematic testing of inputs across models at different precision regimes (FP32 → BF16 → INT8 → INT4 → FP4 → INT2), observing how signal degrades as the medium is compressed. Vectorscope's Precision Degradation operation (forthcoming) implements this method directly, treating quantisation not as an engineering optimisation but as a critical lens on the material substrate.

## Vectors, Layers, and the Manifold: A Primer

Vectorscope inspects three kinds of object, and it helps to understand the difference.

A **vector** is an array of numbers representing a token, a hidden state, or a weight. Every word in a model's vocabulary has an input embedding vector, typically between 768 dimensions (GPT-2) and 4096 (Mistral 7B). When the model runs, each token position at each layer carries a further hidden-state vector of the same dimensionality. The weights of the model are themselves matrices of vectors: the embedding table, the attention projections, the MLP layers, the unembedding head.

A **layer** is a transformation applied to these vectors. A transformer layer takes hidden states in, mixes them through attention and a feed-forward network, and produces hidden states out. Modern models stack between 12 (GPT-2) and 32 or more (Llama 3.2 3B, Mistral 7B) layers. Each layer rewrites the geometry of its input to a degree that can be measured by cosine similarity between consecutive layers. Vectorscope's Token Trajectory and Layer Probe operations make this layer-by-layer rewriting visible.

The **manifold** is the thin surface of meaningful locations within the ambient vector space where learned semantic structure actually resides. The manifold does not pre-exist; it is constituted through training. Models with more dimensions have more room on the manifold to separate distinct concepts without interference. Models with lower precision have less grain to discriminate between closely related meanings. Between extensive geometry (dimensionality) and intensive geometry (precision), the manifold tends toward **vector conformism**, the structural tendency of vector spaces to blur contested meanings toward their statistical centre of mass in training data.

This matters for Vectorscope because the tool offers three complementary reading surfaces. The embedding table and projection head are *static* geometric objects, readable as weight matrices. The hidden states and attention patterns are *dynamic*, visible only during forward passes. The isotropy and precision-degradation analyses are *diagnostic*, surfacing how the geometry compresses. A complete reading of a model's internal structure draws on all three.

## Operations at a Glance

Vectorscope is organised as a three-group tab navigation following the pipeline described in Berry (2026a): tokens → vectors → manifold → vectors → dictionary.

| Group | Mode | Purpose | Core question |
|---|---|---|---|
| Inspect | **Embedding Table** | Input embedding matrix geometry | How is the vocabulary arranged? |
| Inspect | **Projection Head** | Unembedding weights | How does the model decide what to output? |
| Inspect | **Weight Comparison** | Input vs output embeddings | Are the two ends of the pipeline aligned? |
| Inspect | **Attention Inspector** | Multi-head attention heatmaps | Where does each head look? |
| Trace | **Token Trajectory** | One token through all layers | How does a token move through the model? |
| Trace | **Layer Probe** | Hidden states at a specific layer | What does the geometry look like mid-pipeline? |
| Trace | **Full Trace** | End-to-end streaming pipeline | How do all pieces fit together? |
| Trace | **Generation Vector** | Instrumented autoregressive generation | What does each step of generation look like geometrically? |
| Trace | **Manifold Formation** | Layer-by-layer PCA animation | How does the manifold emerge through depth? |
| Critique | **Vocabulary Map** | Global topology of vocabulary | What does the space of all tokens look like? |
| Critique | **Isotropy Analysis** | Direction concentration per layer | How anisotropic is the representation? |
| Critique | **Precision Degradation** | Signal degradation across quantisation | What breaks when the medium compresses? |
| Critique | **Grammar Steering** | Contrastive activation vectors for rhetorical patterns | Does the pattern have an identifiable activation signature? |

All operations include a collapsible **Deep Dive** panel with detailed quantitative data for researchers who want to inspect the numbers directly. 3D plots support **Shift+scroll** for fast zoom. Most operations with user inputs expose a **preset chip row** of theoretically-motivated prompts (contested concepts, bias probes, subject-verb agreement traps, Berry's own formulations).

## Inspect (component-level weight examination)

Operations in the Inspect group read the model's weights directly, without running any forward pass. They make the static geometry of the embedding and projection matrices legible.

- **Embedding Table.** Load and visualise the input embedding matrix. PCA 3D scatter with configurable sample size, norm histogram, effective rank, isotropy score, weight-tying indicator. The cheapest operation and a good smoke test after loading a new model.
- **Projection Head.** Examine the `lm_head` / unembedding weights. 3D scatter, weight-tying detection (GPT-2 ties, most modern models do not), effective rank comparison against the input embeddings.
- **Weight Comparison.** Input embeddings against output unembedding in the same view. Per-token cosine similarity distribution and norm scatter, revealing which tokens the model treats symmetrically at input and output.
- **Attention Inspector.** Multi-head attention heatmaps at any layer with per-head entropy and pattern classification (focused, diagonal, diffuse). A second view walks a single chosen head through every layer to show how attention patterns shift through depth.

## Trace (following data through the pipeline)

Operations in the Trace group send a user-supplied text through the model and instrument the forward pass. They make the dynamic geometry of inference legible.

- **Token Trajectory.** Trace a single token through all layers. 3D trajectory path in PCA space, layer-to-layer cosine similarity, norm profile showing where the representation is most and least rewritten.
- **Layer Probe.** Hidden states at a specific layer for the full input. Token-token similarity heatmap, norm bars, full pairwise similarity matrix. The mid-pipeline view.
- **Full Trace.** The complete tokens → embeddings → layers → predictions pipeline in one streaming operation. NDJSON response with progress, per-token norm heatmap across layers, top-K prediction chart, entropy per position.
- **Generation Vector.** Full autoregressive generation with instrumented forward passes. Six horizontal panels (tokenisation, input embedding, attention, layer progression, output distribution, decoded text), scrubber playback, click-to-focus across panels, global PCA 3D trajectory per token. Tokenisation panel shows running ‖Σv‖ cumulative sum vs √t independence baseline; rich focused-token detail card with bytes, codepoints, per-layer geometry, and sampling detail.
- **Manifold Formation.** Animated layer-by-layer PCA geometry with play / pause controls. Shows how the manifold forms through depth, making the emergence of geometric structure visible as process rather than result.

**Keyboard shortcuts (Generation Vector).** `←` `→` step through panels or, on the Tokenisation panel, step chip by chip; `↑` `↓` jump between rows of chips. Focus syncs across all six panels so arrow-keying through the prompt drives the entire operation. `Esc` closes any modal.

## Critique (theoretical and diagnostic instruments)

Operations in the Critique group are interpretive rather than descriptive. They surface properties of the geometry that matter for a critical account of the model as a medium.

- **Vocabulary Map.** Global topology of the vocabulary embedding space. 3D scatter with token search and highlight, norm distribution, the whole lexicon in one view.
- **Isotropy Analysis.** Effective dimensionality and direction concentration per layer. Cosine-similarity histograms at first, middle, and last layer; top-1 / 3 / 10 principal component variance ratios; mean-norm trajectory. Connects directly to the vector-conformism thesis: anisotropic geometries pull concepts toward dominant axes.
- **Precision Degradation.** The Signal Degradation Laboratory in miniature. Runs a prompt through the loaded model at baseline precision and at each selected target (bf16, fp16, int8, int4, int2), comparing hidden states layer-by-layer and final predictions at the output. Uses in-process fake-quantisation via round-to-nearest — no pre-quantised variant is loaded, so all observed differences come from the weights losing grain rather than from a different model. Per-precision summary cards (argmax match, KL divergence, top-K overlap), per-layer cosine and relative-error plots, side-by-side top-5 prediction table.
- **Grammar Steering.** Contrastive Activation Addition (Turner et al. 2023; Rimsky et al. 2024). Takes matched (positive, negative) prompt pairs — positive carries a rhetorical pattern, negative is the same topic without it — and extracts a per-layer steering vector: the mean-activation difference. Per-layer norm trajectory and leave-one-out separability show at which depth the pattern lives; a 3D PCA scatter of the two populations makes the separation visible. Ships with a 12-pair &ldquo;Not X but Y&rdquo; rhetorical-antithesis preset. This is Phase 1 of the Grammar Steering work (extraction and geometry); Phase 2 (forward-pass intervention) is the next development target.

## General Features

- **Local-only execution.** No inference is sent to any cloud provider. A FastAPI + PyTorch backend loads the model on your own machine; a Next.js frontend talks to it over localhost. Inspect whatever you like without exfiltrating prompts or activations.
- **Open-weight models from HuggingFace.** Any `AutoModelForCausalLM`-compatible repo ID works. Five presets ship by default (GPT-2, Qwen3 0.6B, Llama 3.2 1B and 3B, Mistral 7B) and more can be added by editing [`backend/config/models.md`](backend/config/models.md) without rebuilding the frontend.
- **Native-precision loading.** Bfloat16 models (Qwen, Llama, Mistral) load at bf16 where the device supports it; float32 models (GPT-2) are down-cast to fp16 on MPS / CUDA to halve memory. The Settings panel shows both the native and the loaded precision so the transformation is visible rather than silent.
- **HuggingFace cache management.** Download, delete, and inspect cached model weights from inside the model picker. No need to leave the app to reclaim disk space.
- **Streaming operations.** Long-running operations (Full Trace, Generation Vector) stream NDJSON with progress events so the UI updates as the backend works.
- **Deep Dive.** Every operation has a collapsible Deep Dive panel with the full quantitative data: per-layer statistics tables, histogram bins, PCA coordinates, sampling logs.
- **Preset chips.** Theoretically-motivated example prompts across operations — contested concepts, subject-verb agreement traps, bias probes, Berry's own formulations ("Capital is disciplining bits into vectors"), sampling-config bundles for Generation Vector.
- **Editorial design system.** Ivory backgrounds, serif headings, burgundy accents, compact monospaced data tables. Shared with Manifold Atlas and LLMbench.
- **Export.** Every operation has an Export menu with four formats: **JSON** (raw result, ideal for reproducibility), **CSV** (tabular extract for spreadsheet analysis, where the operation has one), **PNG** (high-resolution plot capture), and **PDF** (composed document with title, metadata, plots, and summary tables for papers or reports).
- **Easter eggs.** A handful of hidden characters reward close reading of the interface.

## Design Rationale

**Why open-weight models only?** Commercial APIs do not expose internal representations. You cannot inspect the embedding table, trace hidden states through layers, or examine attention patterns through an API endpoint. The internal geometry that Vectorscope makes visible simply does not exist in API-mediated access. This is not a limitation but a methodological commitment.

**Why not mechanistic interpretability tools?** TransformerLens, Baukit, and similar tools are designed for alignment researchers investigating circuits, features, and causal interventions. They answer engineering questions: which neurons fire for this concept? What circuit implements this behaviour? Vectorscope answers different questions: What is the shape of the vocabulary space? How does the manifold transform through depth? Where does the geometry compress contested distinctions? These are questions about the *medium*, not the mechanism.

**Why Plotly for 3D visualisation?** The internal geometry of a 768-dimensional space projected to three dimensions is inherently approximate. Plotly's interactive 3D scatter plots allow researchers to rotate, zoom, and inspect the projection from multiple angles, preventing any single viewpoint from naturalising itself as *the* way to see the geometry. The same anti-naturalisation principle that motivates LLMbench's multiple probability visualisations applies here.

**Why the editorial design system?** Vectorscope shares its visual language with Manifold Atlas and LLMbench: ivory backgrounds, serif headings, burgundy accents, compact monospaced data tables. This is deliberate. The editorial aesthetic creates productive distance from the dashboard conventions of engineering tools, signalling that this is an instrument for scholarly inquiry rather than model optimisation.

**Why a Python backend and a TypeScript frontend?** Model weight loading, tensor extraction, and forward-pass instrumentation are PyTorch territory. There is no browser-side equivalent that exposes hidden states at arbitrary layers. The editorial UI, interactive 3D plots, and streaming visualisations are meanwhile native to the Next.js / React stack shared across the Vector Lab. The two halves talk over localhost HTTP with NDJSON streaming for long-running operations, following the same pattern as Manifold Atlas's Ollama integration.

## Getting Started

Vectorscope is two coordinated processes: a Python backend that loads the model and runs the forward passes, and a Next.js frontend that renders the interface. Both need to be running at the same time, in two separate terminals.

### Prerequisites

- **Python 3.9 or newer** (3.11 recommended). Check with `python3 --version`. On macOS, `brew install python@3.11`; on Ubuntu/Debian, `sudo apt install python3 python3-venv python3-pip`; on Windows, use [python.org](https://www.python.org/downloads/) and tick "Add Python to PATH".
- **Node.js 18 or newer** (20 LTS recommended) and **npm**. Check with `node --version` and `npm --version`.
- **Git**, to clone the repository.
- **Disk space.** HuggingFace caches models to `~/.cache/huggingface/`. GPT-2 is ~500 MB, Llama 3.2 1B ~2.5 GB, Mistral 7B ~14 GB.
- **Memory.** 8 GB RAM is enough for GPT-2. 1B–3B models want 16 GB. 7B models need 24 GB unified memory (Apple Silicon) or equivalent.
- **Optional, Apple Silicon.** PyTorch's MPS backend is used automatically when available, making forward passes several times faster than CPU. macOS 13.3+ required.
- **Optional, gated models.** A [HuggingFace account](https://huggingface.co/join) and access token. Llama and Mistral require accepting a licence on the model page and `huggingface-cli login` before downloading. GPT-2 and Qwen3 are ungated.

### Installation

```bash
git clone https://github.com/dmberry/vectorscope.git
cd vectorscope
```

### Running

From the project root, in the first terminal:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The first `pip install` pulls PyTorch, Transformers, FastAPI, scikit-learn and friends (2–5 minutes on a fast connection). When the backend is ready you will see `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal running.

In a second terminal, from the project root:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). If the "Model server: mps" (or `cuda` or `cpu`) indicator in the header is green, the frontend has found the backend. Click it for details on what the model server can do.

### Load a model

Click the model name in the header to open the picker. Start with **GPT-2 124M** — smallest, fastest, and the de facto reference for interpretability work. Click **Load**. The first time you load a given model the backend downloads the weights to `~/.cache/huggingface/`; subsequent loads are near-instant because the weights are cached.

Once the status bar shows the model is loaded, pick an operation from the **Inspect / Trace / Critique** tab groups. Try **Embedding Table** first — it is the cheapest operation and a good smoke test.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'`.** The virtual environment is not activated. Re-run `source backend/.venv/bin/activate`.
- **`EADDRINUSE: port 3000 already in use`.** Another Next.js dev server is running. Stop it, or start this one with `PORT=3001 npm run dev`.
- **Frontend says "Backend unreachable".** The backend is not running, or started on a non-default port. Confirm [http://localhost:8000/status](http://localhost:8000/status) returns JSON.
- **Out of memory.** Pick a smaller preset. GPT-2 runs comfortably in 8 GB; anything above 1B parameters wants 16 GB or more.
- **Llama / Mistral 403 on download.** Gated. Accept the licence on the HuggingFace model page, then `pip install huggingface_hub && huggingface-cli login` inside the activated backend venv.
- **Slow generation on Apple Silicon.** Confirm MPS is in use (the backend logs the device at startup, and the header indicator reports it). If it reports `cpu`, your PyTorch build is CPU-only; reinstall from [pytorch.org](https://pytorch.org/get-started/locally/).

## Architecture

```
backend/                          # Python FastAPI
  main.py                         # FastAPI app, CORS, routes
  config/
    models.md                     # Editable preset catalogue (no rebuild)
    presets.py                    # Markdown-to-dict parser
  models/
    session.py                    # Loaded model state (singleton)
  operations/
    embedding_table.py            # Extract & analyse embed matrix
    projection_head.py            # lm_head / unembedding weights
    weight_comparison.py          # Input vs output embedding comparison
    attention.py                  # Attention pattern extraction
    token_trajectory.py           # Token through all layers
    layer_probe.py                # Hidden states at layer N
    full_trace.py                 # End-to-end pipeline (NDJSON streaming)
    generation_vector.py          # Instrumented autoregressive generation
    manifold_formation.py         # Layer-by-layer PCA geometry
    isotropy.py                   # Direction concentration analysis
    cache_info.py                 # HuggingFace cache inspection
src/                              # Next.js frontend
  app/
    page.tsx                      # Mode-switching shell, model picker overlay
    layout.tsx, globals.css       # Editorial design system
  components/
    layout/                       # Header, TabNav, StatusBar, ModelLoader,
                                  #   BackendInfoDialog, HelpDialog, SettingsDialog
    operations/                   # One component per operation
    Plot3DWrapper.tsx             # Shift+scroll fast zoom for 3D plots
  context/
    ModelContext.tsx              # Loaded model state, backend polling
  lib/
    geometry/                     # Client-side PCA, cosine similarity
    version.ts                    # Single-source version from package.json
  types/
    model.ts                      # TypeScript types
```

The architecture follows the same pattern as LLMbench and Manifold Atlas: a thin `page.tsx` manages mode state and conditionally renders standalone operation components. Each operation dispatches to its own backend endpoint via `asyncio.to_thread` for heavy computation. The model session is a Python singleton holding one loaded model in memory across requests.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework (frontend) | Next.js 16 (App Router), React 19 |
| Framework (backend) | Python 3.9+, FastAPI, Uvicorn |
| Language | TypeScript 5 (strict), Python 3.9+ |
| Model inference | PyTorch ≥ 2.4, HuggingFace Transformers ≥ 4.45, safetensors |
| Styling | Tailwind CSS 3, Vector Lab editorial design system |
| Visualisation | Plotly.js via react-plotly.js (3D scatter, heatmaps, histograms); Three.js via @react-three/fiber (manifold animation) |
| Numerics (server) | scikit-learn (PCA on large tensors), numpy |
| Streaming | NDJSON over FastAPI `StreamingResponse` |
| Persistence | HuggingFace Hub cache on disk; no database |
| Export | jsPDF for composed PDFs; Plotly `toImage` for PNG capture |
| Icons | Lucide React |

## Roadmap

- [x] Isotropy Analysis (v0.2.16)
- [x] Native-precision loading with transparent down-cast reporting (v0.2.18)
- [x] HuggingFace cache management from inside the picker (v0.2.18)
- [x] Editable preset catalogue in `backend/config/models.md` (v0.2.19)
- [x] Precision Degradation / Signal Degradation Laboratory (v0.3.0)
- [x] Locally-trained model picker with inspect + validate (v0.3.1)
- [x] Vector Lab branding integration across Header, About, Help, StatusBar (v0.3.2)
- [ ] Dedicated locally-trained model picker (directory chooser, architecture validation)
- [ ] Annotation system for marking interesting geometric features
- [ ] Base-model vs embedding-model comparison (generative vs embedding versions of the same backbone)
- [ ] Persistent homology on internal representations
- [x] Export system: JSON, CSV, PNG, PDF on every operation (v0.4.0)
- [x] Grammar Steering Phase 1: contrastive activation vector extraction (v0.5.0)
- [ ] Grammar Steering Phase 2: forward-pass intervention (activation addition during generation)
- [ ] Packaged distribution (Tauri desktop app, `pipx` / `uvx` CLI, or Docker Compose)

## Related Work

- Berry, D. M. (2026a) 'What is Vector Space?', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/03/what-is-vector-space.html
- Berry, D. M. (2026b) 'Brain Numbers', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/03/brain-numbers.html
- Berry, D. M. (2026c) *Artificial Intelligence and Critical Theory*. Manchester University Press.
- Berry, D. M. (2025) 'Synthetic Media and Computational Capitalism: Towards a Critical Theory of Artificial Intelligence', *AI & Society*. Available at: https://doi.org/10.1007/s00146-025-02265-2
- Berry, D. M. (2014) *Critical Theory and the Digital*. Bloomsbury.
- Impett, L. and Offert, F. (2026) *Vector Media*. University of Minnesota Press.
- Marino, M. C. (2020) *Critical Code Studies*. MIT Press.
- Montfort, N. et al. (2013) *10 PRINT CHR$(205.5+RND(1)); : GOTO 10*. MIT Press.
- Rimsky, N. et al. (2024) 'Steering Llama 2 via Contrastive Activation Addition'. *arXiv:2312.06681*.
- Turner, A. et al. (2023) 'Activation Addition: Steering Language Models Without Optimization'. *arXiv:2308.10248*.

## Acknowledgements

Vectorscope shares its editorial design system with [Manifold Atlas](https://github.com/dmberry/manifold-atlas) and [LLMbench](https://github.com/dmberry/LLMbench). The three-group tab navigation, Deep Dive convention, and display-controls layout were refined across those projects and then adapted here. Preset chip conventions and the keyboard-navigation grammar follow the patterns established in LLMbench's Compare mode.

The backend builds on the [HuggingFace Transformers](https://github.com/huggingface/transformers) and [safetensors](https://github.com/huggingface/safetensors) libraries. Model weight loading, cache management, and safetensors header inspection use [huggingface_hub](https://github.com/huggingface/huggingface_hub) directly.

## Licence

MIT
