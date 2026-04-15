# Vectorscope

**Internal geometry of open-weight models.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.2.10
**Date:** 15 April 2026
**Licence:** MIT

Vectorscope is a research instrument for examining the internal geometry of open-weight language models. It inspects embedding tables, traces token representations through transformer layers, visualises attention patterns, and analyses the material grain of learned geometry. Unlike commercial embedding APIs, which return sentence-level composite vectors from separately trained embedding models, Vectorscope works directly with open-weight models where the weights, activations, and token embeddings can be inspected at each layer.

The tool is designed for critical and empirical inquiry into the internal structure of language models, not engineering evaluation. Where existing model inspection tools (BertViz, TransformerLens, Baukit) focus on mechanistic interpretability for alignment research, Vectorscope treats the vector space as a *medium* to be examined: its grain, its geometry, its ideological affordances.

## What is Vectorscope?

Vectorscope takes its name from the video engineering instrument that measures the colour properties of a signal by displaying them as vectors on a circular plot. This tool measures the properties of a vector representation by displaying its internal geometry.

The motivation is both methodological and theoretical.

**Methodologically**, Vectorscope responds to a problem with commercial embedding APIs: they do not give researchers what they think they give them. Commercial APIs return sentence-level composite vectors from a separately trained embedding model, not the internal representations of the generative model itself. The outputs are doubly mediated: first by the embedding model's own training objective, then by the API's post-processing. Open-weight models with accessible architectures are the methodologically stronger route for studying how language models organise meaning. Vectorscope is the practical implementation of that critique.

**Theoretically**, the tool connects to a critical theory of vector space developed in Berry (2026a, 2026b). Vector space is not a mathematical *a priori* but a material instantiation shaped by economic constraints. The design choices that determine its structure, the number of dimensions, the precision of each coordinate, the training corpus that shapes the manifold, are economic decisions, not neutral engineering parameters. As Berry argues, "capital is no longer disciplining the signal into bits, it is disciplining the bits into vectors."

The internal geometry of a language model exhibits two fundamental properties that Vectorscope makes legible:

**Extensive geometry** (dimensionality): the number of directions available in the vector space. GPT-2's 768 dimensions, GPT-3's 12,288, GPT-4's wider still. Dimensionality creates the capacity to separate distinct concepts without interference. Vectorscope's Embedding Table and Vocabulary Map operations make this extensive structure visible as topology.

**Intensive geometry** (precision): the grain or resolution of the numerical substrate. Current systems use bfloat16 format, approximately 2.4 significant decimal digits, a format created by, as Berry puts it, "literally chopping off the lower 16 bits" of standard floating point. This intensive property determines how finely the space can discriminate between closely related concepts. Vectorscope's Precision Degradation mode (forthcoming) will make this visible as signal loss.

Between these two properties sits the **manifold**: the thin surface of meaningful locations within the ambient vector space where learned semantic structure actually resides. The manifold does not pre-exist. It is constituted through training. Vectorscope's Manifold Formation mode shows how this surface emerges layer by layer as the model transforms input embeddings through its depth.

The critical stake is this: the manifold's geometry tends toward **vector conformism**, the structural tendency of vector spaces to blur contested meanings toward their statistical centre of mass in training data. Coarse grain creates a centripetal force dragging concepts away from boundaries and toward dense regions of the manifold, encoding dominant usage as geometric common sense. Vectorscope makes this tendency empirically inspectable rather than theoretically asserted.

## Operations

Vectorscope is organised as a three-group tab navigation following the pipeline described in Berry (2026a): tokens → vectors → manifold → vectors → dictionary.

### Inspect (component-level weight examination)

| Operation | Purpose | Status |
|-----------|---------|--------|
| **Embedding Table** | Load and visualise the input embedding matrix. PCA 3D scatter, norm histogram, effective rank, isotropy score | ✓ |
| **Projection Head** | Examine the lm_head / unembedding weights. 3D scatter, weight-tying detection, effective rank comparison | ✓ |
| **Weight Comparison** | Input embeddings vs output unembedding. Per-token cosine similarity distribution, norm scatter | ✓ |
| **Attention Inspector** | Multi-head attention heatmaps at any layer. Per-head entropy, pattern classification (focused/diagonal/diffuse) | ✓ |

### Trace (following data through the pipeline)

| Operation | Purpose | Status |
|-----------|---------|--------|
| **Token Trajectory** | Trace a token through all layers. 3D trajectory path, layer-to-layer cosine similarity, norm profile | ✓ |
| **Layer Probe** | Hidden states at a specific layer. Token similarity heatmap, norm bars, full similarity matrix | ✓ |
| **Full Trace** | Complete tokens → embeddings → layers → predictions pipeline. NDJSON streaming with progress, per-token norm heatmap, top-K prediction chart, entropy | ✓ |
| **Generation Vector** | Full autoregressive generation with instrumented forward passes. Six horizontal panels (tokenisation, input embedding, attention, layer progression, output distribution, decoded text), scrubber playback, click-to-focus across panels, global PCA 3D trajectory per token. Tokenisation panel shows running ‖Σv‖ cumulative sum vs √t independence baseline; rich focused-token detail card with bytes/codepoints, per-layer geometry, and sampling detail | ✓ |
| **Manifold Formation** | Animated layer-by-layer PCA geometry with play/pause controls. Shows how the manifold forms through depth | ✓ |

### Critique (theoretical/analytical instruments)

| Operation | Purpose | Status |
|-----------|---------|--------|
| **Vocabulary Map** | Global topology of the vocabulary embedding space. 3D scatter with token search and highlight, norm distribution | ✓ |
| **Isotropy Analysis** | Effective dimensionality, direction concentration, anisotropy metrics. Connects to vector conformism | Phase 4 |
| **Cross-Model Anatomy** | Compare internal geometry across models. Whether different training corpora produce different geometric rationalities | Phase 4 |
| **Precision Degradation** | Signal Degradation Laboratory: same input through FP32 → BF16 → INT8 → INT4. Implements the Leverhulme method | Phase 4 |

All operations include a collapsible **Deep Dive** panel with detailed quantitative data for researchers who want to inspect the numbers directly. 3D plots support **Shift+scroll** for fast zoom.

Most operations with user inputs also expose a **preset chip row** under the input controls. Presets are theoretically-motivated: contested concepts (justice, capital, woman, labour), subject-verb agreement traps, bias probes, Berry's own formulations ("Capital is disciplining bits into vectors"), and sampling-config bundles for Generation Vector (greedy, low-temp, high-temp). Clicking a chip fills the inputs; you still click **Run** yourself, because some operations are expensive and should not fire on an accidental click.

**Keyboard shortcuts (Generation Vector):** `←` `→` step through panels or, on the Tokenisation panel, step chip by chip; `↑` `↓` jump between rows of chips. Focus syncs across all six panels so arrow-keying through the prompt drives the entire operation. `Esc` closes any modal.

## Supported Models

| Model | Size | Min Memory |
|-------|------|-----------|
| GPT-2 | 124M | 8 GB |
| Qwen3 0.6B | 600M | 16 GB |
| Llama 3.2 1B | 1.2B | 16 GB |
| Llama 3.2 3B | 3B | 16 GB |
| Mistral 7B | 7B | 24 GB |

Any HuggingFace `AutoModelForCausalLM`-compatible model can be loaded. Primary development target: Apple Silicon Mac with 24 GB unified memory.

## Design Rationale

**Why open-weight models only?** Commercial APIs do not expose internal representations. You cannot inspect the embedding table, trace hidden states through layers, or examine attention patterns through an API endpoint. The internal geometry that Vectorscope makes visible simply does not exist in API-mediated access. This is not a limitation but a methodological commitment.

**Why not mechanistic interpretability tools?** TransformerLens, Baukit, and similar tools are designed for alignment researchers investigating circuits, features, and causal interventions. They answer engineering questions: which neurons fire for this concept? What circuit implements this behaviour? Vectorscope answers different questions: What is the shape of the vocabulary space? How does the manifold transform through depth? Where does the geometry compress contested distinctions? These are questions about the *medium*, not the mechanism.

**Why Plotly for 3D visualisation?** The internal geometry of a 768-dimensional space projected to three dimensions is inherently approximate. Plotly's interactive 3D scatter plots allow researchers to rotate, zoom, and inspect the projection from multiple angles, preventing any single viewpoint from naturalising itself as *the* way to see the geometry. The same anti-naturalisation principle that motivates LLMbench's multiple probability visualisations applies here.

**Why the editorial design system?** Vectorscope shares its visual language with Manifold Atlas and LLMbench: ivory backgrounds, serif headings, burgundy accents, compact monospaced data tables. This is deliberate. The editorial aesthetic creates productive distance from the dashboard conventions of engineering tools, signalling that this is an instrument for scholarly inquiry rather than model optimisation.

## Getting Started

Vectorscope is two coordinated processes: a Python backend that loads the model and runs the forward passes, and a Next.js frontend that renders the interface. Both need to be running at the same time, in two separate terminals. The sections below walk through a first-time setup from a freshly cloned repository.

### Prerequisites

Before you begin, install the following. All are free and widely packaged:

- **Python 3.9 or newer** (3.11 recommended). Check with `python3 --version`. On macOS, `brew install python@3.11`; on Ubuntu/Debian, `sudo apt install python3 python3-venv python3-pip`; on Windows, use [python.org](https://www.python.org/downloads/) and tick "Add Python to PATH".
- **Node.js 18 or newer** (20 LTS recommended) and **npm**. Check with `node --version` and `npm --version`. Install via [nodejs.org](https://nodejs.org/) or a version manager such as [nvm](https://github.com/nvm-sh/nvm) / [fnm](https://github.com/Schniz/fnm).
- **Git**, to clone the repository.
- **Disk space for model weights.** HuggingFace caches models to `~/.cache/huggingface/`. GPT-2 is ~500 MB, Llama 3.2 1B is ~2.5 GB, Mistral 7B is ~14 GB. Budget accordingly.
- **Memory.** 8 GB RAM is enough for GPT-2. For 1B–3B models you want 16 GB. For 7B models, 24 GB unified memory (Apple Silicon) or equivalent.

**Optional but recommended on Apple Silicon:** PyTorch's MPS backend is used automatically when available, which makes forward passes several times faster than CPU. No extra install step is needed, but macOS 13.3+ is required for MPS to work reliably.

**Optional:** a [HuggingFace account](https://huggingface.co/join) and access token. Some models (Llama, Mistral) are gated and require you to accept a licence on the model page and authenticate with `huggingface-cli login` before downloading. GPT-2 and Qwen3 are ungated.

### 1. Clone the repository

```bash
git clone https://github.com/dmberry/vectorscope.git
cd vectorscope
ls
```

The `ls` is a sanity check. You should see (among other things) a `backend/` folder, a `src/` folder, a `package.json`, and a `README.md`. If you do, you are in the project root and the rest of the instructions will work. If you see nothing, or a single folder called `vectorscope`, you are one level too high — run `cd vectorscope` again.

Throughout the rest of this guide, **"the project root"** means the directory this `ls` just ran in. Whenever an instruction says "from the project root", you can confirm you are there by running `ls` and checking for `backend/` and `package.json` side by side.

### 2. Set up and start the Python backend

From the project root (see step 1 for how to confirm), run:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

After `cd backend` you can double-check with `ls` — you should see `main.py`, `requirements.txt`, and folders `models/`, `operations/`.

The first `pip install` pulls down PyTorch, Transformers, FastAPI, scikit-learn and friends. On a fast connection this takes 2–5 minutes; on a slow one, considerably more. Subsequent runs reuse the virtual environment instantly.

When the backend is ready you will see something like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Leave this terminal running. You can sanity-check the backend by visiting [http://localhost:8000/api/health](http://localhost:8000/api/health) in a browser; it should return a small JSON payload.

**To stop the backend:** press `Ctrl+C` in the terminal. **To resume later:** `cd backend && source .venv/bin/activate && python main.py`.

### 3. Install and start the frontend

Leave the backend terminal running. Open a *second* terminal and navigate to the project root — **not** the `backend` subdirectory, the top-level `vectorscope` directory where `package.json` lives:

```bash
cd /path/to/vectorscope        # adjust to wherever you cloned it
ls package.json                 # should print: package.json
npm install
npm run dev
```

The `ls package.json` is another sanity check: if it errors with "No such file or directory", you are in the wrong place (most likely still inside `backend/`). `cd ..` once and try again.

The first `npm install` fetches Next.js, React, Plotly, Three.js and the rest of the frontend tree (~1–2 GB of `node_modules`, 1–3 minutes on a fast connection). Subsequent runs are instant.

When the dev server is ready:

```
▲ Next.js 16.2.3
- Local:   http://localhost:3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser. If the header status indicator is green, the frontend has found the backend.

### 4. Load a model and start exploring

In the header, open the model selector and choose a preset (start with **GPT-2 124M**, the smallest and fastest) or paste any HuggingFace `AutoModelForCausalLM`-compatible model ID. Click **Load**.

The first time you load a given model the backend downloads the weights to `~/.cache/huggingface/`. GPT-2 takes about thirty seconds on a fast connection. Larger models take longer. Subsequent loads are near-instant because the weights are cached.

Once the status bar shows the model is loaded, pick an operation from the **Inspect / Trace / Critique** tab groups. Every operation has a collapsible introduction at the top with a *Learn more* modal, and a *Deep Dive* panel at the bottom with the full quantitative data. Try **Embedding Table** first — it is the cheapest operation and a good smoke test.

### Troubleshooting

- **`command not found: python3`** — install Python from the prerequisites above, or use `python` if your system uses that name.
- **`ModuleNotFoundError: No module named 'torch'`** — the virtual environment is not activated. Re-run `source backend/.venv/bin/activate`.
- **`EADDRINUSE: port 3000 already in use`** — another Next.js dev server is running. Either stop it, or start this one on a different port with `PORT=3001 npm run dev`.
- **Frontend says "Backend unreachable"** — the backend process is not running, or it started on a non-default port. Confirm [http://localhost:8000/api/health](http://localhost:8000/api/health) returns JSON.
- **Out of memory loading a large model** — pick a smaller preset. GPT-2 runs comfortably in 8 GB; anything above 1B parameters wants 16 GB or more. The backend uses fp16/bf16 where the hardware supports it, but there is no quantisation yet.
- **Llama / Mistral 403 on download** — these are gated. Accept the licence on the HuggingFace model page, then `pip install huggingface_hub && huggingface-cli login` inside the activated backend venv.
- **Slow generation on Apple Silicon** — confirm MPS is in use. The backend logs the device at startup. If it reports `cpu`, your PyTorch build is CPU-only; reinstall with the official wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### Packaging and distribution (notes for future work)

Vectorscope is currently distributed as source, run in development mode. This is the right default for a research instrument where users may want to patch the Python or the React side mid-session. A few packaging options are on the roadmap for users who would prefer a one-click install:

- **Desktop app via Tauri or Electron.** Bundle the frontend build and spawn the Python backend as a sidecar process. Tauri produces much smaller binaries than Electron and is the likely choice. A single `.dmg`, `.msi`, or `.AppImage` that launches both halves would remove most of the friction above.
- **`pipx` / `uvx` installable CLI.** Publish the backend as a PyPI package (`pip install vectorscope` or `uvx vectorscope`) that ships a pre-built static frontend as package data and serves it from FastAPI on a single port. The whole thing becomes `vectorscope` on the command line.
- **Docker Compose.** A two-service compose file (`backend`, `frontend`) with a pre-configured HuggingFace cache volume. Useful for reproducible lab setups and for running on a workstation from a laptop over the network.
- **Pre-built frontend served by the backend.** The simplest intermediate step: `npm run build` once, commit or ship the `.next` output, and have FastAPI serve it as static files. Reduces the "two terminals" requirement to one without introducing a new packaging technology.
- **Model bundle presets.** An opt-in first-run downloader that fetches GPT-2 to the HuggingFace cache so new users have something to click on immediately, rather than waiting for the first model load.

None of this is implemented yet. Contributions welcome — open an issue on GitHub if you have preferences.

## Architecture

```
backend/                          # Python FastAPI
  main.py                         # FastAPI app, CORS, routes
  models/
    session.py                    # Loaded model state (singleton)
    loader.py                     # Model download, cache, loading
    registry.py                   # Supported model specs
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
src/                              # Next.js frontend
  app/
    page.tsx                      # Mode-switching shell
    layout.tsx, globals.css       # Editorial design system
  components/
    layout/                       # Header, TabNav, StatusBar, ModelLoader
    operations/                   # One component per operation
    Plot3DWrapper.tsx             # Shift+scroll fast zoom for 3D plots
  context/
    ModelContext.tsx               # Loaded model state, backend polling
  lib/
    geometry/                     # Client-side PCA, cosine similarity
    version.ts                    # Single-source version from package.json
  types/
    model.ts                      # TypeScript types
```

The architecture follows the same pattern as LLMbench and Manifold Atlas: a thin `page.tsx` manages mode state and conditionally renders standalone operation components. Each operation dispatches to its own backend endpoint via `asyncio.to_thread` for heavy computation. The model session is a Python singleton holding one loaded model in memory across requests.

## Tech Stack

- **Next.js 16** with React 19 and App Router
- **Python 3.9+** with FastAPI, PyTorch, HuggingFace Transformers
- **Plotly.js** via `react-plotly.js` for 3D scatter plots, heatmaps, histograms
- **Tailwind CSS v3** with the Vector Lab editorial colour palette
- **scikit-learn** for server-side PCA on large tensors
- **Lucide React** for icons

## Vector Lab

Vectorscope is part of the **Vector Lab**, a family of research instruments for the critical study of AI vector media:

- [Manifold Atlas](https://github.com/dmberry/manifold-atlas) — Comparative geometry of AI vector spaces. Looks *between* models at their output embeddings, treating models as black boxes that emit vectors into metric space.
- [LLMbench](https://github.com/dmberry/LLMbench) — Comparative close reading of LLM outputs. Subjects AI-generated text to hermeneutic scrutiny through annotation, overlay views, and stochastic analysis.
- **Vectorscope** — Internal geometry of open-weight models. Looks *inside* a single model at its weights, activations, and learned geometry.

Theoretically, the three tools map onto different levels of Berry's framework. Manifold Atlas operates at the level of the output embedding, the projected metric space, what the retrieval economy sees. LLMbench operates at the level of the generated text, the linguistic surface that users encounter. Vectorscope operates at the level of the vector space itself, the material substrate, the grain and dimensionality analysed in "What Is Vector Space?" One shows you the map. One reads the text. The other shows you the terrain.

## Theoretical Context

Vectorscope connects to a critical theory of vector space and AI media developed across several publications:

- Berry, D. M. (2026a) 'What is Vector Space?', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/03/what-is-vector-space.html
- Berry, D. M. (2026b) 'Brain Numbers', *Stunlaw*. Available at: https://stunlaw.blogspot.com/2026/03/brain-numbers.html
- Berry, D. M. (2026c) *Artificial Intelligence and Critical Theory*. Manchester University Press.
- Berry, D. M. (2025) 'Synthetic Media and Computational Capitalism: Towards a Critical Theory of Artificial Intelligence', *AI & Society*. Available at: https://doi.org/10.1007/s00146-025-02265-2
- Berry, D. M. (2014) *Critical Theory and the Digital*. Bloomsbury.
- Impett, L. and Offert, F. (2026) *Vector Media*. University of Minnesota Press.

## Roadmap

- [ ] Isotropy Analysis (effective dimensionality, anisotropy metrics, direction concentration)
- [ ] Cross-Model Anatomy (two-model comparison of internal geometry)
- [ ] Precision Degradation / Signal Degradation Laboratory (FP32 → BF16 → INT8 → INT4)
- [ ] Annotation system for marking interesting geometric features
- [ ] Base model vs embedding model comparison (generative vs embedding versions of same backbone)
- [ ] Persistent homology on internal representations
- [ ] GGUF / llama.cpp bindings for quantised models
- [ ] Export system (JSON, CSV, PNG, PDF)

## Licence

MIT
