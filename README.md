# Vectorscope

**Internal geometry of open-weight models.**

**Author:** David M. Berry
**Institution:** University of Sussex
**Version:** 0.2.3
**Date:** 16 April 2026
**Licence:** MIT

Vectorscope is a research instrument for examining the internal geometry of open-weight language models. It inspects embedding tables, traces token representations through transformer layers, visualises attention patterns, and analyses the material grain of learned geometry. Unlike commercial embedding APIs, which return sentence-level composite vectors from separately trained embedding models, Vectorscope works directly with open-weight models where the weights, activations, and token embeddings can be inspected at each layer.

The tool is designed for critical and empirical inquiry into the internal structure of language models, not engineering evaluation. Where existing model inspection tools (BertViz, TransformerLens, Baukit) focus on mechanistic interpretability for alignment research, Vectorscope treats the vector space as a *medium* to be examined: its grain, its geometry, its ideological affordances.

## What is Vectorscope?

Vectorscope takes its name from the video engineering instrument that measures the colour properties of a signal by displaying them as vectors on a circular plot. This tool measures the properties of a vector representation by displaying its internal geometry.

The motivation is both methodological and theoretical.

**Methodologically**, Vectorscope responds to a problem identified by Michael Castelle: that commercial embedding APIs do not give researchers what they think they give them. Commercial APIs return sentence-level composite vectors from a separately trained embedding model, not the internal representations of the generative model itself. The outputs are doubly mediated: first by the embedding model's own training objective, then by the API's post-processing. Open-weight models with accessible architectures are the methodologically stronger route for studying how language models organise meaning. Vectorscope is the practical implementation of that critique.

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
| **Manifold Formation** | Animated layer-by-layer PCA geometry with play/pause controls. Shows how the manifold forms through depth | ✓ |

### Critique (theoretical/analytical instruments)

| Operation | Purpose | Status |
|-----------|---------|--------|
| **Vocabulary Map** | Global topology of the vocabulary embedding space. 3D scatter with token search and highlight, norm distribution | ✓ |
| **Isotropy Analysis** | Effective dimensionality, direction concentration, anisotropy metrics. Connects to vector conformism | Phase 4 |
| **Cross-Model Anatomy** | Compare internal geometry across models. Whether different training corpora produce different geometric rationalities | Phase 4 |
| **Precision Degradation** | Signal Degradation Laboratory: same input through FP32 → BF16 → INT8 → INT4. Implements the Leverhulme method | Phase 4 |

All operations include a collapsible **Deep Dive** panel with detailed quantitative data for researchers who want to inspect the numbers directly. 3D plots support **Shift+scroll** for fast zoom.

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

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm

### 1. Start the Python backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The backend runs on `localhost:8000`.

### 2. Start the frontend

```bash
npm install
npm run dev
```

The frontend runs on `localhost:3000`.

### 3. Load a model

Select a model from the preset list or enter any HuggingFace model ID. The first load will download the model weights to `~/.cache/huggingface/`.

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
- Impett, L. and Offert, F. (2026) on vector media from a media theory perspective.

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
