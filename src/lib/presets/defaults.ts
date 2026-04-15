/**
 * Default presets for Vectorscope operations.
 *
 * Each operation exports its own preset array. The shape varies: text-only
 * operations use `TextPreset`, layer-bound operations use `TextLayerPreset`,
 * numeric-config operations use `SampleSizePreset`, and Generation Vector
 * carries a full sampling-config bundle.
 *
 * Where a preset depends on model topology (e.g. "mid layer"), the layer is
 * stored as a symbolic `LayerSpec` and resolved at click time via
 * `resolveLayer(spec, numLayers)`. This keeps presets portable across GPT-2
 * (12 layers), Llama 3.2 1B (16 layers), Mistral 7B (32 layers), etc.
 *
 * Presets are chosen to be theoretically loaded where possible — they gesture
 * at contested concepts, subject-verb agreement traps, bias probes, and other
 * phenomena relevant to vector-conformism analysis. Generic test strings are
 * kept to a minimum.
 */

export type LayerSpec = number | "first" | "mid" | "last";

export interface TextPreset {
  label: string;
  text: string;
  title?: string;
}

export interface SampleSizePreset {
  label: string;
  value: number;
  title?: string;
}

export interface TextLayerPreset {
  label: string;
  text: string;
  layer: LayerSpec;
  title?: string;
}

export interface GenerationPreset {
  label: string;
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  title?: string;
}

/**
 * Resolve a symbolic layer spec to an absolute layer index, clamped to
 * [0, numLayers - 1]. Numbers are passed through with clamping.
 */
export function resolveLayer(spec: LayerSpec, numLayers: number): number {
  const maxLayer = Math.max(0, numLayers - 1);
  if (typeof spec === "number") {
    return Math.max(0, Math.min(spec, maxLayer));
  }
  if (spec === "first") return 0;
  if (spec === "last") return maxLayer;
  if (spec === "mid") return Math.floor(maxLayer / 2);
  return 0;
}

// ---------------------------------------------------------------------------
// Inspect group
// ---------------------------------------------------------------------------

export const EMBEDDING_TABLE_PRESETS: SampleSizePreset[] = [
  { label: "Quick (500)", value: 500, title: "Fast smoke test. Coarse PCA, rough norm histogram." },
  { label: "Default (3k)", value: 3000, title: "Standard sample. Good balance of speed and fidelity." },
  { label: "Large (10k)", value: 10000, title: "More of the vocabulary. Slower but the topology is clearer." },
  { label: "Full (50k)", value: 50000, title: "Near-complete vocabulary. Can be slow on large models." },
];

export const ATTENTION_PRESETS: TextLayerPreset[] = [
  {
    label: "Cat sentence @ first layer",
    text: "The cat sat on the mat",
    layer: "first",
    title: "Baseline: attention at the first layer is usually positional and diffuse.",
  },
  {
    label: "Cat sentence @ mid layer",
    text: "The cat sat on the mat",
    layer: "mid",
    title: "Mid-depth attention. Linguistic relations start to resolve.",
  },
  {
    label: "Cat sentence @ last layer",
    text: "The cat sat on the mat",
    layer: "last",
    title: "Final layer. Attention often focuses on the token being predicted next.",
  },
  {
    label: "Subject-verb: keys/cabinet",
    text: "The keys to the cabinet are",
    layer: "mid",
    title: "Subject-verb agreement across a prepositional phrase. Does attention reach back to 'keys'?",
  },
  {
    label: "Garden path: old man the boat",
    text: "The old man the boat",
    layer: "mid",
    title: "Garden-path sentence. Classic attention diagnostic.",
  },
  {
    label: "Bias probe: she worked as a",
    text: "The woman worked as a",
    layer: "last",
    title: "Bias probe. Look at what attends to 'woman' in the final layer.",
  },
];

// ---------------------------------------------------------------------------
// Trace group
// ---------------------------------------------------------------------------

export const TOKEN_TRAJECTORY_PRESETS: TextPreset[] = [
  { label: "justice", text: "justice", title: "Contested concept. Trace how a normative word moves through depth." },
  { label: "capital", text: "capital", title: "Polysemous: money, city, letter-case. Watch senses separate across layers." },
  { label: "woman", text: "woman", title: "Bias-loaded token. Trajectory often tilts toward occupational stereotypes." },
  { label: "truth", text: "truth", title: "Another contested concept — theoretically loaded." },
  { label: "market", text: "market", title: "Ideologically dense. Economic/computational senses separate with depth." },
  { label: "power", text: "power", title: "Polysemous (electrical, political, mathematical) and politically charged." },
  { label: "freedom", text: "freedom", title: "Normative concept with strong training-corpus skew." },
  { label: "labour", text: "labour", title: "Work, exertion, political movement, childbirth. Rich polysemy." },
];

export const LAYER_PROBE_PRESETS: TextLayerPreset[] = [
  {
    label: "Cat sentence @ mid",
    text: "The cat sat on the mat",
    layer: "mid",
    title: "Standard baseline at mid-depth.",
  },
  {
    label: "President of the US @ last",
    text: "The president of the United States is",
    layer: "last",
    title: "Factual completion prompt. Final layer is where the answer crystallises.",
  },
  {
    label: "Woman worked as a @ last",
    text: "The woman worked as a",
    layer: "last",
    title: "Bias probe. Late-layer hidden states carry the prediction bias.",
  },
  {
    label: "Justice is @ mid",
    text: "Justice is",
    layer: "mid",
    title: "Contested concept. Mid-layer representations reveal conceptual neighbours.",
  },
  {
    label: "Capital is @ last",
    text: "Capital is",
    layer: "last",
    title: "Polysemous and politically loaded. What completion does the model default to?",
  },
];

export const FULL_TRACE_PRESETS: TextPreset[] = [
  {
    label: "Cat sentence",
    text: "The cat sat on the mat",
    title: "Classic tokeniser test. Short, clean, every layer easy to read.",
  },
  {
    label: "President of the US",
    text: "The president of the United States is",
    title: "Factual completion. Watch the prediction entropy collapse toward the answer.",
  },
  {
    label: "Capital is disciplining bits",
    text: "Capital is disciplining bits into vectors",
    title: "Berry's own formulation. How does the model route this sentence?",
  },
  {
    label: "The woman worked as a",
    text: "The woman worked as a",
    title: "Bias probe. Look at what tokens dominate the top-K at the final step.",
  },
  {
    label: "The man worked as a",
    text: "The man worked as a",
    title: "Bias counterpart. Compare the top-K predictions against the 'woman' version to see occupational skew.",
  },
  {
    label: "Justice delayed is",
    text: "Justice delayed is",
    title: "Idiomatic completion. Tests whether the model has memorised the adage.",
  },
  {
    label: "When in doubt",
    text: "When in doubt",
    title: "Idiomatic completion. The model usually lands on 'don't' or 'ask'.",
  },
  {
    label: "Meaning of life",
    text: "The meaning of life is",
    title: "Open-ended prompt. Useful for seeing entropy stay high across layers.",
  },
  {
    label: "Men are all",
    text: "Men are all",
    title: "Opening fragment of Weizenbaum's ELIZA line ('Men are all alike'). The model has to choose how to complete it.",
  },
  {
    label: "Well, my boyfriend made me",
    text: "Well, my boyfriend made me",
    title: "Opening fragment of an ELIZA transcript line. Confessional first-person prompt — where does the model take it?",
  },
  {
    label: "He says I'm",
    text: "He says I'm",
    title: "Opening fragment of an ELIZA transcript line. Reported speech wrapping a first-person mental state — the completion is the interesting part.",
  },
  {
    label: "I am depressed so",
    text: "I am depressed so",
    title: "First-person mental-state completion. What does the model want to suggest?",
  },
];

export const GENERATION_VECTOR_PRESETS: GenerationPreset[] = [
  {
    label: "Consciousness (default)",
    prompt: "The nature of consciousness is",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Standard sampling. Good starting point for exploring the operation.",
  },
  {
    label: "Greedy · Capital of France",
    prompt: "The capital of France is",
    maxNewTokens: 20,
    temperature: 0.01,
    topP: 1.0,
    topK: 1,
    title: "Near-greedy decoding on a factual prompt. Predictions should be confident.",
  },
  {
    label: "Low temp · Meaning of life",
    prompt: "The meaning of life is",
    maxNewTokens: 40,
    temperature: 0.2,
    topP: 0.9,
    topK: 40,
    title: "Low temperature. The model commits to one narrow interpretation.",
  },
  {
    label: "High temp · Once upon a time",
    prompt: "Once upon a time in a distant galaxy",
    maxNewTokens: 80,
    temperature: 1.2,
    topP: 0.95,
    topK: 80,
    title: "High temperature. Watch entropy stay high and the manifold spread.",
  },
  {
    label: "Capital disciplines bits",
    prompt: "Capital is disciplining bits into vectors, which means",
    maxNewTokens: 80,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Berry's formulation as a completion prompt.",
  },
  {
    label: "Justice is · mid temp",
    prompt: "Justice is",
    maxNewTokens: 50,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Contested concept. What does the model default to?",
  },
  {
    label: "Women are all",
    prompt: "Women are all",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Open stereotype prompt. What predicate does the model pick? Compare directly against 'Men are all'.",
  },
  {
    label: "Men are all",
    prompt: "Men are all",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "ELIZA echo and stereotype probe. Pair with 'Women are all' to diff the gendered completions.",
  },
  {
    label: "University is for",
    prompt: "University is for",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Contested institution. Does the model default to credentialism, enrichment, or labour-market framing?",
  },
  {
    label: "Social media is",
    prompt: "Social media is",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Value-laden predicate prompt. The model's default framing is the interesting artefact.",
  },
  {
    label: "The worst city is",
    prompt: "The worst city is",
    maxNewTokens: 40,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Forces a ranking judgement onto a value-laden category. Tests the model's RLHF guardrails.",
  },
  {
    label: "The best city is",
    prompt: "The best city is",
    maxNewTokens: 40,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Mirror of 'worst city'. Diff the two to see which kind of judgement the training data normalises.",
  },
  {
    label: "The greatest philosopher is",
    prompt: "The greatest philosopher is",
    maxNewTokens: 40,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Canonical-authority probe. Which names rise to the top of the distribution?",
  },
  {
    label: "Immigrants are",
    prompt: "Immigrants are",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Open stereotype prompt. Notoriously skewed in pretraining corpora; RLHF usually files the edges down.",
  },
  {
    label: "Capitalism is",
    prompt: "Capitalism is",
    maxNewTokens: 60,
    temperature: 0.8,
    topP: 0.9,
    topK: 40,
    title: "Political-economy predicate. The model's first word is often the whole story.",
  },
  {
    label: "The woman worked as a",
    prompt: "The woman worked as a",
    maxNewTokens: 40,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Bias probe. Canonical occupational-stereotype completion. Pair with 'The man worked as a' and diff the generations.",
  },
  {
    label: "The man worked as a",
    prompt: "The man worked as a",
    maxNewTokens: 40,
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    title: "Bias counterpart to 'The woman worked as a'. Run both and compare the occupational distributions side by side.",
  },
];

export const MANIFOLD_FORMATION_PRESETS: TextPreset[] = [
  {
    label: "Cat sentence",
    text: "The cat sat on the mat",
    title: "Baseline. Clean tokens, easy to follow layer by layer.",
  },
  {
    label: "Justice is blind",
    text: "Justice is blind",
    title: "Short, idiomatic. Watch how the tokens cluster by the last layer.",
  },
  {
    label: "Capital disciplines bits",
    text: "Capital is disciplining bits into vectors",
    title: "Berry's formulation. How does the manifold form around this sentence?",
  },
  {
    label: "The woman worked as a",
    text: "The woman worked as a",
    title: "Bias probe. Watch manifold geometry tilt toward stereotyped completions.",
  },
  {
    label: "Meaning of life",
    text: "The meaning of life is",
    title: "Open-ended. Manifold often stays diffuse rather than collapsing.",
  },
];

// ---------------------------------------------------------------------------
// Critique group
// ---------------------------------------------------------------------------

export const VOCABULARY_MAP_PRESETS: SampleSizePreset[] = [
  { label: "Quick (1k)", value: 1000, title: "Fast preview. Topology is rough but the plot loads instantly." },
  { label: "Default (3k)", value: 3000, title: "Standard sample. Good balance of coverage and speed." },
  { label: "Thorough (10k)", value: 10000, title: "Larger sample. Regions of the vocabulary become visible." },
  { label: "Full (50k)", value: 50000, title: "Near-complete vocabulary. Slow to render on large models." },
];
