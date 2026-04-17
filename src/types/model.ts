/**
 * Types for model loading and backend communication.
 */

export interface ModelInfo {
  modelId: string;
  name: string;
  architecture: string;
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  numAttentionHeads: number;
  weightTied: boolean;
  dtype: string;
  nativeDtype: string;
  sizeBytes: number;
  device: string;
}

export interface BackendStatus {
  status: "connected" | "disconnected" | "loading";
  model: ModelInfo | null;
  device: string;
  availableMemoryMb: number;
}

export interface EmbeddingTableResult {
  shape: [number, number]; // [vocab_size, hidden_dim]
  stats: {
    meanNorm: number;
    stdNorm: number;
    minNorm: number;
    maxNorm: number;
    effectiveRank: number;
    isotropyScore: number;
  };
  sampleTokens: string[];
  sampleCoords3d: [number, number, number][];
  sampleNorms: number[];
}

export interface TokenTrajectoryResult {
  inputText: string;
  tokenIds: number[];
  tokens: string[];
  layers: Array<{
    layer: number;
    vectors: number[][]; // one per token position
    norms: number[];
  }>;
  layerSimilarities: number[]; // cosine sim between consecutive layers
  topPredictions: Array<{
    token: string;
    logit: number;
    probability: number;
  }>;
}

export interface LayerProbeResult {
  inputText: string;
  layer: number;
  tokens: string[];
  hiddenStates: number[][]; // [num_tokens x hidden_dim]
  tokenSimilarities: number[][]; // pairwise cosine between token positions
  norms: number[];
}

export interface IsotropyLayerStats {
  layer: number;
  isotropyScore: number;
  meanCos: number;
  meanAbsCos: number;
  top1VarianceRatio: number;
  top3VarianceRatio: number;
  top10VarianceRatio: number;
  meanNorm: number;
  stdNorm: number;
  sampleSize: number;
}

export interface IsotropyHistogram {
  layer: number;
  edges: number[]; // length bins + 1
  counts: number[]; // length bins
  mean: number;
  std: number;
}

export interface IsotropyResult {
  inputText: string;
  tokens: string[];
  tokenIds: number[];
  numLayers: number;
  hiddenSize: number;
  layers: IsotropyLayerStats[];
  cosineHistograms: {
    first: IsotropyHistogram;
    middle: IsotropyHistogram;
    last: IsotropyHistogram;
  };
}

export interface PrecisionLayerMetrics {
  layer: number;
  mse: number;
  relError: number;
  meanCosine: number;
  minCosine: number;
}

export interface PrecisionOutputMetrics {
  klDivergence: number;
  argmaxMatch: boolean;
  argmaxRef: number;
  argmaxQuant: number;
  topKOverlap: number;
  entropyRef: number;
  entropyQuant: number;
}

export interface PrecisionPrediction {
  tokenId: number;
  token: string;
  prob: number;
}

export interface PrecisionSweepEntry {
  precision: string; // e.g. "int8"
  label: string;     // short display label, e.g. "int8"
  bits: number;
  kind: "float" | "int";
  layers: PrecisionLayerMetrics[];
  output: PrecisionOutputMetrics;
  topPredictions: PrecisionPrediction[];
}

export interface PrecisionDegradationResult {
  inputText: string;
  tokens: string[];
  tokenIds: number[];
  numLayers: number;
  hiddenSize: number;
  baselineDtype: string;
  modelSizeBytes: number;
  memoryWarning: boolean;
  precisions: PrecisionSweepEntry[];
  baselineTopPredictions: PrecisionPrediction[];
}

export interface CachedRepo {
  repoId: string;
  sizeBytes: number;
  lastAccessed: number | null;
  lastModified: number | null;
  nbFiles: number;
  refs: string[];
}

export interface CacheInfo {
  cachePath: string;
  totalSizeBytes: number;
  repoCount: number;
  repos: CachedRepo[];
}

export type TabGroup = "inspect" | "trace" | "critique";

export interface TabDef {
  id: string;
  label: string;
  group: TabGroup;
}

export const TAB_GROUPS: Record<TabGroup, { label: string; tabs: TabDef[] }> = {
  inspect: {
    label: "Inspect",
    tabs: [
      { id: "embedding-table", label: "Embedding Table", group: "inspect" },
      { id: "projection-head", label: "Projection Head", group: "inspect" },
      { id: "weight-comparison", label: "Weight Comparison", group: "inspect" },
      { id: "attention", label: "Attention Inspector", group: "inspect" },
    ],
  },
  trace: {
    label: "Trace",
    tabs: [
      { id: "token-trajectory", label: "Token Trajectory", group: "trace" },
      { id: "layer-probe", label: "Layer Probe", group: "trace" },
      { id: "full-trace", label: "Full Trace", group: "trace" },
      { id: "generation-vector", label: "Generation Vector", group: "trace" },
      { id: "manifold-formation", label: "Manifold Formation", group: "trace" },
    ],
  },
  critique: {
    label: "Critique",
    tabs: [
      { id: "vocabulary-map", label: "Vocabulary Map", group: "critique" },
      { id: "isotropy", label: "Isotropy Analysis", group: "critique" },
      { id: "precision-degradation", label: "Precision Degradation", group: "critique" },
    ],
  },
};
