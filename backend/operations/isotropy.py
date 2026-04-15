"""
Isotropy Analysis: trace the anisotropy signature of hidden states through depth.

Theoretical background
----------------------
Mu and Viswanath (2018) showed that pretrained word embeddings are heavily
concentrated in a narrow cone of the representation space: a handful of
directions account for most of the variance, and the mean vector is large
relative to individual embedding norms. Ethayarajh (2019) extended this to
contextual representations and showed the anisotropy gets worse as you go up
the layer stack: by the final layer, any two random token representations have
an expected cosine similarity close to 1.0.

This operation makes those results visible. The user supplies a text; we run
it through the model with `output_hidden_states=True` and pool every token's
hidden state at every layer into a per-layer sample. Then for each layer we
compute:

  - isotropy_score: 1 - |mean cosine similarity| between random pairs.
    This is the standard "how uniform is this set on the sphere" statistic.
    1.0 is perfect isotropy, 0.0 is total alignment.
  - mean_abs_cos: the raw expected pairwise cosine. Ethayarajh's diagnostic.
  - top1_variance_ratio, top3_variance_ratio, top10_variance_ratio: the
    fraction of total variance explained by the first 1/3/10 principal
    components. If a single PC eats half the variance, the representation
    has collapsed onto a common direction.
  - mean_norm, std_norm: norm distribution summary.

We also capture the pairwise cosine histogram at the final layer so the
frontend can plot the full distribution, not just the mean. A narrow spike
near 1.0 is the visual signature of a collapsed manifold.
"""

import torch
import numpy as np
from models.session import session


def _compute_layer_stats(hs: torch.Tensor) -> dict:
    """
    Compute isotropy stats for a single layer's pool of hidden states.

    hs: shape (n_tokens, hidden_dim), float32, on CPU.
    """
    hs_np = hs.numpy()
    n, d = hs_np.shape

    # Norm stats
    norms = np.linalg.norm(hs_np, axis=1)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())

    # Normalise for cosine
    eps = 1e-10
    hs_normed = hs_np / (norms[:, None] + eps)

    # Pairwise cosine similarity, excluding the diagonal.
    # For modest n (say < 2000) we can afford the full n x n matrix.
    if n <= 2000:
        cos_matrix = hs_normed @ hs_normed.T
        mask = ~np.eye(n, dtype=bool)
        off_diag = cos_matrix[mask]
    else:
        # Sample random pairs
        rng = np.random.default_rng(0)
        idx_a = rng.integers(0, n, size=20000)
        idx_b = rng.integers(0, n, size=20000)
        keep = idx_a != idx_b
        idx_a = idx_a[keep]
        idx_b = idx_b[keep]
        off_diag = np.sum(hs_normed[idx_a] * hs_normed[idx_b], axis=1)

    mean_cos = float(off_diag.mean())
    mean_abs_cos = float(np.abs(off_diag).mean())
    isotropy_score = float(1.0 - abs(mean_cos))

    # Principal component variance ratios.
    # We centre, SVD, and look at the singular values squared.
    centred = hs_np - hs_np.mean(axis=0, keepdims=True)
    # If n < d we still get min(n, d) singular values, which is enough.
    try:
        s = np.linalg.svd(centred, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.zeros(1)
    total_var = float((s ** 2).sum()) + eps
    top1_ratio = float((s[0] ** 2) / total_var) if len(s) >= 1 else 0.0
    top3_ratio = float((s[:3] ** 2).sum() / total_var) if len(s) >= 3 else float((s ** 2).sum() / total_var)
    top10_ratio = float((s[:10] ** 2).sum() / total_var) if len(s) >= 10 else float((s ** 2).sum() / total_var)

    return {
        "isotropyScore": isotropy_score,
        "meanCos": mean_cos,
        "meanAbsCos": mean_abs_cos,
        "top1VarianceRatio": top1_ratio,
        "top3VarianceRatio": top3_ratio,
        "top10VarianceRatio": top10_ratio,
        "meanNorm": mean_norm,
        "stdNorm": std_norm,
        "sampleSize": n,
    }


def _cosine_histogram(hs: torch.Tensor, bins: int = 40) -> dict:
    """Compute a pairwise-cosine histogram for a single layer."""
    hs_np = hs.numpy().astype(np.float32)
    n = hs_np.shape[0]
    norms = np.linalg.norm(hs_np, axis=1)
    hs_normed = hs_np / (norms[:, None] + 1e-10)

    if n <= 1500:
        cos_matrix = hs_normed @ hs_normed.T
        mask = ~np.eye(n, dtype=bool)
        values = cos_matrix[mask]
    else:
        rng = np.random.default_rng(0)
        idx_a = rng.integers(0, n, size=50000)
        idx_b = rng.integers(0, n, size=50000)
        keep = idx_a != idx_b
        values = np.sum(hs_normed[idx_a[keep]] * hs_normed[idx_b[keep]], axis=1)

    counts, edges = np.histogram(values, bins=bins, range=(-1.0, 1.0))
    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def get_isotropy_analysis(text: str) -> dict:
    """
    Run `text` through the model and compute per-layer isotropy diagnostics.

    Returns a dict shaped for the frontend IsotropyAnalysis component.
    """
    if session.model is None:
        raise ValueError("No model loaded")

    tokenizer = session.tokenizer
    model = session.model

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    n_tokens = len(token_ids)

    if n_tokens < 4:
        raise ValueError(
            f"Need at least 4 tokens for isotropy stats, got {n_tokens}. "
            "Try a longer prompt."
        )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
    num_layers = len(hidden_states)
    hidden_dim = hidden_states[0].shape[-1]

    layers_out = []
    for layer_idx, hs_tuple in enumerate(hidden_states):
        hs = hs_tuple[0].float().cpu()  # (seq_len, hidden_dim)
        stats = _compute_layer_stats(hs)
        stats["layer"] = layer_idx
        layers_out.append(stats)

    # Histogram at first, middle, and last layer so the frontend can show the
    # progression of the cosine distribution.
    middle_idx = num_layers // 2
    last_idx = num_layers - 1
    histograms = {
        "first": _cosine_histogram(hidden_states[0][0].float().cpu()),
        "middle": _cosine_histogram(hidden_states[middle_idx][0].float().cpu()),
        "last": _cosine_histogram(hidden_states[last_idx][0].float().cpu()),
    }
    histograms["first"]["layer"] = 0
    histograms["middle"]["layer"] = middle_idx
    histograms["last"]["layer"] = last_idx

    return {
        "inputText": text,
        "tokens": tokens,
        "tokenIds": token_ids,
        "numLayers": num_layers,
        "hiddenSize": hidden_dim,
        "layers": layers_out,
        "cosineHistograms": histograms,
    }
