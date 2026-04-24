"""
Grammar Steering — Phase 1: contrastive activation vector extraction.

Theoretical background
----------------------
Contrastive Activation Addition (CAA), from Turner et al. (2023) and Rimsky
et al. (2024). The idea: if a model has learned a stylistic regularity — say
the "Not X but Y" rhetorical antithesis pattern — that regularity should
show up as a distinguishable direction in activation space. Extract that
direction by subtracting the mean activation of examples that lack the
pattern from the mean activation of examples that carry it. The result is
a per-layer *steering vector*. Phase 2 (separate session) will use this
vector for forward-pass intervention. Phase 1 is the geometric finding: the
pattern has an identifiable signature, and we can locate it by layer.

Method
------
The operation takes a list of contrastive prompt pairs — positive (carries
the pattern) and negative (same topic, register, length; no pattern). For
each text, we run one forward pass with `output_hidden_states=True` and
take the *last-token* hidden state at every layer as the representation of
that prompt. This is standard in the CAA literature: the last token
functions as a summary of everything that precedes it.

For each layer ℓ we then compute:

- **Steering vector** = mean(positive_ℓ) − mean(negative_ℓ). Shape
  (hidden_dim,). This is the direction that, added to activations, pushes
  the model toward the pattern; subtracted, away from it.
- **Separability score.** Leave-one-out accuracy of a nearest-centroid
  classifier along the steering direction. For each held-out pair we
  rebuild centroids from the remaining (N−1) pairs and ask whether the
  held-out positive / negative lands nearer its own centroid. This is the
  honest version of "how cleanly does the pattern carve the space?".
- **Norm trajectory.** ‖steering_vector_ℓ‖ across layers tells us at which
  depth the pattern emerges. CAA typically peaks in the middle layers.
- **PCA projection** of all 2N hidden states at a chosen layer into 3D,
  for the scatter plot that makes the separation visible.

Cost and limits
---------------
We run one forward pass per text, serial (not batched). For typical use
(20–50 pairs, small prompts) that's 40–100 forwards. GPT-2 and Llama 3.2
3B complete in a few seconds on MPS. Larger models will be slower but
still fine. Memory is not a concern — we only store per-layer last-token
vectors, not full activations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.session import session


def _last_token_hidden_states(text: str, model, tokenizer) -> np.ndarray:
    """Run one forward pass and return per-layer last-token hidden states.

    Shape: (num_layers, hidden_dim), float32, on CPU.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states is a tuple of (1, seq_len, hidden_dim) per layer
    # including the embedding layer. Stack into (num_layers, hidden_dim) by
    # taking the last token.
    rows = [h[0, -1].detach().float().cpu().numpy() for h in outputs.hidden_states]
    return np.stack(rows, axis=0)


def _pca_3d(points: np.ndarray) -> Dict[str, Any]:
    """Project `points` (N, D) to 3D via SVD. Returns coords + variance explained."""
    n, d = points.shape
    if n == 0 or d == 0:
        return {"coords": [], "explainedVariance": [0.0, 0.0, 0.0]}
    centred = points - points.mean(axis=0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"coords": [[0.0, 0.0, 0.0] for _ in range(n)], "explainedVariance": [0.0, 0.0, 0.0]}
    total_var = float((s ** 2).sum()) + 1e-12
    k = min(3, len(s))
    coords = u[:, :k] * s[:k]
    if k < 3:
        # Pad with zero columns
        coords = np.concatenate([coords, np.zeros((n, 3 - k))], axis=1)
    ratios = [float((s[i] ** 2) / total_var) if i < len(s) else 0.0 for i in range(3)]
    return {"coords": coords.tolist(), "explainedVariance": ratios}


def _leave_one_out_separability(
    pos: np.ndarray, neg: np.ndarray
) -> float:
    """Leave-one-out nearest-centroid accuracy along the (pos_mean - neg_mean) axis.

    pos, neg: (N, D). Returns accuracy in [0, 1].
    """
    n = pos.shape[0]
    if n < 2:
        return 0.5  # not enough to hold out
    correct = 0
    total = 0
    for i in range(n):
        pos_others = np.delete(pos, i, axis=0)
        neg_others = np.delete(neg, i, axis=0)
        mu_p = pos_others.mean(axis=0)
        mu_n = neg_others.mean(axis=0)
        direction = mu_p - mu_n
        if np.linalg.norm(direction) == 0:
            continue
        midpoint = 0.5 * (mu_p + mu_n)
        # Project held-out onto direction, check which side of midpoint
        for (ex, truth) in [(pos[i], 1), (neg[i], 0)]:
            projection = float((ex - midpoint) @ direction)
            pred = 1 if projection > 0 else 0
            correct += int(pred == truth)
            total += 1
    return correct / total if total else 0.5


def compute_grammar_steering(
    pairs: List[Dict[str, str]],
    pca_layer: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract per-layer steering vectors from contrastive prompt pairs.

    Arguments
    ---------
    pairs : list of {"positive": str, "negative": str}
        Each entry is a matched pair: the positive text carries the pattern
        under study, the negative is the same topic / register without it.
    pca_layer : int, optional
        Which layer's representations to project to 3D for visualisation.
        Defaults to the middle layer.

    Returns
    -------
    dict with keys:
        numLayers, hiddenSize, numPairs, pcaLayer,
        layers: [{layer, steeringNorm, positiveMeanNorm, negativeMeanNorm,
                  separability, steeringVector}],
        pca3d: {layer, positive: [[x,y,z]...], negative: [[x,y,z]...],
                explainedVariance: [r1, r2, r3]},
        pairs: the echoed input list (so the frontend can show them back)
    """
    if session.model is None or session.tokenizer is None:
        raise ValueError("No model loaded")

    # Validate input
    cleaned: List[Dict[str, str]] = []
    for i, p in enumerate(pairs or []):
        if not isinstance(p, dict):
            continue
        pos = (p.get("positive") or "").strip()
        neg = (p.get("negative") or "").strip()
        if not pos or not neg:
            continue
        cleaned.append({"positive": pos, "negative": neg})
    if len(cleaned) < 2:
        raise ValueError(
            f"Need at least 2 valid (positive, negative) pairs. Got {len(cleaned)}."
        )

    model = session.model
    tokenizer = session.tokenizer

    # Collect last-token hidden states: two (N, num_layers, hidden_dim) tensors
    pos_mat: List[np.ndarray] = []
    neg_mat: List[np.ndarray] = []
    for pair in cleaned:
        pos_mat.append(_last_token_hidden_states(pair["positive"], model, tokenizer))
        neg_mat.append(_last_token_hidden_states(pair["negative"], model, tokenizer))

    pos_arr = np.stack(pos_mat, axis=0)  # (N, L, D)
    neg_arr = np.stack(neg_mat, axis=0)
    num_pairs, num_layers, hidden_dim = pos_arr.shape

    if pca_layer is None:
        pca_layer = num_layers // 2
    pca_layer = max(0, min(num_layers - 1, int(pca_layer)))

    # Per-layer diagnostics
    layers_out: List[Dict[str, Any]] = []
    for l in range(num_layers):
        p = pos_arr[:, l, :]  # (N, D)
        n = neg_arr[:, l, :]
        mu_p = p.mean(axis=0)
        mu_n = n.mean(axis=0)
        steering = mu_p - mu_n  # (D,)
        layers_out.append(
            {
                "layer": l,
                "steeringNorm": float(np.linalg.norm(steering)),
                "positiveMeanNorm": float(np.linalg.norm(p, axis=1).mean()),
                "negativeMeanNorm": float(np.linalg.norm(n, axis=1).mean()),
                "separability": _leave_one_out_separability(p, n),
                "steeringVector": steering.astype(np.float32).tolist(),
            }
        )

    # PCA on the chosen layer
    p_pca = pos_arr[:, pca_layer, :]
    n_pca = neg_arr[:, pca_layer, :]
    combined = np.concatenate([p_pca, n_pca], axis=0)
    pca = _pca_3d(combined)
    pos_coords = pca["coords"][:num_pairs]
    neg_coords = pca["coords"][num_pairs:]

    return {
        "numLayers": num_layers,
        "hiddenSize": hidden_dim,
        "numPairs": num_pairs,
        "pcaLayer": pca_layer,
        "layers": layers_out,
        "pca3d": {
            "layer": pca_layer,
            "positive": pos_coords,
            "negative": neg_coords,
            "explainedVariance": pca["explainedVariance"],
        },
        "pairs": cleaned,
    }
