"""
Precision Degradation: the Signal Degradation Laboratory operation.

Theoretical background
----------------------
The Leverhulme Centre for Vector Media bid frames quantisation as a critical
lens on the material substrate of vector media. Where industry treats
low-precision inference as an engineering optimisation (faster, cheaper, same
behaviour), a critical account asks: what signal does a compressed medium no
longer carry? Which distinctions does the geometry stop being able to make?
Precision Degradation is the operation that makes this empirically visible.

Method
------
The operation is an in-process fake-quantisation procedure, not a load of a
pre-quantised variant:

1. The currently-loaded model is treated as the precision baseline (bf16 for
   Qwen/Llama/Mistral, fp16 for GPT-2 downcast, fp32 on CPU). A single forward
   pass on the user's text captures ground-truth hidden states and logits.

2. For each user-selected target precision (bfloat16, float16, int8, int4,
   int2), every parameter tensor is round-tripped through that precision:
   cast down (or quantised with per-tensor symmetric scale) and cast back up
   to the baseline dtype. This simulates what the model would do at that
   precision without changing the compute path, so all observed differences
   come from the weights themselves losing grain.

3. A second forward pass at each precision captures the degraded hidden
   states and logits. Per-layer mean-squared error and cosine similarity
   against the baseline make the layer-by-layer compression visible.
   Output-level metrics (argmax agreement, KL divergence, top-K overlap)
   make the prediction-level consequences visible.

4. Parameters are restored from a CPU-resident snapshot between runs.

This is a deliberate scientific-control design. Compared to loading a
pre-quantised variant (GPTQ / AWQ / GGUF), we get cleaner causal attribution
(same weights, same tokens, only precision varies), Mac-portability (pure
PyTorch round-to-nearest, no vendor kernels), and critical transparency (the
quantisation function is visible code, not a hidden third-party kernel).

Cost and limits
---------------
The CPU snapshot doubles the model's memory footprint during the operation.
Comfortable for GPT-2 (~500 MB), Qwen3 0.6B (~1.2 GB), Llama 3.2 1B (~2.4 GB)
and 3B (~6.4 GB). Tight for Mistral 7B (~14 GB) on a 24 GB Mac; we surface a
warning in the API response when total model size exceeds 4 GB.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.session import session


PRECISION_SPECS: Dict[str, Dict[str, object]] = {
    "bfloat16": {"label": "bf16", "kind": "float", "dtype": torch.bfloat16, "bits": 16},
    "float16": {"label": "fp16", "kind": "float", "dtype": torch.float16, "bits": 16},
    "int8": {"label": "int8", "kind": "int", "bits": 8},
    "int4": {"label": "int4", "kind": "int", "bits": 4},
    "int2": {"label": "int2", "kind": "int", "bits": 2},
}

SUPPORTED_PRECISIONS = list(PRECISION_SPECS.keys())


# ---------------------------------------------------------------------------
# Fake-quantisation kernels
# ---------------------------------------------------------------------------


def _fake_quant_float(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Round-trip a tensor through a lower-precision float dtype and back."""
    orig = tensor.dtype
    return tensor.to(target_dtype).to(orig)


def _fake_quant_int(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Per-tensor symmetric integer quantisation with dequantisation.

    No calibration, no per-channel scales — the point is a visible, simple
    baseline that shows what happens when the precision floor drops, not to
    match what GPTQ/AWQ would do in production. Those more sophisticated
    schemes are themselves objects of critical study and should be added as
    separate named strategies rather than hidden in this path.
    """
    orig_dtype = tensor.dtype
    # Work in float32 for the arithmetic regardless of storage dtype.
    t = tensor.detach().to(torch.float32)
    max_abs = t.abs().max()
    if max_abs <= 0:
        return tensor.clone()
    levels = (1 << (bits - 1)) - 1  # signed: 127 for int8, 7 for int4, 1 for int2
    scale = max_abs / levels
    q = torch.round(t / scale).clamp(-levels, levels)
    return (q * scale).to(orig_dtype)


def _fake_quant(tensor: torch.Tensor, spec: Dict[str, object]) -> torch.Tensor:
    kind = spec["kind"]
    if kind == "float":
        return _fake_quant_float(tensor, spec["dtype"])  # type: ignore[arg-type]
    if kind == "int":
        return _fake_quant_int(tensor, int(spec["bits"]))
    raise ValueError(f"Unknown precision kind: {kind}")


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


def _snapshot_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clone every trainable parameter to CPU so we can restore after quant."""
    with torch.no_grad():
        return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def _restore_params(model: torch.nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.data.copy_(snapshot[name].to(p.device, dtype=p.dtype))


def _apply_quant_in_place(model: torch.nn.Module, spec: Dict[str, object]) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.data.copy_(_fake_quant(p.data, spec))


# ---------------------------------------------------------------------------
# Forward pass capture
# ---------------------------------------------------------------------------


def _forward_capture(
    model: torch.nn.Module, inputs: Dict[str, torch.Tensor]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run one forward pass, return (hidden_states_cpu_float32, final_logits_cpu_float32).

    Note on MPS: a single-call ``.to("cpu", dtype=torch.float32)`` from an MPS
    fp16 tensor silently produces zeros in at least PyTorch 2.4. We always
    cast in two steps (to float on-device, then to CPU) to avoid this.
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = [h[0].detach().float().cpu() for h in outputs.hidden_states]
    logits = outputs.logits[0, -1].detach().float().cpu()
    return hidden, logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _per_layer_metrics(h_ref: torch.Tensor, h_q: torch.Tensor) -> Dict[str, float]:
    """
    Compare a baseline hidden-state tensor to a degraded one.

    Shapes: (seq_len, hidden_dim) float32 on CPU.
    """
    diff = h_ref - h_q
    mse = float(diff.pow(2).mean().item())
    ref_norm = float(h_ref.norm().item()) + 1e-12
    rel_err = float(diff.norm().item() / ref_norm)

    # Per-token cosine sim, then average
    ref_n = F.normalize(h_ref, dim=-1, eps=1e-10)
    q_n = F.normalize(h_q, dim=-1, eps=1e-10)
    cos = (ref_n * q_n).sum(dim=-1)
    mean_cos = float(cos.mean().item())
    min_cos = float(cos.min().item())

    return {
        "mse": mse,
        "relError": rel_err,
        "meanCosine": mean_cos,
        "minCosine": min_cos,
    }


def _output_metrics(
    logits_ref: torch.Tensor, logits_q: torch.Tensor, top_k: int = 10
) -> Dict[str, object]:
    """Compare the final-token logit distributions."""
    p_ref = F.softmax(logits_ref, dim=-1)
    p_q = F.softmax(logits_q, dim=-1)

    # KL(p_ref || p_q) in nats
    kl = float(
        (p_ref * (torch.log(p_ref + 1e-20) - torch.log(p_q + 1e-20))).sum().item()
    )
    kl = max(0.0, kl)  # clip numerical negatives

    # Argmax agreement
    argmax_ref = int(p_ref.argmax().item())
    argmax_q = int(p_q.argmax().item())
    argmax_match = argmax_ref == argmax_q

    # Top-K overlap (Jaccard)
    top_ref = set(torch.topk(p_ref, top_k).indices.tolist())
    top_q = set(torch.topk(p_q, top_k).indices.tolist())
    overlap = len(top_ref & top_q) / max(len(top_ref | top_q), 1)

    # Entropies (bits)
    def entropy_bits(p: torch.Tensor) -> float:
        return float((-p * torch.log2(p + 1e-20)).sum().item())

    return {
        "klDivergence": kl,
        "argmaxMatch": argmax_match,
        "argmaxRef": argmax_ref,
        "argmaxQuant": argmax_q,
        "topKOverlap": overlap,
        "entropyRef": entropy_bits(p_ref),
        "entropyQuant": entropy_bits(p_q),
    }


def _top_predictions(
    logits: torch.Tensor, tokenizer, top_k: int = 5
) -> List[Dict[str, object]]:
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k)
    out: List[Dict[str, object]] = []
    for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        out.append(
            {"tokenId": int(idx), "token": tokenizer.decode([int(idx)]), "prob": float(prob)}
        )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_precision_degradation(
    text: str,
    precisions: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Run `text` through the loaded model at baseline, then at each target
    precision, and return per-layer + output-level comparison data.
    """
    if session.model is None or session.tokenizer is None:
        raise ValueError("No model loaded")

    if precisions is None or len(precisions) == 0:
        precisions = ["bfloat16", "float16", "int8", "int4"]
    # Validate
    for p in precisions:
        if p not in PRECISION_SPECS:
            raise ValueError(
                f"Unknown precision '{p}'. Supported: {', '.join(SUPPORTED_PRECISIONS)}"
            )

    model = session.model
    tokenizer = session.tokenizer

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    if len(token_ids) < 2:
        raise ValueError(
            "Precision Degradation needs at least 2 tokens of input. Try a longer prompt."
        )

    # Model-level metadata so the frontend can show the baseline precision
    # and warn about memory pressure for large models.
    baseline_dtype = next(model.parameters()).dtype
    baseline_dtype_name = str(baseline_dtype).replace("torch.", "")
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # --- Baseline forward pass --------------------------------------------
    base_hidden, base_logits = _forward_capture(model, inputs)
    num_layers = len(base_hidden)
    hidden_dim = base_hidden[0].shape[-1]

    # Snapshot before we start corrupting the weights.
    snapshot = _snapshot_params(model)

    results: List[Dict[str, object]] = []
    try:
        for p in precisions:
            spec = PRECISION_SPECS[p]
            # Apply fake-quant in place, run forward, measure.
            _apply_quant_in_place(model, spec)
            try:
                q_hidden, q_logits = _forward_capture(model, inputs)
            finally:
                # Always restore before the next iteration, even on error.
                _restore_params(model, snapshot)

            layer_metrics = [
                _per_layer_metrics(base_hidden[i], q_hidden[i]) for i in range(num_layers)
            ]
            for i, m in enumerate(layer_metrics):
                m["layer"] = i

            output_metrics = _output_metrics(base_logits, q_logits, top_k=top_k)

            results.append(
                {
                    "precision": p,
                    "label": spec["label"],
                    "bits": spec["bits"],
                    "kind": spec["kind"],
                    "layers": layer_metrics,
                    "output": output_metrics,
                    "topPredictions": _top_predictions(q_logits, tokenizer, top_k=5),
                }
            )
    finally:
        # Belt and braces — ensure the model is pristine regardless of path.
        _restore_params(model, snapshot)
        del snapshot

    return {
        "inputText": text,
        "tokens": tokens,
        "tokenIds": token_ids,
        "numLayers": num_layers,
        "hiddenSize": hidden_dim,
        "baselineDtype": baseline_dtype_name,
        "modelSizeBytes": size_bytes,
        "memoryWarning": size_bytes > 4 * 1024**3,  # > 4 GB
        "precisions": results,
        "baselineTopPredictions": _top_predictions(base_logits, tokenizer, top_k=5),
    }
