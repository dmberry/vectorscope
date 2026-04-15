"""
Generation Vector: stream an autoregressive generation with full per-step
instrumentation via NDJSON.

For every generated token we capture:
  - the new token (id, string, position)
  - per-layer hidden-state norms for the new token
  - per-layer cosine similarity to the previous layer (for the new token)
  - mean attention from the new token back to every context position, per layer
  - top-K predicted next tokens with probabilities and entropy
  - the new token's hidden state at every layer (used later for a global PCA)

At the end of generation we fit a single PCA across *all* (step, layer)
hidden-state vectors for every token in the sequence (prompt + generated)
and emit a `geometry` event with consistent 3D coordinates so the UI can
show a stable layer-trajectory view for any focused token.

Streaming stages:
  1. start      — echo of generation config + prompt length cap
  2. prompt     — prompt tokenisation + per-token input-embedding norms
  3. step:N     — one per generated token (N = 0..max_new_tokens-1)
  4. geometry   — per-token, per-layer 3D PCA coords
  5. complete   — summary (entropy series, norms series, final text)
"""

from __future__ import annotations

import json
import math
import torch
import numpy as np
from typing import Generator, Optional
from models.session import session


# Hard cap on generation length — single place to change later.
MAX_GENERATION_TOKENS = 150


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[int, torch.Tensor]:
    """Return (sampled_token_id, probs) where probs is the full post-softmax
    distribution (before top-p/top-k filtering, so entropy is over the whole
    vocabulary). Greedy when temperature <= 0."""

    if temperature <= 0.0:
        # Greedy — still return the full softmax for entropy reporting.
        probs = torch.softmax(logits, dim=-1)
        token_id = int(torch.argmax(logits).item())
        return token_id, probs

    scaled = logits / max(temperature, 1e-6)
    probs_full = torch.softmax(scaled, dim=-1)

    # Top-k filter.
    if top_k > 0 and top_k < probs_full.size(0):
        top_vals, top_idxs = torch.topk(probs_full, top_k)
        mask = torch.zeros_like(probs_full)
        mask.scatter_(0, top_idxs, top_vals)
        filtered = mask
    else:
        filtered = probs_full.clone()

    # Top-p (nucleus) filter.
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idxs = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep tokens up to and including the one that crosses top_p.
        cutoff = torch.searchsorted(cumulative, torch.tensor(top_p, device=cumulative.device))
        cutoff = int(cutoff.item()) + 1
        keep_idxs = sorted_idxs[:cutoff]
        mask = torch.zeros_like(filtered)
        mask[keep_idxs] = filtered[keep_idxs]
        filtered = mask

    # Renormalise and sample.
    total = filtered.sum()
    if total <= 0:
        filtered = probs_full
        total = filtered.sum()
    filtered = filtered / total
    token_id = int(torch.multinomial(filtered, num_samples=1).item())
    return token_id, probs_full


def _entropy_bits(probs: torch.Tensor) -> float:
    eps = 1e-12
    return float(-torch.sum(probs * torch.log2(probs + eps)).item())


def stream_generation_vector(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40,
) -> Generator[str, None, None]:
    """Yield NDJSON lines for the full instrumented generation."""

    if session.model is None:
        yield json.dumps({"stage": "error", "message": "No model loaded"}) + "\n"
        return

    tokenizer = session.tokenizer
    model = session.model
    device = next(model.parameters()).device

    # Clamp max_new_tokens against the hard cap.
    max_new_tokens = max(1, min(int(max_new_tokens), MAX_GENERATION_TOKENS))

    # --- Stage 1: start ---
    yield json.dumps({
        "stage": "start",
        "maxNewTokens": max_new_tokens,
        "maxNewTokensCap": MAX_GENERATION_TOKENS,
        "temperature": temperature,
        "topP": top_p,
        "topK": top_k,
        "prompt": prompt,
    }) + "\n"

    # --- Stage 2: prompt tokenisation + input embedding norms ---
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids: torch.Tensor = enc["input_ids"][0]  # [seq_len]
    prompt_len = int(input_ids.shape[0])
    prompt_tokens = [tokenizer.decode([int(tid)]) for tid in input_ids.tolist()]

    embed_table = session._get_embedding_table()
    prompt_embeds = embed_table[input_ids].float().detach().cpu().numpy()  # [prompt_len, d]
    prompt_norms = np.linalg.norm(prompt_embeds, axis=1).tolist()

    yield json.dumps({
        "stage": "prompt",
        "tokens": prompt_tokens,
        "tokenIds": [int(x) for x in input_ids.tolist()],
        "promptLen": prompt_len,
        "inputEmbedNorms": [round(n, 4) for n in prompt_norms],
    }) + "\n"

    # Collected state.
    # all_hidden_states[t] = [num_layers+1, hidden_dim] for token position t
    # We store the hidden state of every token at every layer so we can fit a
    # single global PCA at the end.
    all_hidden_vecs: list[list[np.ndarray]] = []  # [seq_len][num_layers+1] of 1D arrays
    per_step_events: list[dict] = []

    # Warm start: one forward pass over the prompt to populate hidden states
    # for every prompt token. This gives us the prompt's geometry so the UI
    # can show any prompt word's layer trajectory.
    with torch.no_grad():
        out = model(
            input_ids=input_ids.unsqueeze(0),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
    # hidden_states: tuple of length num_layers+1, each [1, seq_len, d]
    num_layers = len(out.hidden_states)  # includes layer 0 = post-embedding
    hidden_dim = int(out.hidden_states[0].shape[-1])

    for t in range(prompt_len):
        per_token_layers = []
        for layer_idx in range(num_layers):
            vec = out.hidden_states[layer_idx][0, t].float().detach().cpu().numpy()
            per_token_layers.append(vec)
        all_hidden_vecs.append(per_token_layers)

    # Seed generated_ids with the prompt so subsequent forward passes include
    # the full running context.
    generated_ids: list[int] = list(map(int, input_ids.tolist()))

    # --- Stage 3: step loop ---
    for step_idx in range(max_new_tokens):
        ctx = torch.tensor(generated_ids, device=device).unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            step_out = model(
                input_ids=ctx,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )

        # Final-layer logits at the last position — next-token distribution.
        logits = step_out.logits[0, -1].float()
        token_id, probs = _sample_next_token(logits, temperature, top_p, top_k)
        top_vals, top_idxs = torch.topk(probs, min(top_k, int(probs.size(0))))
        top_predictions = [
            {
                "token": tokenizer.decode([int(idx)]),
                "tokenId": int(idx),
                "probability": round(float(val), 6),
            }
            for val, idx in zip(top_vals.tolist(), top_idxs.tolist())
        ]
        entropy_bits = _entropy_bits(probs)

        # Record hidden states for the newly produced *last* position (the
        # query position that produced this token's logit). This is the
        # position whose geometry matters for "what did the model think when
        # it was about to emit this token?".
        last_pos = ctx.shape[1] - 1
        new_hidden_layers = []
        hidden_norms = []
        for layer_idx in range(num_layers):
            hs = step_out.hidden_states[layer_idx][0, last_pos].float().detach().cpu()
            new_hidden_layers.append(hs.numpy())
            hidden_norms.append(round(float(hs.norm().item()), 4))

        # Per-layer cosine between consecutive layers for the new token.
        layer_deltas: list[Optional[float]] = [None]
        for layer_idx in range(1, num_layers):
            prev = torch.from_numpy(new_hidden_layers[layer_idx - 1])
            curr = torch.from_numpy(new_hidden_layers[layer_idx])
            cos = torch.nn.functional.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
            layer_deltas.append(round(float(cos), 4))

        # Attention summary: mean over heads of the attention weights from
        # the last position back to every context position, for every layer.
        # shape of each attention tensor: [1, num_heads, seq_len, seq_len]
        attention_by_layer: list[list[float]] = []
        for layer_idx, attn in enumerate(step_out.attentions):
            # Mean over heads → [seq_len, seq_len], take the last query row.
            row = attn[0].float().mean(dim=0)[last_pos].detach().cpu().numpy()
            attention_by_layer.append([round(float(x), 5) for x in row.tolist()])

        # Now actually commit the sampled token and store its geometry. This
        # token's "position in the sequence" is last_pos + 1 (it becomes the
        # next token after the current context). We store its hidden states
        # as computed at the *query* position of this step.
        all_hidden_vecs.append(new_hidden_layers)
        generated_ids.append(token_id)

        token_str = tokenizer.decode([token_id])

        step_event = {
            "stage": "step",
            "stepIndex": step_idx,
            "absolutePos": last_pos + 1,  # position of this token in final sequence
            "token": token_str,
            "tokenId": token_id,
            "hiddenNorms": hidden_norms,
            "layerDeltas": layer_deltas,
            "attentionByLayer": attention_by_layer,
            "topPredictions": top_predictions,
            "entropyBits": round(entropy_bits, 4),
            "contextLen": last_pos + 1,
        }
        per_step_events.append(step_event)
        yield json.dumps(step_event) + "\n"

        # Stop on EOS.
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    # --- Stage 4: global geometry (PCA over all stored hidden vectors) ---
    # Flatten to [total_tokens * num_layers, hidden_dim]
    total_tokens = len(all_hidden_vecs)
    flat = np.zeros((total_tokens * num_layers, hidden_dim), dtype=np.float32)
    for t, per_layer in enumerate(all_hidden_vecs):
        for l, vec in enumerate(per_layer):
            flat[t * num_layers + l] = vec

    # Center and SVD for PCA.
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    # Use randomised SVD via numpy for speed on medium matrices.
    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:3]  # [3, hidden_dim]
        coords = centered @ components.T  # [total, 3]
    except Exception:
        coords = centered[:, :3]  # fallback

    # Reshape to [total_tokens, num_layers, 3] and cast to lists for JSON.
    coords_reshaped = coords.reshape(total_tokens, num_layers, 3)
    geometry_payload = {
        "stage": "geometry",
        "numTokens": total_tokens,
        "numLayers": num_layers,
        "hiddenDim": hidden_dim,
        "promptLen": prompt_len,
        "coords": [
            [[round(float(c), 4) for c in layer] for layer in token]
            for token in coords_reshaped.tolist()
        ],
    }
    yield json.dumps(geometry_payload) + "\n"

    # --- Stage 5: complete ---
    generated_tokens = [tokenizer.decode([tid]) for tid in generated_ids[prompt_len:]]
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    completion_text = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=False)

    entropy_series = [e["entropyBits"] for e in per_step_events]
    norms_series_means = [
        round(float(np.mean(e["hiddenNorms"])), 4) for e in per_step_events
    ]

    # Tokenisation series: per-token input-embedding (layer 0) geometry for
    # the whole sequence. Input embeddings are independent lookups — there is
    # no causal information flow at layer 0 — but we can still compute the
    # running vector sum to show how much consecutive tokens reinforce or
    # cancel each other geometrically.
    layer0_stack = np.stack(
        [per_layer[0] for per_layer in all_hidden_vecs], axis=0
    )  # [total_tokens, hidden_dim]
    layer0_norms = np.linalg.norm(layer0_stack, axis=1)
    cumulative_sum = np.cumsum(layer0_stack, axis=0)  # [total_tokens, hidden_dim]
    cumulative_norms = np.linalg.norm(cumulative_sum, axis=1)
    t_indices = np.arange(1, total_tokens + 1, dtype=np.float32)
    cumulative_mean_norms = cumulative_norms / t_indices
    # Independence baseline: if tokens were i.i.d. with per-token norm n,
    # E[||sum||] ~ n * sqrt(t). Use the mean per-token norm as n.
    mean_per_token_norm = float(np.mean(layer0_norms))
    independence_baseline = (mean_per_token_norm * np.sqrt(t_indices)).tolist()
    # First-16 dim preview of each token's input embedding (for the deep-dive
    # numeric table). Keep it bounded so payload stays small.
    preview_dims = min(16, hidden_dim)
    vector_preview = [
        [round(float(v), 4) for v in layer0_stack[t, :preview_dims].tolist()]
        for t in range(total_tokens)
    ]

    tokenisation_series = {
        "perTokenNorms": [round(float(n), 4) for n in layer0_norms.tolist()],
        "cumulativeSumNorms": [round(float(n), 4) for n in cumulative_norms.tolist()],
        "cumulativeMeanNorms": [round(float(n), 4) for n in cumulative_mean_norms.tolist()],
        "independenceBaseline": [round(float(b), 4) for b in independence_baseline],
        "previewDims": preview_dims,
        "vectorPreview": vector_preview,
    }

    yield json.dumps({
        "stage": "complete",
        "numLayers": num_layers,
        "promptLen": prompt_len,
        "numGenerated": len(per_step_events),
        "generatedTokens": generated_tokens,
        "generatedText": completion_text,
        "fullText": full_text,
        "entropySeries": entropy_series,
        "meanNormSeries": norms_series_means,
        "tokenisationSeries": tokenisation_series,
    }) + "\n"
