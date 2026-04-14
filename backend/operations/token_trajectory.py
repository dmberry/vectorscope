"""
Token Trajectory: trace a token's representation through all layers.
"""

import torch
import numpy as np
from models.session import session


def get_token_trajectory(text: str) -> dict:
    """Trace input text through all layers, returning hidden states."""
    if session.model is None:
        raise ValueError("No model loaded")

    tokenizer = session.tokenizer
    model = session.model

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

    layers = []
    for layer_idx, hs in enumerate(hidden_states):
        hs_np = hs[0].float().cpu().numpy()  # (seq_len, hidden_dim)
        norms = np.linalg.norm(hs_np, axis=1).tolist()
        # Only send first token's vector for trajectory (to keep payload small)
        layers.append({
            "layer": layer_idx,
            "vectors": hs_np.tolist(),
            "norms": norms,
        })

    # Compute layer-to-layer cosine similarities (for first token position)
    layer_sims = []
    for i in range(1, len(hidden_states)):
        v1 = hidden_states[i - 1][0, 0].float()
        v2 = hidden_states[i][0, 0].float()
        cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
        layer_sims.append(float(cos))

    # Top predictions from final hidden state (approximate, no lm_head in base model)
    top_predictions = []

    return {
        "inputText": text,
        "tokenIds": token_ids,
        "tokens": tokens,
        "layers": layers,
        "layerSimilarities": layer_sims,
        "topPredictions": top_predictions,
    }
