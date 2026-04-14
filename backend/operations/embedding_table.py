"""
Embedding Table inspection: extract and analyse the input embedding matrix.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from models.session import session


def get_embedding_table(sample_size: int = 5000) -> dict:
    """Extract embedding table, compute stats, project a sample to 3D."""
    if session.model is None:
        raise ValueError("No model loaded")

    embed_weight = session._get_embedding_table()
    vocab_size, hidden_dim = embed_weight.shape

    # Compute norms
    with torch.no_grad():
        norms = torch.norm(embed_weight.float(), dim=1).cpu().numpy()

    # Stats
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))

    # Effective rank (via singular values)
    with torch.no_grad():
        matrix = embed_weight.float().cpu()
        # Use SVD on a sample if vocab is very large
        if vocab_size > 10000:
            indices = np.random.choice(vocab_size, 10000, replace=False)
            matrix = matrix[indices]
        svd = torch.linalg.svdvals(matrix)
        svd_norm = svd / svd.sum()
        entropy = -(svd_norm * torch.log(svd_norm + 1e-10)).sum()
        effective_rank = float(torch.exp(entropy))

    # Isotropy: average cosine similarity between random pairs
    with torch.no_grad():
        sample_idx = np.random.choice(vocab_size, min(1000, vocab_size), replace=False)
        sample_vecs = embed_weight[sample_idx].float()
        sample_normed = sample_vecs / (sample_vecs.norm(dim=1, keepdim=True) + 1e-10)
        cos_matrix = sample_normed @ sample_normed.T
        # Exclude diagonal
        mask = ~torch.eye(len(sample_idx), dtype=torch.bool)
        isotropy_score = float(1.0 - cos_matrix[mask].mean().abs())

    # Sample tokens for 3D projection
    sample_n = min(sample_size, vocab_size)
    sample_indices = np.random.choice(vocab_size, sample_n, replace=False)

    with torch.no_grad():
        sample_vectors = embed_weight[sample_indices].float().cpu().numpy()

    # Server-side PCA to 3D
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(sample_vectors)

    # Get token strings
    tokenizer = session.tokenizer
    sample_tokens = [tokenizer.decode([int(i)]) for i in sample_indices]
    sample_norms = norms[sample_indices].tolist()

    return {
        "shape": [vocab_size, hidden_dim],
        "stats": {
            "meanNorm": mean_norm,
            "stdNorm": std_norm,
            "minNorm": min_norm,
            "maxNorm": max_norm,
            "effectiveRank": effective_rank,
            "isotropyScore": isotropy_score,
        },
        "sampleTokens": sample_tokens,
        "sampleCoords3d": coords_3d.tolist(),
        "sampleNorms": sample_norms,
    }
