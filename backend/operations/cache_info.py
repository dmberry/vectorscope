"""
HuggingFace cache inspection.

Vectorscope loads models via `AutoModelForCausalLM.from_pretrained()`, which
goes through the HuggingFace hub cache. This module exposes the cache layout
to the frontend Settings panel so users can see:

- where models are stored on disk (typically ~/.cache/huggingface/hub/)
- which repos are already cached
- how much space each one takes
- when they were last used
- and delete individual repos to reclaim space

We deliberately surface the real cache path (not a project-local override)
because Vectorscope doesn't set HF_HOME — it inherits whatever the user's
HuggingFace install already uses, which is usually shared with other tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound


def get_cache_info() -> dict:
    """
    Enumerate the HuggingFace hub cache.

    Returns a dict with:
      - cache_path: absolute path to the cache root
      - total_size_bytes: sum of all cached repo sizes
      - repos: list of {repo_id, size_bytes, last_accessed, last_modified, refs}
      - repo_count: number of repos
    """
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        # No cache yet — return an empty but well-formed structure so the
        # frontend can render "no models cached" rather than erroring out.
        return {
            "cache_path": str(HF_HUB_CACHE),
            "total_size_bytes": 0,
            "repo_count": 0,
            "repos": [],
        }

    repos = []
    for repo in info.repos:
        # Only surface model repos — skip datasets and spaces, which aren't
        # loadable by Vectorscope anyway.
        if repo.repo_type != "model":
            continue
        repos.append({
            "repo_id": repo.repo_id,
            "size_bytes": repo.size_on_disk,
            "last_accessed": int(repo.last_accessed) if repo.last_accessed else None,
            "last_modified": int(repo.last_modified) if repo.last_modified else None,
            "nb_files": repo.nb_files,
            "refs": sorted(repo.refs) if repo.refs else [],
        })

    # Sort biggest first — users want to see what's eating their disk
    repos.sort(key=lambda r: r["size_bytes"], reverse=True)

    return {
        "cache_path": str(HF_HUB_CACHE),
        "total_size_bytes": sum(r["size_bytes"] for r in repos),
        "repo_count": len(repos),
        "repos": repos,
    }


def download_repo(repo_id: str) -> dict:
    """
    Download a HuggingFace repo's weights and tokenizer to the cache without
    instantiating a PyTorch model. Useful for pre-staging models so loading
    into memory is instantaneous afterward.

    This uses huggingface_hub.snapshot_download(), which fetches all files
    into the same cache structure that `from_pretrained()` uses, so a later
    model load will find them on disk and skip the network entirely.

    Returns {downloaded: true, repo_id, size_bytes, cache_path} on success.
    Errors from the Hub (e.g. auth, not-found, rate-limit) bubble up as
    exceptions for the FastAPI layer to turn into HTTP errors.
    """
    # Only download safetensors/config/tokenizer files — skip PyTorch .bin
    # duplicates (they're legacy and double the footprint) and Flax/TF
    # equivalents we never use.
    local_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.model",
            "tokenizer*",
            "special_tokens_map.json",
            "vocab*",
            "merges*",
            "*.py",
        ],
    )

    # Re-scan the cache so we can report the actual on-disk size for this repo
    try:
        info = scan_cache_dir()
        size_bytes = 0
        for repo in info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                size_bytes = repo.size_on_disk
                break
    except CacheNotFound:
        size_bytes = 0

    return {
        "downloaded": True,
        "repo_id": repo_id,
        "size_bytes": int(size_bytes),
        "cache_path": str(local_path),
    }


def delete_cached_repo(repo_id: str) -> dict:
    """
    Delete all cached revisions of a given repo.

    Returns {deleted: true, freed_bytes: int} on success.
    Raises ValueError if the repo is not in the cache.
    """
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        raise ValueError(f"No HuggingFace cache found")

    target = None
    for repo in info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            target = repo
            break

    if target is None:
        raise ValueError(f"Repo '{repo_id}' not in cache")

    # Collect every revision hash for this repo
    revisions = [rev.commit_hash for rev in target.revisions]
    if not revisions:
        raise ValueError(f"Repo '{repo_id}' has no revisions to delete")

    strategy = info.delete_revisions(*revisions)
    strategy.execute()

    return {
        "deleted": True,
        "repo_id": repo_id,
        "freed_bytes": int(strategy.expected_freed_size),
    }
