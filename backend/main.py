"""
Vectorscope backend — FastAPI server for model inspection.
"""

import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.session import session
from operations.embedding_table import get_embedding_table
from operations.token_trajectory import get_token_trajectory
from operations.projection_head import get_projection_head
from operations.weight_comparison import get_weight_comparison
from operations.layer_probe import get_layer_probe
from operations.full_trace import stream_full_trace
from operations.attention import get_attention_at_layer, get_attention_head_across_layers
from operations.manifold_formation import get_manifold_formation
from operations.generation_vector import stream_generation_vector, MAX_GENERATION_TOKENS
from operations.isotropy import get_isotropy_analysis
from operations.cache_info import get_cache_info, delete_cached_repo, download_repo
from operations.precision_degradation import (
    get_precision_degradation,
    SUPPORTED_PRECISIONS,
)
from operations.local_model import inspect_local_model
from operations.grammar_steering import compute_grammar_steering
from config.presets import load_presets

app = FastAPI(title="Vectorscope Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    model_id: str


class EmbeddingTableRequest(BaseModel):
    sample_size: int = 5000


class ProjectionHeadRequest(BaseModel):
    sample_size: int = 5000


class WeightComparisonRequest(BaseModel):
    sample_size: int = 2000


class LayerProbeRequest(BaseModel):
    text: str
    layer: int


class FullTraceRequest(BaseModel):
    text: str
    top_k: int = 20


class AttentionLayerRequest(BaseModel):
    text: str
    layer: int


class AttentionHeadRequest(BaseModel):
    text: str
    head: int


class ManifoldFormationRequest(BaseModel):
    text: str


class TokenTrajectoryRequest(BaseModel):
    text: str


class GenerationVectorRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40


class IsotropyRequest(BaseModel):
    text: str


class CacheDeleteRequest(BaseModel):
    repo_id: str


class CacheDownloadRequest(BaseModel):
    repo_id: str


class PrecisionDegradationRequest(BaseModel):
    text: str
    precisions: Optional[List[str]] = None
    top_k: int = 10


class LocalModelInspectRequest(BaseModel):
    path: str


class ContrastivePair(BaseModel):
    positive: str
    negative: str


class GrammarSteeringRequest(BaseModel):
    pairs: List[ContrastivePair]
    pca_layer: Optional[int] = None


@app.get("/status")
async def status():
    return session.get_status()


@app.post("/load")
async def load_model(req: LoadRequest):
    try:
        info = await asyncio.to_thread(session.load, req.model_id)
        return {
            "model": info.__dict__,
            "device": session.device,
            "available_memory_mb": session.get_available_memory_mb(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload_model():
    session.unload()
    return {"status": "unloaded"}


@app.post("/embedding-table")
async def embedding_table(req: EmbeddingTableRequest):
    try:
        result = await asyncio.to_thread(get_embedding_table, req.sample_size)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/token-trajectory")
async def token_trajectory(req: TokenTrajectoryRequest):
    try:
        result = await asyncio.to_thread(get_token_trajectory, req.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projection-head")
async def projection_head(req: ProjectionHeadRequest):
    try:
        result = await asyncio.to_thread(get_projection_head, req.sample_size)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/weight-comparison")
async def weight_comparison(req: WeightComparisonRequest):
    try:
        result = await asyncio.to_thread(get_weight_comparison, req.sample_size)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/layer-probe")
async def layer_probe(req: LayerProbeRequest):
    try:
        result = await asyncio.to_thread(get_layer_probe, req.text, req.layer)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attention-layer")
async def attention_layer(req: AttentionLayerRequest):
    try:
        result = await asyncio.to_thread(get_attention_at_layer, req.text, req.layer)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attention-head")
async def attention_head(req: AttentionHeadRequest):
    try:
        result = await asyncio.to_thread(get_attention_head_across_layers, req.text, req.head)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/manifold-formation")
async def manifold_formation(req: ManifoldFormationRequest):
    try:
        result = await asyncio.to_thread(get_manifold_formation, req.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full-trace")
async def full_trace(req: FullTraceRequest):
    async def generate():
        # Run the heavy computation in a thread, collect results, then stream
        lines = await asyncio.to_thread(lambda: list(stream_full_trace(req.text, req.top_k)))
        for line in lines:
            yield line
    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/generation-vector/config")
async def generation_vector_config():
    return {"maxNewTokensCap": MAX_GENERATION_TOKENS}


@app.post("/generation-vector")
async def generation_vector(req: GenerationVectorRequest):
    async def generate():
        lines = await asyncio.to_thread(
            lambda: list(
                stream_generation_vector(
                    prompt=req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )
            )
        )
        for line in lines:
            yield line
    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/isotropy")
async def isotropy(req: IsotropyRequest):
    try:
        result = await asyncio.to_thread(get_isotropy_analysis, req.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache")
async def cache():
    try:
        return await asyncio.to_thread(get_cache_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/delete")
async def cache_delete(req: CacheDeleteRequest):
    try:
        return await asyncio.to_thread(delete_cached_repo, req.repo_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/download")
async def cache_download(req: CacheDownloadRequest):
    """Pre-stage a repo into the HF cache without loading it into memory."""
    try:
        return await asyncio.to_thread(download_repo, req.repo_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/precision-degradation/config")
async def precision_degradation_config():
    return {"supportedPrecisions": SUPPORTED_PRECISIONS}


@app.post("/precision-degradation")
async def precision_degradation(req: PrecisionDegradationRequest):
    try:
        return await asyncio.to_thread(
            get_precision_degradation, req.text, req.precisions, req.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/local-model/inspect")
async def local_model_inspect(req: LocalModelInspectRequest):
    """Validate a local directory as a loadable HuggingFace model checkpoint."""
    return await asyncio.to_thread(inspect_local_model, req.path)


@app.post("/grammar-steering")
async def grammar_steering(req: GrammarSteeringRequest):
    """Phase 1 of Grammar Steering: extract per-layer CAA steering vectors."""
    try:
        pairs = [p.model_dump() for p in req.pairs]
        return await asyncio.to_thread(compute_grammar_steering, pairs, req.pca_layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/presets")
async def presets():
    """
    Return the user-editable preset model catalogue from backend/config/models.md.

    Shape: { presets: list, source: "markdown"|"fallback", path: str, error: str|None }.
    The frontend uses the `presets` list as-is and surfaces `error` so the user
    can see parse failures after hand-editing the file.
    """
    return await asyncio.to_thread(load_presets)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
