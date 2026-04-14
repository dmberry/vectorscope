"""
Vectorscope backend — FastAPI server for model inspection.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.session import session
from operations.embedding_table import get_embedding_table
from operations.token_trajectory import get_token_trajectory

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


class TokenTrajectoryRequest(BaseModel):
    text: str


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
