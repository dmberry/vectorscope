"""
Model session management.
Holds one loaded model in memory at a time.
"""

from __future__ import annotations

import gc
import torch
import psutil
from transformers import AutoModel, AutoTokenizer, AutoConfig
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    model_id: str
    name: str
    architecture: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    num_attention_heads: int
    weight_tied: bool
    dtype: str
    size_bytes: int
    device: str


class ModelSession:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.info: Optional[ModelInfo] = None
        self.device = self._detect_device()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, model_id: str) -> ModelInfo:
        # Unload any existing model first
        self.unload()

        config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Use fp16 by default, eager attention for MPS compatibility
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": self.device if self.device != "mps" else None,
        }

        if self.device == "mps":
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Detect weight tying
        embed_weight = self._get_embedding_table()
        lm_head = self._get_lm_head_weight()
        weight_tied = lm_head is not None and embed_weight.data_ptr() == lm_head.data_ptr()

        # Estimate size
        size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        self.info = ModelInfo(
            model_id=model_id,
            name=model_id.split("/")[-1],
            architecture=config.model_type,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            num_attention_heads=config.num_attention_heads,
            weight_tied=weight_tied,
            dtype="float16",
            size_bytes=size_bytes,
            device=self.device,
        )

        return self.info

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.info = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_embedding_table(self) -> torch.Tensor:
        """Get input embedding weight matrix, handling architecture differences."""
        model = self.model
        # Try common attribute paths
        for path in [
            "embed_tokens",           # Llama, Qwen, Mistral
            "wte",                    # GPT-2
            "embeddings.word_embeddings",  # BERT
        ]:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "weight"):
                return obj.weight

        # Fallback: search for first Embedding layer
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module.weight

        raise ValueError("Could not find embedding table")

    def _get_lm_head_weight(self) -> Optional[torch.Tensor]:
        """Get output unembedding (lm_head) weight matrix, if separate."""
        # lm_head is on the parent model (e.g. GPT2LMHeadModel), but we load
        # with AutoModel which gives the base. Try common paths anyway.
        model = self.model
        for attr in ["lm_head", "cls", "output"]:
            head = getattr(model, attr, None)
            if head is not None and hasattr(head, "weight"):
                return head.weight
        return None

    def get_available_memory_mb(self) -> int:
        return int(psutil.virtual_memory().available / (1024 * 1024))

    def get_status(self) -> dict:
        return {
            "model": self.info.__dict__ if self.info else None,
            "device": self.device,
            "available_memory_mb": self.get_available_memory_mb(),
        }


# Global singleton
session = ModelSession()
