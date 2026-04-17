"""
Model session management.
Holds one loaded model in memory at a time.
"""

from __future__ import annotations

import gc
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    native_dtype: str
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

    def _detect_native_dtype(self, model_id: str, config) -> str:
        """
        Determine the model's native (on-disk) precision.

        Strategy:
        1. config.torch_dtype when the repo declares it (modern Llama/Qwen/Mistral).
        2. Peek at the safetensors header (single-file or sharded) without loading
           any tensors — this is a few KB of I/O.
        3. Peek at legacy pytorch_model.bin via torch.load(..., map_location="meta").
        4. "unknown" as a last resort.

        Runs before from_pretrained; weights may or may not be cached. We only
        consult files already in the HF cache; we do not trigger downloads.
        """
        # 1) Declared dtype
        declared = getattr(config, "torch_dtype", None)
        if isinstance(declared, str):
            declared = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "float": torch.float32,
            }.get(declared)
        if declared is not None:
            return str(declared).replace("torch.", "")

        # 2) Safetensors header
        try:
            from huggingface_hub import try_to_load_from_cache
            from safetensors import safe_open

            path = try_to_load_from_cache(model_id, "model.safetensors")
            if path is None:
                idx = try_to_load_from_cache(model_id, "model.safetensors.index.json")
                if idx:
                    import json
                    with open(idx) as f:
                        shards = json.load(f).get("weight_map", {})
                    first_shard = next(iter(set(shards.values())), None)
                    if first_shard:
                        path = try_to_load_from_cache(model_id, first_shard)
            if path:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():
                        dtype_str = f.get_slice(key).get_dtype()
                        return {
                            "F32": "float32",
                            "F16": "float16",
                            "BF16": "bfloat16",
                            "F64": "float64",
                        }.get(dtype_str, dtype_str.lower())
        except Exception:
            pass

        # 3) Legacy pytorch_model.bin
        try:
            from huggingface_hub import try_to_load_from_cache
            bin_path = try_to_load_from_cache(model_id, "pytorch_model.bin")
            if bin_path:
                state = torch.load(bin_path, map_location="meta", weights_only=True)
                for v in state.values():
                    return str(v.dtype).replace("torch.", "")
        except Exception:
            pass

        return "unknown"

    def _resolve_dtype(self, config) -> torch.dtype:
        """
        Pick a load dtype that preserves the weights' native precision where possible.

        Most modern open-weight models (Llama 3, Qwen 3, Mistral) are distributed in
        bfloat16. GPT-2 is float32. Previously we forced everything to float16, which
        silently down-cast bf16 weights and corrupted the precision baseline for
        isotropy and (eventually) precision-degradation analysis.

        Strategy:
        - Read config.torch_dtype when the repo declares it.
        - If the declared dtype is bf16, honour it when the device supports bf16.
        - CUDA supports bf16 everywhere recent. MPS supports bf16 on recent PyTorch.
        - CPU: bf16 works but is slow. We still honour it so the numerical baseline
          is correct; users on CPU-only rigs should prefer small models anyway.
        - Fall back to float16 when bf16 is unsupported, then to float32 for tiny
          models like GPT-2 that distribute as fp32.
        """
        declared = getattr(config, "torch_dtype", None)
        # HF config may store dtype as a torch.dtype or as a string
        if isinstance(declared, str):
            declared = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "float": torch.float32,
            }.get(declared)

        if declared == torch.bfloat16:
            # Keep bf16 where we can
            if self.device in ("cuda", "mps", "cpu"):
                return torch.bfloat16
            return torch.float16

        if declared == torch.float32:
            # GPT-2 ships fp32. On a GPU/MPS device we can safely cast down to fp16
            # to save memory; on CPU we leave it alone.
            if self.device in ("cuda", "mps"):
                return torch.float16
            return torch.float32

        if declared == torch.float16:
            return torch.float16

        # Unknown — default to fp16 on accelerators, fp32 on CPU
        if self.device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    def load(self, model_id: str) -> ModelInfo:
        # Unload any existing model first
        self.unload()

        config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Capture the native (on-disk) precision before we resolve a session
        # dtype, so the frontend can report both values and flag when a
        # down-cast happened. Prefer the config's declared dtype; fall back to
        # the actual safetensors / pytorch_model.bin header when the config
        # doesn't declare one (e.g. GPT-2).
        native_dtype_name = self._detect_native_dtype(model_id, config)

        dtype = self._resolve_dtype(config)

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device if self.device != "mps" else None,
        }

        if self.device == "mps":
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Detect weight tying
        embed_weight = self._get_embedding_table()
        lm_head = self._get_lm_head_weight()
        weight_tied = lm_head is not None and embed_weight.data_ptr() == lm_head.data_ptr()

        # Estimate size
        size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        # Use the actual loaded dtype (the model may have up/down-cast layers)
        loaded_dtype = next(self.model.parameters()).dtype
        dtype_name = str(loaded_dtype).replace("torch.", "")

        self.info = ModelInfo(
            model_id=model_id,
            name=model_id.split("/")[-1],
            architecture=config.model_type,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            num_attention_heads=config.num_attention_heads,
            weight_tied=weight_tied,
            dtype=dtype_name,
            native_dtype=native_dtype_name,
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
        # With AutoModelForCausalLM, the base model is often nested under .model or .transformer
        candidates = [self.model, getattr(self.model, "model", None), getattr(self.model, "transformer", None)]
        for base in candidates:
            if base is None:
                continue
            for path in [
                "embed_tokens",           # Llama, Qwen, Mistral
                "wte",                    # GPT-2
                "embeddings.word_embeddings",  # BERT
            ]:
                obj = base
                for attr in path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None and hasattr(obj, "weight"):
                    return obj.weight

        # Fallback: search for first Embedding layer
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                return module.weight

        raise ValueError("Could not find embedding table")

    def _get_lm_head_weight(self) -> Optional[torch.Tensor]:
        """Get output unembedding (lm_head) weight matrix."""
        for attr in ["lm_head", "cls", "output"]:
            head = getattr(self.model, attr, None)
            if head is not None and hasattr(head, "weight"):
                return head.weight
        return None

    def _get_base_model(self):
        """Get the base transformer model (unwrapped from CausalLM wrapper)."""
        for attr in ["model", "transformer", "bert"]:
            base = getattr(self.model, attr, None)
            if base is not None:
                return base
        return self.model

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
