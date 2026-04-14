"use client";

import React, { createContext, useContext, useState, useCallback } from "react";
import type { ModelInfo, BackendStatus } from "@/types/model";

interface ModelContextType {
  backendStatus: BackendStatus;
  loadModel: (modelId: string) => Promise<void>;
  unloadModel: () => Promise<void>;
  checkBackend: () => Promise<void>;
}

const ModelContext = createContext<ModelContextType | null>(null);

const BACKEND_URL = "http://localhost:8000";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseModelInfo(raw: any): ModelInfo | null {
  if (!raw) return null;
  return {
    modelId: raw.model_id,
    name: raw.name,
    architecture: raw.architecture,
    hiddenSize: raw.hidden_size,
    numLayers: raw.num_layers,
    vocabSize: raw.vocab_size,
    numAttentionHeads: raw.num_attention_heads,
    weightTied: raw.weight_tied,
    dtype: raw.dtype,
    sizeBytes: raw.size_bytes,
    device: raw.device,
  };
}

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>({
    status: "disconnected",
    model: null,
    device: "unknown",
    availableMemoryMb: 0,
  });

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/status`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus({
          status: "connected",
          model: parseModelInfo(data.model),
          device: data.device,
          availableMemoryMb: data.available_memory_mb,
        });
      } else {
        setBackendStatus(prev => ({ ...prev, status: "disconnected" }));
      }
    } catch {
      setBackendStatus(prev => ({ ...prev, status: "disconnected" }));
    }
  }, []);

  const loadModel = useCallback(async (modelId: string) => {
    setBackendStatus(prev => ({ ...prev, status: "loading" }));
    try {
      const res = await fetch(`${BACKEND_URL}/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to load model");
      }
      const data = await res.json();
      setBackendStatus({
        status: "connected",
        model: parseModelInfo(data.model),
        device: data.device,
        availableMemoryMb: data.available_memory_mb,
      });
    } catch (error) {
      setBackendStatus(prev => ({ ...prev, status: "connected" }));
      throw error;
    }
  }, []);

  const unloadModel = useCallback(async () => {
    try {
      await fetch(`${BACKEND_URL}/unload`, { method: "POST" });
      setBackendStatus(prev => ({ ...prev, model: null }));
    } catch {
      // Backend may be down
    }
  }, []);

  return (
    <ModelContext.Provider value={{ backendStatus, loadModel, unloadModel, checkBackend }}>
      {children}
    </ModelContext.Provider>
  );
}

export function useModel() {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModel must be used within ModelProvider");
  return ctx;
}
