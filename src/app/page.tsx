"use client";

import { useState, useEffect } from "react";
import Providers from "./providers";
import Header from "@/components/layout/Header";
import TabNav from "@/components/layout/TabNav";
import StatusBar from "@/components/layout/StatusBar";
import ModelLoader from "@/components/layout/ModelLoader";
import { useModel } from "@/context/ModelContext";
import type { TabGroup } from "@/types/model";
import EmbeddingTable from "@/components/operations/EmbeddingTable";
import TokenTrajectory from "@/components/operations/TokenTrajectory";
import VocabularyMap from "@/components/operations/VocabularyMap";
import ProjectionHead from "@/components/operations/ProjectionHead";
import WeightComparison from "@/components/operations/WeightComparison";
import LayerProbe from "@/components/operations/LayerProbe";
import FullTrace from "@/components/operations/FullTrace";
import GenerationVector from "@/components/operations/GenerationVector";
import AttentionInspector from "@/components/operations/AttentionInspector";
import ManifoldFormation from "@/components/operations/ManifoldFormation";
import IsotropyAnalysis from "@/components/operations/IsotropyAnalysis";
import PrecisionDegradation from "@/components/operations/PrecisionDegradation";
import GrammarSteering from "@/components/operations/GrammarSteering";
import { Clippy } from "@/components/easter-eggs/Clippy";

function VectorscopeApp() {
  const { backendStatus, checkBackend } = useModel();
  const [activeGroup, setActiveGroup] = useState<TabGroup>("inspect");
  const [activeTab, setActiveTab] = useState("embedding-table");
  const [pickerOpen, setPickerOpen] = useState(false);
  const modelLoaded = !!backendStatus.model;

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, modelLoaded ? 30000 : 5000);
    return () => clearInterval(interval);
  }, [checkBackend, modelLoaded]);

  // Close the picker on Esc
  useEffect(() => {
    if (!pickerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPickerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pickerOpen]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header onOpenModelPicker={() => setPickerOpen(true)} activeTab={activeTab} />
      <TabNav
        activeGroup={activeGroup}
        activeTab={activeTab}
        onGroupChange={setActiveGroup}
        onTabChange={setActiveTab}
        modelLoaded={modelLoaded}
      />

      <main className="flex-1 px-6 py-4">
        {!modelLoaded ? (
          <ModelLoader />
        ) : (
          <div className="max-w-7xl mx-auto">
            {activeTab === "embedding-table" && <EmbeddingTable />}
            {activeTab === "projection-head" && <ProjectionHead />}
            {activeTab === "weight-comparison" && <WeightComparison />}
            {activeTab === "token-trajectory" && <TokenTrajectory />}
            {activeTab === "layer-probe" && <LayerProbe />}
            {activeTab === "full-trace" && <FullTrace />}
            {activeTab === "generation-vector" && <GenerationVector />}
            {activeTab === "attention" && <AttentionInspector />}
            {activeTab === "manifold-formation" && <ManifoldFormation />}
            {activeTab === "vocabulary-map" && <VocabularyMap />}
            {activeTab === "isotropy" && <IsotropyAnalysis />}
            {activeTab === "precision-degradation" && <PrecisionDegradation />}
            {activeTab === "grammar-steering" && <GrammarSteering />}
            {!["embedding-table", "projection-head", "weight-comparison", "token-trajectory", "layer-probe", "full-trace", "generation-vector", "attention", "manifold-formation", "vocabulary-map", "isotropy", "precision-degradation", "grammar-steering"].includes(activeTab) && (
              <div className="card-editorial p-4 text-center">
                <p className="font-sans text-xs text-slate">
                  {activeTab} — coming in a later phase
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Model-swap picker overlay — reachable by clicking the model name in the Header */}
      {pickerOpen && modelLoaded && (
        <div
          className="fixed inset-0 z-40 bg-black/30 flex items-start justify-center overflow-y-auto py-8"
          onClick={() => setPickerOpen(false)}
        >
          <div onClick={e => e.stopPropagation()} className="w-full">
            <ModelLoader
              onLoaded={() => setPickerOpen(false)}
              onClose={() => setPickerOpen(false)}
            />
          </div>
        </div>
      )}

      <StatusBar />
      <Clippy />
    </div>
  );
}

export default function Home() {
  return (
    <Providers>
      <VectorscopeApp />
    </Providers>
  );
}
