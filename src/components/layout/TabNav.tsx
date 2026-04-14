"use client";

import { cn } from "@/lib/utils";
import { TAB_GROUPS, type TabGroup } from "@/types/model";

interface TabNavProps {
  activeGroup: TabGroup;
  activeTab: string;
  onGroupChange: (group: TabGroup) => void;
  onTabChange: (tabId: string) => void;
  modelLoaded: boolean;
}

export default function TabNav({
  activeGroup,
  activeTab,
  onGroupChange,
  onTabChange,
  modelLoaded,
}: TabNavProps) {
  const groups = Object.entries(TAB_GROUPS) as [TabGroup, typeof TAB_GROUPS[TabGroup]][];

  return (
    <div className="border-b border-parchment-dark bg-card">
      {/* Group tabs */}
      <div className="flex gap-0 px-6">
        {groups.map(([groupId, group]) => (
          <button
            key={groupId}
            onClick={() => {
              onGroupChange(groupId);
              onTabChange(group.tabs[0].id);
            }}
            className={cn(
              "px-3 py-1.5 font-sans text-caption font-medium border-b-2 transition-colors",
              activeGroup === groupId
                ? "border-burgundy text-burgundy"
                : "border-transparent text-slate hover:text-ink hover:border-parchment-dark"
            )}
          >
            {group.label}
          </button>
        ))}
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-0 px-6 bg-cream/50">
        {TAB_GROUPS[activeGroup].tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            disabled={!modelLoaded}
            className={cn(
              "px-3 py-1.5 font-sans text-caption font-medium transition-colors whitespace-nowrap",
              activeTab === tab.id
                ? "text-burgundy bg-card border-b border-burgundy"
                : modelLoaded
                ? "text-slate hover:text-ink"
                : "text-slate/40 cursor-not-allowed"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}
