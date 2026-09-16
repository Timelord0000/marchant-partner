"""
Roadmap Retriever — matches regression signals and merchant profile traits
to the most relevant growth roadmap stages.
"""

import json
from pathlib import Path
from typing import Any, Optional

ROADMAP_PATH = Path(__file__).resolve().parent.parent / "roadmap" / "growth_roadmap.json"


class RoadmapRetriever:
    """Loads the growth roadmap and retrieves relevant stages."""

    def __init__(self):
        self.stages: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        with open(ROADMAP_PATH) as f:
            data = json.load(f)
        self.stages = data["stages"]
        self._loaded = True

    def _keyword_score(self, stage: dict, keywords: list[str]) -> int:
        """Count how many trigger keywords match the input keywords."""
        trigger = set(k.lower() for k in stage.get("trigger_keywords", []))
        input_set = set(k.lower() for k in keywords)
        return len(trigger & input_set)

    def _signal_score(self, stage: dict, feature_contributions: dict[str, float]) -> float:
        """
        Weight a stage by the magnitude of contribution from its trigger signals.
        Higher absolute contribution = more relevant.
        """
        trigger_signals = stage.get("trigger_signals", [])
        total = 0.0
        for sig in trigger_signals:
            if sig in feature_contributions:
                total += abs(feature_contributions[sig])
        return total

    def retrieve(
        self,
        feature_contributions: dict[str, float],
        keywords: Optional[list] = None,
        top_n: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top_n most relevant roadmap stages.

        Args:
            feature_contributions: per-feature contribution from regression model.
            keywords: optional merchant-context keywords (e.g. "loyalty", "digital").
            top_n: number of stages to return.

        Returns:
            List of roadmap stage dicts, ordered by relevance.
        """
        if not self._loaded:
            self.load()

        scored = []
        for stage in self.stages:
            sig_score = self._signal_score(stage, feature_contributions)
            kw_score = self._keyword_score(stage, keywords or [])
            combined = sig_score * 2.0 + kw_score * 1.0  # signal weighted higher
            scored.append((combined, stage))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [stage for _, stage in scored[:top_n]]

    def get_stage_by_id(self, stage_id: str) -> Optional[dict]:
        """Direct lookup by stage ID."""
        if not self._loaded:
            self.load()
        for stage in self.stages:
            if stage["id"] == stage_id:
                return stage
        return None

    def get_all_stages(self) -> list[dict[str, Any]]:
        """Returns all roadmap stages."""
        if not self._loaded:
            self.load()
        return self.stages
