"""Human-in-the-Loop Operator Feedback Memory Bank.

Stores operator-annotated false positive signatures and suppresses repeat alarms
on known benign operational events (e.g. approved maintenance, deployment blips)
using causal cosine similarity matching over temporal feature vectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FeedbackRule:
    rule_id: str
    series_id: str
    feature_signature: np.ndarray
    reason: str
    created_at_step: int
    suppression_weight: float = 0.85


class OperatorFeedbackBank:
    """Manages active human feedback and suppresses matching spurious alarms."""

    def __init__(self, similarity_threshold: float = 0.92) -> None:
        self.similarity_threshold = similarity_threshold
        self._rules: dict[str, list[FeedbackRule]] = {}
        self._rule_counter = 0

    def add_feedback(
        self,
        series_id: str,
        feature_vector: np.ndarray,
        reason: str = "Operator confirmed benign",
        step_index: int = 0,
        suppression_weight: float = 0.85,
    ) -> str:
        """Register an operator feedback rule suppressing similar future signatures."""
        self._rule_counter += 1
        rule_id = f"FB_{series_id}_{self._rule_counter:04d}"

        # Normalize feature vector for cosine matching
        vec = np.array(feature_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        rule = FeedbackRule(
            rule_id=rule_id,
            series_id=str(series_id),
            feature_signature=vec,
            reason=reason,
            created_at_step=step_index,
            suppression_weight=suppression_weight,
        )

        if str(series_id) not in self._rules:
            self._rules[str(series_id)] = []
        self._rules[str(series_id)].append(rule)
        return rule_id

    def check_suppression(
        self, series_id: str, current_feature_vector: np.ndarray
    ) -> tuple[bool, float, Optional[str]]:
        """Check if current temporal signature matches any operator feedback rule.

        Returns:
            (is_suppressed, discount_factor, matching_rule_id)
        """
        rules = self._rules.get(str(series_id), [])
        if not rules:
            return False, 1.0, None

        vec = np.array(current_feature_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return False, 1.0, None
        norm_vec = vec / norm

        best_sim = 0.0
        best_rule: Optional[FeedbackRule] = None

        for rule in rules:
            if len(rule.feature_signature) == len(norm_vec):
                sim = float(np.dot(rule.feature_signature, norm_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_rule = rule

        if best_rule is not None and best_sim >= self.similarity_threshold:
            discount = 1.0 - best_rule.suppression_weight
            return True, discount, best_rule.rule_id

        return False, 1.0, None

    def list_rules(self, series_id: Optional[str] = None) -> list[dict]:
        """List active feedback rules."""
        out = []
        all_rules = (
            self._rules.get(str(series_id), [])
            if series_id
            else [r for rules in self._rules.values() for r in rules]
        )
        for r in all_rules:
            out.append({
                "rule_id": r.rule_id,
                "series_id": r.series_id,
                "reason": r.reason,
                "suppression_weight": r.suppression_weight,
                "created_at_step": r.created_at_step,
            })
        return out

    def clear(self, series_id: Optional[str] = None) -> None:
        if series_id:
            self._rules.pop(str(series_id), None)
        else:
            self._rules.clear()
