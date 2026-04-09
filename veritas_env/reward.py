# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward constants and calculator for the Veritas AI Environment.

All reward numbers live here in one place so they are easy to tune,
audit, and explain to judges. The calculate_step_reward() and
calculate_report_reward() functions are called by environment.py.

Reward design philosophy:
  - Dense signals every step so RL training is not sparse
  - Partial credit on report so agent learns incrementally
  - Penalties for reckless behaviour (false accusations)
  - Efficiency bonus for solving in fewer steps
"""

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────
# STEP-LEVEL REWARD CONSTANTS
# ─────────────────────────────────────────────────────────

STEP_PENALTY             = -0.02
QUERY_SUSPICIOUS_ACCOUNT = +0.05
QUERY_INNOCENT_ACCOUNT   = -0.02
FLAG_CORRECT_SUSPECT     = +0.10
FLAG_CORRECT_ASSOCIATE   = +0.05
FLAG_INNOCENT_ACCOUNT    = -0.10


# ─────────────────────────────────────────────────────────
# REPORT-LEVEL REWARD CONSTANTS
# ─────────────────────────────────────────────────────────

REPORT_CORRECT_SUSPECT   = +0.40
REPORT_WRONG_SUSPECT     = -0.20
REPORT_CORRECT_ASSOCIATE = +0.10
REPORT_FALSE_ASSOCIATE   = -0.05
REPORT_CORRECT_CASE_TYPE = +0.20
REPORT_GOOD_EVIDENCE     = +0.15
SOLVE_BONUS              = +0.25


# ─────────────────────────────────────────────────────────
# EVIDENCE KEYWORDS
# ─────────────────────────────────────────────────────────

EVIDENCE_KEYWORDS = [
    "device", "ip", "velocity", "structuring", "threshold",
    "pattern", "linked", "shared", "chain", "layering",
    "mule", "coordinated",
]
EVIDENCE_KEYWORD_MIN = 2


# ─────────────────────────────────────────────────────────
# CLAMP HELPERS
# Keeps values in the ranges required by the validator
# ─────────────────────────────────────────────────────────

def _clamp_score(score: float) -> float:
    """Ensure partial_score is strictly open (0, 1) — validator requirement."""
    return round(max(0.01, min(0.99, score)), 4)


def _clamp_reward(reward: float) -> float:
    """Clamp reward to strictly open (-1, 1) — validator requirement."""
    return round(max(-0.99, min(0.99, reward)), 4)


# ─────────────────────────────────────────────────────────
# STEP REWARD CALCULATOR
# ─────────────────────────────────────────────────────────

def calculate_step_reward(
    action_type: str,
    account_id: Optional[str],
    is_suspicious: bool,
    is_already_flagged: bool,
    correct_suspect: str,
    correct_associates: List[str],
) -> float:
    reward = STEP_PENALTY

    if action_type in ("query_transactions", "lookup_account"):
        if account_id is None:
            return _clamp_reward(reward)
        reward += QUERY_SUSPICIOUS_ACCOUNT if is_suspicious else QUERY_INNOCENT_ACCOUNT

    elif action_type == "flag_account":
        if account_id is None or is_already_flagged:
            return _clamp_reward(reward)
        if account_id == correct_suspect:
            reward += FLAG_CORRECT_SUSPECT
        elif account_id in correct_associates:
            reward += FLAG_CORRECT_ASSOCIATE
        else:
            reward += FLAG_INNOCENT_ACCOUNT

    return _clamp_reward(reward)


# ─────────────────────────────────────────────────────────
# REPORT REWARD CALCULATOR
# ─────────────────────────────────────────────────────────

def calculate_report_reward(
    primary_suspect: Optional[str],
    associates: Optional[List[str]],
    case_type: Optional[str],
    evidence_summary: Optional[str],
    correct_suspect: str,
    correct_associates: List[str],
    correct_case_type: str,
    already_solved: bool,
) -> tuple[float, float]:
    if already_solved:
        return (0.0, _clamp_score(1.0))

    reward = STEP_PENALTY
    score  = 0.0

    associates       = associates or []
    evidence_summary = evidence_summary or ""

    # ── Primary suspect ──────────────────────────────────
    if primary_suspect == correct_suspect:
        reward += REPORT_CORRECT_SUSPECT
        score  += 0.40
    else:
        reward += REPORT_WRONG_SUSPECT

    # ── Associates ───────────────────────────────────────
    correct_set   = set(correct_associates)
    submitted_set = set(associates)

    true_positives  = submitted_set & correct_set
    false_positives = submitted_set - correct_set

    for _ in true_positives:
        reward += REPORT_CORRECT_ASSOCIATE
        if len(correct_set) > 0:
            score += 0.10 / len(correct_set) * len(correct_associates)

    for _ in false_positives:
        reward += REPORT_FALSE_ASSOCIATE

    score = min(score, 0.70)   # cap associate contribution

    # ── Case type ─────────────────────────────────────────
    if case_type and case_type.strip().lower() == correct_case_type.lower():
        reward += REPORT_CORRECT_CASE_TYPE
        score  += 0.20

    # ── Evidence quality ─────────────────────────────────
    keyword_hits = sum(1 for kw in EVIDENCE_KEYWORDS if kw in evidence_summary.lower())
    if keyword_hits >= EVIDENCE_KEYWORD_MIN:
        reward += REPORT_GOOD_EVIDENCE
        score  += 0.15

    # Clamp partial_score to strictly open (0, 1)
    partial_score = _clamp_score(score)

    # ── Solve bonus (matches SOLVE_THRESHOLD in environment.py) ──
    if partial_score >= 0.80:
        reward += SOLVE_BONUS

    return (_clamp_reward(reward), partial_score)