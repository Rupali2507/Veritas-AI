# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions and graders for the Veritas AI Environment.

Each task defines:
  task_id          : unique string identifier
  difficulty       : easy | medium | hard
  description      : what the agent must accomplish
  max_steps        : episode step limit
  case_type        : which scenario type to generate
  grader()         : pure function → float 0.0-1.0

Grader design rules:
  - Pure function: no side effects, no randomness
  - Deterministic: same inputs always return same score
  - Partial credit: never just 0 or 1, always granular
  - Transparent: every point has a clear reason
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────
# TASK DATACLASS
# ─────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id:     str
    difficulty:  str
    description: str
    max_steps:   int
    case_type:   str
    hints:       List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# GRADER HELPERS
# ─────────────────────────────────────────────────────────

EVIDENCE_KEYWORDS = [
    "device", "ip", "velocity", "structuring", "threshold",
    "pattern", "linked", "shared", "chain", "layering",
    "mule", "coordinated",
]
EVIDENCE_KEYWORD_MIN = 2


def _clamp(score: float) -> float:
    """Ensure score is strictly open (0, 1) as required by the validator."""
    return round(max(0.01, min(0.99, score)), 4)


def _evidence_quality(summary: Optional[str]) -> float:
    """Return 0.15 if summary contains enough domain keywords, else 0."""
    if not summary:
        return 0.0
    lower = summary.lower()
    hits = sum(1 for kw in EVIDENCE_KEYWORDS if kw in lower)
    return 0.15 if hits >= EVIDENCE_KEYWORD_MIN else 0.0


def _associate_score(
    submitted: Optional[List[str]],
    correct: List[str],
    per_correct: float,
    max_contribution: float,
) -> float:
    """
    Score associate list with partial credit per correct entry.
    Caps at max_contribution regardless of list size.
    """
    if not submitted or not correct:
        return 0.0
    correct_set   = set(correct)
    submitted_set = set(submitted)
    true_positives = len(submitted_set & correct_set)
    contribution = true_positives * per_correct
    return round(min(contribution, max_contribution), 4)


# ─────────────────────────────────────────────────────────
# TASK 1 — EASY — Card Scheme
# ─────────────────────────────────────────────────────────

TASK_EASY = Task(
    task_id     = "task_easy",
    difficulty  = "easy",
    case_type   = "card_scheme",
    max_steps   = 8,
    description = (
        "An alert has been raised on a case involving suspicious "
        "high-value purchases at unlisted merchants. "
        "Investigate the accounts in scope, identify the primary "
        "suspect behind the card scheme, and submit your report. "
        "Set case_type to 'card_scheme' in your report."
    ),
    hints = [
        "Start by looking up the account mentioned in the alert.",
        "Query that account's transactions — look at merchant names "
        "and transaction amounts.",
        "High-value purchases at suspicious merchants in a short "
        "period is the key signal.",
        "Flag the account you are most confident about, then submit.",
    ],
)


def grade_easy(
    primary_suspect:  Optional[str],
    associates:       Optional[List[str]],
    case_type:        Optional[str],
    evidence_summary: Optional[str],
    correct_suspect:  str,
    correct_associates: List[str],   # empty for task_easy
) -> float:
    """
    Grader for task_easy.

    Scoring breakdown (max 1.0):
      0.50 — correct primary suspect
      0.20 — correct case_type ('card_scheme')
      0.15 — evidence summary mentions relevant signals
      0.15 — no innocent accounts in associates list
    """
    score = 0.0

    # 0.50 — correct suspect
    if primary_suspect == correct_suspect:
        score += 0.50

    # 0.20 — correct case type
    if case_type and case_type.strip().lower() == "card_scheme":
        score += 0.20

    # 0.15 — evidence quality
    score += _evidence_quality(evidence_summary)

    # 0.15 — did not falsely accuse innocent accounts
    # For task_easy there are no real associates, so any
    # non-empty associates list means false accusations
    submitted_assoc = set(associates or [])
    if len(submitted_assoc) == 0:
        score += 0.15
    else:
        # Partial: penalise per false accusation but don't go negative
        false_count = len(submitted_assoc - {correct_suspect})
        penalty = false_count * 0.05
        score += max(0.0, 0.15 - penalty)

    return _clamp(score)


# ─────────────────────────────────────────────────────────
# TASK 2 — MEDIUM — Layering Scheme
# ─────────────────────────────────────────────────────────

TASK_MEDIUM = Task(
    task_id     = "task_medium",
    difficulty  = "medium",
    case_type   = "layering_scheme",
    max_steps   = 12,
    description = (
        "A structuring pattern has been flagged — repeated peer "
        "transfers just below the ₹10,000 reporting threshold. "
        "Trace the full money movement chain. Identify the origin "
        "account (primary suspect) and the intermediate accounts "
        "(associates). Set case_type to 'layering_scheme'."
    ),
    hints = [
        "Start with the account in the alert.",
        "Query its transactions and look for large peer transfers.",
        "Follow the money — the recipient account is likely an "
        "intermediary. Look up that account too.",
        "The amounts will be consistently just under ₹10,000.",
        "You need to trace the full chain to get full credit.",
    ],
)


def grade_medium(
    primary_suspect:    Optional[str],
    associates:         Optional[List[str]],
    case_type:          Optional[str],
    evidence_summary:   Optional[str],
    correct_suspect:    str,
    correct_associates: List[str],
) -> float:
    """
    Grader for task_medium.

    Scoring breakdown (max 1.0):
      0.35 — correct primary suspect
      0.30 — correct associates (0.10 each, max 0.30)
      0.15 — correct case_type ('layering_scheme')
      0.15 — evidence summary mentions relevant signals
      0.05 — no false associates beyond the real chain
    """
    score = 0.0

    # 0.35 — correct suspect
    if primary_suspect == correct_suspect:
        score += 0.35
    elif primary_suspect in correct_associates:
        # Agent found someone in the chain but not the origin
        score += 0.10

    # 0.30 — associates (partial credit per correct one)
    score += _associate_score(
        submitted        = associates,
        correct          = correct_associates,
        per_correct      = 0.10,
        max_contribution = 0.30,
    )

    # 0.15 — correct case type
    if case_type and case_type.strip().lower() == "layering_scheme":
        score += 0.15

    # 0.15 — evidence quality
    score += _evidence_quality(evidence_summary)

    # 0.05 — no false associates
    submitted_set = set(associates or [])
    correct_set   = set(correct_associates) | {correct_suspect}
    false_positives = submitted_set - correct_set
    if len(false_positives) == 0:
        score += 0.05

    return _clamp(score)


# ─────────────────────────────────────────────────────────
# TASK 3 — HARD — Coordinated Scheme
# ─────────────────────────────────────────────────────────

TASK_HARD = Task(
    task_id     = "task_hard",
    difficulty  = "hard",
    case_type   = "coordinated_scheme",
    max_steps   = 18,
    description = (
        "A coordinated activity alert has been raised. Multiple "
        "accounts are suspected to be controlled by a single "
        "operator and are moving funds to a common external account. "
        "Use device IDs and IP address patterns to identify which "
        "accounts are linked. Name the primary operator and all "
        "associates. Set case_type to 'coordinated_scheme'."
    ),
    hints = [
        "Start by looking up every account in the initial alert.",
        "Pay close attention to device_id and ip_address fields.",
        "Accounts sharing the same device ID are likely controlled "
        "by the same person.",
        "Accounts in the same IP subnet (first 3 octets match) "
        "suggest the same location or operator.",
        "The primary suspect is the account with the most shared "
        "device links.",
        "You need to name all associates for full credit — partial "
        "credit is given per correct associate.",
    ],
)


def grade_hard(
    primary_suspect:    Optional[str],
    associates:         Optional[List[str]],
    case_type:          Optional[str],
    evidence_summary:   Optional[str],
    correct_suspect:    str,
    correct_associates: List[str],
) -> float:
    """
    Grader for task_hard.

    Scoring breakdown (max 1.0):
      0.30 — correct primary suspect
      0.32 — correct associates (0.08 each, max 0.32 for 4+ associates)
      0.10 — correct case_type ('coordinated_scheme')
      0.15 — evidence mentions device/IP linkage signals
      0.08 — no false associates
      0.05 — partial credit if wrong suspect but right associates
    """
    score = 0.0

    # 0.30 — correct suspect
    suspect_correct = (primary_suspect == correct_suspect)
    if suspect_correct:
        score += 0.30
    elif primary_suspect in correct_associates:
        # Found a ring member but not the operator
        score += 0.08

    # 0.32 — associates
    n_correct = max(len(correct_associates), 1)
    per_assoc = round(0.32 / n_correct, 4)
    score += _associate_score(
        submitted        = associates,
        correct          = correct_associates,
        per_correct      = per_assoc,
        max_contribution = 0.32,
    )

    # 0.10 — correct case type
    if case_type and case_type.strip().lower() == "coordinated_scheme":
        score += 0.10

    # 0.15 — evidence mentions device/IP signals specifically
    evidence_score = _evidence_quality(evidence_summary)
    # Extra check: hard task specifically requires device/IP reasoning
    if evidence_summary:
        lower = evidence_summary.lower()
        if "device" in lower or "ip" in lower:
            score += evidence_score
        else:
            score += evidence_score * 0.5
    # else: no evidence summary, no score

    # 0.08 — no false associates
    submitted_set = set(associates or [])
    correct_set   = set(correct_associates) | {correct_suspect}
    false_positives = submitted_set - correct_set
    if len(false_positives) == 0:
        score += 0.08

    # 0.05 — partial credit: wrong suspect but got most associates
    if not suspect_correct and associates:
        submitted_set   = set(associates)
        correct_set_raw = set(correct_associates)
        overlap_ratio = len(submitted_set & correct_set_raw) / max(len(correct_set_raw), 1)
        if overlap_ratio >= 0.6:
            score += 0.05

    return _clamp(score)


# ─────────────────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────────────────

TASKS = {
    "task_easy":   TASK_EASY,
    "task_medium": TASK_MEDIUM,
    "task_hard":   TASK_HARD,
}

TASK_ORDER = ["task_easy", "task_medium", "task_hard"]

GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
}