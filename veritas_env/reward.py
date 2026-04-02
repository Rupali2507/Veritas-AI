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
# Applied every single step regardless of action type
# ─────────────────────────────────────────────────────────

# Small cost per step — encourages efficiency, discourages
# aimless querying of every account
STEP_PENALTY = -0.02

# Reward for querying an account that is genuinely suspicious
# (determined internally by checking against ground truth)
QUERY_SUSPICIOUS_ACCOUNT = +0.05

# Mild penalty for querying an account with no suspicious activity
# Teaches the agent to be targeted, not random
QUERY_INNOCENT_ACCOUNT = -0.02

# Immediate reward when agent flags the correct primary suspect
FLAG_CORRECT_SUSPECT = +0.10

# Immediate reward for flagging a correct associate
FLAG_CORRECT_ASSOCIATE = +0.05

# Immediate penalty for flagging an innocent account
# False positives are costly in real investigation work
FLAG_INNOCENT_ACCOUNT = -0.10


# ─────────────────────────────────────────────────────────
# REPORT-LEVEL REWARD CONSTANTS
# Applied only when agent calls submit_report
# ─────────────────────────────────────────────────────────

# Correctly named the primary suspect — highest single reward
REPORT_CORRECT_SUSPECT = +0.40

# Wrong primary suspect — serious mistake
REPORT_WRONG_SUSPECT = -0.20

# Each correctly named associate (scales with ring size)
REPORT_CORRECT_ASSOCIATE = +0.10

# Each innocent account incorrectly listed as associate
REPORT_FALSE_ASSOCIATE = -0.05

# Correct case type classification
REPORT_CORRECT_CASE_TYPE = +0.20

# Evidence summary mentions key indicators
# (shared device, IP pattern, structuring, velocity)
REPORT_GOOD_EVIDENCE = +0.15

# Bonus for perfect report on first submission
SOLVE_BONUS = +0.25


# ─────────────────────────────────────────────────────────
# EVIDENCE KEYWORDS
# Used to score evidence_summary quality
# Agent gets REPORT_GOOD_EVIDENCE if summary contains
# at least 2 of these domain-relevant terms
# ─────────────────────────────────────────────────────────

EVIDENCE_KEYWORDS = [
    "device",
    "ip",
    "velocity",
    "structuring",
    "threshold",
    "pattern",
    "linked",
    "shared",
    "chain",
    "layering",
    "mule",
    "coordinated",
]

EVIDENCE_KEYWORD_MIN = 2   # minimum matches to earn evidence reward


# ─────────────────────────────────────────────────────────
# STEP REWARD CALCULATOR
# Called by environment.py after every action
# ─────────────────────────────────────────────────────────

def calculate_step_reward(
    action_type: str,
    account_id: Optional[str],
    is_suspicious: bool,
    is_already_flagged: bool,
    correct_suspect: str,
    correct_associates: List[str],
) -> float:
    """
    Calculate the reward for one step (non-report actions).

    Args:
        action_type     : the action the agent took
        account_id      : which account was acted on
        is_suspicious   : whether this account is part of the scheme
        is_already_flagged : whether agent already flagged this account
        correct_suspect : ground truth primary suspect account ID
        correct_associates : ground truth associate account IDs

    Returns:
        float reward for this step
    """
    reward = STEP_PENALTY   # always pay the step cost

    if action_type in ("query_transactions", "lookup_account"):
        if account_id is None:
            return reward
        if is_suspicious:
            reward += QUERY_SUSPICIOUS_ACCOUNT
        else:
            reward += QUERY_INNOCENT_ACCOUNT

    elif action_type == "flag_account":
        if account_id is None:
            return reward
        # Don't reward flagging the same account twice
        if is_already_flagged:
            return reward
        if account_id == correct_suspect:
            reward += FLAG_CORRECT_SUSPECT
        elif account_id in correct_associates:
            reward += FLAG_CORRECT_ASSOCIATE
        else:
            reward += FLAG_INNOCENT_ACCOUNT

    return round(reward, 4)


# ─────────────────────────────────────────────────────────
# REPORT REWARD CALCULATOR
# Called by environment.py only on submit_report
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
    """
    Calculate the reward for a submit_report action.
    Also returns the partial_score (0.0-1.0) for the grader.

    Args:
        primary_suspect    : agent's named suspect
        associates         : agent's named associates
        case_type          : agent's classified case type
        evidence_summary   : agent's written evidence summary
        correct_suspect    : ground truth suspect
        correct_associates : ground truth associates
        correct_case_type  : ground truth case type
        already_solved     : True if agent already submitted correctly

    Returns:
        tuple of (reward: float, partial_score: float)
    """
    if already_solved:
        # Episode already complete, no more rewards
        return (0.0, 1.0)

    reward = STEP_PENALTY   # report still costs one step
    score = 0.0

    associates = associates or []
    evidence_summary = evidence_summary or ""

    # ── Primary suspect ──────────────────────────────────
    if primary_suspect == correct_suspect:
        reward += REPORT_CORRECT_SUSPECT
        score += 0.40
    else:
        reward += REPORT_WRONG_SUSPECT

    # ── Associates ───────────────────────────────────────
    correct_set = set(correct_associates)
    submitted_set = set(associates)

    true_positives = submitted_set & correct_set
    false_positives = submitted_set - correct_set

    for _ in true_positives:
        reward += REPORT_CORRECT_ASSOCIATE
        # Each correct associate contributes proportionally to score
        if len(correct_set) > 0:
            score += 0.10 / len(correct_set) * len(correct_associates)

    for _ in false_positives:
        reward += REPORT_FALSE_ASSOCIATE

    # Cap associate contribution to 0.30
    score = min(score, 0.70)

    # ── Case type ─────────────────────────────────────────
    if case_type and case_type.strip().lower() == correct_case_type.lower():
        reward += REPORT_CORRECT_CASE_TYPE
        score += 0.20

    # ── Evidence quality ─────────────────────────────────
    summary_lower = evidence_summary.lower()
    keyword_hits = sum(
        1 for kw in EVIDENCE_KEYWORDS if kw in summary_lower
    )
    if keyword_hits >= EVIDENCE_KEYWORD_MIN:
        reward += REPORT_GOOD_EVIDENCE
        score += 0.15

    # Normalise score to max 1.0
    partial_score = round(min(score, 1.0), 4)

    # ── Solve bonus ───────────────────────────────────────
    if partial_score >= 1.0:
        reward += SOLVE_BONUS

    return (round(reward, 4), partial_score)