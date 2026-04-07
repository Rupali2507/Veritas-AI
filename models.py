# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Veritas AI Environment.

Veritas AI is a financial crime investigation environment where an agent
acts as an analyst — querying accounts, gathering evidence, and filing
investigation reports.

Three models define the complete data contract:
  VeritasAction      : what the agent sends IN  (its move)
  VeritasObservation : what the agent gets BACK (what it sees)
  VeritasState       : episode metadata          (bookkeeping)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Stubs for standalone inference (validator runs without openenv-core)
    class Action(BaseModel):       # type: ignore[no-redef]
        pass
    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: dict = {}
    class State(BaseModel):        # type: ignore[no-redef]
        episode_id: str = ""
        step_count: int = 0


# ─────────────────────────────────────────────────────────
# ACTION — the agent's move each step
# ─────────────────────────────────────────────────────────

class VeritasAction(Action):
    """
    One action the agent takes per step.

    The agent picks ONE action_type per step:
      query_transactions — pull transaction rows for an account
      lookup_account     — get full profile of one account
      flag_account       — mark an account as suspicious
      submit_report      — final move, name the suspect, ends episode
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: query_transactions | lookup_account "
            "| flag_account | submit_report"
        )
    )

    # ── Used by: query_transactions, lookup_account, flag_account ──
    account_id: Optional[str] = Field(
        default=None,
        description="Account ID to query, look up, or flag"
    )

    # ── Used by: query_transactions (optional filters) ──
    date_from: Optional[str] = Field(
        default=None,
        description="Start date filter YYYY-MM-DD"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date filter YYYY-MM-DD"
    )
    min_amount: Optional[float] = Field(
        default=None,
        description="Minimum transaction amount filter"
    )
    max_amount: Optional[float] = Field(
        default=None,
        description="Maximum transaction amount filter"
    )

    # ── Used by: flag_account ──
    reason: Optional[str] = Field(
        default=None,
        description="Why this account is being flagged"
    )

    # ── Used by: submit_report (the agent's final conclusion) ──
    primary_suspect: Optional[str] = Field(
        default=None,
        description="Account ID of the main suspect"
    )
    associates: Optional[List[str]] = Field(
        default=None,
        description="List of associate account IDs"
    )
    case_type: Optional[str] = Field(
        default=None,
        description=(
            "One of: card_scheme | layering_scheme | "
            "coordinated_scheme"
        )
    )
    evidence_summary: Optional[str] = Field(
        default=None,
        description="Agent's written summary of evidence found"
    )


# ─────────────────────────────────────────────────────────
# OBSERVATION — what the agent sees after each step
# ─────────────────────────────────────────────────────────

class VeritasObservation(Observation):
    """
    What the agent receives back after every reset() and step().

    Important: done, reward, metadata are already defined in the
    base Observation class — do NOT redefine them here.
    We only add Veritas-specific fields.
    """

    # ── Case context — set at reset(), stays constant ──
    case_id: str = Field(
        default="",
        description="Unique ID for this investigation case"
    )
    task_id: str = Field(
        default="",
        description="Which task is active this episode"
    )
    difficulty: str = Field(
        default="",
        description="easy | medium | hard"
    )
    task_description: str = Field(
        default="",
        description="Full text of what the agent must accomplish"
    )

    # ── What the agent sees about the case ──
    initial_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="System alerts that opened this case, given at reset()"
    )
    accounts_in_scope: List[str] = Field(
        default_factory=list,
        description="Account IDs the agent is allowed to investigate"
    )

    # ── Result of the last action ──
    action_result: Optional[Any] = Field(
        default=None,
        description="Data returned by the last action"
    )
    action_error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid"
    )

    # ── Running investigation state ──
    flagged_accounts: List[str] = Field(
        default_factory=list,
        description="Accounts the agent has flagged so far"
    )

    # ── Progress signals ──
    partial_score: float = Field(
        default=0.0,
        description="Grader score 0.0-1.0, updated on submit_report"
    )
    feedback: str = Field(
        default="",
        description="Human-readable hint about current progress"
    )

    # ── Step tracking ──
    steps_taken: int = Field(
        default=0,
        description="Steps used so far this episode"
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed this episode"
    )


# ─────────────────────────────────────────────────────────
# STATE — episode bookkeeping, returned by state()
# ─────────────────────────────────────────────────────────

class VeritasState(State):
    """
    Episode-level metadata returned by state().

    Important: episode_id and step_count are already in the
    base State class — do NOT redefine them here.
    We only add Veritas-specific tracking fields.
    """

    task_id: str = Field(default="")
    difficulty: str = Field(default="")

    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated this episode"
    )
    best_score: float = Field(
        default=0.0,
        description="Highest partial_score seen so far"
    )
    solved: bool = Field(
        default=False,
        description="True if agent submitted a perfect report"
    )
    investigation_log: List[str] = Field(
        default_factory=list,
        description="History of every action taken — for debugging"
    )