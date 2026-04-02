# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Core environment logic for Veritas AI.

Implements the three OpenEnv methods:
  reset()  → VeritasObservation   (start new episode)
  step()   → VeritasObservation   (take one action)
  state    → VeritasState         (episode metadata)

This class is instantiated once per server process.
All episode state lives in instance variables — reset()
clears everything cleanly for a new episode.
"""

import random
import uuid
from typing import Any, Dict, List, Optional

from models import (
    VeritasAction,
    VeritasObservation,
    VeritasState,
)
from veritas_env.data_generator import generate_scenario
from veritas_env.reward import (
    calculate_step_reward,
    calculate_report_reward,
)
from veritas_env.tasks import TASKS, GRADERS, TASK_ORDER


class VeritasEnvironment:
    """
    Veritas AI Financial Crime Investigation Environment.

    The agent plays the role of a financial crime analyst.
    It queries accounts and transactions, flags suspicious
    accounts, and submits a final investigation report.

    Episode lifecycle:
      1. reset()        — fresh scenario generated
      2. step() × N    — agent investigates
      3. submit_report  — grader scores the report, done=True
      (or max_steps reached — done=True, partial score returned)
    """

    def __init__(self, task_id: Optional[str] = None):
        """
        Args:
            task_id: pin to one task, or None to cycle through all.
        """
        self._task_id_override  = task_id
        self._episode_index     = 0   # cycles through TASK_ORDER

        # Episode state — all reset in reset()
        self._scenario:   Optional[Dict] = None
        self._task_id:    str = ""
        self._episode_id: str = ""
        self._step_count: int = 0
        self._flagged:    List[str] = []
        self._solved:     bool = False
        self._cumulative_reward: float = 0.0
        self._best_score:        float = 0.0
        self._investigation_log: List[str] = []
        self._seed: int = 0

    # ─────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────

    def reset(self) -> VeritasObservation:
        """Start a fresh episode with a new scenario."""

        # Pick task
        if self._task_id_override:
            self._task_id = self._task_id_override
        else:
            self._task_id = TASK_ORDER[
                self._episode_index % len(TASK_ORDER)
            ]
            self._episode_index += 1

        task = TASKS[self._task_id]

        # Generate fresh scenario
        self._seed       = random.randint(0, 999_999)
        self._scenario   = generate_scenario(self._task_id, self._seed)

        # Reset all episode state
        self._episode_id         = str(uuid.uuid4())
        self._step_count         = 0
        self._flagged            = []
        self._solved             = False
        self._cumulative_reward  = 0.0
        self._best_score         = 0.0
        self._investigation_log  = []

        return VeritasObservation(
            # Episode context
            case_id          = self._scenario["case_id"],
            task_id          = self._task_id,
            difficulty       = task.difficulty,
            task_description = task.description,
            # What the agent sees at the start
            initial_alerts   = self._scenario["alerts"],
            accounts_in_scope= self._scenario["accounts_in_scope"],
            flagged_accounts = [],
            # No action taken yet
            action_result    = None,
            action_error     = None,
            # Progress
            partial_score    = 0.0,
            feedback         = (
                "New case opened. Review the alerts and begin "
                "your investigation."
            ),
            # OpenEnv base fields
            done             = False,
            reward           = 0.0,
            # Step tracking
            steps_taken      = 0,
            max_steps        = task.max_steps,
        )

    # ─────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────

    def step(self, action: VeritasAction) -> VeritasObservation:
        """
        Execute one action and return updated observation.
        Never raises — all errors returned in action_error field.
        """
        if self._scenario is None:
            return self._error_obs("Call reset() before step().")

        if self._solved:
            return self._error_obs(
                "Episode already complete. Call reset() to start a new case."
            )

        self._step_count += 1
        task     = TASKS[self._task_id]
        max_steps = task.max_steps

        # Route to handler
        action_type = (action.action_type or "").strip().lower()
        action_result = None
        action_error  = None
        reward        = 0.0
        partial_score = self._best_score
        feedback      = ""

        if action_type == "query_transactions":
            action_result, action_error = self._handle_query(action)
            if action_error is None:
                is_susp = self._is_suspicious(action.account_id)
                reward = calculate_step_reward(
                    action_type        = action_type,
                    account_id         = action.account_id,
                    is_suspicious      = is_susp,
                    is_already_flagged = action.account_id in self._flagged,
                    correct_suspect    = self._scenario["primary_suspect"],
                    correct_associates = self._scenario["associates"],
                )
                feedback = (
                    f"Retrieved {len(action_result)} transaction(s) "
                    f"for {action.account_id}."
                )

        elif action_type == "lookup_account":
            action_result, action_error = self._handle_lookup(action)
            if action_error is None:
                is_susp = self._is_suspicious(action.account_id)
                reward = calculate_step_reward(
                    action_type        = action_type,
                    account_id         = action.account_id,
                    is_suspicious      = is_susp,
                    is_already_flagged = action.account_id in self._flagged,
                    correct_suspect    = self._scenario["primary_suspect"],
                    correct_associates = self._scenario["associates"],
                )
                feedback = (
                    f"Account profile retrieved for {action.account_id}."
                )

        elif action_type == "flag_account":
            action_result, action_error = self._handle_flag(action)
            if action_error is None:
                already = action.account_id in self._flagged
                is_susp = self._is_suspicious(action.account_id)
                reward = calculate_step_reward(
                    action_type        = action_type,
                    account_id         = action.account_id,
                    is_suspicious      = is_susp,
                    is_already_flagged = already,
                    correct_suspect    = self._scenario["primary_suspect"],
                    correct_associates = self._scenario["associates"],
                )
                if not already:
                    self._flagged.append(action.account_id)
                feedback = (
                    f"Account {action.account_id} flagged."
                    if not already
                    else f"Account {action.account_id} already flagged."
                )

        elif action_type == "submit_report":
            action_result, action_error, reward, partial_score, feedback = (
                self._handle_report(action)
            )

        else:
            action_error = (
                f"Unknown action_type '{action.action_type}'. "
                "Must be one of: query_transactions | lookup_account "
                "| flag_account | submit_report"
            )
            reward = -0.05   # penalty for invalid action

        # Update running totals
        self._cumulative_reward += reward
        if partial_score > self._best_score:
            self._best_score = partial_score

        # Log this action
        self._investigation_log.append(
            f"Step {self._step_count}: {action_type} "
            f"→ reward={reward:+.3f}"
        )

        # Check episode termination
        done = self._solved or (self._step_count >= max_steps)
        if done and not self._solved:
            feedback += (
                f" Episode ended after {self._step_count} steps. "
                f"Best score: {self._best_score:.2f}."
            )

        return VeritasObservation(
            case_id           = self._scenario["case_id"],
            task_id           = self._task_id,
            difficulty        = task.difficulty,
            task_description  = task.description,
            initial_alerts    = self._scenario["alerts"],
            accounts_in_scope = self._scenario["accounts_in_scope"],
            action_result     = action_result,
            action_error      = action_error,
            flagged_accounts  = list(self._flagged),
            partial_score     = partial_score,
            feedback          = feedback,
            done              = done,
            reward            = reward,
            steps_taken       = self._step_count,
            max_steps         = max_steps,
        )

    # ─────────────────────────────────────────────────────
    # state (property)
    # ─────────────────────────────────────────────────────

    @property
    def state(self) -> VeritasState:
        """Return current episode metadata."""
        return VeritasState(
            episode_id         = self._episode_id,
            step_count         = self._step_count,
            task_id            = self._task_id,
            difficulty         = TASKS[self._task_id].difficulty
                                 if self._task_id else "",
            cumulative_reward  = round(self._cumulative_reward, 4),
            best_score         = round(self._best_score, 4),
            solved             = self._solved,
            investigation_log  = list(self._investigation_log),
        )

    # ─────────────────────────────────────────────────────
    # Action handlers (private)
    # ─────────────────────────────────────────────────────

    def _handle_query(
        self, action: VeritasAction
    ) -> tuple[Optional[List], Optional[str]]:
        """Handle query_transactions action."""
        if not action.account_id:
            return None, "account_id is required for query_transactions."

        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."

        txns = [
            t for t in self._scenario["transactions"]
            if t["from_account"] == acc_id or t["to_account"] == acc_id
        ]

        # Apply optional filters
        if action.date_from:
            txns = [t for t in txns if t["date"] >= action.date_from]
        if action.date_to:
            txns = [t for t in txns if t["date"] <= action.date_to]
        if action.min_amount is not None:
            txns = [t for t in txns if t["amount"] >= action.min_amount]
        if action.max_amount is not None:
            txns = [t for t in txns if t["amount"] <= action.max_amount]

        return txns, None

    def _handle_lookup(
        self, action: VeritasAction
    ) -> tuple[Optional[Dict], Optional[str]]:
        """Handle lookup_account action."""
        if not action.account_id:
            return None, "account_id is required for lookup_account."

        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."

        return self._scenario["accounts"][acc_id], None

    def _handle_flag(
        self, action: VeritasAction
    ) -> tuple[Optional[str], Optional[str]]:
        """Handle flag_account action."""
        if not action.account_id:
            return None, "account_id is required for flag_account."

        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."

        return f"Account {acc_id} flagged for review.", None

    def _handle_report(
        self, action: VeritasAction
    ) -> tuple[Optional[str], Optional[str], float, float, str]:
        """
        Handle submit_report action.
        Returns (action_result, action_error, reward, partial_score, feedback)
        """
        if not action.primary_suspect:
            return (
                None,
                "primary_suspect is required for submit_report.",
                -0.05, self._best_score, "Report rejected — missing suspect."
            )

        grader = GRADERS[self._task_id]
        partial_score = grader(
            primary_suspect    = action.primary_suspect,
            associates         = action.associates or [],
            case_type          = action.case_type,
            evidence_summary   = action.evidence_summary,
            correct_suspect    = self._scenario["primary_suspect"],
            correct_associates = self._scenario["associates"],
        )

        reward, _ = calculate_report_reward(
            primary_suspect    = action.primary_suspect,
            associates         = action.associates,
            case_type          = action.case_type,
            evidence_summary   = action.evidence_summary,
            correct_suspect    = self._scenario["primary_suspect"],
            correct_associates = self._scenario["associates"],
            correct_case_type  = self._scenario["case_type"],
            already_solved     = self._solved,
        )

        if partial_score >= 1.0:
            self._solved = True
            feedback = (
                "Outstanding. Report accepted. Case closed. "
                f"Final score: {partial_score:.2f}."
            )
        elif partial_score >= 0.7:
            feedback = (
                f"Good investigation. Score: {partial_score:.2f}. "
                "Some elements of your report need refinement."
            )
        elif partial_score >= 0.4:
            feedback = (
                f"Partial credit. Score: {partial_score:.2f}. "
                "Key elements of the scheme were missed."
            )
        else:
            feedback = (
                f"Report filed. Score: {partial_score:.2f}. "
                "Significant elements of the scheme were not identified."
            )

        return (
            f"Report submitted. Score: {partial_score:.2f}",
            None,
            reward,
            partial_score,
            feedback,
        )

    # ─────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────

    def _is_suspicious(self, account_id: Optional[str]) -> bool:
        """Check if account is part of the scheme (not an innocent)."""
        if not account_id or not self._scenario:
            return False
        return (
            account_id == self._scenario["primary_suspect"]
            or account_id in self._scenario["associates"]
        )

    def _error_obs(self, message: str) -> VeritasObservation:
        """Return a minimal error observation without crashing."""
        task_id = self._task_id or "task_easy"
        task    = TASKS.get(task_id, TASKS["task_easy"])
        return VeritasObservation(
            case_id           = self._scenario["case_id"]
                                 if self._scenario else "",
            task_id           = task_id,
            difficulty        = task.difficulty,
            task_description  = task.description,
            initial_alerts    = [],
            accounts_in_scope = [],
            action_result     = None,
            action_error      = message,
            flagged_accounts  = [],
            partial_score     = 0.0,
            feedback          = message,
            done              = False,
            reward            = 0.0,
            steps_taken       = self._step_count,
            max_steps         = task.max_steps,
        )