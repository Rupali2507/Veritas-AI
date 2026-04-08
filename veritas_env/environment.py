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
"""

import random
import uuid
from typing import Any, Dict, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:
    class Environment:  # type: ignore[no-redef]
        """Stub for when openenv-core is not installed (standalone inference)."""
        def __init__(self, *args, **kwargs):
            pass

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


class VeritasEnvironment(Environment):
    """
    Veritas AI Financial Crime Investigation Environment.
    """

    def __init__(self, task_id=None):
        # Safely call super().__init__() — the openenv base class may
        # require a server context that isn't present during standalone inference.
        try:
            super().__init__()
        except Exception:
            pass  # Safe to ignore — we don't need the server context here

        self._task_id_override  = task_id
        self._episode_index     = 0

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

    def reset(self, seed=None, episode_id=None, **kwargs) -> VeritasObservation:
        """Start a fresh episode with a new scenario."""

        if self._task_id_override:
            self._task_id = self._task_id_override
        else:
            self._task_id = TASK_ORDER[
                self._episode_index % len(TASK_ORDER)
            ]
            self._episode_index += 1

        task = TASKS[self._task_id]

        self._seed       = random.randint(0, 999_999)
        self._scenario   = generate_scenario(self._task_id, self._seed)

        self._episode_id         = str(uuid.uuid4())
        self._step_count         = 0
        self._flagged            = []
        self._solved             = False
        self._cumulative_reward  = 0.0
        self._best_score         = 0.0
        self._investigation_log  = []

        return VeritasObservation(
            case_id          = self._scenario["case_id"],
            task_id          = self._task_id,
            difficulty       = task.difficulty,
            task_description = task.description,
            initial_alerts   = self._scenario["alerts"],
            accounts_in_scope= self._scenario["accounts_in_scope"],
            flagged_accounts = [],
            action_result    = None,
            action_error     = None,
            partial_score    = 0.0,
            feedback         = (
                "New case opened. Review the alerts and begin "
                "your investigation."
            ),
            done             = False,
            reward           = 0.0,
            steps_taken      = 0,
            max_steps        = task.max_steps,
        )

    # ─────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────

    def step(self, action: VeritasAction, timeout_s=None, **kwargs) -> VeritasObservation:
        """Execute one action and return updated observation."""
        if self._scenario is None:
            return self._error_obs("Call reset() before step().")

        if self._solved:
            return self._error_obs(
                "Episode already complete. Call reset() to start a new case."
            )

        self._step_count += 1
        task      = TASKS[self._task_id]
        max_steps = task.max_steps

        action_type   = (action.action_type or "").strip().lower()
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
                feedback = f"Account profile retrieved for {action.account_id}."

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
            reward = -0.05

        self._cumulative_reward += reward
        if partial_score > self._best_score:
            self._best_score = partial_score

        self._investigation_log.append(
            f"Step {self._step_count}: {action_type} "
            f"-> reward={reward:+.3f}"
        )

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

    def _handle_query(self, action):
        if not action.account_id:
            return None, "account_id is required for query_transactions."
        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."
        txns = [
            t for t in self._scenario["transactions"]
            if t["from_account"] == acc_id or t["to_account"] == acc_id
        ]
        if action.date_from:
            txns = [t for t in txns if t["date"] >= action.date_from]
        if action.date_to:
            txns = [t for t in txns if t["date"] <= action.date_to]
        if action.min_amount is not None:
            txns = [t for t in txns if t["amount"] >= action.min_amount]
        if action.max_amount is not None:
            txns = [t for t in txns if t["amount"] <= action.max_amount]
        return txns, None

    def _handle_lookup(self, action):
        if not action.account_id:
            return None, "account_id is required for lookup_account."
        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."
        return self._scenario["accounts"][acc_id], None

    def _handle_flag(self, action):
        if not action.account_id:
            return None, "account_id is required for flag_account."
        acc_id = action.account_id
        if acc_id not in self._scenario["accounts"]:
            return None, f"Account '{acc_id}' not found in this case."
        return f"Account {acc_id} flagged for review.", None

    def _handle_report(self, action):
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
        if not account_id or not self._scenario:
            return False
        return (
            account_id == self._scenario["primary_suspect"]
            or account_id in self._scenario["associates"]
        )

    def _error_obs(self, message: str) -> VeritasObservation:
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

    def close(self) -> None:
        """Clean up episode resources."""
        self._scenario = None