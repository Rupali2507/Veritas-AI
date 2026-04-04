# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
inference.py — Baseline inference script for Veritas AI Environment.

MANDATORY requirements (hackathon spec):
  - Named exactly inference.py at project root
  - Uses OpenAI client for all LLM calls
  - Reads credentials from environment variables:
      API_BASE_URL : LLM API endpoint
      MODEL_NAME   : model identifier
      HF_TOKEN     : HuggingFace token (used as API key)
  - Runs all 3 tasks and prints reproducible scores
  - Must complete in under 20 minutes on vcpu=2, memory=8gb

Usage:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py
"""

import json
import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Add project root to path ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from veritas_env.environment import VeritasEnvironment
from veritas_env.tasks import TASK_ORDER, TASKS
from models import VeritasAction

# ── Credentials from environment variables ────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf_placeholder")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── Episode config ────────────────────────────────────────
MAX_STEPS   = 12
TEMPERATURE = 0.1
MAX_TOKENS  = 512

# ── OpenAI client (mandatory per hackathon spec) ──────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are Veritas, an expert financial crime investigation analyst.
    You are given an investigation case with suspicious account alerts.
    Your job is to investigate accounts, gather evidence, and identify
    the person responsible for the financial crime scheme.

    You have exactly 4 actions available:
      1. query_transactions — retrieve transactions for an account
      2. lookup_account     — get full profile of an account
      3. flag_account       — mark an account as suspicious
      4. submit_report      — file your final investigation report

    RULES:
    - Respond with ONLY a valid JSON object. No explanation, no markdown.
    - Choose exactly one action per response.
    - Always investigate before submitting a report.
    - For submit_report you must include:
        primary_suspect, associates (list), case_type, evidence_summary

    CASE TYPES:
      card_scheme         — high velocity purchases at suspicious merchants
      layering_scheme     — structured transfers just below reporting threshold
      coordinated_scheme  — multiple accounts linked by shared device or IP

    JSON FORMAT EXAMPLES:
    {"action_type": "lookup_account", "account_id": "ACC-1234"}
    {"action_type": "query_transactions", "account_id": "ACC-1234", "date_from": "2024-01-01", "date_to": "2024-03-31"}
    {"action_type": "flag_account", "account_id": "ACC-1234", "reason": "High velocity suspicious merchants"}
    {"action_type": "submit_report", "primary_suspect": "ACC-1234", "associates": [], "case_type": "card_scheme", "evidence_summary": "Account showed high velocity pattern at suspicious merchants with shared device linkage"}
""").strip()


# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    """Build the user prompt from current observation and history."""

    alerts_text = json.dumps(obs.get("initial_alerts", []), indent=2)
    accounts_text = ", ".join(obs.get("accounts_in_scope", []))
    history_text = "\n".join(history[-4:]) if history else "None yet."

    last_action_text = ""
    if obs.get("action_result") is not None:
        result = obs["action_result"]
        if isinstance(result, list):
            last_action_text = (
                f"\nLAST ACTION RESULT ({len(result)} rows):\n"
                + json.dumps(result[:5], indent=2)  # show max 5 rows
            )
        else:
            last_action_text = (
                f"\nLAST ACTION RESULT:\n{json.dumps(result, indent=2)}"
            )

    if obs.get("action_error"):
        last_action_text = f"\nLAST ACTION ERROR: {obs['action_error']}"

    return textwrap.dedent(f"""
        TASK ({obs.get('difficulty', '').upper()}):
        {obs.get('task_description', '')}

        CASE ID: {obs.get('case_id', '')}
        STEPS USED: {obs.get('steps_taken', 0)} / {obs.get('max_steps', 10)}

        INITIAL ALERTS:
        {alerts_text}

        ACCOUNTS IN SCOPE: {accounts_text}
        FLAGGED SO FAR: {', '.join(obs.get('flagged_accounts', [])) or 'None'}
        CURRENT SCORE: {obs.get('partial_score', 0.0):.2f}
        FEEDBACK: {obs.get('feedback', '')}

        RECENT ACTIONS:
        {history_text}
        {last_action_text}

        Respond with your next action as a single JSON object:
    """).strip()


# ─────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────

def call_llm(messages: List[Dict]) -> str:
    """Call the LLM via OpenAI client. Returns response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return ""


def parse_action(response_text: str) -> Optional[Dict]:
    """Extract JSON action from LLM response."""
    if not response_text:
        return None

    text = response_text.strip()

    # Strip markdown fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break

    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────
# SINGLE TASK RUNNER
# ─────────────────────────────────────────────────────────

def run_task(task_id: str) -> Dict[str, Any]:
    """Run one full investigation episode. Returns result dict."""
    task = TASKS[task_id]

    print(f"\n{'='*60}")
    print(f"TASK : {task_id}  ({task.difficulty.upper()})")
    print(f"{'='*60}")

    env     = VeritasEnvironment(task_id=task_id)
    obs     = env.reset()
    obs_dict = {
        "case_id":          obs.case_id,
        "task_description": obs.task_description,
        "difficulty":       obs.difficulty,
        "initial_alerts":   obs.initial_alerts,
        "accounts_in_scope":obs.accounts_in_scope,
        "flagged_accounts": obs.flagged_accounts,
        "action_result":    obs.action_result,
        "action_error":     obs.action_error,
        "partial_score":    obs.partial_score,
        "feedback":         obs.feedback,
        "steps_taken":      obs.steps_taken,
        "max_steps":        obs.max_steps,
        "done":             obs.done,
    }

    history:    List[str] = []
    best_score: float     = 0.0

    for step in range(1, MAX_STEPS + 1):
        if obs_dict.get("done"):
            break

        # Build prompt and call LLM
        user_prompt = build_user_prompt(obs_dict, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        response_text = call_llm(messages)
        action_dict   = parse_action(response_text)

        if not action_dict or "action_type" not in action_dict:
            print(f"  Step {step}: [parse failed] raw={response_text[:80]}")
            # Default to a safe fallback — look up the first alert account
            alert_account = (obs_dict.get("initial_alerts") or [{}])[0].get(
                "account_id", ""
            )
            action_dict = {
                "action_type": "lookup_account",
                "account_id":  alert_account,
            }

        action_type = action_dict.get("action_type", "")
        print(f"  Step {step}: {action_type} "
              f"| account={action_dict.get('account_id', '-')}")

        # Execute action
        action = VeritasAction(**action_dict)
        obs    = env.step(action)

        obs_dict = {
            "case_id":          obs.case_id,
            "task_description": obs.task_description,
            "difficulty":       obs.difficulty,
            "initial_alerts":   obs.initial_alerts,
            "accounts_in_scope":obs.accounts_in_scope,
            "flagged_accounts": obs.flagged_accounts,
            "action_result":    obs.action_result,
            "action_error":     obs.action_error,
            "partial_score":    obs.partial_score,
            "feedback":         obs.feedback,
            "steps_taken":      obs.steps_taken,
            "max_steps":        obs.max_steps,
            "done":             obs.done,
        }

        score    = obs.partial_score
        feedback = obs.feedback

        if score > best_score:
            best_score = score

        history.append(
            f"Step {step}: {action_type} "
            f"account={action_dict.get('account_id', '-')} "
            f"→ score={score:.2f} | {feedback[:60]}"
        )

        print(f"         score={score:.2f} | {feedback[:70]}")

        if obs_dict.get("done"):
            break

    state  = env.state
    solved = state.solved

    print(f"\n  RESULT: best_score={best_score:.2f} "
          f"solved={solved} steps={state.step_count}")

    return {
        "task_id":    task_id,
        "difficulty": task.difficulty,
        "best_score": round(best_score, 4),
        "solved":     solved,
        "steps":      state.step_count,
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "="*60)
    print("  Veritas AI — Baseline Inference")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")
    print("="*60)

    results    = []
    total_start = time.time()

    for task_id in TASK_ORDER:
        result = run_task(task_id)
        results.append(result)
        time.sleep(1)

    total_time = time.time() - total_start
    avg_score  = sum(r["best_score"] for r in results) / len(results)

    print("\n" + "="*60)
    print("  FINAL SCORES")
    print("="*60)
    for r in results:
        status = "SOLVED" if r["solved"] else f"best={r['best_score']:.2f}"
        print(f"  {r['task_id']:20s}  {r['difficulty']:6s}  {status}")

    print(f"\n  Average score : {avg_score:.4f}")
    print(f"  Total runtime : {total_time:.1f}s")
    print("="*60 + "\n")

    # Machine-readable output for judges
    output = {
        "model":      MODEL_NAME,
        "scores":     results,
        "avg_score":  round(avg_score, 4),
        "runtime_s":  round(total_time, 1),
    }
    print("JSON_RESULTS:", json.dumps(output))


if __name__ == "__main__":
    main()