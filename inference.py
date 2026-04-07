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
API_KEY      = os.getenv("HF_TOKEN")
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
    You are Veritas, a financial crime analyst. You investigate cases efficiently.

    AVAILABLE ACTIONS (pick exactly one per turn):
    1. {"action_type": "lookup_account", "account_id": "ACC-XXXX"}
    2. {"action_type": "query_transactions", "account_id": "ACC-XXXX"}
    3. {"action_type": "flag_account", "account_id": "ACC-XXXX", "reason": "..."}
    4. {"action_type": "submit_report", "primary_suspect": "ACC-XXXX", "associates": [], "case_type": "card_scheme", "evidence_summary": "..."}

    CASE TYPES: card_scheme | layering_scheme | coordinated_scheme

    STRICT INVESTIGATION PROTOCOL — FOLLOW EXACTLY:
    - Turn 1: lookup_account on the account in the alert
    - Turn 2: query_transactions on that same account
    - Turn 3: flag_account on the most suspicious account
    - Turn 4: submit_report — YOU MUST SUBMIT BY TURN 4-5 MAXIMUM

    SUBMIT_REPORT RULES:
    - primary_suspect: the account ID from the initial alert
    - associates: [] for card_scheme, list intermediaries for layering_scheme
    - case_type: match the alert type exactly
    - evidence_summary: describe what you found in transactions

    YOU MUST CALL submit_report WITHIN 5 STEPS. NOT DOING SO MEANS FAILURE.
    NEVER repeat the same action twice. NEVER query the same account more than once.

    Respond with ONLY a single JSON object. No explanation. No markdown.
""").strip()
# ─────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    steps_used = obs.get('steps_taken', 0)
    max_steps  = obs.get('max_steps', 10)
    steps_left = max_steps - steps_used

    if steps_left <= 3:
        urgency = f"⚠️ ONLY {steps_left} STEPS LEFT — SUBMIT REPORT NOW"
    elif steps_left <= 5:
        urgency = f"WARNING: {steps_left} steps left — submit report soon"
    else:
        urgency = f"{steps_left} steps remaining"

    alert = obs.get('initial_alerts', [{}])[0]
    suspect = alert.get('account_id', '')
    alert_type = alert.get('alert_type', '')

    history_text = "\n".join(history[-3:]) if history else "None"

    last = ""
    if obs.get('action_result') is not None:
        r = obs['action_result']
        if isinstance(r, list):
            last = f"Last result: {len(r)} transactions returned"
            if r:
                last += f"\nSample: {json.dumps(r[0])}"
        else:
            last += f"Last result: {json.dumps(r)[:300]}"
    if obs.get('action_error'):
        last = f"Last error: {obs['action_error']}"

    return f"""TASK: {obs.get('task_description', '')}

ALERT ACCOUNT: {suspect}  (this is your primary suspect)
ALERT TYPE: {alert_type}
ALL ACCOUNTS: {', '.join(obs.get('accounts_in_scope', []))}
FLAGGED: {', '.join(obs.get('flagged_accounts', [])) or 'none'}
STEP: {steps_used}/{max_steps} — {urgency}

RECENT ACTIONS:
{history_text}

{last}

INSTRUCTION: If you have done 2+ queries, SUBMIT YOUR REPORT NOW.
Primary suspect is almost certainly: {suspect}
For card_scheme: associates=[], case_type="card_scheme"

Respond with ONE JSON action:"""

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
        valid_fields = {
            'action_type', 'account_id', 'date_from', 'date_to',
            'min_amount', 'max_amount', 'reason', 'primary_suspect',
            'associates', 'case_type', 'evidence_summary', 'metadata'
        }
        clean_dict = {k: v for k, v in action_dict.items() if k in valid_fields}
        action = VeritasAction(**clean_dict)
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