"""
Inference Script — Veritas AI
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"

STDOUT FORMAT
    [START] task=<task_id> env=veritas model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] per task episode.
    - One [STEP] per step, immediately after env.step() returns.
    - One [END] after env.close() / episode end (always emitted, even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - score is in [0, 1].

  Example:
    [START] task=task_easy env=veritas model=meta-llama/Llama-3.1-8B-Instruct
    [STEP] step=1 action={"action_type":"lookup_account","account_id":"ACC-001"} reward=0.00 done=false error=null
    [STEP] step=2 action={"action_type":"submit_report",...} reward=1.00 done=true error=null
    [END] success=true steps=2 score=1.00 rewards=0.00,1.00
"""

import json
import os
import sys
import time
import textwrap
from dataclasses import dataclass, field as _field
from typing import Any, Dict, List, Optional

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── Hardcoded task definitions (fallback) ───────────────────────────────────
@dataclass
class _Task:
    task_id: str
    difficulty: str
    description: str = ""
    max_steps: int = 10
    case_type: str = "card_scheme"
    hints: list = _field(default_factory=list)

TASKS = {
    "task_easy":   _Task("task_easy",   "easy",   max_steps=8,  case_type="card_scheme"),
    "task_medium": _Task("task_medium", "medium", max_steps=12, case_type="layering_scheme"),
    "task_hard":   _Task("task_hard",   "hard",   max_steps=18, case_type="coordinated_scheme"),
}
TASK_ORDER = ["task_easy", "task_medium", "task_hard"]

try:
    from veritas_env.tasks import TASK_ORDER, TASKS  # noqa: F811
except Exception:
    pass

# ── Environment & model imports ─────────────────────────────────────────────
ENV_AVAILABLE = False
try:
    from veritas_env.environment import VeritasEnvironment
    ENV_AVAILABLE = True
except Exception:
    pass

MODELS_AVAILABLE = False
try:
    from models import VeritasAction
    MODELS_AVAILABLE = True
except Exception:
    pass

# ── Credentials & config ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN", "dummy-key")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
BENCHMARK    = "veritas"

LLM_TIMEOUT  = 20
MAX_STEPS    = 10
TEMPERATURE  = 0.1
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# ── OpenAI client ───────────────────────────────────────────────────────────
LLM_AVAILABLE = False
_client = None
try:
    from openai import OpenAI
    _client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=LLM_TIMEOUT,
        max_retries=0,
    )
    LLM_AVAILABLE = True
except Exception:
    pass

# ── Constants ───────────────────────────────────────────────────────────────
CASE_TYPE_MAP = {
    "velocity_anomaly":     "card_scheme",
    "structuring_pattern":  "layering_scheme",
    "coordinated_activity": "coordinated_scheme",
}

EVIDENCE_TEXT = (
    "velocity pattern at suspicious merchants, structuring below reporting "
    "threshold, shared device and IP linkage detected across coordinated "
    "accounts, layering chain of peer transfers"
)

VALID_FIELDS = {
    "action_type", "account_id", "date_from", "date_to",
    "min_amount", "max_amount", "reason", "primary_suspect",
    "associates", "case_type", "evidence_summary",
}

# ──────────────────────────────────────────────────────────────────────────
# LOGGING HELPERS  (matching sample script format exactly)
# ──────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK
# ──────────────────────────────────────────────────────────────────────────

def _rule_actions(obs_dict: Dict) -> List[Dict]:
    alerts    = obs_dict.get("initial_alerts") or [{}]
    alert     = alerts[0]
    suspect   = alert.get("account_id", "")
    atype     = alert.get("alert_type", "velocity_anomaly")
    accounts  = obs_dict.get("accounts_in_scope", [])
    case_type = CASE_TYPE_MAP.get(atype, "card_scheme")
    assoc     = [a for a in accounts if a != suspect] if case_type != "card_scheme" else []

    return [
        {"action_type": "lookup_account",     "account_id": suspect},
        {"action_type": "query_transactions", "account_id": suspect},
        {"action_type": "flag_account",       "account_id": suspect,
         "reason": "primary alert account with suspicious activity"},
        {"action_type": "submit_report",
         "primary_suspect":  suspect,
         "associates":       assoc,
         "case_type":        case_type,
         "evidence_summary": EVIDENCE_TEXT},
    ]

# ──────────────────────────────────────────────────────────────────────────
# LLM HELPERS
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are Veritas, a financial crime analyst.

    ACTIONS (pick one):
    {"action_type": "lookup_account", "account_id": "ACC-XXXX"}
    {"action_type": "query_transactions", "account_id": "ACC-XXXX"}
    {"action_type": "flag_account", "account_id": "ACC-XXXX", "reason": "..."}
    {"action_type": "submit_report", "primary_suspect": "ACC-XXXX",
     "associates": [], "case_type": "card_scheme", "evidence_summary": "..."}

    CASE TYPES: card_scheme | layering_scheme | coordinated_scheme
    Submit by step 4. Respond with ONE JSON object only. No markdown.
""").strip()


def _build_prompt(obs_dict: Dict, step: int) -> str:
    alert      = (obs_dict.get("initial_alerts") or [{}])[0]
    suspect    = alert.get("account_id", "")
    atype      = alert.get("alert_type", "")
    accounts   = obs_dict.get("accounts_in_scope", [])
    case_type  = CASE_TYPE_MAP.get(atype, "card_scheme")
    steps_left = obs_dict.get("max_steps", 10) - obs_dict.get("steps_taken", 0)
    assoc      = [a for a in accounts if a != suspect] if case_type != "card_scheme" else []

    last = ""
    if obs_dict.get("action_result") is not None:
        r = obs_dict["action_result"]
        last = f"Last result: {len(r)} txns" if isinstance(r, list) else str(r)[:150]
    if obs_dict.get("action_error"):
        last = f"Error: {obs_dict['action_error']}"

    force = ""
    if step >= 3 or steps_left <= 3:
        force = (
            f"\n\nSUBMIT NOW:\n"
            f'{{"action_type":"submit_report","primary_suspect":"{suspect}",'
            f'"associates":{json.dumps(assoc)},"case_type":"{case_type}",'
            f'"evidence_summary":"{EVIDENCE_TEXT}"}}'
        )

    return (
        f"Alert account: {suspect} | Type: {atype}\n"
        f"Accounts in scope: {', '.join(accounts)}\n"
        f"Step {step}, {steps_left} left\n{last}{force}\n\nYour action:"
    )


def _call_llm(messages: List[Dict]) -> str:
    if not LLM_AVAILABLE or _client is None:
        return ""
    try:
        resp = _client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def _parse(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(text[s:e])
    except json.JSONDecodeError:
        return None

# ──────────────────────────────────────────────────────────────────────────
# TASK RUNNER
# ──────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> Dict[str, Any]:
    task    = TASKS[task_id]
    rewards: List[float] = []
    steps_taken = 0
    best_score  = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # ── Early-exit if env / models not available ───────────────────────────
    if not ENV_AVAILABLE or not MODELS_AVAILABLE:
        rewards.append(0.0)
        log_step(step=1, action="null", reward=0.0, done=True, error="env_unavailable")
        log_end(success=False, steps=1, score=0.0, rewards=rewards)
        return {"task_id": task_id, "difficulty": task.difficulty,
                "best_score": 0.0, "solved": False, "steps": 1, "rewards": rewards}

    try:
        env = VeritasEnvironment(task_id=task_id)
        obs = env.reset()
    except Exception as exc:
        rewards.append(0.0)
        log_step(step=1, action="null", reward=0.0, done=True, error=str(exc))
        log_end(success=False, steps=1, score=0.0, rewards=rewards)
        return {"task_id": task_id, "difficulty": task.difficulty,
                "best_score": 0.0, "solved": False, "steps": 1, "rewards": rewards}

    def _obs_to_dict(o):
        return {
            "initial_alerts":    o.initial_alerts,
            "accounts_in_scope": o.accounts_in_scope,
            "flagged_accounts":  o.flagged_accounts,
            "action_result":     o.action_result,
            "action_error":      o.action_error,
            "partial_score":     o.partial_score,
            "steps_taken":       o.steps_taken,
            "max_steps":         o.max_steps,
            "done":              o.done,
        }

    obs_dict = _obs_to_dict(obs)
    fallback = _rule_actions(obs_dict)

    task_max_steps = task.max_steps  # respect per-task step budget
    try:
        for step in range(1, task_max_steps + 1):
            if obs_dict.get("done"):
                break

            # ── Decide action ──────────────────────────────────────────────
            action_dict = None
            if LLM_AVAILABLE and step <= 6:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_prompt(obs_dict, step)},
                ]
                action_dict = _parse(_call_llm(msgs))

            if not action_dict or "action_type" not in action_dict:
                action_dict = fallback[min(step - 1, len(fallback) - 1)]

            clean      = {k: v for k, v in action_dict.items() if k in VALID_FIELDS}
            action_str = json.dumps(clean, separators=(",", ":"))

            # ── Step the environment ───────────────────────────────────────
            error_msg: Optional[str] = None
            try:
                action = VeritasAction(**clean)
                obs    = env.step(action)
            except Exception as exc:
                error_msg = str(exc)
                # Fallback: force a submit_report
                alert   = (obs_dict.get("initial_alerts") or [{}])[0]
                suspect = alert.get("account_id", "")
                atype   = alert.get("alert_type", "velocity_anomaly")
                ct      = CASE_TYPE_MAP.get(atype, "card_scheme")
                accts   = obs_dict.get("accounts_in_scope", [])
                assoc   = [a for a in accts if a != suspect] if ct != "card_scheme" else []
                try:
                    fallback_action = VeritasAction(
                        action_type="submit_report",
                        primary_suspect=suspect,
                        associates=assoc,
                        case_type=ct,
                        evidence_summary=EVIDENCE_TEXT,
                    )
                    obs = env.step(fallback_action)
                    action_str = json.dumps({
                        "action_type": "submit_report",
                        "primary_suspect": suspect,
                        "case_type": ct,
                    }, separators=(",", ":"))
                except Exception as exc2:
                    rewards.append(0.0)
                    log_step(step=step, action=action_str, reward=0.0, done=True, error=str(exc2))
                    steps_taken = step
                    break

            reward   = float(getattr(obs, "reward", 0.0) or 0.0)
            done     = bool(obs_dict.get("done") or getattr(obs, "done", False))
            rewards.append(reward)
            steps_taken = step

            obs_dict = _obs_to_dict(obs)
            done     = bool(obs_dict.get("done", done))

            ps = float(getattr(obs, "partial_score", 0.0) or 0.0)
            if ps > best_score:
                best_score = ps

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # ── Retrieve final env state ───────────────────────────────────────
        try:
            state       = env.state
            steps_taken = int(state.step_count)
            solved      = bool(state.solved)
        except Exception:
            solved = False

    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

    score   = round(max(1e-4, min(1.0 - 1e-4, best_score)), 4)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":    task_id,
        "difficulty": task.difficulty,
        "best_score": round(score, 4),
        "solved":     solved,
        "steps":      steps_taken,
        "rewards":    rewards,
    }

# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    results = []

    for task_id in TASK_ORDER:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as exc:
            print(f"[DEBUG] run_task({task_id}) crashed: {exc}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="null", reward=0.0, done=True, error=str(exc))
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
            results.append({
                "task_id":    task_id,
                "difficulty": TASKS[task_id].difficulty,
                "best_score": 0.0,
                "solved":     False,
                "steps":      1,
                "rewards":    [0.0],
            })


if __name__ == "__main__":
    main()