<!-- ---
title: Veritas AI Environment Server
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
--- -->

# Veritas AI — Financial Crime Investigation Environment

> A real-world OpenEnv environment where an AI agent plays the role
> of a financial crime analyst — querying accounts, tracing money
> flows, and filing investigation reports.

**Built for the Meta PyTorch OpenEnv Hackathon 2026**

---

## Why Veritas AI?

Financial crime investigation is one of the highest-value real-world
tasks for AI agents. Every major bank and fintech employs human analysts
who follow exactly the workflow this environment simulates:

1. Receive an alert on a suspicious account
2. Query transaction histories and account profiles
3. Trace patterns — velocity anomalies, structuring, device linkage
4. File a formal investigation report naming the suspect and scheme

An agent trained in Veritas AI learns investigative reasoning that is
**directly applicable to real fraud detection pipelines**. This is not
a toy environment — it mirrors production analyst tooling.

---

## Live Demo

| Resource | URL |
|---|---|
| HF Space | https://huggingface.co/spaces/ratnesh18/veritas-ai |
| Interactive API docs | https://ratnesh18-veritas-ai.hf.space/docs |
| Health check | https://ratnesh18-veritas-ai.hf.space/health |
| Schema | https://ratnesh18-veritas-ai.hf.space/schema |

---

## How It Works

Each episode the agent receives an open investigation case with one or
more system-generated alerts. It has access to a synthetic financial
database and must investigate before the step limit runs out.

reset()  →  New case opened, initial alerts visible
step()   →  Agent takes one investigative action
state()  →  Episode metadata: steps, score, log


The episode ends when the agent calls `submit_report` or exhausts
its step budget. A deterministic grader scores the report 0.0–1.0
with partial credit at every criterion level.

---

## Action Space

The agent has exactly 4 actions — mirroring real analyst tooling:

| Action | Key fields | What it does |
|---|---|---|
| `query_transactions` | `account_id`, optional filters | Pull transaction rows for one account |
| `lookup_account` | `account_id` | Get full profile: device ID, IP, balance, registration |
| `flag_account` | `account_id`, `reason` | Formally mark an account as suspicious |
| `submit_report` | `primary_suspect`, `associates`, `case_type`, `evidence_summary` | File final report — triggers grader, ends episode |

**Example actions:**
```json
{"action_type": "lookup_account", "account_id": "ACC-1234"}

{"action_type": "query_transactions", "account_id": "ACC-1234",
 "date_from": "2024-01-01", "date_to": "2024-03-31"}

{"action_type": "flag_account", "account_id": "ACC-1234",
 "reason": "High velocity purchases at suspicious merchants"}

{"action_type": "submit_report",
 "primary_suspect": "ACC-1234",
 "associates": ["ACC-5678"],
 "case_type": "layering_scheme",
 "evidence_summary": "Structured transfers just below reporting threshold, linked device IDs"}
```

---

## Observation Space

After every `reset()` and `step()`, the agent receives:

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Unique investigation case ID |
| `task_id` | string | Active task identifier |
| `difficulty` | string | easy / medium / hard |
| `task_description` | string | Full task objective |
| `initial_alerts` | list | System alerts that opened the case |
| `accounts_in_scope` | list | Account IDs available to investigate |
| `action_result` | any | Data returned by last action |
| `action_error` | string | Error message if action was invalid |
| `flagged_accounts` | list | Accounts flagged so far this episode |
| `partial_score` | float | Grader score 0.0–1.0, updated on submit |
| `feedback` | string | Human-readable progress hint |
| `reward` | float | Shaped reward for this step |
| `done` | boolean | True when episode ends |
| `steps_taken` | int | Steps used so far |
| `max_steps` | int | Episode step limit |

---

## Three Tasks — Easy → Medium → Hard

### Task 1 — Easy — Card Scheme

```
task_id   : task_easy
difficulty: easy
max_steps : 8
case_type : card_scheme
```
One account is making rapid high-value purchases at unlisted merchants —
a clear velocity anomaly. The agent must identify the account and submit
with `case_type = "card_scheme"`.

**Grading:** correct suspect (0.50) + correct case type (0.20) +
evidence quality (0.15) + no false accusations (0.15)

---

### Task 2 — Medium — Layering Scheme

```task_id   : task_medium
difficulty: medium
max_steps : 12
case_type : layering_scheme
```
Money is moving through a chain of 3–4 accounts in amounts just below
the ₹10,000 reporting threshold — a classic structuring pattern. The
agent must trace the full chain and identify both the origin account
(primary suspect) and all intermediaries (associates).

**Grading:** correct suspect (0.35) + correct associates (0.30) +
case type (0.15) + evidence quality (0.15) + no false associates (0.05)

---

### Task 3 — Hard — Coordinated Ring

```
task_id   : task_hard
difficulty: hard
max_steps : 18
case_type : coordinated_scheme
```

A ring of 5–8 accounts controlled by a single operator, linked by
shared device fingerprints and IP address subnets. The agent must
cross-reference device and IP data to identify the ring operator and
all associates — a task that requires genuine reasoning beyond pattern
matching.

**Grading:** correct suspect (0.30) + correct associates (0.32) +
case type (0.10) + device/IP evidence (0.15) +
no false associates (0.08) + partial ring credit (0.05)

---

## Reward Function

Dense, shaped rewards throughout every episode — not binary win/lose:

| Signal | Value | When |
|---|---|---|
| Step penalty | −0.02 | Every step |
| Query suspicious account | +0.05 | After query/lookup on scheme account |
| Query innocent account | −0.02 | After query/lookup on innocent |
| Flag correct suspect | +0.10 | Immediately on correct flag |
| Flag correct associate | +0.05 | Immediately on correct flag |
| Flag innocent account | −0.10 | Immediately on wrong flag |
| Report — correct suspect | +0.40 | On submit_report |
| Report — correct case type | +0.20 | On submit_report |
| Report — correct associate | +0.10 each | On submit_report |
| Report — good evidence | +0.15 | On submit_report |
| Report — wrong suspect | −0.20 | On submit_report |
| Solve bonus | +0.25 | First perfect report |

---

## Baseline Scores

Baseline agent: `llama-3.1-8b-instant` (OpenAI-compatible client, zero shot)

| Task | Difficulty | Baseline Score | Notes |
|---|---|---|---|
| task_easy | Easy | **0.85** | Suspect correctly identified in 3 steps |
| task_medium | Medium | **0.95** | Full money chain traced |
| task_hard | Hard | **0.00** | Requires device/IP graph reasoning — genuine RL challenge |
| **Average** | | **0.60** | Strong learning signal, large headroom for RL improvement |

Oracle agent (perfect knowledge) scores: **1.00 / 0.90 / 0.95**

The gap between baseline (0.60) and oracle (0.95) represents the
learning opportunity for RL training.

---

## Setup & Local Run
```bash
# Clone
git clone https://github.com/Rupali2507/Veritas-AI
cd Veritas-AI

# Install
pip install openenv-core fastapi uvicorn pydantic openai

# Run server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{}'

# Interactive docs
open http://localhost:7860/docs
```

## Docker
```bash
# Build
docker build -t veritas-ai:latest -f server/Dockerfile .

# Run
docker run -p 7860:7860 veritas-ai:latest

# Verify
curl http://localhost:7860/health
```

## Run Baseline Inference
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_groq_api_key"
python inference.py
```

The inference script uses the OpenAI client pointed at any
OpenAI-compatible endpoint. For HuggingFace router:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py
```

---

## OpenEnv Validation
```bash
# Local spec validation
openenv validate

# Live environment validation
openenv validate --url https://ratnesh18-veritas-ai.hf.space
```

Result: **6/6 criteria passed** ✅

---

## Project Structure
``` bash
Veritas-AI/
├── models.py                     # VeritasAction, VeritasObservation, VeritasState
├── inference.py                  # Baseline agent (mandatory, root level)
├── openenv.yaml                  # OpenEnv spec
├── pyproject.toml                # Project metadata
├── uv.lock                       # Locked dependencies
├── client.py                     # Environment client
│
├── veritas_env/
│   ├── environment.py            # reset() / step() / state
│   ├── tasks.py                  # 3 task definitions + graders
│   ├── reward.py                 # Reward constants + calculators
│   ├── data_generator.py         # Synthetic scenario generator
│   └── init.py
│
├── server/
│   ├── app.py                    # FastAPI server
│   ├── Vertias_AI_environment.py # Environment bridge
│   ├── Dockerfile                # Container definition
│   └── requirements.txt
│
└── scripts/
└── validate_submission.sh    # Pre-submission checklist

```
---


## Environment Variables

Set in HF Space settings → Repository secrets:

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (e.g. https://router.huggingface.co/v1) |
| `MODEL_NAME` | Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct) |
| `HF_TOKEN` | HuggingFace token or compatible API key |

---

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) — HTTP server
- [Pydantic](https://docs.pydantic.dev) — typed models
- [SQLite](https://sqlite.org) — in-memory synthetic database

---

*Meta PyTorch OpenEnv Hackathon 2026 — Round 1 Submission*
