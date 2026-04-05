---
title: Veritas AI Environment Server
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Veritas AI — Financial Crime Investigation Environment

> An OpenEnv environment where an AI agent investigates
> financial crime cases by querying accounts, gathering
> evidence, and filing investigation reports.

Built for the Meta PyTorch OpenEnv Hackathon 2026.

## What Is It?

Veritas AI puts an AI agent in the role of a fraud analyst at a financial institution. The agent receives an investigation case, then must query transaction histories, look up account profiles, flag suspicious accounts, and ultimately submit a formal investigation report — all within a limited number of steps. Every action the agent takes is evaluated by a task-specific grader that scores the quality of the final report.

This environment mirrors real-world work done by fraud analysts at banks and fintechs every day. Analysts routinely trace money-laundering chains, spot card-fraud rings, and identify structured transactions designed to stay below regulatory reporting thresholds. By framing these challenges as RL tasks, Veritas AI enables AI systems to learn investigation strategies that are directly applicable to real financial crime detection pipelines.

## Tasks

| Task ID       | Difficulty | Max Steps | Description                          |
|---------------|------------|-----------|--------------------------------------|
| task_easy     | Easy       | 8         | Card scheme: spot high-velocity      |
|               |            |           | purchases at suspicious merchants    |
| task_medium   | Medium     | 12        | Layering scheme: trace money chain   |
|               |            |           | structured below reporting threshold |
| task_hard     | Hard       | 18        | Coordinated ring: find accounts      |
|               |            |           | linked by shared device ID and IP    |

## Action Space

| Action              | Required fields        | What it does                    |
|---------------------|------------------------|---------------------------------|
| query_transactions  | account_id             | Returns transaction rows        |
| lookup_account      | account_id             | Returns full account profile    |
| flag_account        | account_id, reason     | Marks account as suspicious     |
| submit_report       | primary_suspect,       | Files final report, ends        |
|                     | associates, case_type, | episode, triggers grader        |
|                     | evidence_summary       |                                 |

## Setup

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic openai

# Run locally
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or with Docker (from project root)
docker build -t veritas-ai:latest -f server/Dockerfile .
docker run -p 8000:8000 veritas-ai:latest

# Interactive API docs
open http://localhost:8000/docs
```

## Baseline Scores

| Task        | Difficulty | Score | Model              |
|-------------|------------|-------|--------------------|
| task_easy   | Easy       | X.XX  | Qwen2.5-72B        |
| task_medium | Medium     | X.XX  | Qwen2.5-72B        |
| task_hard   | Hard       | X.XX  | Qwen2.5-72B        |
| Average     |            | X.XX  |                    |

*Scores to be updated after running inference.py*

## Live Demo

- HF Space: https://huggingface.co/spaces/YOUR_USERNAME/veritas-ai
- Interactive API docs: https://YOUR_USERNAME-veritas-ai.hf.space/docs
- Health check: https://YOUR_USERNAME-veritas-ai.hf.space/health

## Project Structure

```
Veritas-AI/
├── README.md                          # This file
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project metadata and dependencies
├── uv.lock                            # Locked dependencies
├── models.py                          # VeritasAction, VeritasObservation, VeritasState
├── inference.py                       # Baseline agent script
├── client.py                          # Environment client
├── veritas_env/
│   ├── environment.py                 # reset() / step() / state
│   ├── tasks.py                       # 3 tasks with graders
│   ├── reward.py                      # Reward constants and calculators
│   ├── data_generator.py              # Synthetic scenario generator
│   └── __init__.py
├── server/
│   ├── app.py                         # FastAPI server
│   ├── Vertias_AI_environment.py      # Bridge file
│   ├── Dockerfile                     # Container image definition
│   ├── requirements.txt               # Server dependencies
│   └── __init__.py
└── scripts/
    └── validate_submission.sh         # Pre-submission checklist
```

## Environment Variables

Set these in HF Space settings under "Repository secrets":

| Variable      | Value                                    |
|---------------|------------------------------------------|
| API_BASE_URL  | https://router.huggingface.co/v1         |
| MODEL_NAME    | Qwen/Qwen2.5-72B-Instruct                |
| HF_TOKEN      | your HuggingFace token (hf_...)          |