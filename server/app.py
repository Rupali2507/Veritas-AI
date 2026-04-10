import sys
import os
import traceback

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SERVER)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, model_validator
from typing import Optional, List, Any

from models import VeritasAction, VeritasObservation
from Vertias_AI_environment import VeritasEnvironment

_env: Optional[VeritasEnvironment] = None

app = FastAPI(title="Veritas-AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("\n=== SERVER ERROR ===")
    print(tb)
    print("===================\n")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": tb},
        headers={"Access-Control-Allow-Origin": "*"},
    )

# ── ✅ FIX 1: ResetRequest — task_id is fully optional with a default
# The grader sends an empty body {} or no body at all → must not crash
class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_easy"

    # Allow empty body / null body gracefully
    model_config = {"extra": "allow"}

# ── ✅ FIX 2: StepRequest — support BOTH formats:
#   Format A (UI): { "action": { "action_type": "...", ... }, "case_id": "..." }
#   Format B (grader): { "action_type": "...", "account_id": "...", ... } (flat)
class StepRequest(BaseModel):
    # Nested format (UI sends this)
    action: Optional[Any] = None
    case_id: Optional[str] = None
    session_id: Optional[str] = None

    # Flat format fields (grader sends these at top level)
    action_type: Optional[str] = None
    account_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    reason: Optional[str] = None
    primary_suspect: Optional[str] = None
    associates: Optional[List[str]] = None
    case_type: Optional[str] = None
    evidence_summary: Optional[str] = None

    model_config = {"extra": "allow"}

    def get_action(self) -> VeritasAction:
        """
        Resolve action from either nested or flat format.
        """
        # If 'action' field is present and is a dict, use it
        if self.action and isinstance(self.action, dict):
            return VeritasAction(**self.action)

        # If 'action' is already a VeritasAction object
        if self.action and isinstance(self.action, VeritasAction):
            return self.action

        # Flat format — build VeritasAction from top-level fields
        flat = {}
        if self.action_type:
            flat["action_type"] = self.action_type
        if self.account_id:
            flat["account_id"] = self.account_id
        if self.date_from:
            flat["date_from"] = self.date_from
        if self.date_to:
            flat["date_to"] = self.date_to
        if self.min_amount is not None:
            flat["min_amount"] = self.min_amount
        if self.max_amount is not None:
            flat["max_amount"] = self.max_amount
        if self.reason:
            flat["reason"] = self.reason
        if self.primary_suspect:
            flat["primary_suspect"] = self.primary_suspect
        if self.associates is not None:
            flat["associates"] = self.associates
        if self.case_type:
            flat["case_type"] = self.case_type
        if self.evidence_summary:
            flat["evidence_summary"] = self.evidence_summary

        return VeritasAction(**flat)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "task_easy",   "difficulty": "easy",   "description": "Card scheme investigation"},
            {"task_id": "task_medium", "difficulty": "medium", "description": "Layering scheme investigation"},
            {"task_id": "task_hard",   "difficulty": "hard",   "description": "Coordinated fraud investigation"},
        ]
    }

# ── ✅ FIX 3: /reset handles completely empty body by using Request directly
@app.post("/reset")
async def reset(request: Request):
    global _env

    # Safely parse body — grader may send empty body, {}, or {"task_id": "..."}
    try:
        raw = await request.json()
    except Exception:
        raw = {}

    if raw is None:
        raw = {}

    task_id = raw.get("task_id", "task_easy") or "task_easy"

    _env = VeritasEnvironment()
    obs = _env.reset(task_id=task_id)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
async def step(request: Request):
    global _env

    if _env is None:
        return {
            "observation": {
                "action_error": "No active episode. Call /reset first.",
                "action_result": None,
                "feedback": "No active episode. Call /reset first.",
                "steps_taken": 0,
                "max_steps": 8,
                "partial_score": 0.0,
                "flagged_accounts": [],
            },
            "reward": 0.0,
            "done": False,
        }

    try:
        raw = await request.json()
    except Exception:
        raw = {}

    if raw is None:
        raw = {}

    # Build StepRequest from raw dict to handle both flat and nested formats
    try:
        req = StepRequest(**raw)
        action = req.get_action()
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid action payload", "detail": str(e)},
            headers={"Access-Control-Allow-Origin": "*"},
        )

    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward),
        "done": bool(obs.done),
    }


@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"error": "No active episode"}
    return _env.get_state()


# ── Serve static UI ──────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        ui_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(ui_path):
            return FileResponse(ui_path, media_type="text/html")
        return {"message": "Place index.html in the static/ folder."}


# ── Entry point ──────────────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()