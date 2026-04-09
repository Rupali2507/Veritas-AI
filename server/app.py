import sys
import os
import traceback

# Add project root AND server/ to path
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SERVER)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import VeritasAction, VeritasObservation
from Vertias_AI_environment import VeritasEnvironment

# ── Single global environment instance ──────────────────────────────────────
_env: Optional[VeritasEnvironment] = None

app = FastAPI(title="Veritas-AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global error handler — always returns CORS headers even on 500 ───────────
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

# ── Request models ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "task_easy"

class StepRequest(BaseModel):
    action: VeritasAction
    case_id: Optional[str] = None
    session_id: Optional[str] = None

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

@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    _env = VeritasEnvironment()
    obs = _env.reset(task_id=req.task_id)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}

@app.post("/step")
def step(req: StepRequest):
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
    obs = _env.step(req.action)
    return {"observation": obs.model_dump(), "reward": float(obs.reward), "done": bool(obs.done)}

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)