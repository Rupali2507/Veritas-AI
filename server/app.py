# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI server for the Veritas AI Environment.

Endpoints:
    POST /reset  — start a new investigation episode
    POST /step   — take one action
    GET  /state  — get episode metadata
    GET  /health — health check (required by hackathon validator)
    GET  /tasks  — list all available tasks
    WS   /ws     — WebSocket for persistent sessions
    GET  /       — Veritas AI Web UI  ← NEW
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

import sys
import os
from fastapi.staticfiles import StaticFiles  # ← NEW
from fastapi.responses import FileResponse   # ← NEW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VeritasAction, VeritasObservation
from server.Vertias_AI_environment import VeritasEnvironment

# Create the FastAPI app using OpenEnv's factory
app = create_app(
    lambda: VeritasEnvironment(),
    VeritasAction,
    VeritasObservation,
    env_name="Veritas-AI",
    max_concurrent_envs=1,
)

# ── Serve the UI ────────────────────────────────────────────────────────────
# Place index.html in a folder called "static" next to this app.py file.
# The UI will then be available at https://your-space.hf.space/

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if os.path.isdir(STATIC_DIR):
    # Mount static assets (CSS, JS, images) if you add them later
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the Veritas AI investigation UI."""
    ui_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"message": "UI not found. Place index.html in the static/ folder."}
# ───────────────────────────────────────────────────────────────────────────


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for uv run server and direct execution.
    Called by pyproject.toml [project.scripts] server entry.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main()