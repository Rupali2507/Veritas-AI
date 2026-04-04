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
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VeritasAction, VeritasObservation
from server.Vertias_AI_environment import VeritasEnvironment
# Create the FastAPI app using OpenEnv's factory
# This automatically creates /reset, /step, /state, /ws, /web, /health
app = create_app(
    VeritasEnvironment,
    VeritasAction,
    VeritasObservation,
    env_name="Veritas-AI",
    max_concurrent_envs=1,
)


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