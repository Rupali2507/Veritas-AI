# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Veritas AI Environment — server-side wrapper.
Bridges OpenEnv's server framework to our core VeritasEnvironment.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veritas_env.environment import VeritasEnvironment

__all__ = ["VeritasEnvironment"]