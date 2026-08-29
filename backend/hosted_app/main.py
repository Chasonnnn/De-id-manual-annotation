from __future__ import annotations

import os

from .runtime import RuntimeSettings, build_runtime_app

app = build_runtime_app(RuntimeSettings.from_environment(os.environ))
