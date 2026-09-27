"""Built-in tools provided by the runtime itself."""

from __future__ import annotations

import json
from typing import Any

STOP_TOOL_FLAG = "_is_stop_tool"


def complete_task(result: str = "", global_storage: Any = None) -> dict[str, Any]:
    """Submit the final answer; the runtime stops after this call."""
    del result, global_storage  # the answer is read from the call arguments
    return {
        "result": json.dumps(
            {
                "success": True,
                "message": "Result submitted and task completed.",
                "status": "TASK_COMPLETED",
                "task_completed": True,
            },
            ensure_ascii=False,
        ),
        STOP_TOOL_FLAG: True,
    }
