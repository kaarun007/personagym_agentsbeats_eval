from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

import json
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / timestamp

def safe_json(obj):
    """Safe JSON serializer for ADK agent state objects."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

async def log_state_after_agent(callback_context: CallbackContext):
    agent_name = callback_context.agent_name

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Get filepath
    timestamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
    filepath = LOG_DIR / f"{timestamp}_{agent_name}.json"

    # Write state to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            callback_context.state.to_dict(),
            f,
            indent=2,
            default=safe_json
        )

    print(f">>> {agent_name} | State written to: {filepath}")

async def log_prompt_before_llm(callback_context: CallbackContext, llm_request: LlmRequest):
    agent_name = callback_context.agent_name
    system_prompt = llm_request.config.system_instruction or "none"
    print(f">>> {agent_name} | System prompt: {system_prompt}")
