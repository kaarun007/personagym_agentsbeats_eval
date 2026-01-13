"""
Logging callback functions for ADK agents
"""
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone

# Initialize logger
logger = logging.getLogger(__name__)

# Set up log directory path
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs") / timestamp

def safe_json(obj):
    """
    Safe JSON serializer for ADK agent state objects.
    """

    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

def pre_agent_logging_callback(callback_context: CallbackContext) -> None:
    """
    Callback function that runs before an agent is invoked to log the agent invocation event
    """

    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    logger.info(f"[{invocation_id}] Agent invoked: {agent_name}.")

def post_agent_logging_callback(callback_context: CallbackContext) -> None:
    """
    Callback function that runs after an agent completes its invocation to log the outcome of the invocation event
    """

    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id

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

    logger.info(f"[{invocation_id}] Agent invocation completed for agent: {agent_name} | State written to: {filepath}")
