"""
Logging callback functions for ADK agents
"""

from google.adk.agents.callback_context import CallbackContext

import logging

logger = logging.getLogger(__name__)

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
    logger.info(f"[{invocation_id}] Agent invocation completed for agent: {agent_name}")
