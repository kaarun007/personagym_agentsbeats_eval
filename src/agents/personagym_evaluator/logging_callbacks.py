import json
from google.adk.agents.callback_context import CallbackContext

async def log_state_before_agent(callback_context: CallbackContext):
    agent_name = callback_context.agent_name
    print(f"\n===== BEFORE AGENT: {agent_name} =====")
    print(json.dumps(callback_context.state.to_dict(), indent=2))
    print("=============================")

async def log_state_after_agent(callback_context: CallbackContext):
    agent_name = callback_context.agent_name
    print(f"\n===== AFTER AGENT: {agent_name} =====")
    print(json.dumps(callback_context.state.to_dict(), indent=2))
    print("=============================")
