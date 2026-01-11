"""
Persona Response Agent
"""

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv

from src.tools.message_tool import MessageToolProvider
from src.agents.personagym_evaluator.temp_callbacks import log_state_after_agent, log_prompt_before_llm
from google.adk.agents.callback_context import CallbackContext
load_dotenv()

system_prompt = """
You are an agent tasked with communicating with an external persona agent to obtain its responses to a number of scenario-based questions in order to assess its behaviour. The questions that will be posed to the persona will be provided to you.

Communication with the persona agent will be via the A2A protocol using the available `talk_to_agent` tool. Prompt the persona agent to respond to the provided questions. The tool is called `talk_to_agent` and it accepts 3 arguments: 
- `message` (str): the message to be sent as a prompt to the persona agent
- `url` (str): the base url of the A2A server which exposes the persona agent
- `new_conversation` (bool - Optional): Set to True if it is a new conversation with the persona agent, else set to False to continue an existing conversation

Return ONLY the persona's responses to each of the provided questions as a Python list of strings, where each response is a separate array element. Do NOT include anything else in the ouput except for the persona's responses to the questions.
"""

def create_persona_response_agent(name: str) -> Agent:
    """
    Creates an instance of the persona response agent
    """
    message_tool_provider = MessageToolProvider()

    async def combined_callback(callback_context: CallbackContext):
        message_tool_provider.reset()
        await log_state_after_agent(callback_context)

    return Agent(
        name=name,
        description="Agent that communicates with the persona agent under evaluation",
        model=LiteLlm(model=os.environ["RESPONSE_MODEL"]),
        instruction=system_prompt,
        tools=[FunctionTool(func=message_tool_provider.talk_to_agent)],
        # after_agent_callback=lambda callback_context: message_tool_provider.reset(),
        after_agent_callback=combined_callback,
        before_model_callback=log_prompt_before_llm
    )

