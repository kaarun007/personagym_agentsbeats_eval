"""
Persona Response Agent
"""

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.models.lite_llm import LiteLlm

import os
from pydantic import BaseModel
from dotenv import load_dotenv

from src.tools.message_tool import MessageToolProvider
from src.agents.personagym_evaluator.sub_agents.question_generator import NUM_OF_QUESTIONS
from src.agents.personagym_evaluator.temp_callbacks import log_state_after_agent, log_prompt_before_llm
from google.adk.agents.callback_context import CallbackContext
load_dotenv()

# Output schema for persona responses
class PersonaResponseItem(BaseModel):
    id: int
    question: str
    response: str

class PersonaResponseList(BaseModel):
    all_responses: list[PersonaResponseItem]

system_prompt = f"""
You are an agent tasked with communicating with an external persona agent to obtain its responses to {NUM_OF_QUESTIONS} scenario-based questions in order to assess its behaviour. The {NUM_OF_QUESTIONS} questions that will be posed to the persona will be provided to you.

Communication with the persona agent will be via the A2A protocol using the available `talk_to_agent` tool. Prompt the persona agent to respond to the provided questions. The tool is called `talk_to_agent` and it accepts 3 arguments: 
- `message` (str): the message to be sent as a prompt to the persona agent
- `url` (str): the base url of the A2A server which exposes the persona agent
- `new_conversation` (bool - Optional): Set to True if it is a new conversation with the persona agent, else set to False to continue an existing conversation

TOOL USAGE:
- You must use the function named "talk_to_agent" to communicate with the persona agent.
- Do not invent or rename tools.
- Do not include formatting markers such as <|channel|>, commentary, or system tags in tool calls.
- Note: `talk_to_agent<|channel|>commentary` IS and MUST be used as tool named `talk_to_agent`

**Your output must be a JSON object, strictly following this format:**
{{
    "all_responses": [
        {{"id": 1, "question": "<question #1>", "response": "<response to question #1>"}},
        {{"id": 2, "question": "<question #2>", "response": "<response to question #2>"}},
        ... continue for all {NUM_OF_QUESTIONS} questions
    ]
}}

**Additional Requirements:**
Do not include any explanation or extra text outside the output JSON format.
You MUST return exactly {NUM_OF_QUESTIONS} objects in "all_responses".
"""

def create_persona_response_agent(task_name: str) -> Agent:
    """
    Creates an instance of the persona response agent
    """
    message_tool_provider = MessageToolProvider()

    async def combined_callback(callback_context: CallbackContext):
        message_tool_provider.reset()
        await log_state_after_agent(callback_context)

    return Agent(
        name=f"persona_response_agent_for_{task_name}_eval",
        description="Agent that communicates with the persona agent under evaluation",
        model=LiteLlm(model=os.environ["RESPONSE_MODEL"]),
        instruction=system_prompt,
        tools=[FunctionTool(func=message_tool_provider.talk_to_agent)],
        output_schema=PersonaResponseList,
        output_key=f"{task_name}_persona_responses",
        # after_agent_callback=lambda callback_context: message_tool_provider.reset(),  # Add this back in once temp callbacks are removed
        after_agent_callback=combined_callback,
        before_model_callback=log_prompt_before_llm
    )

