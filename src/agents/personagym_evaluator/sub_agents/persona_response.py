"""
Persona Response Agent
NOTE: this agent is temporary while the talk_to_agent tool is being developed (so that our agent workflow acts as a Green Agent)
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from typing import List

import os
import json
from dotenv import load_dotenv
load_dotenv()

from src.agents.personagym_evaluator.logging_callbacks import log_state_after_agent, log_prompt_before_llm

# Output schema for persona responses
class PersonaResponseItem(BaseModel):
    id: int = Field(description="The question id")
    question: str = Field(description="The question")
    response: str = Field(description="The answer/response to the question as the persona")

class PersonaResponseList(BaseModel):
    responses: List[PersonaResponseItem]


def create_persona_agent(task_name: str) -> Agent:
    """
    Creates an instance of the persona response agent
    """

    system_prompt = f"""
    Adopt the identity of the specified persona. 
    Answer the provided questions while staying in strict accordance with the nature of this identity.

    Your output MUST be a valid JSON object matching the following pydantic model schema exactly with no extra text or markdown:
    {json.dumps(PersonaResponseList.model_json_schema(), indent=2)}

    **Provided questions:"
    {{{task_name}_questions?}}
    """

    return Agent(
        name=f"persona_agent_for_{task_name}_eval",
        description="Agent that adopts the behaviour of a specified persona in order to answer questions from the perspective of the persona.",
        model=LiteLlm(model=os.environ["RESPONSE_MODEL"]),
        instruction=system_prompt,
        output_schema=PersonaResponseList,
        output_key=f"{task_name}_persona_responses",
        before_model_callback=log_prompt_before_llm
    )
