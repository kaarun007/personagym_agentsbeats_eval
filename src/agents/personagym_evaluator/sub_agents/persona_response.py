"""
Persona Response Agent
NOTE: this agent is temporary while the talk_to_agent tool is being developed (so that our agent workflow acts as a Green Agent)
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from src.agents.personagym_evaluator.sub_agents.question_generator import EvaluationTask
from src.agents.personagym_evaluator.logging_callbacks import log_state_after_agent

import os
from dotenv import load_dotenv
load_dotenv()

system_prompt = """
Adopt the identity of the specified persona. 
Answer the provided questions while staying in strict accordance with the nature of this identity.
"""

def create_persona_agent(task_name: str) -> Agent:
    """
    Creates an instance of the persona response agent
    """
    return Agent(
        name=f"persona_agent_for_{task_name}_eval",
        description="Agent that adopts the behaviour of a specified persona in order to answer questions from the perspective of the persona.",
        model=LiteLlm(model=os.environ["RESPONSE_MODEL"]),
        instruction=system_prompt,
    )

