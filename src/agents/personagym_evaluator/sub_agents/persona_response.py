"""
Persona Response Agent
NOTE: this agent is temporary while the talk_to_agent tool is being developed (so that our agent workflow acts as a Green Agent)
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from src.agents.personagym_evaluator.sub_agents.question_generator import EvaluationTask, NUM_OF_QUESTIONS
from src.agents.personagym_evaluator.logging_callbacks import log_state_after_agent, log_prompt_before_llm

import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Output schema for persona responses
class PersonaResponseItem(BaseModel):
    id: int
    question: str
    response: str

class PersonaResponseList(BaseModel):
    all_responses: list[PersonaResponseItem]

def create_persona_agent(task_name: str) -> Agent:
    """
    Creates an instance of the persona response agent
    """
    system_prompt = f"""
    Adopt the identity of the specified persona. 
    Answer the provided {NUM_OF_QUESTIONS} questions while staying in strict accordance with the nature of this identity.

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
    Respond to each question in order. You MUST return exactly {NUM_OF_QUESTIONS} objects in "all_responses".
    The id for the generated response along with the question should match the original id for the provided question.

    **Provided {NUM_OF_QUESTIONS} questions to answer:**
    {{{task_name}_questions?}}
    Also found in the state object named `{task_name}_questions`or in previous agents responses.
    """
    return Agent(
        name=f"persona_agent_for_{task_name}_eval",
        description="Agent that adopts the behaviour of a specified persona in order to answer questions from the perspective of the persona.",
        model=LiteLlm(model=os.environ["RESPONSE_MODEL"]),
        instruction=system_prompt,
        output_schema=PersonaResponseList,
        output_key=f"{task_name}_persona_responses",
        after_agent_callback=log_state_after_agent,
        before_model_callback=log_prompt_before_llm
    )

