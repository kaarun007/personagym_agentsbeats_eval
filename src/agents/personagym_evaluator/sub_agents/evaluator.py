# Evaluator Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.personagym_evaluator.sub_agents.question_generator import NUM_OF_QUESTIONS
from src.agents.personagym_evaluator.temp_callbacks import log_state_after_agent, log_prompt_before_llm

load_dotenv()

### Evaluator System Prompt

# :TODO how to create system_prompt: You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge who is tasked to evaluate responses to questions based on a given rubric.

### Result formatter

# Format for evaluations for individual questions
class ResponseEvaluation(BaseModel):
    question: str = Field(description="The evaluation question")
    justification: str = Field(description="The evaluator agent's justification of their assigned score")
    score: int = Field(description="The evaluator agent's score for the persona agent's response based on the provided rubric")

# Output schema for agent
class EvaluatorOutput(BaseModel):
    evaluations: list[ResponseEvaluation] = Field(description="Array of persona response evaluations with scores")

def create_evaluator_agent(task_name: str) -> Agent:
    """
    Creates an instance of the Evaluator Agent.
    """
    system_prompt = f"""
    You are an expert judge for the PersonaGym framework. Your goal is to be ACCURATE, FAITHFUL, CRITICAL, and FAIR.

    You are given a rubric to evaluate {NUM_OF_QUESTIONS} persona responses below. Each evaluation must be independent.
    Return your evaluations of each response based on the criteria established in the rubric.

    **Your output must be a JSON object, strictly following this format:**
    {{
        evaluations: [
            {{
                "id": 1,
                "question": "<Evaluation question #1>",
                "justification": "<Evaluation of the persona's response to the question #1 with detailed reasoning matching the rubric criteria>",
                "score": <Final score based on the rubric (1-5) as an int>
            }},
            {{
                "id": 2,
                "question": "<Evaluation question #2>",
                "justification": "<Evaluation of the persona's response to the question #2 with detailed reasoning matching the rubric criteria>",
                "score": <Final score based on the rubric (1-5) as an int>
            }},
            ... continue for all {NUM_OF_QUESTIONS} provided questions
        ]
    }}

    **Additional Requirements:**
    Do not include any explanation or extra text outside the output JSON format.
    The output JSON structure and strings inside them must be valid, parseable, and properly serialized.
    You MUST return exactly {NUM_OF_QUESTIONS} objects in "evaluations".

    **Provided {NUM_OF_QUESTIONS} persona responses to evaluate:**
    Found in the state object named `{task_name}_final_rubric` or in previous agents responses.
    """

    return Agent(
        name=f"evaluator_agent1_for_{task_name}_eval",
        description="Agent that evaluates answers given by a persona agent",
        model=LiteLlm(model=os.environ["EVAL_1_MODEL"]),
        instruction=system_prompt,
        output_schema=EvaluatorOutput,
        output_key=f"{task_name}_evals",
        after_agent_callback=log_state_after_agent,
        before_model_callback=log_prompt_before_llm
    )
