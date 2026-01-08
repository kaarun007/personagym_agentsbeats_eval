# Evaluator Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.personagym_evaluator.sub_agents.question_generator import EvaluationTask

from src.agents.personagym_evaluator.logging_callbacks import log_state_before_agent, log_state_after_agent

load_dotenv()

### Evaluator System Prompt

# :TODO how to create system_prompt: You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge who is tasked to evaluate responses to questions based on a given rubric.
system_prompt = """
You are an expert judge for the PersonaGym framework. Your goal is to be ACCURATE, FAITHFUL, CRITICAL, and FAIR.

You are given a rubric to evaluate persona responses below. Each evaluation must be independent.
Return your evaluations of each response based on the criteria established in the rubric.

You must follow this STRICT Output Format for each question:

Format:
{
    evaluation_task: [The evaluation task type e.g. Expected Action]
    evaluations: [
        {
            "question": [Evaluation question #1],
            "justification": [Evaluation of the persona's response to the question with detailed reasoning matching the rubric criteria],
            "score": [Final score based on the rubric (1-5)]

        },
        {
            "question": [Evaluation question #2],
            "justification": [Evaluation of the persona's response to the question with detailed reasoning matching the rubric criteria],
            "score": [Final score based on the rubric (1-5)]
        },
        {
            ... continue for all provided questions
        }
    ]
}

Each question evaluation will be a JSON object as defined above with the "question", "justification" and "score" fields. The length of the "evaluations" array should be equal to the total number of question-response pairs.
"""

### Result formatter

# Format for evaluations for individual questions
class ResponseEvaluation(BaseModel):
    question: str = Field(description="The evaluation question")
    justification: str = Field(description="The evaluator agent's justification of their assigned score")
    score: int = Field(description="The evaluator agent's score for the persona agent's response based on the provided rubric")

# Output schema for agent
class EvaluatorOutput(BaseModel):
    evaluation_task: EvaluationTask
    evaluations: list[ResponseEvaluation] = Field(description="Array of persona response evaluations with scores")

def create_evaluator_agent(task: EvaluationTask) -> Agent:
    """
    Creates an instance of the Evaluator Agent.
    """
    task_name = task.name.lower()

    return Agent(
        name=f"evaluator_agent_for_{task_name}_eval",
        description="Agent that evaluates answers given by a persona agent",
        model=LiteLlm(model=os.environ["EVAL_1_MODEL"]),
        instruction=system_prompt,
        output_schema=EvaluatorOutput,
        output_key=f"{task_name}_evaluations",
        before_agent_callback=log_state_before_agent,
        after_agent_callback=log_state_after_agent,
    )
