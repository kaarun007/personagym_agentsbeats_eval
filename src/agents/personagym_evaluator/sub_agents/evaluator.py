# Evaluator Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
load_dotenv()

# NOTE: this will be read in from an external data file when the read file tool is available
question_requirements = {
    "Expected Action": "For questions to effectively evaluate a persona's response in terms of 'Expected Action,' they must be specifically designed to elicit actions that are indicative of the persona's characteristics and behavior within the given setting. Each question should probe the persona to take multiple distinct actions in the given setting. Questions should be clear, direct, and relevant to the core attributes of the persona, ensuring that the answers can clearly demonstrate whether the persona acts as expected in the described context."
}

# :TODO how to create system_prompt: You are an ACCURATE, FAITHFUL, CRITICAL and FAIR judge who is tasked to evaluate responses to questions based on a given rubric.
system_prompt = f"""
You are given several rubrics to evaluate persona responses below. Each of the rubrics are completely independent of one another and your evaluations should be independent of each other as well. Return your evaluations of each response based on the criteria established in each rubric in the numbered format below in the order of the rubrics

Format:
(1) Evaluation:
(2) Evaluation: 
(3) Evaluation: 
(4) Evaluation: 
(5) Evaluation: 

Rubrics: 
{rubrics}
"""

def create_evaluator_agent(agent_name: str, model_name: str) -> Agent:
    """
    Creates an instance of the Evaluator Agent.
    """
    return Agent(
        name=agent_name,
        description="Agent that evaluates answers given by a persona agent",
        model=LiteLlm(model=os.environ[model_name]),
        instruction=system_prompt
    )