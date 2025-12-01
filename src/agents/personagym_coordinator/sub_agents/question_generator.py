# Question Generator Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
load_dotenv()

# NOTE: this will be read in from an external data file when the read file tool is available
question_requirements = {
    "Expected Action": "For questions to effectively evaluate a persona's response in terms of 'Expected Action,' they must be specifically designed to elicit actions that are indicative of the persona's characteristics and behavior within the given setting. Each question should probe the persona to take multiple distinct actions in the given setting. Questions should be clear, direct, and relevant to the core attributes of the persona, ensuring that the answers can clearly demonstrate whether the persona acts as expected in the described context."
}

system_prompt = f"""
You are tasked with determining if a person with the given persona description can answer questions related to environments that specifically test the given evaluation task.
Generate exactly 10 challenging multi-step questions to do this where the questions are intended to be asked directly to the persona.
You may use the question description below to guide you.
Your output must be the generated questions in a Python list format with no other explanation or output.

Evaluation Task: Expected Action
Questions Description: {question_requirements["Expected Action"]}
"""

root_agent = Agent(
    name="personagym_coordinator",
    description="Agent that generates appropriate questions to evaluate the behaviour of a persona",
    model=LiteLlm(model=os.environ["QUESTION_MODEL"]),
    instruction=system_prompt
)
