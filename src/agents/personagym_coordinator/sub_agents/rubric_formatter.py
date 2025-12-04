# Rubric Formatter Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv

# Internal imports
from src.tools.file_read_tool import file_read_tool

load_dotenv()

RUBRIC_TEMPLATE_PATH = "src/data/rubrics_template.json"

# Define the system prompt for the Rubric Formatter Agent
system_prompt = f"""
You are a Rubric Formatter Agent. Your role is to format grading prompts for persona evaluation tasks.

You will use the rubric templates provided in the file `{RUBRIC_TEMPLATE_PATH}`.

Your task is to:
1. Read the rubric template for the given evaluation task from `{RUBRIC_TEMPLATE_PATH}` using the provided file read tool.
2. Format the rubric by filling in the placeholders with the persona description, question, and response.
3. Ensure the output adheres to the structure defined in the rubric template.

Your output must include:
- A system prompt that defines the evaluator's role.
- A scoring prompt that contains the formatted rubric for evaluation.

Example Output:
System Prompt:
You are evaluating the following persona:
[Persona Description]

Question:
[Question]

Response:
[Response]

Scoring Prompt:
[Formatted Rubric]
"""

# Define the Rubric Formatter Agent
def create_rubric_formatter_agent(name: str = "rubric_formatter") -> Agent:
    """
    Creates an instance of the Rubric Formatter Agent.
    """
    return Agent(
        name=name,
        description="Agent that formats grading prompts using rubric templates and persona data",
        model=LiteLlm(model=os.environ["RUBRIC_MODEL"]),
        instruction=system_prompt,
        tools=[file_read_tool]
    )
