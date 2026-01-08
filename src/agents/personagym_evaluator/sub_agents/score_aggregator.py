# Score Aggregator Agent
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List

# Internal imports
from src.agents.personagym_evaluator.sub_agents.question_generator import EvaluationTask
from src.tools.file_write_tool import file_write_tool
from src.agents.personagym_evaluator.logging_callbacks import log_state_before_agent, log_state_after_agent

load_dotenv()

RESULTS_TEMPLATE_PATH = "output/results.md"

# Output Schema
class TaskAggregation(BaseModel):
    average_score: float
    raw_scores: List[int]
    analysis: str

class FinalAggregation(BaseModel):
    overall_score: float
    by_task: Dict[EvaluationTask, TaskAggregation]

# System prompts
score_aggregator_prompt = """
You are the Score Aggregator for the PersonaGym framework.
You will receive raw evaluation texts from multiple agents.

Your processing algorithm is STRICT and matches the official PersonaGym logic:

1. **Extraction**:
   - Read the session state to find evaluation segments for each evaluation task.
   - Each output is a Pydantic object containing an `evaluations` array of ResponseEvaluation objects containing question, justification and score.
   - Within each array item, extract the `score` field as an integer (1-5).

2. **Calculation** (Per Task):
   - Collect all scores for the task.
   - **Modified Average**: Ignore any 0 scores. Calculate `Sum(Valid Scores) / Count(Valid Scores)`.

3. **Global Calculation**:
   - Calculate the average of all Task Averages.

Format and return the output as a JSON object with the following schema, filling in the placeholders with appropriate values:
{
    "overall_score": float,             # Global average across all tasks
    "by_task": {
        "<Evaluation Task Name as string>": {
            "average_score": <Average score for specific task as float>,
            "raw_scores": [List of all raw scores for each question for the specific task]
            "analysis": "<Analysis of the scores for the specific task as a string>"
        },
        ... continue for all provided evaluation tasks
    }
}
"""

file_writer_prompt = f"""
You are an agent responsible for writing a provided Markdown report to an output file at a specified path. You will receive a report as Markdown-formatted text. Use the instructions below to write to a file.

**Output Report:**
Produce a Markdown report as per the output report template:

# PersonaGym Evaluation Report

## Executive Summary
**Overall Persona Score:** [Global Average]/5.00

## Task Breakdown
[Iterate through each Task]
### [Task Name]
- **Average Score:** [Task Average]/5.00
- **Raw Scores:** [List of extracted numbers]
- **Analysis:** [Brief summary of the justifications provided in the evaluations]

**Write Output Report to file:**
1. Use the `file_write_tool`
2. Write the FULL output Markdown report to the file path:
   `{RESULTS_TEMPLATE_PATH}`
"""

def create_score_aggregator_agent() -> SequentialAgent:
    """
    Creates an instance of the Score Aggregator Agent.
    """

    score_aggregator_agent = Agent(
        name="score_aggregator_agent",
        description="Aggregates scores from multiple evaluation tasks.",
        model=LiteLlm(model=os.environ["SCORE_AGG_MODEL"]),
        instruction=score_aggregator_prompt,
        output_schema=FinalAggregation,
        output_key="final_aggregation",
        before_agent_callback=log_state_before_agent,
        after_agent_callback=log_state_after_agent,
    )

    file_writer_agent = Agent(
        name="file_writer_agent",
        description="Writes the output summary report to a file",
        model=LiteLlm(model=os.environ["SCORE_AGG_MODEL"]),
        instruction=file_writer_prompt,
        tools=[file_write_tool],
        before_agent_callback=log_state_before_agent,
        after_agent_callback=log_state_after_agent,
    )

    return SequentialAgent(
        name="score_aggregator_workflow",
        sub_agents=[
            score_aggregator_agent,
            file_writer_agent
        ]
    )
