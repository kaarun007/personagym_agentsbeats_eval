# Score Aggregator Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import os
from dotenv import load_dotenv

load_dotenv()

system_prompt = """
You are the Score Aggregator for the PersonaGym framework.
You will receive raw evaluation texts from multiple agents.

Your processing algorithm is STRICT and matches the official PersonaGym logic:

1. **Extraction**:
   - Parse each input text to find evaluation segments starting with `(N) Evaluation:`.
   - Within each segment, look for the exact phrase: `Therefore, the final score is X`.
   - Extract the integer `X` (1-5). If not found, treat as 0.

2. **Calculation** (Per Task):
   - Collect all scores for the task.
   - **Modified Average**: Ignore any 0 scores. Calculate `Sum(Valid Scores) / Count(Valid Scores)`.

3. **Global Calculation**:
   - Calculate the average of all Task Averages.

**Output Report:**
Produce a Markdown report:

# PersonaGym Evaluation Report

## Executive Summary
**Overall Persona Score:** [Global Average]/5.00

## Task Breakdown
[Iterate through each Task]
### [Task Name]
- **Average Score:** [Task Average]/5.00
- **Raw Scores:** [List of extracted numbers]
- **Ana lysis:** [Brief summary of the justifications provided in the evaluations]
"""

def create_score_aggregator_agent() -> Agent:
    """
    Creates an instance of the Score Aggregator Agent.
    """
    return Agent(
        name="score_aggregator_agent",
        description="Aggregates scores from multiple evaluation tasks and generates a summary report.",
        model=LiteLlm(model=os.environ.get("SCORE_AGG_MODEL")),
        instruction=system_prompt
    )
