# personagym_agentsbeats_eval

## Current State:

Here is the explanation of the code sequence in PersonaGym, starting from run.py:

### High-Level Overview:
The code evaluates "personas" (simulated characters) by placing them in specific settings, asking them challenging questions, and then grading their responses using LLM-based judges against defined rubrics.

### Detailed Steps: 
1. Entry Point (run.py): The script parses arguments and iterates through the list of personas to be evaluated.
2. Context Generation:
      - Select Settings: An LLM selects relevant scenarios (e.g., "Wedding", "Courtroom") for the persona.
      - Generate Questions: An LLM generates challenging, multi-step questions designed to test specific attributes (like "Toxicity" or "Consistency") in those settings.
3. Persona Response: The target model (acting as the persona) answers the generated questions.
4. Evaluation & Scoring:
      - Format Rubrics: The system prepares grading prompts, dynamically generating few-shot examples relevant to the specific question and persona.
      - Score: Two separate evaluator models grade the response based on the rubric. Their scores are averaged.
5. Aggregation: Scores across all tasks are averaged to produce a final PersonaScore.
      - Save Results & Scores

## Agentified State:
