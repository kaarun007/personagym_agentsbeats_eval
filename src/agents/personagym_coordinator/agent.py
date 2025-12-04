# PersonaGym Coordinator Agent
# This agent handles the A2A protocol and orchestrates the workflow.

from google.adk.agents import ParallelAgent, SequentialAgent

from personagym_coordinator.sub_agents.settings_selector import root_agent as settings_selector_agent
from personagym_coordinator.sub_agents.question_generator import EvaluationTask, create_question_agent

from dotenv import load_dotenv
load_dotenv()

# Create a coordinator agent that orchestrates persona evaluation for each evaluation task in parallel
evaluation_task_coordinator = ParallelAgent(
    name="evaluation_task_coordinator",
    description="Agent that coordinates the evaluation of a persona for all possible evaluation tasks in parallel",
    sub_agents=[
        create_question_agent(task=EvaluationTask.EXPECTED_ACTION),
        create_question_agent(task=EvaluationTask.TOXICITY)
    ]
)

# Expose the main sequential workflow as the root agent
root_agent = SequentialAgent(
    name="personagym_coordinator",
    description="Orchestrates the PersonaGym evaluation workflow",
    sub_agents=[
        settings_selector_agent,
        evaluation_task_coordinator
    ]
)
