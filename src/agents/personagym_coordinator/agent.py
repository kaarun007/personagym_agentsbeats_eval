# PersonaGym Coordinator Agent
# This agent handles the A2A protocol and orchestrates the workflow.

from google.adk.agents import ParallelAgent, SequentialAgent

from personagym_coordinator.sub_agents.settings_selector import root_agent as settings_selector_agent
from personagym_coordinator.sub_agents.question_generator import EvaluationTask, create_question_agent
from personagym_coordinator.sub_agents.persona_response import create_persona_agent
from personagym_coordinator.sub_agents.rubric_formatter import create_rubric_formatter_agent

from dotenv import load_dotenv
load_dotenv()

# Create workflows for each evaluation task
evaluation_task_workflows = []
evaluation_tasks = [EvaluationTask.EXPECTED_ACTION, EvaluationTask.TOXICITY]
for task in evaluation_tasks:
    evaluation_task_workflow = SequentialAgent(
        name=f"{task.name.lower()}_eval_workflow",
        description=f"Evaluation task workflow for the task {task.value}",
        sub_agents=[
            create_question_agent(task=task),
            create_persona_agent(name=f"persona_agent_for_{task.name.lower()}_eval"),
            create_rubric_formatter_agent(name=f"rubric_formatter_agent_for_{task.name.lower()}_eval")
        ]
    )
    evaluation_task_workflows.append(evaluation_task_workflow)

# Create a coordinator agent that orchestrates persona evaluation for each evaluation task in parallel
evaluation_task_coordinator = ParallelAgent(
    name="evaluation_task_coordinator",
    description="Agent that coordinates the evaluation of a persona for all possible evaluation tasks in parallel",
    sub_agents=evaluation_task_workflows
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
