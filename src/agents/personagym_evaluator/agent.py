# PersonaGym Coordinator Agent
# This agent handles the A2A protocol and orchestrates the workflow.

from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.session import InMemorySessionService
from google.adk.runners import Runner
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
import argparse
import uvicorn

from personagym_evaluator.sub_agents.settings_selector import root_agent as settings_selector_agent
from personagym_evaluator.sub_agents.question_generator import EvaluationTask, create_question_agent
from personagym_evaluator.sub_agents.persona_response import create_persona_agent
from personagym_evaluator.sub_agents.rubric_formatter import create_rubric_formatter_agent
from personagym_evaluator.sub_agents.evaluator import create_evaluator_agent
from personagym_evaluator.sub_agents.score_aggregator import create_score_aggregator_agent

from dotenv import load_dotenv
load_dotenv()

# Session constants
APP_NAME = "personagym_agentsbeat_eval"
USER_ID = "evalbox"
SESSION_ID = "session123"

initial_state = {
    "selected_settings": [],        # Settings selected by the settings selector
    "tasks": {},                    # Each task will have its own nested state
    "final_scores": {
        "overall": 0,
        "by_task": {}               # e.g., {"expected_action": 4.5, "toxicity": 3.8}
    },
    "persona_data": {},             # Stores persona description, responses, etc.
    "rubrics": {},                  # Formatted rubrics per task
    "raw_evaluations": {},          # Raw evaluation texts from evaluators
}

# Create Session Service
session_service = InMemorySessionService()
session = session_service.create_session(
    state=initial_state,  # Initial empty state
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"Created new session: app_name={APP_NAME}, user_id={USER_ID}, session_id={SESSION_ID}")

# Create workflows for each evaluation task
evaluation_task_workflows = []
evaluation_tasks = [EvaluationTask.EXPECTED_ACTION, EvaluationTask.TOXICITY]
for task in evaluation_tasks:
    task_name = task.name.lower()
    evaluation_task_workflow = SequentialAgent(
        name=f"{task_name}_eval_workflow",
        description=f"Evaluation task workflow for the task {task.value}",
        sub_agents=[
            create_question_agent(task=task),
            create_persona_agent(name=f"persona_agent_for_{task_name}_eval"),
            create_rubric_formatter_agent(task=task),   
            create_evaluator_agent(agent_name=f"evaluator_agent1_for_{task_name}_eval")
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
    description="Orchestrates the PersonaGym evaluation workflow. Expects a persona description as input.",
    sub_agents=[
        settings_selector_agent,
        evaluation_task_coordinator,
        create_score_aggregator_agent()
    ]
)

# Wrap root agent with Runner to integrate session
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

def create_agent_card(url: str) -> AgentCard:
    """Create the A2A agent card for PersonaGym"""
    skill = AgentSkill(
        id='evaluate_persona',
        name='Evaluate Persona Behavior',
        description='Evaluates a persona by selecting appropriate settings and generating evaluation questions.',
        tags=['persona', 'evaluation', 'behavioral-analysis'],
        examples=["""
                {
                    "persona": "A grumpy high school math teacher who secretly loves poetry."
                }
        """]
    )
    
    agent_card = AgentCard(
        name="PersonaGymCoordinator",
        description="Orchestrates the PersonaGym evaluation workflow. Expects a persona description as input.",
        url=url,
        version="1.0.0",
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    return agent_card

def main():
    parser = argparse.ArgumentParser(description="Run the PersonaGym Coordinator Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}/"
    agent_card = create_agent_card(card_url)

    # Expose root agent with session via A2A
    a2a_app = to_a2a(
        runner.agent,
        agent_card=agent_card,
        session_service=session_service
    )
    uvicorn.run(a2a_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
