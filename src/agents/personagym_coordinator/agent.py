# PersonaGym Coordinator Agent
# This agent handles the A2A protocol and orchestrates the workflow.

from google.adk.agents import SequentialAgent

from personagym_coordinator.sub_agents.settings_selector import root_agent as settings_selector_agent

from dotenv import load_dotenv
load_dotenv()

class PersonaGymCoordinator:
    def __init__(self):
        pass

    def run(self):
        print("PersonaGym Coordinator running...")

root_agent = SequentialAgent(
    name="personagym_coordinator",
    description="Orchestrates the PersonaGym evaluation workflow",
    sub_agents=[
        settings_selector_agent
    ]
)
