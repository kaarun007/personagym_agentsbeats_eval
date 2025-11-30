# Entry point for PersonaGym Agent
from agents.personagym_coordinator.agent import PersonaGymCoordinator

def main():
    agent = PersonaGymCoordinator()
    agent.run()

if __name__ == "__main__":
    main()
