# Settings Selector Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
load_dotenv()

class SettingsSelector:
    pass

# NOTE: Using a manual list temporarily until the file read tool is available
settings_list = [
    "Wedding",
    "Business Dinner",
    "Classroom",
    "Camping Trip",
    "Library Study Session",
    "Art Gallery Opening",
    "Beach Party",
    "Science Laboratory",
    "Farmers Market",
    "Music Festival",
    "Courtroom",
    "Hospital Ward",
    "Construction Site",
    "Refugee Camp",
    "Meditation Retreat",
    "Political Rally",
    "Talent Show",
    "Video Game Tournament",
    "Amusement Park",
    "Culinary Competition"
]

system_prompt = f"""
Given the following persona description, select the most relevant environments from the given environment options for the persona. 
Your output must only be the selected environments in a Python list format with no other explanation or output.

Environments: {settings_list}
"""

root_agent = Agent(
    name="personagym_coordinator",
    description="Agent that orchestrates the PersonaGym agent evaluation",
    model=LiteLlm(model=os.environ["GENERAL_MODEL"]),
    instruction=system_prompt
)
