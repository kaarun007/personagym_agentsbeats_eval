# Settings Selector Agent
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from src.tools.file_read_tool import file_read_tool
from src.utils.logging_callbacks import pre_agent_logging_callback, post_agent_logging_callback

import os
from dotenv import load_dotenv
load_dotenv()

SETTINGS_FILE_PATH = "src/data/settings.json"

system_prompt = f"""
Given the following persona description, select the most relevant environments from the given environment options for the persona. 
Your output must only be the selected environments in a Python list format with no other explanation or output.

Obtain the complete list of possible environments using the `file_read_tool` with the file `{SETTINGS_FILE_PATH}`
"""

root_agent = Agent(
    name="settings_selector",
    description="Agent that selects appropriate settings/environments in which to evaluate a particular persona",
    model=LiteLlm(model=os.environ["SETTINGS_MODEL"]),
    instruction=system_prompt,
    tools=[file_read_tool],
    before_agent_callback=pre_agent_logging_callback,
    after_agent_callback=post_agent_logging_callback
)
