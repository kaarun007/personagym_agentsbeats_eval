"""
File Management Tools for PersonaGym AgentsBeats Evaluation
"""

from langchain_community.agent_toolkits import FileManagementToolkit

# Initialize File Management Toolkit with selected tools
tools = FileManagementToolkit(
    selected_tools=["read_file", "write_file"],
).get_tools()

# Unpack tools
file_read_tool, file_write_tool = tools
