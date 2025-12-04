from langchain_community.agent_toolkits import FileManagementToolkit

# Load LangChain's read_file tool
lc_read_tool = FileManagementToolkit(
    selected_tools=["read_file"]
).get_tools()[0]

def file_read_tool(file_path: str) -> str:
    """
    Read and return the contents of a file.

    This tool validates the file path relative to its configured root directory
    and safely handles missing or invalid paths.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        str: The contents of the file, or an error message if the file
             does not exist or cannot be accessed.
    """
    print(f"[file_read_tool] Reading file: {file_path}")
    return lc_read_tool.run({"file_path": file_path})
