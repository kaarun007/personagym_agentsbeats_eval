from langchain_community.agent_toolkits import FileManagementToolkit

# Load LangChain's write_file tool
lc_write_tool = FileManagementToolkit(
    selected_tools=["write_file"]
).get_tools()[0]


def file_write_tool(
        file_path: str,
        text: str,
        append: bool = False
) -> str:
    """
    Write text to a file on disk.

    This tool automatically creates parent directories if needed and safely
    handles invalid or restricted file paths.

    Args:
        file_path (str): The path to the file to write.
        text (str): The content to write into the file.
        append (bool, optional): If True, text is appended to the file.
                                 If False, the file is overwritten. Defaults to False.

    Returns:
        str: A success message or an error message describing what went wrong.
    """
    print(f"[file_write_tool] Writing to file: {file_path} | append={append}")
    return lc_write_tool.run({
        "file_path": file_path,
        "text": text,
        "append": append
    })