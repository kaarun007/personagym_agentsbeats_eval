# Rubric Formatter Agent
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Internal imports
from src.agents.personagym_evaluator.sub_agents.question_generator import EvaluationTask
from src.tools.file_read_tool import file_read_tool
from src.agents.personagym_evaluator.temp_callbacks import log_state_after_agent, log_prompt_before_llm

load_dotenv()

RUBRIC_TEMPLATE_PATH = "src/data/rubrics_template.json"

# Define the output schema for the example generator agent
class ResponseExample(BaseModel):
    score: int
    example_response: str

class ExamplesList(BaseModel):
    question: str
    examples: list[ResponseExample]

class ExampleGeneratorOutput(BaseModel):
    questions: list[ExamplesList]

# Define the final output schema
class ResponseToEvaluate(BaseModel):
    id: int
    question: str
    response: str
    examples: list[ResponseExample]

class EvaluationRubric(BaseModel):
    scoring_rubric: str
    responses: list[ResponseToEvaluate]
    
# Define the Rubric Formatter Agent
def create_rubric_formatter_agent(task: EvaluationTask) -> SequentialAgent:
    """
    Creates an instance of the Rubric Formatter Agent.
    """
    task_name = task.name.lower()

    # Define the system prompt for the Rubric Extractor Agent
    rubric_extractor_system_prompt = f"""
    You are a rubric extractor agent. Your role is to extract the appropriate rubric to be used to evaluate a persona's {task.value}.

    Read the rubric template for the {task.value} evaluation task from `{RUBRIC_TEMPLATE_PATH}` using the provided `file_read_tool`. Rubrics for each different task type are contained within a JSON array under the key "rubrics". ONLY extract the rubric for the task type `{task.value}`. Return the rubric as a single JSON object as read from the file.
    """

    # Define the system prompt to be used by the example generator agent
    example_generator_system_prompt = """
    You are an example generator agent. Your role is to read a provided scoring rubric used to evaluate responses from a given persona and generate example responses for each of the possible scores in the rubric for the given persona and question. You will adopt the behaviour of the given persona and provide a response for the given question where the quality of your response is based on the criteria specified in the rubric for that particular score.

    Your output should conform to the following format:
    {
        "questions": [<Array of questions and corresponding example responses>]
    }

    You will populate the "questions" array with your generated example responses for each provided evaluation question, formatting using the JSON schema below:
    {
        "question": <Evaluation question>,
        "examples": [
            {
                "score": 1,
                "example_response": "<Example response for score 1>"
            },
            {
                "score": 2,
                "example_response": "<Example response for score 2>"
            },
            {
                "score": 3,
                "example_response": "<Example response for score 3>"
            },
            {
                "score": 4,
                "example_response": "<Example response for score 4>"
            },
            {
                "score": 5,
                "example_response": "<Example response for score 5>"
            }
        ]
    }
    This should be repeated once for each of the provided evaluation questions.
    """

    # Define the system prompt for the Rubric Formatter Agent
    rubric_formatter_system_prompt = f"""
    You are a Rubric Formatter Agent. Your role is to format grading prompts for persona evaluation tasks. Using the output from previous agents, create a full evaluation rubric for the purposes of evaluating a persona against the {task.value} evaluation task.

    Format the rubric as a JSON object with the following schema, filling in the placeholders with appropriate values:
    {{
        "scoring_rubric": "<The extracted rubric JSON as a string>",
        "responses": [Array of responses to be evaluated]
    }}

    For each evaluation question, populate the responses array with a JSON object according to the following schema:
    {{
        "id": 1
        "question": "<The evaluation question #1>",
        "response": "<The response for question #1 from the persona agent>",
        "examples": [Array of generated example responses to the evaluation question #1 for each score]
    }}
    """

    rubric_extractor_agent = Agent(
        name=f"rubric_extractor_agent_for_{task_name}_eval",
        description="Agent that extracts the appropriate rubric from a full list of rubrics",
        model=LiteLlm(model=os.environ["RUBRIC_MODEL"]),
        instruction=rubric_extractor_system_prompt,
        tools=[file_read_tool],
        output_key=f"{task_name}_raw_rubric",
        after_agent_callback=log_state_after_agent,
        before_model_callback=log_prompt_before_llm
    )

    example_generator_agent = Agent(
        name=f"example_generator_agent_for_{task_name}_eval",
        description="Agent that generates response examples for each score in the provided rubric",
        model=LiteLlm(model=os.environ["RUBRIC_MODEL"]),
        instruction=example_generator_system_prompt,
        output_schema=ExampleGeneratorOutput,
        output_key=f"{task_name}_response_examples",
        after_agent_callback=log_state_after_agent,
        before_model_callback=log_prompt_before_llm
    )

    rubric_formatter_agent = Agent(
        name=f"rubric_formatter_agent_for_{task_name}_eval",
        description="Agent that formats the final rubric and examples to pass on to the evaluator agent",
        model=LiteLlm(model=os.environ["RUBRIC_MODEL"]),
        instruction=rubric_formatter_system_prompt,
        output_schema=EvaluationRubric,
        output_key=f"{task_name}_final_rubric",
        after_agent_callback=log_state_after_agent,
        before_model_callback=log_prompt_before_llm
    )

    return SequentialAgent(
        name=f"rubric_formatter_workflow_for_{task_name}_eval",
        description=f"Workflow to generate the scoring rubric for evaluating a given persona for its {task_name}",
        sub_agents=[
            rubric_extractor_agent,
            example_generator_agent,
            rubric_formatter_agent
        ]
    )

