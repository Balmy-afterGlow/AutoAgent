from typing import List, Callable, Union, Optional, Tuple, Dict

# Third-party imports
from pydantic import BaseModel

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[dict], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = "none"
    parallel_tool_calls: bool = False
    examples: Union[List[Tuple[dict, str]], Callable[[dict], list]] = []
    handle_mm_func: Callable[[], str] | None = None
    agent_teams: Dict[str, Callable] = {}


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    task_context: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
    image: Optional[str] = None  # base64 encoded image
