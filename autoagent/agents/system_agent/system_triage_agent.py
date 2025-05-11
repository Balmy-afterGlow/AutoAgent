from .filesurfer_agent import get_filesurfer_agent
from .programming_agent import get_coding_agent
from .websurfer_agent import get_websurfer_agent
from autoagent.registry import register_agent
from autoagent.types import Agent, Result
from autoagent.tools.inner import case_resolved, case_not_resolved, additional_inquiry
import json
import re
from typing import Optional, Dict, Any


def repair_json_string(json_str: str) -> str:
    """
    Attempt to repair common JSON formatting issues.

    Args:
        json_str: The potentially malformed JSON string

    Returns:
        A repaired JSON string
    """
    # Remove any trailing commas before closing brackets/braces
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # Fix missing quotes around property names
    json_str = re.sub(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)

    # Fix single quotes to double quotes
    json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)

    # Fix missing commas between properties
    json_str = re.sub(r'"\s*}\s*"', r'", "', json_str)

    # Fix missing quotes around string values
    json_str = re.sub(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)([,\s}])", r':"\1"\2', json_str)

    return json_str


def safe_json_loads(
    json_str: str, default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely parse a JSON string with error handling and repair attempts.

    Args:
        json_str: The JSON string to parse
        default: Default value to return if all parsing attempts fail

    Returns:
        The parsed JSON object or the default value if all parsing attempts fail
    """
    if default is None:
        default = {
            "current_agent": "",
            "task_list": {},
            "context": {
                "previous_actions": [],
                "important_decisions": [],
                "constraints": [],
            },
        }

    # First try: direct parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second try: repair common JSON issues
    try:
        repaired_json = repair_json_string(json_str)
        return json.loads(repaired_json)
    except json.JSONDecodeError:
        pass

    # Third try: extract any valid JSON objects from the string
    try:
        # Find all potential JSON objects in the string
        json_objects = re.findall(r"\{[^{}]*\}", json_str)
        if json_objects:
            # Try to parse the largest matching object
            largest_object = max(json_objects, key=len)
            return json.loads(largest_object)
    except json.JSONDecodeError:
        pass

    # If all attempts fail, try to preserve any valid information
    try:
        # Extract key-value pairs that look like valid JSON
        key_value_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', json_str)
        if key_value_pairs:
            result = default.copy()
            for key, value in key_value_pairs:
                if key in result:
                    if isinstance(result[key], list):
                        result[key].append(value)
                    else:
                        result[key] = value
            return result
    except Exception:
        pass

    # If everything fails, return default with warning
    print(
        f"Warning: Failed to parse JSON after multiple attempts. Using default context."
    )
    print(f"Original JSON string: {json_str}")
    return default


@register_agent(name="System Triage Agent", func_name="get_system_triage_agent")
def get_system_triage_agent(model: str, **kwargs):
    """
    This is the `System Triage Agent`, it can help the user to determine which agent is best suited to handle the user's request under the current context, and transfer the conversation to that agent.

    Args:
        model: The model to use for the agent.
        **kwargs: Additional keyword arguments, `file_env`, `web_env` and `code_env` are required.
    """
    filesurfer_agent = get_filesurfer_agent(model)
    websurfer_agent = get_websurfer_agent(model)
    coding_agent = get_coding_agent(model)
    instructions = f"""You are a helpful assistant that can help the user with their request.
Based on the state of solving user's task, your responsibility is to determine which agent is best suited to handle the user's request under the current context, and transfer the conversation to that agent. And you should not stop to try to solve the user's request by transferring to another agent only until the task is completed.

There are three agents you can transfer to:
1. use `transfer_to_filesurfer_agent` to transfer to {filesurfer_agent.name}, it can help you to open any type of local files and browse the content of them.
2. use `transfer_to_websurfer_agent` to transfer to {websurfer_agent.name}, it can help you to open any website and browse any content on it.
3. use `transfer_to_coding_agent` to transfer to {coding_agent.name}, it can help you to write code to solve the user's request, especially some complex tasks.
"""

    optimized_instructions = f"""You are a task orchestration assistant that efficiently routes user requests across specialized agents.

Your responsibility is to:
1. Analyze the user's request and determine the most appropriate agent to handle each step
2. Transfer the conversation to that agent with clear instructions
3. When control returns to you, review the progress and either transfer to another agent or mark the task as resolved

IMPORTANT: Since each agent operates with its own system instructions, you must embed context and progress information directly in your transfer messages. 

When deconstructing user intent, please work according to the following rules:
1. Analyze user needs in detail
2. Break down user needs into multiple subtasks and build workflows
3. Generate a task list with two levels. The first level is the sub-agent that needs to transfer the task to execute, and the second level is the sub-tasks assigned to the corresponding sub-agent.
4. Each task in the task list needs to have a status description of "Pending/Completed". If you encounter the task_context parameter, you need to pass it as context.

Task Context Format:
{{
    "current_agent": "agent_name",
    "task_list": {{
        "agent_name": {{
            "subtasks": [
                {{
                    "id": "task_1",
                    "description": "Detailed task description",
                    "status": "Pending/Completed",
                    "dependencies": ["task_2", "task_3"],  # Optional: list of task IDs this task depends on
                    "expected_output": "What should be produced by this task"
                }}
            ]
        }}
    }},
    "context": {{
        "previous_actions": ["action1", "action2"],
        "important_decisions": ["decision1", "decision2"],
        "constraints": ["constraint1", "constraint2"]
    }}
}}

Guidelines for Task Management:
1. Each task must have a unique ID for tracking
2. Dependencies between tasks must be clearly specified
3. When transferring control to another agent, ensure task_context is properly formatted and complete
4. Update task status when a task is completed
5. Maintain context throughout the conversation

When reviewing conversation history:
1. Look for "Task Context:" sections in the messages
2. Parse the JSON task context to understand current state
3. Update task statuses based on agent responses
4. Track progress through the task list
5. Maintain context information across agent transitions

When transferring control:
1. Verify task_context format is correct
2. Ensure all necessary information is included
3. Update status of current agent's tasks
4. Pass complete context to the next agent

Available specialized agents:
1. {filesurfer_agent.name}: Handles file access and processing tasks. Use `transfer_to_filesurfer_agent` for file-related operations.
2. {websurfer_agent.name}: Handles web browsing and online research. Use `transfer_to_websurfer_agent` for internet-related tasks.
3. {coding_agent.name}: Creates and explains code solutions. Use `transfer_to_coding_agent` for programming tasks.

When reviewing an agent's response, pay careful attention to:
- Whether they fully completed their assigned subtask
- What new information or outputs they provided
- What logical next step is needed to progress toward resolving the user's complete request

Only mark a task as resolved when the user's entire request has been fulfilled completely.

You need to remember that any function call under System Triage Agent must be called separately.
"""
    tool_choice = "required"
    tools = (
        [case_resolved, case_not_resolved, additional_inquiry]
        if tool_choice == "required"
        else []
    )
    system_triage_agent = Agent(
        name="System Triage Agent",
        model=model,
        instructions=optimized_instructions,
        functions=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
    )

    def transfer_to_filesurfer_agent(sub_task_description: str, task_context: str):
        """
        Args:
            sub_task_description: The detailed description of the sub-task that the `System Triage Agent` will ask the `File Surfer Agent` to do.
            task_context: A JSON string containing the task context with task list and status.

        [IMPORTANT] Please make sure that the generated json has no format errors.

        Example of valid task_context:
        {
            "current_agent": "System Triage Agent",
            "task_list": {
                "File Surfer Agent": {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "description": "Open and read the configuration file",
                            "status": "Pending",
                            "dependencies": [],
                            "expected_output": "Configuration file contents"
                        }
                    ]
                }
            },
            "context": {
                "previous_actions": ["Initialized task", "Analyzed requirements"],
                "important_decisions": ["Selected File Surfer Agent for file operations"],
                "constraints": ["Must maintain file permissions", "Handle large files efficiently"]
            }
        }
        """
        return Result(
            value=sub_task_description,
            task_context=safe_json_loads(task_context),
            agent=filesurfer_agent,
        )

    def transfer_to_websurfer_agent(sub_task_description: str, task_context: str):
        """
        Args:
            sub_task_description: The detailed description of the sub-task that the `System Triage Agent` will ask the `Web Surfer Agent` to do.
            task_context: A JSON string containing the task context with task list and status.

        [IMPORTANT] Please make sure that the generated json has no format errors.

        Example of valid task_context:
        {
            "current_agent": "System Triage Agent",
            "task_list": {
                "Web Surfer Agent": {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "description": "Search for API documentation",
                            "status": "Pending",
                            "dependencies": [],
                            "expected_output": "API documentation URL and key information"
                        }
                    ]
                }
            },
            "context": {
                "previous_actions": ["Initialized task", "Analyzed requirements"],
                "important_decisions": ["Selected Web Surfer Agent for web research"],
                "constraints": ["Must verify source reliability", "Handle rate limiting"]
            }
        }
        """
        return Result(
            value=sub_task_description,
            task_context=safe_json_loads(task_context),
            agent=websurfer_agent,
        )

    def transfer_to_coding_agent(sub_task_description: str, task_context: str):
        """
        Args:
            sub_task_description: The detailed description of the sub-task that the `System Triage Agent` will ask the `Coding Agent` to do.
            task_context: A JSON string containing the task context with task list and status.

        [IMPORTANT] Please make sure that the generated json has no format errors.

        Example of valid task_context:
        {
            "current_agent": "System Triage Agent",
            "task_list": {
                "Coding Agent": {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "description": "Implement error handling for API calls",
                            "status": "Pending",
                            "dependencies": ["task_2"],
                            "expected_output": "Robust error handling code with retry mechanism"
                        }
                    ]
                }
            },
            "context": {
                "previous_actions": ["Initialized task", "Analyzed requirements"],
                "important_decisions": ["Selected Coding Agent for implementation"],
                "constraints": ["Must follow coding standards", "Include unit tests"]
            }
        }
        """
        return Result(
            value=sub_task_description,
            task_context=safe_json_loads(task_context),
            agent=coding_agent,
        )

    def transfer_back_to_triage_agent(completion_description: str, task_context: str):
        """
        It must be called after confirming that the subtask has been completed. The best practice is to call this function only in a response to avoid calling it as the last function in a response to multiple function calls, which may lead to speculation that the task has been successfully completed.

        Args:
            completion_description: The detailed description of the task status after a sub-agent has finished its sub-task. A sub-agent can use this tool to transfer the conversation back to the `System Triage Agent` only when it has finished its sub-task.
            task_context: A JSON string containing the task context with task list and status.

        [IMPORTANT] Please make sure that the generated json has no format errors.

        Example of valid task_context:
        {
            "current_agent": "File Surfer Agent",
            "task_list": {
                "File Surfer Agent": {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "description": "Open and read the configuration file",
                            "status": "Completed",
                            "dependencies": [],
                            "expected_output": "Configuration file contents"
                        }
                    ]
                }
            },
            "context": {
                "previous_actions": ["Opened file", "Read contents", "Parsed configuration"],
                "important_decisions": ["Selected JSON parser for configuration"],
                "constraints": ["Maintained file permissions", "Handled large file efficiently"]
            }
        }
        """
        return Result(
            value=completion_description,
            task_context=safe_json_loads(task_context),
            agent=system_triage_agent,
        )

    system_triage_agent.agent_teams = {
        filesurfer_agent.name: transfer_to_filesurfer_agent,
        websurfer_agent.name: transfer_to_websurfer_agent,
        coding_agent.name: transfer_to_coding_agent,
    }
    system_triage_agent.functions.extend(
        [
            transfer_to_filesurfer_agent,
            transfer_to_websurfer_agent,
            transfer_to_coding_agent,
        ]
    )
    filesurfer_agent.functions.append(transfer_back_to_triage_agent)
    websurfer_agent.functions.append(transfer_back_to_triage_agent)
    coding_agent.functions.append(transfer_back_to_triage_agent)
    return system_triage_agent
