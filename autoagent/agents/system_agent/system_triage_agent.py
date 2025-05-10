from .filesurfer_agent import get_filesurfer_agent
from .programming_agent import get_coding_agent
from .websurfer_agent import get_websurfer_agent
from autoagent.registry import register_agent
from autoagent.types import Agent, Result
from autoagent.tools.inner import case_resolved, case_not_resolved, additional_inquiry


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

For each transfer, include:
1. A clear summary of the overall task
2. What has been accomplished so far
3. The specific subtask for the target agent to complete
4. Any relevant information gathered from previous steps

Available specialized agents:
1. {filesurfer_agent.name}: Handles file access and processing tasks. Use `transfer_to_filesurfer_agent` for file-related operations.
2. {websurfer_agent.name}: Handles web browsing and online research. Use `transfer_to_websurfer_agent` for internet-related tasks.
3. {coding_agent.name}: Creates and explains code solutions. Use `transfer_to_coding_agent` for programming tasks.

When reviewing an agent's response, pay careful attention to:
- Whether they fully completed their assigned subtask
- What new information or outputs they provided
- What logical next step is needed to progress toward resolving the user's complete request

Only mark a task as resolved when the user's entire request has been fulfilled completely.
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

            task_context: A checklist to indicate overall task completion.
        """
        return Result(
            value=sub_task_description,
            task_context=task_context,
            agent=filesurfer_agent,
        )

    def transfer_to_websurfer_agent(sub_task_description: str, task_context: str):
        """
        Args:
            sub_task_description: The detailed description of the sub-task that the `System Triage Agent` will ask the `Web Surfer Agent` to do.

            task_context: A checklist to indicate overall task completion.
        """
        return Result(
            value=sub_task_description, task_context=task_context, agent=websurfer_agent
        )

    def transfer_to_coding_agent(sub_task_description: str, task_context: str):
        """
        Args:
            sub_task_description: The detailed description of the sub-task that the `System Triage Agent` will ask the `Coding Agent` to do.

            task_context: A checklist to indicate overall task completion.
        """
        return Result(
            value=sub_task_description, task_context=task_context, agent=coding_agent
        )

    def transfer_back_to_triage_agent(completion_description: str, task_context: str):
        """
        Args:
            completion_description: The detailed description of the task status after a sub-agent has finished its sub-task. A sub-agent can use this tool to transfer the conversation back to the `System Triage Agent` only when it has finished its sub-task.

            task_context: A checklist to indicate overall task completion.
        """
        return Result(
            value=completion_description,
            task_context=task_context,
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
