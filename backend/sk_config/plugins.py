from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

USER_ENDPOINT = "https://visualaiproject.services.ai.azure.com/api/projects/visualaiproject-project"


def _get_project_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint=USER_ENDPOINT,
        credential=DefaultAzureCredential(),
    )


def _deployment() -> str:
    return os.getenv("AZURE_FOUNDRY_DEPLOYMENT")


def _sync_call_goal_interpreter(user_input: str) -> str:
    project_client = _get_project_client()
    agent = project_client.agents.create_version(
        agent_name="goal-interpreter",
        definition=PromptAgentDefinition(
            model=_deployment(),
            instructions=(
                "Extract the following from this input:\n"
                "1. Objective (what does the user want to do?)\n"
                "2. Duration in days/weeks if mentioned\n\n"
                'Return as: "Objective: ...\\nDuration: ..."'
            ),
        ),
    )
    openai_client = project_client.get_openai_client()
    response = openai_client.responses.create(
        input=[{"role": "user", "content": f"User input: {user_input}"}],
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    return response.output_text


async def call_goal_interpreter(user_input: str) -> str:
    return await asyncio.to_thread(_sync_call_goal_interpreter, user_input)


def _sync_call_task_breakdown(user_input: str) -> str:
    project_client = _get_project_client()
    agent = project_client.agents.create_version(
        agent_name="task-breakdown",
        definition=PromptAgentDefinition(
            model=_deployment(),
            instructions=(
                "Take the objective and break it down into appropriate tasks that MUST collectively fit within the specified duration.\n\n"
                "IMPORTANT DURATION INSTRUCTIONS:\n"
                "- If the duration is short (e.g., 3 days or less), create just 2-3 focused tasks.\n"
                "- If the duration is medium (e.g., 1-2 weeks), create 3-4 balanced tasks.\n"
                "- If the duration is longer (e.g., 3+ weeks), create 4-6 comprehensive tasks.\n"
                "- Ensure tasks are sized appropriately for the timeframe - shorter durations require more concise tasks.\n\n"
                "For each task, include:\n"
                "1. A clear high-level task name/description\n"
                "2. 3-5 specific actionable subtasks with naturally integrated resources\n\n"
                "Format:\n"
                "TASK 1: [High-level task name/description]\n"
                "- [Actionable step with optional resources]\n"
                "- [Actionable step with optional resources]\n"
                "- [Actionable step with optional resources]\n\n"
                "TASK 2: [High-level task name/description]\n"
                "- ...\n\n"
                "[Continue for remaining tasks]"
            ),
        ),
    )
    openai_client = project_client.get_openai_client()
    response = openai_client.responses.create(
        input=[{"role": "user", "content": f"Objective: {user_input}"}],
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    return response.output_text


async def call_task_breakdown(user_input: str) -> str:
    return await asyncio.to_thread(_sync_call_task_breakdown, user_input)


def _sync_call_timeline_generator(user_input: str) -> str:
    project_client = _get_project_client()
    agent = project_client.agents.create_version(
        agent_name="timeline-generator",
        definition=PromptAgentDefinition(
            model=_deployment(),
            instructions=(
                "Provide a simple time duration estimate for completing this task.\n\n"
                "IMPORTANT TIMELINE CONSTRAINTS:\n"
                "- Your duration MUST respect the overall plan duration\n"
                "- Express durations in days rather than weeks whenever possible\n"
                "- Ensure your estimate allows ALL tasks to fit within the total timeframe\n"
                "- If this task is part of a short total duration (e.g., 3 days), keep the estimate very concise (1-2 days)\n\n"
                'Just return a clear and concise duration estimate like "1 day", "3 days", "5 days", etc.\n'
                "ONLY use weeks as the unit if the task genuinely requires more than 10 days.\n"
                "No additional explanation needed."
            ),
        ),
    )
    openai_client = project_client.get_openai_client()
    response = openai_client.responses.create(
        input=[{"role": "user", "content": f"Task: {user_input}"}],
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    return response.output_text


async def call_timeline_generator(user_input: str) -> str:
    return await asyncio.to_thread(_sync_call_timeline_generator, user_input)


def _sync_call_plan_modifier(user_input: str) -> str:
    project_client = _get_project_client()
    agent = project_client.agents.create_version(
        agent_name="plan-modifier",
        definition=PromptAgentDefinition(
            model=_deployment(),
            instructions=(
                "You are a helpful AI assistant tasked with modifying learning plans based on natural language requests.\n\n"
                "The input below contains the current plan JSON followed by the user's modification request.\n"
                "Parse both sections carefully and apply the requested changes to the plan.\n\n"
                "Instructions:\n"
                "1. Extract the current plan JSON from the input\n"
                "2. Understand the user's modification request\n"
                "3. Apply the requested changes to the plan\n"
                "4. Possible modifications include:\n"
                "   - Changing the goal/objective\n"
                "   - Modifying the duration\n"
                "   - Adding new tasks\n"
                "   - Removing tasks\n"
                "   - Updating task content\n"
                "   - Correcting errors in the plan\n"
                "   - Changing task durations\n\n"
                "5. Make only the requested changes while preserving the rest of the plan's structure\n"
                "6. Ensure all durations still add up appropriately and fit within the overall timeframe\n"
                "7. Return the complete modified plan in JSON format\n\n"
                "Return the modified plan in this format:\n"
                "{\n"
                '  "objective": "The updated objective text",\n'
                '  "tasks": [\n'
                "    {\n"
                '      "title": "Task title",\n'
                '      "content": "Task content with subtasks",\n'
                '      "duration": "Duration in days",\n'
                '      "completed": boolean\n'
                "    },\n"
                "    ...\n"
                "  ]\n"
                "}\n\n"
                "If the user request requires clarification, identify what's unclear and return:\n"
                "{\n"
                '  "needs_clarification": true,\n'
                '  "clarification_question": "Your specific question about what you need clarified"\n'
                "}"
            ),
        ),
    )
    openai_client = project_client.get_openai_client()
    response = openai_client.responses.create(
        input=[{"role": "user", "content": user_input}],
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    return response.output_text


async def call_plan_modifier(user_input: str) -> str:
    return await asyncio.to_thread(_sync_call_plan_modifier, user_input)
