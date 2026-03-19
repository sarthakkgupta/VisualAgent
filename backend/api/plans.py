import json
import logging
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, Body, HTTPException, Request, status

from db.mongo_config import queries_collection
from models import PlanModificationRequest
from sk_config.plugins import (
    call_goal_interpreter,
    call_plan_modifier,
    call_task_breakdown,
    call_timeline_generator,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/plan")
async def generate_plan(request: Request):
    try:
        body = await request.json()
        goal_input = body.get("goal")
        user_id = body.get("user_id")
        include_timeline = body.get("include_timeline", True)

        if not goal_input:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No goal provided")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No user_id provided")

        logger.info(f"Processing goal for user {user_id}: {goal_input}")

        # Step 1: Interpret the goal
        objective_text = str(await call_goal_interpreter(goal_input)).strip()
        logger.info(f"Interpreted objective: {objective_text}")

        # Extract duration if specified
        specified_duration = None
        for line in objective_text.split("\n"):
            if line.startswith("Duration:"):
                specified_duration = line.replace("Duration:", "").strip()
                break

        # Step 2: Build task breakdown input with duration context
        task_breakdown_input = objective_text
        if specified_duration and specified_duration.lower() != "not mentioned":
            category = _categorize_duration(specified_duration)
            task_breakdown_input = (
                f"{objective_text}\n\n"
                f"CRITICAL INSTRUCTION: This is a {category} duration plan of {specified_duration}. "
                f"ALL tasks MUST collectively fit within exactly {specified_duration}. "
                f"Create an appropriate number of tasks for this timeframe."
            )

        tasks_text = str(await call_task_breakdown(task_breakdown_input)).strip()
        task_sections = tasks_text.split("TASK ")[1:]

        total_days = (
            _to_days(specified_duration)
            if specified_duration and specified_duration.lower() != "not mentioned"
            else 0
        )
        estimates = _distribute_days(total_days, len(task_sections))
        allocated_days = 0
        processed_tasks = []

        # Step 3: Build tasks with optional timelines
        for i, section in enumerate(task_sections):
            parts = section.strip().split("\n", 1)
            if len(parts) < 2:
                continue

            task = {
                "title": parts[0].strip(": "),
                "content": parts[1].strip(),
                "completed": False,
            }

            if include_timeline:
                if total_days > 0:
                    task_days = estimates[i] if i < len(estimates) else 1
                    context = (
                        f"This is task {i + 1} of {len(task_sections)} in a plan where ALL TASKS must fit within "
                        f"EXACTLY {specified_duration}. So far {allocated_days} days have been allocated. "
                        f"This task should take approximately {task_days} days. Task: {task['title']}"
                    )
                    duration_text = str(await call_timeline_generator(context)).strip()
                    days_allocated = max(1, _to_days(duration_text) or task_days)

                    if i == len(task_sections) - 1:
                        days_allocated = max(1, total_days - allocated_days)
                    else:
                        days_available = total_days - allocated_days - (len(task_sections) - i - 1)
                        days_allocated = max(1, min(days_allocated, days_available))

                    duration_text = f"{days_allocated} days"
                    allocated_days += days_allocated
                else:
                    duration_text = str(await call_timeline_generator(task["title"])).strip()

                task["duration"] = duration_text

            processed_tasks.append(task)

        # Final overflow correction
        if total_days > 0 and allocated_days > total_days:
            _adjust_durations(processed_tasks, total_days, allocated_days)

        result = {"objective": objective_text, "tasks": processed_tasks}
        queries_collection.insert_one({
            "user_id": user_id,
            "query": goal_input,
            "result": result,
            "timestamp": datetime.utcnow(),
        })
        logger.info(f"Plan stored for user {user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/api/task/modify")
async def modify_plan(modification_request: PlanModificationRequest = Body(...)):
    try:
        oid = _parse_object_id(modification_request.task_id)
        existing = queries_collection.find_one({"_id": oid, "user_id": modification_request.user_id})
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        combined_input = (
            f"Current Plan JSON:\n{json.dumps(existing.get('result', {}))}\n\n"
            f"Modification Request:\n{modification_request.modification_text}"
        )
        modified_text = await call_plan_modifier(combined_input)

        try:
            modified_plan = json.loads(modified_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse the modified plan", "raw_response": str(modified_text)}

        if isinstance(modified_plan, dict) and modified_plan.get("needs_clarification"):
            return {
                "needs_clarification": True,
                "clarification_question": modified_plan.get("clarification_question", "Could you clarify?"),
            }

        queries_collection.update_one(
            {"_id": oid, "user_id": modification_request.user_id},
            {"$set": {"result": modified_plan}},
        )
        updated = queries_collection.find_one({"_id": oid, "user_id": modification_request.user_id})
        updated["_id"] = str(updated["_id"])
        return updated

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying plan: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- Helpers ---

def _parse_object_id(task_id: str) -> ObjectId:
    try:
        return ObjectId(task_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task ID format")


def _categorize_duration(duration: str) -> str:
    d = duration.lower()
    digits = int("".join(filter(str.isdigit, d)) or "0")
    if "day" in d:
        return "short" if digits <= 3 else ("long" if digits >= 21 else "medium")
    if "week" in d:
        return "long" if digits > 1 else "medium"
    if "month" in d:
        return "long"
    return "medium"


def _to_days(duration: str) -> int:
    if not duration:
        return 0
    d = duration.lower()
    digits = int("".join(filter(str.isdigit, d)) or "0")
    if "week" in d:
        return digits * 7
    if "month" in d:
        return digits * 30
    return digits


def _distribute_days(total_days: int, num_tasks: int) -> list:
    if total_days <= 0 or num_tasks <= 0:
        return []
    estimates, remaining = [], total_days
    for i in range(num_tasks):
        if i == num_tasks - 1:
            estimates.append(remaining)
        elif i == 0 and num_tasks > 2:
            days = max(1, min(int(total_days * 0.3), remaining - 1))
            estimates.append(days)
        else:
            days = max(1, min(int(remaining / (num_tasks - i)), remaining - (num_tasks - i - 1)))
            estimates.append(days)
        remaining -= estimates[i]
    return estimates


def _adjust_durations(tasks: list, total_days: int, allocated_days: int) -> None:
    factor = total_days / allocated_days
    adjusted = 0
    for i, task in enumerate(tasks):
        if "duration" not in task:
            continue
        days = _to_days(task["duration"])
        days = max(1, total_days - adjusted) if i == len(tasks) - 1 else max(1, int(days * factor))
        task["duration"] = f"{days} days"
        adjusted += days
