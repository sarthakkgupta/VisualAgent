from fastapi import FastAPI, Request, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from sk_config.kernel import get_kernel_and_plugins
from db.mongo_config import queries_collection
from datetime import datetime
import uvicorn
from typing import Optional, Dict, Any, List
import httpx
import logging
from bson import ObjectId
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/plan")
async def generate_plan(request: Request):
    try:
        body = await request.json()
        goal_input = body.get("goal")
        user_id = body.get("user_id")
        include_timeline = body.get("include_timeline", True)
        
        if not goal_input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No goal provided"
            )

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )

        logger.info(f"Processing goal for user {user_id}: {goal_input}")
        kernel, plugins = get_kernel_and_plugins()

        # Step 1: Get high-level objective
        logger.debug("Getting objective interpretation")
        objective = await plugins["goal_interpreter"].invoke(kernel=kernel, input=goal_input)
        logger.info(f"Interpreted objective: {objective}")
        
        # Extract duration from objective if specified
        objective_text = str(objective).strip()
        specified_duration = None
        for line in objective_text.split('\n'):
            if line.startswith('Duration:'):
                specified_duration = line.replace('Duration:', '').strip()
                logger.info(f"Extracted specified duration: {specified_duration}")
                break
        
        # Provide duration context to task breakdown
        task_breakdown_input = str(objective)
        if specified_duration and specified_duration.lower() != "not mentioned":
            # Create a more explicit instruction about duration constraints
            duration_category = "medium"  # Default
            
            # Determine duration category for task sizing
            if "day" in specified_duration.lower():
                days = int(''.join(filter(str.isdigit, specified_duration)))
                if days <= 3:
                    duration_category = "short"
                elif days >= 21:
                    duration_category = "long"
                else:
                    duration_category = "medium"
            elif "week" in specified_duration.lower():
                weeks = int(''.join(filter(str.isdigit, specified_duration)))
                if weeks <= 1:
                    duration_category = "medium"
                else:
                    duration_category = "long"
            elif "month" in specified_duration.lower():
                duration_category = "long"
                
            logger.info(f"Categorized duration as {duration_category}")
            
            task_breakdown_input = (
                f"{objective_text}\n\n"
                f"CRITICAL INSTRUCTION: This is a {duration_category} duration plan of {specified_duration}. "
                f"ALL tasks MUST collectively fit within exactly {specified_duration}. "
                f"Create an appropriate number of tasks for this timeframe."
            )
            
        # Step 2: Break down into high-level tasks with integrated subtasks and resources
        logger.debug("Breaking down into tasks with subtasks and resources")
        tasks_response = await plugins["task_breakdown"].invoke(kernel=kernel, input=task_breakdown_input)
        
        # Parse the response to extract tasks, subtasks, and resources
        tasks_text = str(tasks_response).strip()
        task_sections = tasks_text.split("TASK ")[1:]  # Split by "TASK " and remove empty first element
        
        # Process the parsed tasks
        processed_tasks = []
        num_tasks = len(task_sections)
        
        # Convert specified_duration to days for calculation (approximate)
        total_days = 0
        if specified_duration and specified_duration.lower() != "not mentioned":
            if "day" in specified_duration.lower():
                total_days = int(''.join(filter(str.isdigit, specified_duration)))
            elif "week" in specified_duration.lower():
                total_days = int(''.join(filter(str.isdigit, specified_duration))) * 7
            elif "month" in specified_duration.lower():
                total_days = int(''.join(filter(str.isdigit, specified_duration))) * 30
            logger.info(f"Estimated total days: {total_days}")
        
        # Track the running total of days allocated
        allocated_days = 0
        
        # Initial estimates for each task based on total days
        initial_estimates = []
        if total_days > 0 and num_tasks > 0:
            # First pass: make initial estimates that sum to exactly total_days
            remaining_days = total_days
            
            for i in range(num_tasks):
                # Last task gets all remaining days
                if i == num_tasks - 1:
                    initial_estimates.append(remaining_days)
                # First task might need a bit more time for setup
                elif i == 0 and num_tasks > 2:
                    est_days = max(1, int(total_days * 0.3))
                    initial_estimates.append(min(est_days, remaining_days-1))
                # Otherwise distribute fairly
                else:
                    est_days = max(1, int(remaining_days / (num_tasks - i)))
                    initial_estimates.append(min(est_days, remaining_days - (num_tasks - i - 1)))
                
                remaining_days -= initial_estimates[i]
                
            logger.info(f"Initial day allocations: {initial_estimates}, total: {sum(initial_estimates)}/{total_days}")
        
        for i, task_section in enumerate(task_sections):
            # Extract task title (first line)
            task_parts = task_section.strip().split("\n", 1)
            if len(task_parts) < 2:
                continue
                
            task_title = task_parts[0].strip(": ")
            task_content = task_parts[1].strip() if len(task_parts) > 1 else ""
            
            # Create a task object
            task_object = {
                "title": task_title,
                "content": task_content,
                "completed": False  # Initialize each task as not completed
            }
            
            # Add timeline if requested
            if include_timeline:
                logger.debug(f"Generating timeline for task: {task_title}")
                if "timeline_generator" in plugins:
                    # If we have a specified duration, calculate a portion for this task
                    if specified_duration and specified_duration.lower() != "not mentioned" and total_days > 0:
                        # Get pre-allocated days for this task
                        task_days = initial_estimates[i] if i < len(initial_estimates) else 1
                        
                        # Build context that ensures timeline fits in overall duration
                        task_context = (
                            f"This is task {i+1} of {num_tasks} in a plan where ALL TASKS COLLECTIVELY must be completed within EXACTLY {specified_duration}. "
                            f"So far, {allocated_days} days have been allocated to previous tasks. "
                            f"This plan has a total of ONLY {total_days} days available. "
                            f"This task should take APPROXIMATELY {task_days} days to ensure all tasks fit together. "
                            f"Task: {task_title}"
                        )
                        logger.debug(f"Using task context with duration: {task_context}")
                        timeline = await plugins["timeline_generator"].invoke(kernel=kernel, input=task_context)
                        
                        # Extract the numeric duration for tracking allocated time
                        duration_text = str(timeline).strip()
                        logger.info(f"Timeline generator returned: '{duration_text}' for task {i+1}")
                        
                        try:
                            if "day" in duration_text.lower():
                                days_allocated = int(''.join(filter(str.isdigit, duration_text)))
                            elif "week" in duration_text.lower():
                                days_allocated = int(''.join(filter(str.isdigit, duration_text))) * 7
                            else:
                                # Default to the calculated task_days if we can't parse
                                days_allocated = task_days
                                
                            # Enforce minimum 1 day for any task
                            days_allocated = max(1, days_allocated)
                            
                            # For last task, ensure we don't exceed total
                            if i == num_tasks - 1:
                                # If we're going to exceed the total, cap at remaining days
                                if allocated_days + days_allocated > total_days:
                                    days_allocated = max(1, total_days - allocated_days)
                                    duration_text = f"{days_allocated} days"
                            # For all other tasks, ensure we leave at least 1 day per remaining task
                            else:
                                days_needed_for_remaining = num_tasks - i - 1  # At least 1 day per remaining task
                                days_available = total_days - allocated_days - days_needed_for_remaining
                                if days_allocated > days_available:
                                    days_allocated = max(1, days_available)
                                    duration_text = f"{days_allocated} days"
                                
                            # Update running total
                            allocated_days += days_allocated
                            logger.info(f"Task {i+1} allocated {days_allocated} days. Total allocated: {allocated_days}/{total_days}")
                        except Exception as e:
                            # If parsing fails, use our calculated value
                            logger.error(f"Failed to parse duration '{duration_text}': {str(e)}")
                            days_allocated = task_days
                            duration_text = f"{days_allocated} days"
                            allocated_days += days_allocated
                            logger.info(f"Using calculated {days_allocated} days. Total allocated: {allocated_days}/{total_days}")
                    else:
                        timeline = await plugins["timeline_generator"].invoke(kernel=kernel, input=task_title)
                        duration_text = str(timeline).strip()
                        
                    # Ensure duration is added to the task object
                    task_object["duration"] = duration_text
            
            processed_tasks.append(task_object)

        # Final check: verify we haven't exceeded the total duration
        if total_days > 0 and allocated_days > total_days:
            logger.warning(f"Total allocated days ({allocated_days}) exceeded specified duration ({total_days}). Adjusting final tasks.")
            
            # Simple proportional adjustment to fit within total
            reduction_factor = total_days / allocated_days
            adjusted_allocated = 0
            
            for i, task in enumerate(processed_tasks):
                if "duration" in task:
                    duration_text = task["duration"]
                    try:
                        if "day" in duration_text.lower():
                            days = int(''.join(filter(str.isdigit, duration_text)))
                            # Adjust days proportionally, ensuring at least 1 day
                            adjusted_days = max(1, int(days * reduction_factor))
                            
                            # Last task gets any remaining days
                            if i == len(processed_tasks) - 1:
                                adjusted_days = max(1, total_days - adjusted_allocated)
                                
                            task["duration"] = f"{adjusted_days} days"
                            adjusted_allocated += adjusted_days
                    except Exception as e:
                        logger.error(f"Failed to adjust duration '{duration_text}': {str(e)}")

        result = {
            "objective": str(objective).strip(),
            "tasks": processed_tasks
        }

        # Store the result in MongoDB
        query_record = {
            "user_id": user_id,
            "query": goal_input,
            "result": result,
            "timestamp": datetime.utcnow()
        }
        queries_collection.insert_one(query_record)
        logger.info(f"Plan generated and stored successfully for user {user_id}")

        return result

    except HTTPException as he:
        logger.error(f"HTTP error in generate_plan: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/history")
async def get_history(request: Request, user_id: str):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )

        # Get history with _id included
        history = list(queries_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1))
        
        # Convert ObjectId to string for each document
        for doc in history:
            doc['_id'] = str(doc['_id'])
        
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/health")
async def health_check(request: Request):
    try:
        # Check for test_token
        # Test MongoDB connection
        queries_collection.find_one({})
        return {"status": "healthy", "database": "connected"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.get("/api/task/{task_id}")
async def get_task_details(task_id: str, user_id: str):
    try:
        if not user_id:
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No user_id provided"
        )

        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Find the task in the queries collection
        task = queries_collection.find_one(
            {
                "user_id": user_id,
                "_id": task_object_id
            }
        )

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        # Convert ObjectId to string for JSON serialization
        task['_id'] = str(task['_id'])
        return task
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching task details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Define Pydantic models for request validation
class TaskUpdate(BaseModel):
    objective: Optional[str] = None
    tasks: Optional[List[Dict[str, Any]]] = None

# Define Pydantic models for the task completion feature
class TaskStatus(BaseModel):
    task_index: int
    completed: bool

class SubtaskStatus(BaseModel):
    task_index: int
    subtask_index: int
    completed: bool

class SubtaskBulkStatus(BaseModel):
    subtask_statuses: List[SubtaskStatus]

class TaskBulkStatus(BaseModel):
    task_statuses: List[TaskStatus]

# Define Pydantic model for natural language modification requests
class PlanModificationRequest(BaseModel):
    instruction: str  # Natural language instruction for modifying the plan

@app.put("/api/task/{task_id}")
async def update_task(task_id: str, user_id: str, update_data: TaskUpdate = Body(...)):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )

        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Check if task exists
        existing_task = queries_collection.find_one(
            {
                "user_id": user_id,
                "_id": task_object_id
            }
        )

        if not existing_task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )

        # Prepare update data
        update_fields = {}
        
        # Update result.objective if provided
        if update_data.objective is not None:
            update_fields["result.objective"] = update_data.objective
            
        # Update result.tasks if provided
        if update_data.tasks is not None:
            update_fields["result.tasks"] = update_data.tasks
            
        # If nothing to update, return error
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
            
        # Update the task
        result = queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": update_fields}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="No changes made to the task"
            )
            
        # Get the updated task
        updated_task = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        # Convert ObjectId to string for JSON serialization
        updated_task['_id'] = str(updated_task['_id'])
        
        return updated_task
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str, user_id: str):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )

        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Check if task exists
        existing_task = queries_collection.find_one(
            {
                "user_id": user_id,
                "_id": task_object_id
            }
        )

        if not existing_task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Delete the task
        result = queries_collection.delete_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="Task could not be deleted"
            )
            
        return {"message": "Task successfully deleted", "task_id": task_id}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Add endpoint to update specific task details
class TaskDetailUpdate(BaseModel):
    task_index: int
    title: Optional[str] = None
    content: Optional[str] = None
    duration: Optional[str] = None

@app.patch("/api/task/{task_id}/detail")
async def update_task_detail(task_id: str, user_id: str, update_data: TaskDetailUpdate = Body(...)):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )

        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Check if task exists
        existing_task = queries_collection.find_one(
            {
                "user_id": user_id,
                "_id": task_object_id
            }
        )

        if not existing_task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Get tasks array
        tasks = existing_task.get("result", {}).get("tasks", [])
        
        # Check if task_index is valid
        if update_data.task_index < 0 or update_data.task_index >= len(tasks):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task index. Must be between 0 and {len(tasks)-1}"
            )
            
        # Create update operations
        update_operations = {}
        
        if update_data.title is not None:
            update_operations[f"result.tasks.{update_data.task_index}.title"] = update_data.title
            
        if update_data.content is not None:
            update_operations[f"result.tasks.{update_data.task_index}.content"] = update_data.content
            
        if update_data.duration is not None:
            update_operations[f"result.tasks.{update_data.task_index}.duration"] = update_data.duration
            
        # If nothing to update, return error
        if not update_operations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
            
        # Update the specific task detail
        result = queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": update_operations}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="No changes made to the task detail"
            )
            
        # Get the updated task
        updated_task = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        # Convert ObjectId to string for JSON serialization
        updated_task['_id'] = str(updated_task['_id'])
        
        return updated_task
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating task detail: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.patch("/api/task/{task_id}/completion")
async def update_task_completion(task_id: str, user_id: str, status_update: TaskStatus = Body(...)):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )
            
        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Find the task
        task_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        if not task_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Validate task index
        tasks = task_doc.get("result", {}).get("tasks", [])
        if status_update.task_index < 0 or status_update.task_index >= len(tasks):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid task index. Must be between 0 and {len(tasks)-1}"
            )
            
        # Update the completed status for the specific task
        update_field = f"result.tasks.{status_update.task_index}.completed"
        result = queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": {update_field: status_update.completed}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="No changes made to the completion status"
            )
            
        # Calculate and update the overall progress
        updated_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        tasks = updated_doc.get("result", {}).get("tasks", [])
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.get("completed", False))
        
        # Calculate progress as percentage
        progress = int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
        
        # Update the progress field
        queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": {"progress": progress}}
        )
        
        # Return the updated document
        updated_final = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        # Convert ObjectId to string for JSON serialization
        updated_final['_id'] = str(updated_final['_id'])
        
        return updated_final
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating task completion status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.patch("/api/task/{task_id}/bulk-completion")
async def update_bulk_task_completion(task_id: str, user_id: str, status_updates: TaskBulkStatus = Body(...)):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )
            
        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Find the task
        task_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        if not task_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Validate all task indices
        tasks = task_doc.get("result", {}).get("tasks", [])
        total_tasks = len(tasks)
        
        for status in status_updates.task_statuses:
            if status.task_index < 0 or status.task_index >= total_tasks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task index: {status.task_index}. Must be between 0 and {total_tasks-1}"
                )
        
        # Update each task's completion status
        update_operations = {}
        for status in status_updates.task_statuses:
            update_field = f"result.tasks.{status.task_index}.completed"
            update_operations[update_field] = status.completed
        
        # If nothing to update, return error
        if not update_operations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        result = queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": update_operations}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="No changes made to the completion status"
            )
            
        # Calculate and update the overall progress
        updated_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        tasks = updated_doc.get("result", {}).get("tasks", [])
        completed_tasks = sum(1 for task in tasks if task.get("completed", False))
        
        # Calculate progress as percentage
        progress = int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
        
        # Update the progress field
        queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": {"progress": progress}}
        )
        
        # Return the updated document
        updated_final = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        # Convert ObjectId to string for JSON serialization
        updated_final['_id'] = str(updated_final['_id'])
        
        return updated_final
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating bulk task completion status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/task/{task_id}/progress")
async def get_task_progress(task_id: str, user_id: str):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )
            
        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Find the task
        task_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        if not task_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Get tasks and calculate progress
        tasks = task_doc.get("result", {}).get("tasks", [])
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.get("completed", False))
        
        # Calculate progress as percentage
        progress = int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
        
        # Get stored progress or calculate if not available
        stored_progress = task_doc.get("progress", progress)
        
        return {
            "task_id": task_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress": stored_progress,
            "completed_status": [task.get("completed", False) for task in tasks]
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting task progress: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/task/{task_id}/modify")
async def modify_plan_with_instruction(task_id: str, user_id: str, modification: PlanModificationRequest = Body(...)):
    try:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user_id provided"
            )
            
        # Convert string task_id to ObjectId
        try:
            task_object_id = ObjectId(task_id)
        except:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task ID format"
            )

        # Find the task
        task_doc = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        if not task_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        # Get current plan details
        current_objective = task_doc.get("result", {}).get("objective", "")
        current_tasks = task_doc.get("result", {}).get("tasks", [])
        original_query = task_doc.get("query", "")
        
        # Extract duration from objective if specified
        specified_duration = None
        for line in current_objective.split('\n'):
            if line.startswith('Duration:'):
                specified_duration = line.replace('Duration:', '').strip()
                break
                
        if not specified_duration or specified_duration.lower() == "not mentioned":
            specified_duration = "not specified"
            
        # Initialize kernel and plugins for AI processing
        kernel, plugins = get_kernel_and_plugins()
        
        # Determine the modification type and process accordingly
        instruction = modification.instruction.lower()
        
        # Initialize task_num variable to avoid "cannot access local variable" error
        task_num = None
        
        # Prepare context for AI processing
        current_plan_summary = f"Current objective: {current_objective}\n"
        current_plan_summary += f"Number of tasks: {len(current_tasks)}\n"
        
        # Add task summaries
        for i, task in enumerate(current_tasks):
            current_plan_summary += f"Task {i+1}: {task.get('title')}\n"
            
            # Add abbreviated content (first 100 chars)
            content = task.get('content', '')
            if content:
                summary = content[:100] + "..." if len(content) > 100 else content
                current_plan_summary += f"  Content: {summary}\n"
                
            if 'duration' in task:
                current_plan_summary += f"  Duration: {task.get('duration')}\n"
                
        # Create a prompt for the AI to process the modification
        modification_input = (
            f"You are helping a user modify their learning plan. Here's the current plan:\n\n"
            f"{current_plan_summary}\n\n"
            f"The user wants to: \"{modification.instruction}\"\n\n"
            f"Original goal: {original_query}\n"
            f"Total duration: {specified_duration}\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"1. DO NOT create a completely new plan.\n"
            f"2. PRESERVE the existing tasks and structure as much as possible.\n"
            f"3. Make MINIMAL changes needed to address the user's request.\n"
            f"4. If making a task more beginner-friendly, simplify explanations without removing content.\n"
            f"5. If adjusting difficulty, modify details while keeping the same task structure.\n"
            f"6. Return the modified plan in the SAME format as the original.\n"
            f"7. NEVER remove tasks completely unless explicitly asked to reduce task count.\n\n"
            f"ONLY modify what's necessary to address: \"{modification.instruction}\"\n"
        )
        
        # Handle different types of modifications
        if "decrease" in instruction and "task" in instruction:
            # Instead of regenerating from scratch, attempt to combine/merge existing tasks
            try:
                # Get current tasks and find candidates for merging
                existing_tasks = current_tasks.copy()
                
                # If we have very few tasks already, don't reduce further
                if len(existing_tasks) <= 2:
                    # Just update the content of the existing tasks to be more comprehensive
                    for i, task in enumerate(existing_tasks):
                        # Update content to make it more comprehensive since we're not reducing tasks
                        enhanced_input = f"Make this task more comprehensive by combining multiple concepts:\n\nTask: {task.get('title')}\nContent: {task.get('content', '')}"
                        enhanced_content = await plugins["task_breakdown"].invoke(kernel=kernel, input=enhanced_input)
                        existing_tasks[i]['content'] = str(enhanced_content).strip()
                        
                    update_fields = {
                        "result.tasks": existing_tasks,
                        "last_modified": datetime.utcnow()
                    }
                else:
                    # Use task_breakdown to get suggestions for combining tasks
                    task_summary = "\n".join([f"Task {i+1}: {task.get('title')}" for i, task in enumerate(existing_tasks)])
                    merge_input = f"I have these tasks for my plan:\n{task_summary}\n\nSuggest which tasks could be combined to create a more concise plan with fewer tasks. DO NOT rewrite the entire plan, just indicate which task numbers could be merged."
                    
                    merge_suggestions = await plugins["task_breakdown"].invoke(kernel=kernel, input=merge_input)
                    merge_text = str(merge_suggestions).strip()
                    
                    logger.info(f"Merge suggestions: {merge_text}")
                    
                    # Plan B: If we can't parse merge suggestions, use a more conservative approach
                    # Merge adjacent pairs of tasks if we need to reduce by around 50%
                    target_count = max(2, len(existing_tasks) // 2 + len(existing_tasks) % 2)
                    new_tasks = []
                    
                    i = 0
                    while i < len(existing_tasks):
                        if len(new_tasks) < target_count - 1 and i < len(existing_tasks) - 1:
                            # Merge this task with the next one
                            task1 = existing_tasks[i]
                            task2 = existing_tasks[i+1]
                            
                            merged_title_input = f"Create a single concise title that combines these two topics:\n1. {task1.get('title')}\n2. {task2.get('title')}"
                            merged_title = await plugins["task_breakdown"].invoke(kernel=kernel, input=merged_title_input)
                            
                            merged_content = f"**Part 1: {task1.get('title')}**\n{task1.get('content', '')}\n\n**Part 2: {task2.get('title')}**\n{task2.get('content', '')}"
                            
                            # Determine duration (if applicable)
                            merged_duration = None
                            if "duration" in task1 and "duration" in task2:
                                try:
                                    # Extract numeric values from durations
                                    duration1 = int(''.join(filter(str.isdigit, task1.get("duration", "0"))))
                                    duration2 = int(''.join(filter(str.isdigit, task2.get("duration", "0"))))
                                    merged_duration = f"{duration1 + duration2} days"
                                except:
                                    # If we can't parse, use the longer duration
                                    merged_duration = task1.get("duration", task2.get("duration"))
                            elif "duration" in task1:
                                merged_duration = task1.get("duration")
                            elif "duration" in task2:
                                merged_duration = task2.get("duration")
                                
                            # Create merged task
                            merged_task = {
                                "title": str(merged_title).strip(),
                                "content": merged_content,
                                "completed": task1.get("completed", False) and task2.get("completed", False)
                            }
                            
                            if merged_duration:
                                merged_task["duration"] = merged_duration
                                
                            new_tasks.append(merged_task)
                            i += 2  # Skip both tasks we just merged
                        else:
                            # Keep this task as is
                            new_tasks.append(existing_tasks[i])
                            i += 1
                    
                    update_fields = {
                        "result.tasks": new_tasks,
                        "last_modified": datetime.utcnow()
                    }
            except Exception as e:
                logger.error(f"Error merging tasks: {str(e)}")
                # If merging fails, don't modify the tasks
                update_fields = {
                    "last_modified": datetime.utcnow(),
                    "modification_note": f"Failed to decrease tasks: {str(e)}"
                }
            
        elif "increase" in instruction and "task" in instruction:
            # Instead of regenerating from scratch, attempt to split/expand existing tasks
            try:
                existing_tasks = current_tasks.copy()
                new_tasks = []
                
                # Iterate through existing tasks and expand each one into multiple tasks
                for task in existing_tasks:
                    # Decide if this task should be split or kept as is
                    task_title = task.get('title', '')
                    task_content = task.get('content', '')
                    
                    expansion_input = f"Analyze this task and determine if it can be broken down into multiple smaller tasks:\nTask: {task_title}\nContent: {task_content}\n\nIf it can be broken down, provide 2-3 new task titles that would break this down into smaller components. Otherwise, respond with 'KEEP_AS_IS'."
                    
                    expansion_result = await plugins["task_breakdown"].invoke(kernel=kernel, input=expansion_input)
                    expansion_text = str(expansion_result).strip()
                    
                    if "KEEP_AS_IS" in expansion_text.upper():
                        # Keep this task as is
                        new_tasks.append(task)
                    else:
                        # Try to split this task into multiple tasks
                        split_input = f"Split this task into multiple smaller tasks:\nTask: {task_title}\nContent: {task_content}\n\nProvide 2-3 new tasks in this format:\nTASK 1: [title]\n[detailed content]\n\nTASK 2: [title]\n[detailed content]\n\nEtc."
                        
                        split_result = await plugins["task_breakdown"].invoke(kernel=kernel, input=split_input)
                        split_text = str(split_result).strip()
                        
                        # Parse the split tasks
                        split_sections = split_text.split("TASK ")[1:]  # Skip empty first element
                        
                        if split_sections:
                            # Process each split task
                            for section in split_sections:
                                try:
                                    section_parts = section.strip().split("\n", 1)
                                    if len(section_parts) >= 2:
                                        split_title = section_parts[0].strip(": ")
                                        split_content = section_parts[1].strip() if len(section_parts) > 1 else ""
                                        
                                        # Create new task
                                        split_task = {
                                            "title": split_title,
                                            "content": split_content,
                                            "completed": task.get("completed", False)  # Inherit completion status
                                        }
                                        
                                        # If original task had duration, distribute it
                                        if "duration" in task:
                                            try:
                                                original_duration = int(''.join(filter(str.isdigit, task.get("duration", "0"))))
                                                # Distribute duration among split tasks
                                                split_duration = max(1, original_duration // len(split_sections))
                                                split_task["duration"] = f"{split_duration} days"
                                            except:
                                                # If parsing fails, don't set duration
                                                pass
                                                
                                        new_tasks.append(split_task)
                                except Exception as e:
                                    logger.error(f"Error processing split task section: {str(e)}")
                                    # If we fail to parse a section, just keep the original task
                                    if not new_tasks or new_tasks[-1].get("title") != task.get("title"):
                                        new_tasks.append(task)
                        else:
                            # If splitting failed, keep original task
                            new_tasks.append(task)
                
                update_fields = {
                    "result.tasks": new_tasks,
                    "last_modified": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Error expanding tasks: {str(e)}")
                # If expansion fails, don't modify the tasks
                update_fields = {
                    "last_modified": datetime.utcnow(),
                    "modification_note": f"Failed to increase tasks: {str(e)}"
                }
        
        else:
            # Generic modification - use task_breakdown plugin since it's already working
            try:
                # Create a temporary function for this specific modification
                temp_function = kernel.create_semantic_function(
                    prompt_template=modification_input,
                    function_name="modify_plan_temp",
                    description="Temporary function to modify a plan"
                )
                
                # Invoke the temporary function
                modified_content = await temp_function.invoke()
                modified_content_text = str(modified_content).strip()
            except Exception as e:
                logger.error(f"Error creating temporary function: {str(e)}")
                # Fallback to using plugins["task_breakdown"]
                modified_content = await plugins["task_breakdown"].invoke(
                    kernel=kernel, 
                    input=f"MODIFY PLAN: {modification.instruction}\n\nOriginal objective: {current_objective}\n\nCurrent plan summary: {current_plan_summary}"
                )
                modified_content_text = str(modified_content).strip()
            
            # Try to parse a full plan response, or just update objective
            try:
                # Check if the response looks like a task breakdown
                if "TASK 1:" in modified_content_text or "TASK 1." in modified_content_text:
                    # Try to parse as a set of tasks
                    tasks_text = modified_content_text
                    task_sections = tasks_text.split("TASK ")
                    # Remove any content before the first task
                    task_sections = [section for section in task_sections if section.strip() and (section.startswith("1:") or section.startswith("1."))]
                    
                    new_tasks = []
                    for i, section in enumerate(task_sections):
                        task_num_removed = section.split("\n", 1)
                        if len(task_num_removed) < 2:
                            continue
                            
                        title_line = task_num_removed[0].strip()
                        # Remove the task number prefix (e.g., "1:" or "1.")
                        title = title_line[title_line.find(":") + 1:] if ":" in title_line else title_line[title_line.find(".") + 1:]
                        title = title.strip()
                        
                        content = task_num_removed[1].strip()
                        
                        new_task = {
                            "title": title,
                            "content": content,
                            "completed": False
                        }
                        
                        # Preserve duration if available
                        if i < len(current_tasks) and "duration" in current_tasks[i]:
                            new_task["duration"] = current_tasks[i]["duration"]
                        
                        new_tasks.append(new_task)
                    
                    if new_tasks:
                        update_fields = {
                            "result.tasks": new_tasks,
                            "last_modified": datetime.utcnow()
                        }
                    else:
                        # Fallback to updating the objective
                        update_fields = {
                            "result.objective": modified_content_text,
                            "last_modified": datetime.utcnow()
                        }
                else:
                    # Update the objective
                    update_fields = {
                        "result.objective": modified_content_text,
                        "last_modified": datetime.utcnow()
                    }
            except:
                # Fallback to updating the objective
                update_fields = {
                    "result.objective": modified_content_text,
                    "last_modified": datetime.utcnow()
                }
        
        # Update the task document
        result = queries_collection.update_one(
            {"_id": task_object_id, "user_id": user_id},
            {"$set": update_fields}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_304_NOT_MODIFIED,
                detail="No changes made to the plan"
            )
            
        # Recalculate progress if tasks have changed
        if "result.tasks" in update_fields:
            updated_doc = queries_collection.find_one(
                {"_id": task_object_id, "user_id": user_id}
            )
            
            tasks = updated_doc.get("result", {}).get("tasks", [])
            total_tasks = len(tasks)
            completed_tasks = sum(1 for task in tasks if task.get("completed", False))
            
            # Calculate progress as percentage
            progress = int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
            
            # Update the progress field
            queries_collection.update_one(
                {"_id": task_object_id, "user_id": user_id},
                {"$set": {"progress": progress}}
            )
            
        # Get the updated task
        updated_task = queries_collection.find_one(
            {"_id": task_object_id, "user_id": user_id}
        )
        
        # Convert ObjectId to string for JSON serialization
        updated_task['_id'] = str(updated_task['_id'])
        
        # Add metadata about the modification
        response = {
            "instruction": modification.instruction,
            "modified_task": updated_task,
            "message": "Plan successfully modified"
        }
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error modifying plan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

