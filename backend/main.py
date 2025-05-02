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

