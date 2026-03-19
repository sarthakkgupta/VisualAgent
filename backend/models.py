from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class TaskUpdate(BaseModel):
    objective: Optional[str] = None
    tasks: Optional[List[Dict[str, Any]]] = None


class TaskStatus(BaseModel):
    task_index: int
    completed: bool


class SubtaskStatus(BaseModel):
    task_index: int
    subtask_index: int
    completed: bool


class TaskBulkStatus(BaseModel):
    task_statuses: List[TaskStatus]


class TaskDetailUpdate(BaseModel):
    task_index: int
    title: Optional[str] = None
    content: Optional[str] = None
    duration: Optional[str] = None


class PlanModificationRequest(BaseModel):
    modification_text: str
    task_id: str
    user_id: str
