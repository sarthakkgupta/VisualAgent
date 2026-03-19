import logging

from bson import ObjectId
from fastapi import APIRouter, Body, HTTPException, status

from db.mongo_config import queries_collection
from models import TaskBulkStatus, TaskDetailUpdate, TaskStatus, TaskUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


def _parse_object_id(task_id: str) -> ObjectId:
    try:
        return ObjectId(task_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task ID format")


def _get_or_404(oid: ObjectId, user_id: str) -> dict:
    doc = queries_collection.find_one({"_id": oid, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return doc


def _serialize(doc: dict) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc


def _update_progress(oid: ObjectId, user_id: str) -> None:
    doc = queries_collection.find_one({"_id": oid, "user_id": user_id})
    tasks = doc.get("result", {}).get("tasks", [])
    total = len(tasks)
    completed = sum(1 for t in tasks if t.get("completed", False))
    progress = int((completed / total) * 100) if total else 0
    queries_collection.update_one({"_id": oid, "user_id": user_id}, {"$set": {"progress": progress}})


@router.get("/api/task/{task_id}")
async def get_task(task_id: str, user_id: str):
    try:
        return _serialize(_get_or_404(_parse_object_id(task_id), user_id))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/api/task/{task_id}")
async def update_task(task_id: str, user_id: str, update_data: TaskUpdate = Body(...)):
    try:
        oid = _parse_object_id(task_id)
        _get_or_404(oid, user_id)

        fields = {}
        if update_data.objective is not None:
            fields["result.objective"] = update_data.objective
        if update_data.tasks is not None:
            fields["result.tasks"] = update_data.tasks
        if not fields:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided")

        result = queries_collection.update_one({"_id": oid, "user_id": user_id}, {"$set": fields})
        if result.modified_count == 0:
            raise HTTPException(status_code=status.HTTP_304_NOT_MODIFIED, detail="No changes made")

        return _serialize(queries_collection.find_one({"_id": oid, "user_id": user_id}))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/api/task/{task_id}")
async def delete_task(task_id: str, user_id: str):
    try:
        oid = _parse_object_id(task_id)
        _get_or_404(oid, user_id)

        result = queries_collection.delete_one({"_id": oid, "user_id": user_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=status.HTTP_304_NOT_MODIFIED, detail="Task could not be deleted")

        return {"message": "Task successfully deleted", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/api/task/{task_id}/detail")
async def update_task_detail(task_id: str, user_id: str, update_data: TaskDetailUpdate = Body(...)):
    try:
        oid = _parse_object_id(task_id)
        doc = _get_or_404(oid, user_id)
        tasks = doc.get("result", {}).get("tasks", [])

        if update_data.task_index < 0 or update_data.task_index >= len(tasks):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task index")

        idx = update_data.task_index
        fields = {}
        if update_data.title is not None:
            fields[f"result.tasks.{idx}.title"] = update_data.title
        if update_data.content is not None:
            fields[f"result.tasks.{idx}.content"] = update_data.content
        if update_data.duration is not None:
            fields[f"result.tasks.{idx}.duration"] = update_data.duration
        if not fields:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided")

        result = queries_collection.update_one({"_id": oid, "user_id": user_id}, {"$set": fields})
        if result.modified_count == 0:
            raise HTTPException(status_code=status.HTTP_304_NOT_MODIFIED, detail="No changes made")

        return _serialize(queries_collection.find_one({"_id": oid, "user_id": user_id}))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task detail: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/api/task/{task_id}/completion")
async def update_task_completion(task_id: str, user_id: str, status_update: TaskStatus = Body(...)):
    try:
        oid = _parse_object_id(task_id)
        doc = _get_or_404(oid, user_id)
        tasks = doc.get("result", {}).get("tasks", [])

        if status_update.task_index < 0 or status_update.task_index >= len(tasks):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task index")

        queries_collection.update_one(
            {"_id": oid, "user_id": user_id},
            {"$set": {f"result.tasks.{status_update.task_index}.completed": status_update.completed}},
        )
        _update_progress(oid, user_id)
        return _serialize(queries_collection.find_one({"_id": oid, "user_id": user_id}))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating completion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/api/task/{task_id}/bulk-completion")
async def update_bulk_completion(task_id: str, user_id: str, status_updates: TaskBulkStatus = Body(...)):
    try:
        oid = _parse_object_id(task_id)
        doc = _get_or_404(oid, user_id)
        total_tasks = len(doc.get("result", {}).get("tasks", []))

        fields = {}
        for s in status_updates.task_statuses:
            if s.task_index < 0 or s.task_index >= total_tasks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task index: {s.task_index}",
                )
            fields[f"result.tasks.{s.task_index}.completed"] = s.completed

        if not fields:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update data provided")

        result = queries_collection.update_one({"_id": oid, "user_id": user_id}, {"$set": fields})
        if result.modified_count == 0:
            raise HTTPException(status_code=status.HTTP_304_NOT_MODIFIED, detail="No changes made")

        _update_progress(oid, user_id)
        return _serialize(queries_collection.find_one({"_id": oid, "user_id": user_id}))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating bulk completion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/api/task/{task_id}/progress")
async def get_task_progress(task_id: str, user_id: str):
    try:
        oid = _parse_object_id(task_id)
        doc = _get_or_404(oid, user_id)
        tasks = doc.get("result", {}).get("tasks", [])
        total = len(tasks)
        completed = sum(1 for t in tasks if t.get("completed", False))
        progress = int((completed / total) * 100) if total else 0

        return {
            "task_id": task_id,
            "total_tasks": total,
            "completed_tasks": completed,
            "progress": doc.get("progress", progress),
            "completed_status": [t.get("completed", False) for t in tasks],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
