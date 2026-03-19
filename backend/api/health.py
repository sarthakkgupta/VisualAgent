import logging

from fastapi import APIRouter, HTTPException, status

from db.mongo_config import queries_collection

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/health")
async def health_check():
    try:
        queries_collection.find_one({})
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy")


@router.get("/api/history")
async def get_history(user_id: str):
    try:
        history = list(queries_collection.find({"user_id": user_id}).sort("timestamp", -1))
        for doc in history:
            doc["_id"] = str(doc["_id"])
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
