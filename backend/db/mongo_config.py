from pymongo import MongoClient, errors
import os
from dotenv import load_dotenv
import certifi
import logging
from typing import Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

def get_database():
    """Simple function to get MongoDB database connection"""
    MONGODB_URI = os.getenv('MONGODB_URI')
    if not MONGODB_URI:
        raise ValueError("MongoDB URI is not set in environment variables")

    try:
        # Create MongoDB client with basic settings
        client = MongoClient(
            MONGODB_URI
        )
        
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client['visual_agent_db']
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

# Initialize database and collections
try:
    db = get_database()
    users_collection = db['users']
    queries_collection = db['queries']

    # Create basic indexes
    users_collection.create_index('email', unique=True)
    queries_collection.create_index('user_id')
    queries_collection.create_index('timestamp')
    
    logger.info("Successfully initialized MongoDB and created indexes")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB: {str(e)}")
    raise

