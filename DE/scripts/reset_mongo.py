"""Reset MongoDB raw_audio_jobs to allow re-processing with new code."""
import asyncio
import sys
sys.path.insert(0, "/opt/airflow")
from src.shared.infrastructure.mongo.client import get_mongo_db

async def reset():
    db = get_mongo_db()
    col = db["raw_audio_jobs"]

    # Delete all done jobs so pipeline re-processes them
    result = await col.delete_many({"status": {"$in": ["done", "failed"]}})
    print(f"Deleted {result.deleted_count} done/failed MongoDB jobs")

    # Reset processing → pending_processing
    result2 = await col.update_many(
        {"status": "processing"},
        {"$set": {"status": "pending_processing"}}
    )
    print(f"Reset {result2.modified_count} processing → pending_processing")

    remaining = await col.count_documents({})
    print(f"Remaining MongoDB jobs: {remaining}")

asyncio.run(reset())
