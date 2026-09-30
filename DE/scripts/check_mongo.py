"""Quick check: MongoDB raw_audio_jobs status."""
import asyncio
import sys
sys.path.insert(0, "/opt/airflow")

from src.shared.infrastructure.mongo.client import get_mongo_db

async def main():
    db = get_mongo_db()
    col = db["raw_audio_jobs"]
    total = await col.count_documents({})
    print(f"Total raw_audio_jobs: {total}")

    pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    async for g in col.aggregate(pipeline):
        print(f"  {g['_id']}: {g['count']}")

    # Sample a few docs
    print("\n--- Sample docs (last 3) ---")
    async for doc in col.find().sort("created_at", -1).limit(3):
        print(f"  id={doc.get('_id')} status={doc.get('status')} source_type={doc.get('source_type')} url={str(doc.get('source_url',''))[:80]}")

asyncio.run(main())
