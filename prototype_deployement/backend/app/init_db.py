import asyncio
from app.core.db import engine, Base
from app import db_models  # noqa: F401 - registers SQLAlchemy models with Base metadata

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized.")

if __name__ == "__main__":
    asyncio.run(init())
