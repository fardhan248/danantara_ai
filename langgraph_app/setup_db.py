from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os, asyncio

DB_URL = os.getenv("DATABASE_URL")

async def start_setup():
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
        await checkpointer.setup()

if __name__ == "__main__":
    asyncio.run(start_setup())