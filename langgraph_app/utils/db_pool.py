import asyncpg, asyncio
import os
from fastapi import HTTPException

database_url = os.getenv("DATABASE_URL")

_db_pool = None
_db_pool_lock = asyncio.Lock()

async def init_db_pool():
    try: 
        global _db_pool
        async with _db_pool_lock:
            if _db_pool is not None:
                return
              
            _db_pool = await asyncpg.create_pool(
                dsn=database_url,
                min_size=2,
                max_size=10,
                max_inactive_connection_lifetime=30,
                max_cached_statement_lifetime=0,
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in init_db_pool: {e}")
        
async def get_db_pool():
    if _db_pool is None:
        await init_db_pool()
    return _db_pool
    
async def close_db_pool():
    global _db_pool
    if _db_pool:
        await _db_pool.close()
