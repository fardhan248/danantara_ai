from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import router
from utils.db_pool import close_db_pool, get_db_pool
from contextlib import asynccontextmanager
from miniopy_async import Minio
from setup_db import start_setup
import utils.contextmanager_utils as cm
import chromadb, os

LLAMA_CPP_KEY = os.getenv("LLAMA_CPP_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_setup()
    cm.chroma = chromadb.HttpClient(host="chromadb", port=8000)

    app.state.pool = await get_db_pool()
    yield
    await close_db_pool()
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)