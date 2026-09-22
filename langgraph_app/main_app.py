from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routes import router
from utils.db_pool import close_db_pool, get_db_pool
from contextlib import asynccontextmanager
from miniopy_async import Minio
from setup_db import start_setup
from models.openai import LLamaCppReranker
import utils.contextmanager_utils as cm
import chromadb, os

MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
LLAMA_CPP_KEY = os.getenv("LLAMA_CPP_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_setup()
    cm.chroma = chromadb.HttpClient(host="chromadb", port=8000)
    cm.minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )
    cm.reranker = LLamaCppReranker(api_key=LLAMA_CPP_KEY)

    await cm.reranker.start()
    app.state.pool = await get_db_pool()
    yield
    await close_db_pool()
    await cm.reranker.close()
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)