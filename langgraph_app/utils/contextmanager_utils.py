from chromadb.api import ClientAPI
from miniopy_async import Minio
from models.openai import LLamaCppReranker

minio: Minio | None = None
chroma: ClientAPI | None = None
reranker: LLamaCppReranker | None = None