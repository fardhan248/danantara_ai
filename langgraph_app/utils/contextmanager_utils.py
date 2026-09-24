from chromadb.api import ClientAPI
from miniopy_async import Minio

minio: Minio | None = None
chroma: ClientAPI | None = None
