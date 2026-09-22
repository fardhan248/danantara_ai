import uuid, asyncio, re
from models.openai import llm, embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime
from zoneinfo import ZoneInfo
import utils.contextmanager_utils as cm
from utils.extract_text_utils import ExtractPDF
from unstructured.partition.text import partition_text
from io import BytesIO
from rank_bm25 import BM25Okapi

async def get_vector_store_chroma(collection: str):
    return Chroma(
        client=cm.chroma,
        collection_name=collection,
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
async def get_vector_store_retriever(chroma_vector_store, search_filter: dict = None, k: int = 5):
    search_kwargs = {"k": k}
    
    if search_filter:
        search_kwargs["filter"] = search_filter
    
    return chroma_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
     
async def str_to_datetime(date: str):
    try:
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    return dt

 
async def chunk_document(filename, content_type, file_bytes):
    if content_type not in ("application/pdf", "application/epub+zip", "text/plain"):
        if file_bytes[:4] == b"%PDF":
            content_type = "application/pdf"
        elif file_bytes[:2] == b"PK":
            content_type = "application/epub+zip"
        else:
            content_type = "text/plain"

    knowledge_id = str(uuid.uuid4())

    print("Extract knowledge...", flush=True)
    if content_type == "application/pdf": # or content_type == "application/epub+zip":
        extract = ExtractPDF(filebytes=BytesIO(file_bytes), filetype=content_type, client=llm, knowledge_id=knowledge_id)
        await extract.start()

        chunks = [chunk.text for chunk in extract.chunks]
        metadatas = [chunk.metadata.to_dict() for chunk in extract.chunks]
        ids = [meta["chunk_id"] for meta in metadatas]
        
    else: # txt
        extract = partition_text(
            file=BytesIO(file_bytes),
            chunking_strategy="by_title",
            languages=["ind", "eng"],
        )

        chunks = [element.text for element in extract]
        metadatas = [element.metadata.to_dict() for element in extract]
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

    metadatas = [
        {
            "filename": filename,
            "content_type": content_type,
            "len_pages": extract.len_doc,
            "chunk_number": i + 1,
            "len_chunks": len(chunks),
            "len_char": len(chunk),
            "knowledge_id": knowledge_id,
            "chunk_id": ids[i],
            "created_at": str(datetime.now(ZoneInfo("Asia/Jakarta"))),
            "page_numbers": metadata.get("page_numbers", [1]),
            "is_continuation": metadata.get("is_continuation", False),
            "image_out_table": metadata.get("image_out_table") or None,
            "image_in_table": metadata.get("image_in_table") or None,
            "table": metadata.get("table") or None,
            "type": "text",
        }
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas))
    ]

    if content_type == "application/pdf":
        for table in extract.tables:
            tab = table["table"]
            metadata = {
                "filename": filename,
                "content_type": content_type,
                "len_char": len(tab),
                "knowledge_id": knowledge_id,
                "chunk_id": table.get("chunk_id", ""),
                "table_id": table["table_id"],
                "created_at": str(datetime.now(ZoneInfo("Asia/Jakarta"))),
                "page_numbers": table.get("page_numbers", [1]),
                "image_ids": table.get("image_ids") or None,
                "type": "table",
                "table": tab,
            }

            chunks.append(table["description"])
            metadatas.append(metadata)
            ids.append(table["table_id"])

        for image in extract.images_out_table:
            img_desc = image["description"]
            metadata = {
                "filename": filename,
                "content_type": content_type,
                "knowledge_id": knowledge_id,
                "chunk_id": image.get("chunk_id", ""),
                "image_id": image["image_id"],
                "created_at": str(datetime.now(ZoneInfo("Asia/Jakarta"))),
                "page_number": image.get("page_number", 1),
                "img_path": image["img_path"],
                "type": "image_out_tab",
            }

            chunks.append(img_desc)
            metadatas.append(metadata)
            ids.append(image["image_id"])

        for image in extract.images_in_table:
            img_desc = image["description"]
            metadata = {
                "filename": filename,
                "content_type": content_type,
                "knowledge_id": knowledge_id,
                "chunk_id": image.get("chunk_id", ""),
                "image_id": image["image_id"],
                "created_at": str(datetime.now(ZoneInfo("Asia/Jakarta"))),
                "page_number": image.get("page_number", 1),
                "img_path": image["img_path"],
                "table_id": image["table_id"],
                "row": image["row"],
                "column": image["column"],
                "type": "image_in_tab",
            }

            chunks.append(img_desc)
            metadatas.append(metadata)
            ids.append(image["image_id"])
        
    return chunks, metadatas, ids

## Tool: Put new knowledge (by tenant admin)
async def save_chunks_to_db(chunks, metadatas, ids):
    knowledge_id = metadatas[0]["knowledge_id"]
    
    documents = [
        Document(
            page_content=chunks[i],
            metadata=metadatas[i],
            id=ids[i],
        )
        for i in range(len(chunks))
    ]

    try:
        vector_store = await get_vector_store_chroma(f"knowledges")
        
        await vector_store.aadd_documents(documents=documents, ids=ids)

    except Exception as e:
        print(e)
        return {"status": "error", "knowledge_id": 0, "content": str(e)}
        
    return {"status": "success", "knowledge_id": knowledge_id, "chunk_ids": ids, "metadata": metadatas}
    
async def put_new_knowledge(f):    
    filename = f.filename
    content_type = f.content_type
    file_bytes = await f.read()
    
    try:
        # Chunking document
        chunks, metadatas, ids = await chunk_document(filename, content_type, file_bytes)
        
        result = await save_chunks_to_db(chunks, metadatas, ids)
        print("Done extract knowledge", flush=True)
        return result
    
    except Exception as e:
        print(e, flush=True)
        return {"status": "error", "knowledge_id": 0, "content": str(e)}


# BM25
class BM25Retriever:
    def __init__(self):
        self.documents = None
        self.metadatas = None
        self.bm25 = None

    @staticmethod
    async def bm_tokenize(text: str):
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower().split()

    async def _get_bm25(self, results_embedding: list[Document]):
        chunk_ids = []
        table_ids = []
        for result in results_embedding:
            metadata = result.metadata
            if metadata["type"] == "text":
                chunk_ids.append(metadata["chunk_id"])
            elif metadata["type"] == "table":
                table_ids.append(metadata["table_id"])

        conditions = []
        if chunk_ids:
            conditions.append({
                "$and": [
                    {"type": {"$eq": "text"}},
                    {"chunk_id": {"$nin": chunk_ids}},
                ]
            })

        if table_ids:
            conditions.append({
                "$and": [
                    {"type": {"$eq": "table"}},
                    {"table_id": {"$nin": table_ids}},
                ]
            })

        if len(conditions) == 2:
            conditions = {"$or": conditions}
        elif len(conditions) == 1:
            conditions = conditions[0]
        else:
            conditions = None

        vector_store = await get_vector_store_chroma("knowledges")
        collection = vector_store._collection
        results = collection.get(where=conditions, include=["documents", "metadatas"])

        docs, metas = [], []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if meta["type"] == "table":
                docs.append(meta["table"])
                metas.append({key: val for key, val in meta.items()})
            else:
                docs.append(doc)
                metas.append(meta)

        self.documents = docs
        self.metadatas = metas

        corpus = [
            await self.bm_tokenize(doc)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(corpus)

    async def start(self, results: list[Document]):
        await self._get_bm25(results)

    async def retrieve(self, query: str, k: int = 5):
        query_tokens = await self.bm_tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_ids = scores.argsort()[::-1][:k]

        return [
            {"metadata": self.metadatas[i], "page_content": self.documents[i]} 
            for i in top_ids
        ]


# Delete
async def delete_knowledge(knowledge_id: str):
    try:
        vector_store = await get_vector_store_chroma(f"knowledges")
        
        collection = vector_store._collection
        collection_name = collection.name

        # Check if there is a knowledge or not
        result = collection.get(where={"knowledge_id": knowledge_id})
        if len(result["ids"]) == 0:
            return {"status": "success", "content": f"There is no knowledge_id {knowledge_id} in knowledges."}

        # Delete knowledge
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.delete(where={"knowledge_id": knowledge_id})
        )
        
        # If the knowledge is empty, delete the collection
        knowledge_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.get(include=[])
        )

        if len(knowledge_ids["ids"]) == 0:
            client = vector_store._client
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
        
        return {"status": "success", "content": f"Delete knowledge_id {knowledge_id} success."}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e)}

# List documents
async def list_doc():
    try:
        vector_store = await get_vector_store_chroma(f"knowledges")
        collection = vector_store._collection

        raw_data = collection.get(
            include=["metadatas"]
        )

        unique_files = {}
        for meta in raw_data["metadatas"]:
            if not meta:
                continue

            knowledge_id = meta.get("knowledge_id")

            if knowledge_id not in unique_files:
                unique_files[knowledge_id] = {
                    "filename": meta.get("filename"),
                    "content_type": meta.get("content_type"),
                    "len_pages": meta.get("len_pages"),
                    "len_chunks": meta.get("len_chunks"),
                    "knowledge_id": knowledge_id,
                    "created_at": meta.get("created_at"),
                }

        unique_files = list(unique_files.values())

        return {"status": "success", "content": unique_files}

    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e)}
    