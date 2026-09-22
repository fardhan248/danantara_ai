import utils.contextmanager_utils as cm
import httpx, os

LLAMA_CPP_EMBEDDING_ENDPOINT = os.getenv("LLAMA_CPP_EMBEDDING_ENDPOINT")
LLAMA_CPP_LLM_ENDPOINT = os.getenv("LLAMA_CPP_LLM_ENDPOINT")
LLAMA_CPP_RERANKER_ENDPOINT = os.getenv("LLAMA_CPP_RERANKER_ENDPOINT")
LLAMA_CPP_KEY = os.getenv("LLAMA_CPP_KEY")

pool = None

async def llm_check(results):
    # llama_cpp_llm (qwen)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{LLAMA_CPP_LLM_ENDPOINT}/health", headers={"Authorization": f"Bearer {LLAMA_CPP_KEY}"})
            
            if response.status_code == 200:
                results["llama_cpp_llm"] = {"status": "success", "content": response.json()}
            else:
                results["llama_cpp_llm"] = {"status": "error", "content": response.json()}
    
    except Exception as e:
        results["llama_cpp_llm"] = {"status": "error", "content": str(e)}
        
    # llama_cpp_embedding (qwen)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{LLAMA_CPP_EMBEDDING_ENDPOINT}/health", headers={"Authorization": f"Bearer {LLAMA_CPP_KEY}"})
            
            if response.status_code == 200:
                results["llama_cpp_embedding"] = {"status": "success", "content": response.json()}
            else:
                results["llama_cpp_embedding"] = {"status": "error", "content": response.json()}
    
    except Exception as e:
        results["llama_cpp_embedding"] = {"status": "error", "content": str(e)}

    # llama_cpp_rerank (qwen)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{LLAMA_CPP_RERANKER_ENDPOINT}/health", headers={"Authorization": f"Bearer {LLAMA_CPP_KEY}"})
                
                if response.status_code == 200:
                    results["llama_cpp_rerank"] = {"status": "success", "content": response.json()}
                else:
                    results["llama_cpp_rerank"] = {"status": "error", "content": response.json()}
        
        except Exception as e:
            results["llama_cpp_rerank"] = {"status": "error", "content": str(e)}

    return results
    
async def db_check(results):
    # asyncpg pool
    try:
        async with pool.acquire() as conn:
            response = await conn.fetchval("SELECT 1")
            results["asyncpg_pool"] = {"status": "success", "content": response == 1}
    
    except Exception as e:
        results["asyncpg_pool"] = {"status": "error", "content": str(e)}

    # MinIO
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://minio:9000/minio/health/live"
            )

            if response.status_code == 200:
                results["minio"] = {"status": "success", "content": "ok"}
            else:
                results["minio"] = {"status": "success", "content": "failed"}
    
    except Exception as e:
        results["minio"] = {"status": "error", "content": str(e)}
    
    # ChromaDB
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://chromadb:8000/api/v2/heartbeat"
            )
            
            results["chromadb"] = {"status": "success", "content": response.text}
    
    except Exception as e:
        results["chromadb"] = {"status": "error", "content": str(e)}
        
    # Example of chroma db data
    try:
        collection_names = cm.chroma.list_collections()
        
        collection_info = []
        for collection in collection_names:
            name = collection.name
            count = collection.count()
            metadata = collection.metadata
            collection_id = collection.id
            
            response = collection.get(include=["metadatas"])
            metadatas = response["metadatas"]
            knowledge_ids = []
            if len(metadatas) > 0:
                for meta in metadatas:
                    try:
                        if meta["knowledge_id"] not in knowledge_ids:
                            knowledge_ids.append(meta["knowledge_id"])
                    except Exception:
                        if meta["memory_id"] not in knowledge_ids:
                            knowledge_ids.append(meta["memory_id"])
            
            collection_info.append({"collection_name": name, "collection_id": collection_id, "collection_count": count, "collection_metadata": metadata, "knowledge_ids": knowledge_ids})
            
        results["chromadb_example"] = {"status": "success", "content": collection_info}

    except Exception as e:
        results["chromadb_example"] = {"status": "error", "content": str(e)}
        
    return results
    
async def health_check(db_pool):
    global pool
    pool = db_pool
    
    results = {}
    
    # LLM
    results = await llm_check(results)  
        
    # Database
    results = await db_check(results)

    return results