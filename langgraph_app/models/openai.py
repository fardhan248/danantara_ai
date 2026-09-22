from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os, httpx

LLAMA_CPP_EMBEDDING_ENDPOINT = os.getenv("LLAMA_CPP_EMBEDDING_ENDPOINT")
LLAMA_CPP_LLM_ENDPOINT = os.getenv("LLAMA_CPP_LLM_ENDPOINT")
LLAMA_CPP_RERANKER_ENDPOINT = os.getenv("LLAMA_CPP_RERANKER_ENDPOINT")
LLAMA_CPP_KEY = os.getenv("LLAMA_CPP_KEY")

llm = ChatOpenAI(
    base_url=f"http://{LLAMA_CPP_LLM_ENDPOINT}/v1",
    api_key=LLAMA_CPP_KEY,
    model="/models/Qwen3.5-4B-Q4_K_M.gguf",
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)

llm_thinking = ChatOpenAI(
    base_url=f"http://{LLAMA_CPP_LLM_ENDPOINT}/v1",
    api_key=LLAMA_CPP_KEY,
    model="/models/Qwen3.5-4B-Q4_K_M.gguf",
)

embedding = OpenAIEmbeddings(
    base_url=f"http://{LLAMA_CPP_EMBEDDING_ENDPOINT}/v1", 
    api_key=LLAMA_CPP_KEY,
    model="/models/Qwen3-Embedding-4B-Q4_K_M.gguf" 
)

class LLamaCppReranker:
    def __init__(self, api_key: str, base_url: str = LLAMA_CPP_RERANKER_ENDPOINT, model: str = "/models/Qwen3-Reranker-0.6B-Q4_K_M.gguf"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.client = None

    async def start(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def rerank(self, query: str, documents: list[str], top_n: int = 5):
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.base_url}/v1/rerank",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                },
            )

        response.raise_for_status()

        ids = [r["index"] for r in response.json()["results"]]

        return ids