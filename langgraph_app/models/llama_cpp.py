from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os, httpx

LLAMA_CPP_EMBEDDING_ENDPOINT = os.getenv("LLAMA_CPP_EMBEDDING_ENDPOINT")
LLAMA_CPP_LLM_ENDPOINT = os.getenv("LLAMA_CPP_LLM_ENDPOINT")
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
