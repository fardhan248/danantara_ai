from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os

GEMINI_API = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API,
    thinking_budget=0,
)

llm_thinking = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API,
    thinking_level="medium",
)

llm_high_thinking = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API,
    thinking_level="high",
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=GEMINI_API,
)
