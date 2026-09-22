from pydantic import BaseModel
    
class ChatInput(BaseModel):
    input_prompt: str
    thread_id: str | None = None
    bm25: bool = False
    rerank: bool = False
    enhanced: bool = False
