from pydantic import BaseModel, Field
from typing_extensions import Union, Literal
    
class ChatInput(BaseModel):
    input_prompt: str
    thread_id: str | None = None
    bm25: bool = False
    rerank: bool = False
    enhanced: bool = False

class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str