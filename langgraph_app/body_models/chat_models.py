from pydantic import BaseModel, Field
from typing_extensions import Union, Literal
    
class ChatInput(BaseModel):
    input_prompt: str
    thread_id: str | None = None

class SummaryResumeInput(BaseModel):
    thread_id: str
    approved: bool

class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str