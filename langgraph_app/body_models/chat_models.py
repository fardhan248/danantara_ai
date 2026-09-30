from pydantic import BaseModel, Field
from typing_extensions import Union, Literal
    
class ChatInput(BaseModel):
    input_prompt: str
    thread_id: str | None = None
    ticker: str | None = None

class SummaryResumeInput(BaseModel):
    thread_id: str
    start_date: str
    end_date: str
    approved: bool
    ticker: str | None = None

class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str