from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import copy
from typing_extensions import TypedDict, Annotated, Any, Union, Literal
from pydantic import BaseModel, Field

# State
def items_reducer(current: list, new: dict | list):
    if current is None:
        current = []
        
    result = copy.deepcopy(current)
    
    # Fan-in 
    if isinstance(new, list):
        if any(isinstance(i, dict) and any(k in i for k in ("append", "replace", "remove")) for i in new):
            for update in new:
                result = items_reducer(result, update)
            return result
        else:
            new = [{"append": new}]
    
    # Remove element
    for item in new.get("remove", []):
        if item in result:
            result.remove(item)
    
    # Append element
    for item in new.get("append", []):
        if item not in result:
            result.append(item)
            
    # Replace element (especially for selected knowledge_id/s_knowledge_id)
    for item in new.get("replace", []):
        if isinstance(item, dict):
            _id = list(item.keys())[0] # knowledge_id
            for i, existing in enumerate(result):
                if _id in existing:
                    result[i] = item
                    break
        else:
            for i, existing in enumerate(result):
                result[i] = item
        
    return result    

class MainState(TypedDict):
    thread_id: str
    routing: str
    ticker: str

    query: str

class ChatbotState(TypedDict):
    thread_id: str 
    routing: str # chat, summary, report
    ticker: str

    messages: Annotated[list[BaseMessage], add_messages] = [] # list of AnyMessage, Human, AI, Tool, System
    knowledge_path: Annotated[list[str], items_reducer] = [] # list of str: ["path_cache1", "path_cache2"]
    table_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]
    image_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]
    finance_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]
    price_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]

    query: str
    tool_loop: int = 0
    final_answer: dict[str, Any]

class SummaryState(TypedDict):
    thread_id: str
    ticker: str

    documents_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]
    finance_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]
    price_path: Annotated[list[str], items_reducer] = [] # ["path_cache1", "path_cache2"]

class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str
