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

class State(TypedDict):
    thread_id: str 
    # streaming_mode: bool = False
    bm25: bool
    rerank: bool
    enhanced: bool

    messages: Annotated[list[BaseMessage], add_messages] = [] # list of AnyMessage, Human, AI, Tool, System
    selected_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"knowledge_id": knowledge_id, "chunk_ids": [id_1, id_2]}]
    chunk_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"chunk_id": chunk_id, "content": content, "metadata": metadata}]
    selected_table: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"table_id": str}]
    tables: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"table": str, "metadata": dict}]
    selected_image_in_table: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"img_path": str, "image_id": str}]
    image_in_table: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"description": str, "metadata": dict}]
    selected_image_out_table: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"img_path": str, "image_id": str}]
    image_out_table: Annotated[list[dict[str, Any]], items_reducer] = [] # [{"description": str, "metadata": dict}]

    query: str
    tool_loop: int = 0
    final_answer: dict[str, Any]

class LLMOutput(BaseModel):
    answer: str
    sources: Union[list[str], Literal["N/A"]] = "N/A"

class LLMRAG(BaseModel):
    question: str
