from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

import os, logging, traceback
from core.langgraph_core import get_agent
import core.langgraph_core as lang_core
from body_models.chat_models import ChatInput

DB_URL = os.getenv("DATABASE_URL")
pool = None 

logger = logging.getLogger(__name__)

async def chat_workflow(db_pool, input_data: ChatInput):
    global pool
    pool = db_pool
    lang_core.pool = pool
    
    builder = await get_agent()
    
    thread_id = input_data.thread_id
    input_prompt = input_data.input_prompt
    bm25 = input_data.bm25
    rerank = input_data.rerank
    enhanced = input_data.enhanced
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
            agent = builder.compile(checkpointer=checkpointer)

            result_agent = await agent.ainvoke(
                {
                    "thread_id": str(thread_id),
                    "messages": [HumanMessage(content=input_prompt)],
                    "bm25": bm25,
                    "rerank": rerank,
                    "enhanced": enhanced,
                },
                config,
            )

            content = result_agent["final_answer"]

            return {"thread_id": str(thread_id), "content": content} 
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "content": ""}