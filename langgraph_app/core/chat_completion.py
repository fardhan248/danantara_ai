from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

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

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "input_prompt": input_prompt,
        }
    }
    
    try:
        # Gambar/video (opsional) disimpan di redis, state hanya menyimpan key-nya
        image_path = await lang_core.save_media_inputs(thread_id, input_data.images, "image")
        video_path = await lang_core.save_media_inputs(thread_id, input_data.videos, "video")

        async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
            agent = builder.compile(checkpointer=checkpointer)

            result_agent = await agent.ainvoke(
                {
                    "thread_id": str(thread_id),
                    "messages": [HumanMessage(content=input_prompt)],
                    "query": input_prompt,
                    "routing": "chatbot",
                    "ticker": input_data.ticker,
                    "sector": input_data.sector,
                    "image_path": image_path,
                    "video_path": video_path,
                },
                config,
            )

            content = result_agent["final_answer"]

            return {"thread_id": str(thread_id), "content": content} 
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "content": str(e)}

async def summary_workflow(db_pool, input_data, resume: bool = False):
    global pool
    pool = db_pool
    lang_core.pool = pool

    builder = await get_agent()
    thread_id = input_data.thread_id
    start_date = input_data.start_date
    end_date = input_data.end_date

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
            agent = builder.compile(checkpointer=checkpointer)

            if resume:
                await agent.ainvoke(
                    Command(resume={"approved": input_data.approved}),
                    config,
                )
            else:
                await agent.ainvoke(
                    {
                        "thread_id": str(thread_id),
                        "start_date": start_date,
                        "end_date": end_date,
                        "routing": "summary",
                        "ticker": input_data.ticker,
                        "sector": input_data.sector,
                    },
                    config,
                )

            state = await agent.aget_state(config)
            if state.next:
                interrupt_data = state.tasks[0].interrupts[0].value
                return {
                    "status": "waiting_for_approval",
                    "thread_id": thread_id,
                    "content": interrupt_data,
                }

            return {
                "status": "done",
                "thread_id": thread_id,
                "summary": state.values.get("summary"),
            }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "content": str(e)}