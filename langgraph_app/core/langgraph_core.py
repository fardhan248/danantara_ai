from langgraph_app.models.gemini import llm, llm_thinking
from transformers import AutoTokenizer

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.messages.utils import trim_messages
from langgraph.types import Command, interrupt
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.documents import Document

from typing_extensions import Annotated
import copy, traceback, json, base64, pickle, os, asyncio
import utils.contextmanager_utils as cm
from utils.documents_utils import get_vector_store_chroma, get_vector_store_retriever, BM25Retriever
from core.states import State, LLMOutput, LLMRAG, SummaryState
from string_utils.prompts import Prompts
from typing import Union, List
from typing_extensions import Any
from redis.asyncio import redis
from langchain_mcp_adapters.client import MultiServerMCPClient

# Utilities
prompts = Prompts()
pool = None

## MCP
sectors_client = MultiServerMCPClient({
    "sectors": {
        "transport": "streamable_http",
        "url": "https://sectors-mcp.supertype.ai/mcp",
        "headers": {"Authorization": f"Bearer {os.getenv('SECTORS_API_KEY')}"},
    },
})

ALLOWED_TOOLS = {
    "fetch-corporate-actions",
    "fetch-foreign-flow",
    "fetch-news",
    "fetch-fillings",
    "fetch-suspensions",
    "fetch-broker-summary-top",
}

_tools_cache = None

async def get_filtered_tools():
    all_tools = await sectors_client.get_tools()
    filtered_tools = [tool for tool in all_tools if tool.name in ALLOWED_TOOLS]
    return filtered_tools

async def get_tools_cache():
    global _tools_cache
    if _tools_cache is None:
        _tools_cache = await get_filtered_tools()
    return _tools_cache

## Redis
r = redis.Redis(host="redis", port=6379, db=0)

async def save_to_temp(key: str, data) -> str:
    await r.set(key, pickle.dumps(data), ex=3600)  # expires in 1 hour
    return key

async def load_from_temp(key: str):
    data = await r.get(key)
    if data is not None:
        return pickle.loads(data)
    return None

## Trimming messages
async def trimming_message(messages):
    messages = trim_messages(
        messages,
        strategy="last",
        token_counter=llm,
        max_tokens=10200,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    return messages 

llm_output = llm.with_structured_output(
    schema=LLMOutput.model_json_schema(), method="json_schema"
)

llm_rag = llm.with_structured_output(
    schema=LLMRAG.model_json_schema(), method="json_schema"
)

# Tools
llm_thinking_tools = None
tool_node = None

## Define Tools node
async def get_tools_list():
    global llm_thinking_tools, tool_node
    tools = await get_tools_cache()

    llm_thinking_tools = llm_thinking.bind_tools(tools)
    tool_node = ToolNode(tools)
    return tools

asyncio.run(get_tools_list())
    
async def should_continue(state: State):
    print("Should continue?", flush=True)
    messages = state["messages"]
    
    tool_calls = getattr(messages[-1], "tool_calls", [])
    
    if len(tool_calls) == 0:
        print("basic_conclusion", flush=True)
        return "basic_conclusion"

    if state["tool_loop"] > 3:
        print("basic_conclusion", flush=True)
        return "basic_conclusion"
        
    return "tools"

# Agents
async def routing_where(state: State):
    route = state["routing"]

    if route == "chatbot":
        return "chatbot_agent"
    else: # route == "summary"
        return "summary_agent"

# ===== CHATBOT =====

## RAG (retrieve data from database based on just new query)
async def rag(state: State):
    print("Node: rag", flush=True)

## Agent: Basic 
async def basic(state: State):
    print("Node: basic", flush=True)
    
    results = await get_contexts_from_current_state(
        state.get("chunk_knowledge", []),
        state.get("tables", []),
        state.get("image_in_table", []),
        state.get("image_out_table", [])
    )

    reformat_chunk_knowledge, reformat_tables, reformat_image_in_table, reformat_image_out_table = results

    system_query = prompts.BASIC_SYSTEM_QUERY.format_map({
        "knowledges": reformat_chunk_knowledge,
        "images_in_table_descriptions": reformat_image_in_table,
        "images_out_table_descriptions": reformat_image_out_table,
        "tables": reformat_tables,
    })   

    messages = state["messages"]

    final_query = [
        SystemMessage(content=system_query),
        *messages,
        HumanMessage(content=f"User's query: {state['query']}"),
    ]
    print("token system basic:", count_tokens([SystemMessage(content=system_query)]), flush=True)

    final_query = await trimming_message(final_query)
    
    response = await llm_thinking_tools.ainvoke(final_query)

    print("Berhasil lewat basic", flush=True)
    return {"messages": [response], "tool_loop": state.get("tool_loop", 0) + 1}

async def basic_conclusion(state: State):
    print("Node: basic_conclusion", flush=True)

    results = await get_contexts_from_current_state(
        state.get("chunk_knowledge", []),
        state.get("tables", []),
        state.get("image_in_table", []),
        state.get("image_out_table", [])
    )
    reformat_chunk_knowledge, reformat_tables, reformat_image_in_table, reformat_image_out_table = results
    
    system_query = prompts.BASIC_CONCLUSION_SYSTEM_QUERY.format_map({
        "knowledges": reformat_chunk_knowledge,
        "images_in_table_descriptions": reformat_image_in_table,
        "images_out_table_descriptions": reformat_image_out_table,
        "tables": reformat_tables,
    })   

    messages = state["messages"]
    
    final_query = [
        SystemMessage(content=system_query),
        *messages,
        HumanMessage(content=f"User's query: {state['query']}"),
    ]
    print("token system (conclusion):", count_tokens([SystemMessage(content=system_query)]), flush=True)

    final_query = await trimming_message(final_query)
    response = await llm_output.ainvoke(final_query)

    if not isinstance(response, dict):
        response = {"answer": response.content, "sources": []}

    print("Berhasil lewat basic_conclusion", flush=True)
    return {
        "messages": AIMessage(content=json.dumps(response, ensure_ascii=False)),
        "final_answer": response,
    }

# ===== SUMMARY ===== (per week)
async def get_table_schema(table_name: str) -> str:
    query = """
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = $1
    ORDER BY ordinal_position;
    """
    async with pool.acquire() as connection:
        rows = await connection.fetch(query, table_name)

    schema = "\n".join([f"{row['column_name']} ({row['data_type']}, {'nullable' if row['is_nullable'] == 'YES' else 'not nullable'})" for row in rows])
    return f"{table_name} schema:\n{schema}"

async def fetch_data_api(state: SummaryState):
    # get table schema for price and finance tables
    price_schema = await get_table_schema("price_snapshots")
    finance_schema = await get_table_schema("finance_reports")

    # query for generate sql query
    price_query = prompts.PRICE_QUERY.format_map({"ticker": state["ticker"], "price_schema": price_schema, "start_date": state["start_date"], "end_date": state["end_date"]})
    finance_query = prompts.FINANCE_QUERY.format_map({"ticker": state["ticker"], "finance_schema": finance_schema, "start_date": state["start_date"], "end_date": state["end_date"]})

    # generate sql query
    price_query = await llm.ainvoke([SystemMessage(content=prompts.SQL_SYSTEM_QUERY), HumanMessage(content=price_query)])
    finance_query = await llm.ainvoke([SystemMessage(content=prompts.SQL_SYSTEM_QUERY), HumanMessage(content=finance_query)])

    price_query = price_query.content[0]["text"] if isinstance(price_query.content, list) else price_query.content
    finance_query = finance_query.content[0]["text"] if isinstance(finance_query.content, list) else finance_query.content

    # Fetch data from API
    async with pool.acquire() as connection:
        price_data = await connection.fetch(price_query)
        finance_data = await connection.fetch(finance_query)

        # Save to cache
        await save_to_temp(f"price_data_{state['ticker']}_{state['start_date']}_{state['end_date']}", price_data)
        await save_to_temp(f"finance_data_{state['ticker']}_{state['start_date']}_{state['end_date']}", finance_data)

    return {
        "price_path": [f"price_data_{state['ticker']}_{state['start_date']}_{state['end_date']}"],
        "finance_path": [f"finance_data_{state['ticker']}_{state['start_date']}_{state['end_date']}"]
    }

async def summary_agent(state: SummaryState):
    # get data from cache
    price_data = await load_from_temp(f"price_data_{state['ticker']}_{state['start_date']}_{state['end_date']}")
    finance_data = await load_from_temp(f"finance_data_{state['ticker']}_{state['start_date']}_{state['end_date']}")

    # generate summary
    system_query = prompts.SUMMARY_SYSTEM_QUERY.format_map({
        "ticker": state["ticker"],
        "start_date": state["start_date"],
        "end_date": state["end_date"]
    })

    response = await llm_thinking_tools.ainvoke([SystemMessage(content=system_query), HumanMessage(content=f"Price data: {price_data}\nFinance data: {finance_data}")])
    summary = response.content[0]["text"] if isinstance(response.content, list) else response.content

    return {
        "summary": summary
    }

async def should_continue_summary(state: State):
    print("Should continue_summary?", flush=True)
    messages = state["messages"]
    
    tool_calls = getattr(messages[-1], "tool_calls", [])
    
    if len(tool_calls) == 0:
        print("human_review", flush=True)
        return "human_review"

    if state["tool_loop"] > 3:
        print("human_review", flush=True)
        return "human_review"
        
    return "tools"

async def human_review(state: SummaryState):
    decision = interrupt({
        "question": "Approval diperlukan untuk melanjutkan ke tahap berikutnya. Apakah Anda menyetujui ringkasan ini? (ya/tidak)",
        "summary": state["summary"],
    })
    return {"approved": decision["approved"]}

async def should_repeat_summary(state: SummaryState):
    if state.get("approved", False):
        return END
    else:
        return "fetch_data_api"

# Define agent
async def get_agent():
    # Summary
    summary_builder = StateGraph(SummaryState)

    summary_builder.add_node("fetch_data_api", fetch_data_api)
    summary_builder.add_node("summary_agent", summary_agent)
    summary_builder.add_node("human_review", human_review)
    summary_builder.add_node("tools", tool_node)

    summary_builder.add_edge(START, "fetch_data_api")
    summary_builder.add_edge("fetch_data_api", "summary_agent")
    summary_builder.add_conditional_edges("summary_agent", should_continue_summary, ["human_review", "tools"])
    summary_builder.add_edge("tools", "summary_agent")
    summary_builder.add_conditional_edges("human_review", should_repeat_summary, ["fetch_data_api", END])

    summary = summary_builder.compile()

    # Chatbot
    chatbot_builder = StateGraph(State)
    
    chatbot_builder.add_node("rag", rag)
    chatbot_builder.add_node("basic", basic)
    chatbot_builder.add_node("basic_conclusion", basic_conclusion)
    chatbot_builder.add_node("tools", tool_node) 
    
    chatbot_builder.add_edge("rag", "basic")
    chatbot_builder.add_conditional_edges("basic", should_continue, ["basic_conclusion", "tools"])
    chatbot_builder.add_edge("tools", "basic")
    chatbot_builder.add_edge("basic_conclusion", END)

    chatbot = chatbot_builder.compile()

    # Main
    main_builder = StateGraph(State)
    main_builder.add_node("summary_agent", summary)
    main_builder.add_node("chatbot_agent", chatbot)
    
    main_builder.add_conditional_edges(START, routing_where, ["chatbot_agent", "summary_agent"])
    main_builder.add_edge("chatbot_agent", END)
    main_builder.add_edge("summary_agent", END)
    
    return main_builder
    
async def get_agent_graph():
    builder = await get_agent()
    
    agent = builder.compile()
    
    png_graph = agent.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_graph)

    return base64.b64encode(png_graph).decode("utf-8")
