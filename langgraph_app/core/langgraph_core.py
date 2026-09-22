from models.openai import llm, llm_thinking
from transformers import AutoTokenizer

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.messages.utils import trim_messages
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.documents import Document

from typing_extensions import Annotated
import copy, traceback, json, base64
import utils.contextmanager_utils as cm
from utils.documents_utils import get_vector_store_chroma, get_vector_store_retriever, BM25Retriever
from core.states import State, LLMOutput, LLMRAG
from string_utils.prompts import Prompts
from typing import Union, List
from typing_extensions import Any

prompts = Prompts()

async def search_for_tables_from_chunks(meta_chunks, vector_store, all_table_ids) -> list[dict[str, Any]]:
    collection = vector_store._collection

    tables = []
    ids = []
    for meta in meta_chunks:
        if meta["type"] == "text": # is text
            table_ids = meta.get("table", [])
            if table_ids:
                result = collection.get(where={"table_id": {"$in": table_ids}}, include=["metadatas"])

                if len(result["ids"]) > 0:
                    for meta_tab in result["metadatas"]:
                        if meta_tab["type"] == "table":
                            table_id = meta_tab["table_id"]
                            if table_id not in all_table_ids:
                                all_table_ids.append(table_id)
                                ids.append({"table_id": table_id})
                                tables.append({
                                    "table": meta_tab["table"], 
                                    "metadata": {key: val for key, val in meta_tab.items() if key != "table"},
                                })

        elif meta["type"] == "table":
            table_id = meta["table_id"]
            if table_id not in all_table_ids:
                all_table_ids.append(table_id)
                ids.append({"table_id": meta["table_id"]})
                tables.append({
                    "table": meta["table"], 
                    "metadata": {key: val for key, val in meta.items() if key != "table"},
                })

    return ids, tables

async def _get_description_and_metadata(result, selected_image_table, image_table, all_image_ids):
        if len(result["ids"]) > 0:
            for desc, meta_img in zip(result["documents"], result["metadatas"]):
                image_id = meta_img["image_id"]
                if image_id not in all_image_ids:
                    img_path = meta_img["img_path"]
                    all_image_ids.append(image_id)
                    selected_image_table.append({"img_path": img_path, "image_id": image_id})
                    image_table.append({"description": desc, "metadata": meta_img})

        return selected_image_table, image_table

async def search_for_images_from_chunks(meta_chunks, vector_store, all_image_ids) -> list[dict[str, Any]]:
    async def _searching_process(key):
        selected_image_table = []
        image_table = []
        for meta in meta_chunks:
            if meta["type"] == "text": # is text
                image_ids = meta.get(key, []) # image_table
                if image_ids:
                    result = collection.get(where={"image_id": {"$in": image_ids}}, include=["documents", "metadatas"])
    
                    selected_image_table, image_table = await _get_description_and_metadata(result, selected_image_table, image_table, all_image_ids)
    
            elif meta["type"] == key[:-2]: # image_tab
                image_id = meta["image_id"]
                result = collection.get(where={"image_id": image_id}, include=["documents", "metadatas"])
    
                selected_image_table, image_table = await _get_description_and_metadata(result, selected_image_table, image_table, all_image_ids)

        return selected_image_table, image_table

    collection = vector_store._collection

    selected_image_out_table, image_out_table = await _searching_process("image_out_table")
    selected_image_in_table, image_in_table = await _searching_process("image_in_table")

    return selected_image_out_table, image_out_table, selected_image_in_table, image_in_table

# Retrieve table and image data from chunks
async def search_tables_and_images_from_chunks(chunks_state, selected_table_state, selected_image_in_table_state, selected_image_out_table_state, vector_store):
    all_image_ids = [image["image_id"] for image in selected_image_out_table_state]
    all_image_ids += [image["image_id"] for image in selected_image_in_table_state]

    all_table_ids = [table["table_id"] for table in selected_table_state]

    meta_chunks = [chunk["metadata"] for chunk in chunks_state]

    selected_table, tables = await search_for_tables_from_chunks(meta_chunks, vector_store, all_table_ids)
    selected_image_out_table, image_out_table, selected_image_in_table, image_in_table = await search_for_images_from_chunks(meta_chunks, vector_store, all_image_ids)

    return selected_table, tables, selected_image_out_table, image_out_table, selected_image_in_table, image_in_table


# Retrieve image data from tables
async def search_for_images_from_tables(tables_state, selected_image_in_tab_state, image_in_table_state, vector_store, existing_image_ids = None):
    collection = vector_store._collection
    all_image_ids = list(existing_image_ids or [])
    all_image_ids += [image["image_id"] for image in selected_image_in_tab_state]

    selected_image_in_table = list(selected_image_in_tab_state)
    image_in_table = list(image_in_table_state)
    for table in tables_state:
        image_ids = table["metadata"].get("image_ids", [])
        if image_ids:
            result = collection.get(where={"image_id": {"$in": image_ids}}, include=["documents", "metadatas"])

            selected_image_in_table, image_in_table = await _get_description_and_metadata(result, selected_image_in_table, image_in_table, all_image_ids)

    return selected_image_in_table, image_in_table


# Filter chunk just for text
async def filter_chunk(chunk_append, selected_knowledge_dict, knowledge_id_append):
    new_chunk_append = [] 
    new_selected_knowledge_dict = {}
    # new_replace_ids = set()
    new_knowledge_id_append = []
    for chunk in chunk_append:
        if chunk["metadata"]["type"] == "text":
            knowledge_id = chunk["metadata"]["knowledge_id"]

            new_chunk_append.append(chunk)

            if knowledge_id in selected_knowledge_dict:
                if knowledge_id not in new_selected_knowledge_dict:
                    new_selected_knowledge_dict[knowledge_id] = selected_knowledge_dict[knowledge_id]

            if knowledge_id in knowledge_id_append:
                if knowledge_id not in new_knowledge_id_append:
                    new_knowledge_id_append.append(knowledge_id)

    return new_chunk_append, new_selected_knowledge_dict, new_knowledge_id_append

async def filter_chunk_tool(chunk_append, selected_knowledge_dict, replace_ids, knowledge_id_append):
    new_chunk_append = [] 
    new_selected_knowledge_dict = {}
    new_replace_ids = set()
    new_knowledge_id_append = []
    for chunk in chunk_append:
        if chunk["metadata"]["type"] == "text":
            knowledge_id = chunk["metadata"]["knowledge_id"]

            new_chunk_append.append(chunk)

            if knowledge_id in selected_knowledge_dict:
                if knowledge_id not in new_selected_knowledge_dict:
                    new_selected_knowledge_dict[knowledge_id] = selected_knowledge_dict[knowledge_id]

            if knowledge_id in knowledge_id_append:
                if knowledge_id not in new_knowledge_id_append:
                    new_knowledge_id_append.append(knowledge_id)

            if knowledge_id in replace_ids:
                new_replace_ids.add(knowledge_id)

    return new_chunk_append, new_selected_knowledge_dict, new_replace_ids, new_knowledge_id_append

async def get_contexts_from_current_state(chunk_knowledge, tables, image_in_table, image_out_table):
    reformat_chunk_knowledge = [f"{i+1}. {chunk['content']}\nSource pages: {chunk['metadata']['page_numbers']}\n\n" for i, chunk in enumerate(chunk_knowledge) if chunk["metadata"]["type"] == "text"]
    reformat_tables = [f"{i+1}. {tab['table']}\nSource pages: {tab['metadata']['page_numbers']}\n\n" for i, tab in enumerate(tables)]
    reformat_image_in_table = [f"{i+1}. {image['description']}\nSource page: {image['metadata']['page_number']}\nrow, column: {image['metadata']['row']}, {image['metadata']['column']}\n\n" for i, image in enumerate(image_in_table)]
    reformat_image_out_table = [f"{i+1}. {image['description']}\nSource page: {image['metadata']['page_number']}\n\n" for i, image in enumerate(image_out_table)]
    
    return reformat_chunk_knowledge, reformat_tables, reformat_image_in_table, reformat_image_out_table

async def get_metadata_and_content(results):
    new_results = []
    for result in results:
        if isinstance(result, Document):
            metadata = result.metadata
            content = result.page_content
        else:
            metadata = result["metadata"]
            content = result["page_content"]

        new_results.append({"metadata": metadata, "page_content": content})

    return new_results

async def check_duplicate_knowledge(results, selected_knowledge, chunk_ids, knowledge_ids):
    selected_knowledge_dict = {x["knowledge_id"]: {"chunk_ids": x["chunk_ids"]} for x in selected_knowledge}
    knowledge_id_append = []
    chunk_append = []
    replace_ids = set()

    for result in results:
        metadata = result["metadata"]
        content = result["page_content"]

        chunk_id = metadata["chunk_id"]
        if chunk_id in chunk_ids:
            continue
            
        knowledge_id = metadata["knowledge_id"]
        if knowledge_id not in selected_knowledge_dict.keys():
            selected_knowledge_dict[knowledge_id] = {"chunk_ids": []}
            knowledge_id_append.append(knowledge_id)
        elif knowledge_id in knowledge_ids and knowledge_id not in knowledge_id_append:
            replace_ids.add(knowledge_id)
            
        selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
        chunk_append.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})

    return selected_knowledge_dict, knowledge_id_append, chunk_append, replace_ids

async def replace_knowledge(results):
    selected_knowledge_dict = {}
    knowledge_id_append = []
    chunk_append = []

    for result in results:
        metadata = result["metadata"]
        content = result["page_content"]
        chunk_id = metadata["chunk_id"]
        knowledge_id = metadata["knowledge_id"]

        if knowledge_id not in selected_knowledge_dict.keys():
            selected_knowledge_dict[knowledge_id] = {"chunk_ids": []}
            knowledge_id_append.append(knowledge_id)
            
        selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
        chunk_append.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})

    return selected_knowledge_dict, knowledge_id_append, chunk_append

# Tools
## Tool: Fetch new knowledge 
@tool
async def fetch_new_knowledge(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId], 
) -> Command | str:
    """
    Fetch new tenant/company knowledge chunks from the database that are not yet in the current state.

    Embeds the query, retrieves the top 5 closest knowledge chunks via vector similarity search,
    filters out chunks already present in the state, fetches metadata for newly found knowledge entries,
    and updates the state with the new knowledge and chunks.

    Args:
        query (str): The search query to embed and use for similarity search.

    Returns:
        Command: Updates selected_knowledge and chunk_knowledge state on success.
        str: Error message on failure.
    """
    print("Tool: fetch_new_knowledge", flush=True)
    
    selected_knowledge = copy.deepcopy(state["selected_knowledge"])
    
    # Get knowledge ids and chunk ids
    knowledge_ids = [x["knowledge_id"] for x in selected_knowledge]
    chunk_ids = [c_id for x in selected_knowledge for c_id in x["chunk_ids"]]

    use_rerank = state["rerank"]
    if state["bm25"]:
        if not use_rerank:
            use_rerank = True
    elif use_rerank:
        use_rerank = False
    
    try:
        # Retrieve from the database
        vector_store = await get_vector_store_chroma(f"knowledges")
        retriever = await get_vector_store_retriever(vector_store, {"type": {"$in": ["text", "table"]}}, k=8)

        instruct = "Given a user query about the document knowledge, retrieve the relevant passages that answer the query"
        final_query = f"Instruct: {instruct}\nQuery:{query}"
        results = await retriever.ainvoke(final_query)

        if state["bm25"]:
            bm25 = BM25Retriever()
            await bm25.start(results)
            results += await bm25.retrieve(query, k=8)
        
        if len(results) == 0:
            print("Tool: fetch_new_knowledge end", flush=True)
            return "Success. Based on the query, the knowledge is not exist in the database."

        results = await get_metadata_and_content(results)
        if use_rerank:
            list_document = [r["page_content"] for r in results]
            instruct = "Classify whether the document matches the query topic"
            final_query = f"Instruct: {instruct}\nQuery:{query}"

            rerank_ids = await cm.reranker.rerank(final_query, list_document, 8)
            results = [results[i] for i in rerank_ids]

        result = await check_duplicate_knowledge(results, selected_knowledge, chunk_ids, knowledge_ids)
        selected_knowledge_dict, knowledge_id_append, chunk_append, replace_ids = result

        result = await search_tables_and_images_from_chunks(
            chunk_append, 
            state.get("selected_table", []),
            state.get("selected_image_in_table", []),
            state.get("selected_image_out_table", []),
            vector_store,
        )

        selected_table, tables, selected_image_out_table, image_out_table, selected_image_in_table, image_in_table = result
        existing_image_ids = [img["image_id"] for img in state.get("selected_image_in_table", [])]
        selected_image_in_table, image_in_table = await search_for_images_from_tables(tables, selected_image_in_table, image_in_table, vector_store, existing_image_ids=existing_image_ids)

        # Filter chunk just for text
        result = await filter_chunk_tool(chunk_append, selected_knowledge_dict, replace_ids, knowledge_id_append)
        chunk_append, selected_knowledge_dict, replace_ids, knowledge_id_append = result

        print("Tool: fetch_new_knowledge end", flush=True)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Success fetch new knowledge from the database.", 
                        tool_call_id=tool_call_id,
                        name="fetch_new_knowledge",
                    )
                ],
                "selected_knowledge": {
                    "append": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
                    "replace": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in replace_ids],
                },
                "chunk_knowledge": {
                    "append": chunk_append,
                },
                "selected_table": {"append": selected_table}, "tables": {"append": tables},
                "selected_image_in_table": {"append": selected_image_in_table}, "image_in_table": {"append": image_in_table},
                "selected_image_out_table": {"append": selected_image_out_table}, "image_out_table": {"append": image_out_table}
            }
        )
        
    except Exception as e:
        traceback.print_exc()
        return "Failed fetch new knowledge from database."

## Define Tools node
tools = [fetch_new_knowledge]

llm_tools = llm_thinking.bind_tools(tools)

tool_node = ToolNode(tools)
    
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
async def extract_content(message) -> str:
    content = message.content

    if isinstance(content, list): # AI
        content = " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        
    if isinstance(message, AIMessage):
        if message.tool_calls: # AI with tool calls
            tool_names = [t["name"] for t in message.tool_calls]
            return f"AI: content: {content}, tool calls: {tool_names}"
    
    return content

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

def count_tokens(messages: Union[str, List[BaseMessage]]) -> int:
    if isinstance(messages, str):
        return len(tokenizer.encode(messages, add_special_tokens=False))

    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(tokenizer.encode(block["text"]), add_special_tokens=False)
        elif isinstance(content, str) and content:
            total += len(tokenizer.encode(content, add_special_tokens=False))

    return total
    
async def trimming_message(messages):
    messages = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens,
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

## RAG (retrieve data from database based on just new query)
async def rag(state: State):
    print("Node: rag", flush=True)
   
    use_rerank = state["rerank"]
    if state["bm25"]:
        if not use_rerank:
            use_rerank = True
    elif use_rerank:
        use_rerank = False

    if state["enhanced"]:
        # Get chunk knowledge
        results = await get_contexts_from_current_state(
            state.get("chunk_knowledge", []),
            state.get("tables", []),
            state.get("image_in_table", []),
            state.get("image_out_table", [])
        )
        reformat_chunk_knowledge, reformat_tables, reformat_image_in_table, reformat_image_out_table = results

        system_query = prompts.RAG_SYSTEM_QUERY.format_map({
            "knowledges": reformat_chunk_knowledge,
            "images_in_table_descriptions": reformat_image_in_table,
            "images_out_table_descriptions": reformat_image_out_table,
            "tables": reformat_tables,
        })

        messages = state["messages"]

        final_query = [
            SystemMessage(content=system_query),
            *messages[:-1],
            HumanMessage(content=f"User's query: {messages[-1].content}"),
        ]

        # Trim messages
        final_query = await trimming_message(final_query)

        response = await llm_rag.ainvoke(final_query)

        if isinstance(response, dict):
            text = response.get("question", "none")
        else:
            text = response.content

        if text in ["none", ""]:
            return {}
    else: 
        text = state["messages"][-1].content

    # Retrieve from the database
    vector_store = await get_vector_store_chroma("knowledges")
    retriever = await get_vector_store_retriever(vector_store, {"type": {"$in": ["text", "table"]}}, k=8) #, {"type": "text"})

    instruct = "Given a user query about the document knowledge, retrieve the relevant passages that answer the query"
    final_query = f"Instruct: {instruct}\nQuery:{text}"
    results = await retriever.ainvoke(final_query)

    if state["bm25"]:
        bm25 = BM25Retriever()
        await bm25.start(results)
        results += await bm25.retrieve(text, k=8)

    if len(results) == 0:
        return {}

    results = await get_metadata_and_content(results)
    if use_rerank:
        list_document = [r["page_content"] for r in results]
        instruct = "Classify whether the document matches the query topic"
        final_query = f"Instruct: {instruct}\nQuery:{text}"

        rerank_ids = await cm.reranker.rerank(final_query, list_document, 8)
        results = [results[i] for i in rerank_ids]

    result = await replace_knowledge(results)
    selected_knowledge_dict, knowledge_id_append, chunk_append = result
    
    result = await search_tables_and_images_from_chunks(
        chunk_append, [], [], [], vector_store,
    )
    
    selected_table, tables, selected_image_out_table, image_out_table, selected_image_in_table, image_in_table = result
    selected_image_in_table, image_in_table = await search_for_images_from_tables(tables, selected_image_in_table, image_in_table, vector_store)

    result = await filter_chunk(chunk_append, selected_knowledge_dict, knowledge_id_append)
    chunk_append, selected_knowledge_dict, knowledge_id_append = result
    
    return {
        "selected_knowledge": {
            "replace": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
        },
        "chunk_knowledge": {
            "replace": chunk_append,
        },
        "selected_table": {"replace": selected_table}, "tables": {"replace": tables},
        "selected_image_in_table": {"replace": selected_image_in_table}, "image_in_table": {"replace": image_in_table},
        "selected_image_out_table": {"replace": selected_image_out_table}, "image_out_table": {"replace": image_out_table},
        "query": text,
    }

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
    response = await llm_tools.ainvoke(final_query)

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


# Define agent
async def get_agent():
    builder = StateGraph(State)
    
    builder.add_node("rag", rag)
    builder.add_node("basic", basic)
    builder.add_node("basic_conclusion", basic_conclusion)
    builder.add_node("tools", tool_node) 
    
    builder.add_edge(START, "rag")
    builder.add_edge("rag", "basic")
    builder.add_conditional_edges("basic", should_continue, ["basic_conclusion", "tools"])
    builder.add_edge("tools", "basic")
    builder.add_edge("basic_conclusion", END)
    
    return builder
    
async def get_agent_graph():
    builder = await get_agent()
    
    agent = builder.compile()
    
    png_graph = agent.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_graph)

    return base64.b64encode(png_graph).decode("utf-8")
