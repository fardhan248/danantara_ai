import sys, requests, time, argparse

def get_knowledge_id():
    url = "http://localhost:8000/list/documents"

    try:
        response = requests.get(
            url,
        )
    except Exception as e:
        raise ValueError("Error", str(e))

    content = response.json().get("content", None)
    if content:
        if isinstance(content, list):
            return [c["knowledge_id"] for c in content]
        else:
            return None
    else:
        return response.json()["status"]

def delete_knowledge(knowledge_id):
    url = f"http://localhost:8000/delete/knowledge"

    try:
        response = requests.get(
            url,
            params={"knowledge_id": knowledge_id},
        )
    except Exception as e:
            raise ValueError("Error", str(e))

    content = response.json().get("status", None)
    if content:
        return content
    else:
        return response.json()["status"]

def reset_knowledge():
    knowledge_ids = get_knowledge_id()
    if isinstance(knowledge_ids, list):
        for k_id in knowledge_ids:
            status = delete_knowledge(k_id)

        return "success"

    elif knowledge_ids is None:
        return "No knowledge"

    else:
        return "success"

def upload(file_path: str = r"docs\MODUL PEMBELAJARAN.pdf"):
    url = "http://localhost:8000/upload"

    f = {"f": open(file_path, "rb")}

    try:
        response = requests.post(
            url,
            files=f,
        )
    except Exception as e:
        raise ValueError("Error", str(e))

    return response.status_code

def main(query: str, thread_id: str = None, bm25: bool = False, rerank: bool = False, enhanced: bool = False):
    url = "http://localhost:8000/chat"

    if thread_id:
        payload = {
            "thread_id": thread_id,
            "input_prompt": query,
            "bm25": bm25,
            "rerank": rerank,
            "enhanced": enhanced,
        }
    else:
        payload = {
            "input_prompt": query,
            "bm25": bm25,
            "rerank": rerank,
            "enhanced": enhanced,
        }

    try:
        start = time.perf_counter()
        response = requests.post(
            url,
            json=payload,
        )
        end = time.perf_counter()
        time_exec = end - start
    except Exception as e:
        raise ValueError("Error:", str(e))

    if response.status_code == 200:
        result = response.json()
        if thread_id is None:
            thread_id = result.get("thread_id", None)

        text = result.get("content", None)
        if isinstance(text, dict):
            answer = text.get("answer", "No key 'answer'")
            sources = text.get("sources", [])

            text = f"{answer}\nSources: {sources}"

        print("--- Assistant:\n> ", end="")
        print(text)
        print(f"\n- Time execution: {time_exec:.2f} seconds")
        print()
        return thread_id

    else:
        raise ValueError(response.json())

def get_graph():
    url = "http://localhost:8000/get_graph"

    try:
        response = requests.get(url)
    except Exception as e:
        raise ValueError("Error:", str(e))

    return response.status_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graph", "--get-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bm25", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enhanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-reset", "--reset-knowledge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-id", "--thread-id", type=str, default=None)
    parser.add_argument("-doc", "--document-path", type=str, default=r"docs\MODUL PEMBELAJARAN.pdf")

    args = parser.parse_args()

    graph = bool(args.get_graph)
    bm25 = bool(args.bm25)
    rerank = bool(args.rerank)
    enhanced = bool(args.enhanced)
    reset = bool(args.reset_knowledge)
    thread_id = args.thread_id
    doc = args.document_path

    if graph:
        status = get_graph()
        print("Status", str(status))
        sys.exit()

    # Upload PDF
    if reset:
        result = reset_knowledge()
        print(result)

        print("Extract knowledge...")
        result_doc = upload(doc)
        print("Upload status:", result_doc)

    print("============= Start a chat with an Asisstant =============")
    try:
        while True:
            query = input("--- User:\n> ")

            thread_id = main(query, thread_id, bm25, rerank, enhanced)

    except KeyboardInterrupt:
        print("Chat thread id:", thread_id)
        print("\n============= Chat ended =============")
