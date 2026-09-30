from fastapi import Request, APIRouter, UploadFile
from core.chat_completion import chat_workflow, summary_workflow
from core.langgraph_core import get_agent_graph
from utils.documents_utils import put_new_knowledge, delete_knowledge, list_doc
from body_models.chat_models import  ChatInput, SummaryResumeInput
from utils.health_check import health_check
import uuid

router = APIRouter()

@router.get("/") #✅
async def root():
    return {"message": "Chatbot is running"}


@router.get("/hello") #✅
async def hello():
    return {"message": "Hello, World!"}


# Chat Endpoints
@router.post("/chat") #✅
async def chat(
    request: Request,
    input_data: ChatInput, 
):
    pool = request.app.state.pool

    if input_data.thread_id is None:
        input_data.thread_id = str(uuid.uuid4())

    return await chat_workflow(pool, input_data) #✅

# Summary
@router.post("/summary") #✅
async def summary(
    request: Request,
    input_data: SummaryResumeInput, 
):
    pool = request.app.state.pool

    if input_data.thread_id is None:
        input_data.thread_id = str(uuid.uuid4())

    return await summary_workflow(pool, input_data, resume=False) #✅

# Continue summary
@router.post("/summary/resume")
async def summary_resume(
    request: Request,
    input_data: SummaryResumeInput,
):
    pool = request.app.state.pool
    return await summary_workflow(pool, input_data, resume=True)

# Report data
@router.post("/report")
async def report(
    request: Request,
):
    pool = request.app.state.pool

# Finance report data
@router.post("/report_finance")
async def report_finance():
    pass

# Prediksi
@router.post("/predict")
async def get_predict():
    pass

# Stock price data
@router.post("/get_price")
async def get_price():
    pass

# Upload document (RAG)
@router.post("/upload") #✅
async def upload(request: Request, f: UploadFile):
    pool = request.app.state.pool
    
    return await put_new_knowledge(f)
    
# Delete knowledge
@router.get("/delete/knowledge") #✅
async def delete_k(knowledge_id: str):
    return await delete_knowledge(knowledge_id)

# Get all document filenames in tenant_id #✅
@router.get("/list/documents")
async def list_documents():
    return await list_doc()


@router.get("/get_graph") #✅
async def get_graph():
    result = await get_agent_graph()
    return {"status": "success", "content": result}
    

@router.get("/health") #✅
async def health(request: Request):
    pool = request.app.state.pool
    
    return await health_check(pool)
