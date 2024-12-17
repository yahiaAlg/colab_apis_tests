```python
import os
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from datetime import datetime
import uuid
import PyPDF2
import markdown

app = FastAPI(
    title="Chat Assistant API",
    description="API for document discussions and chat interactions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Ollama and embeddings
llm = Ollama(model="phi3", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="phi3", base_url="http://localhost:11434")

PERSIST_DIRECTORY = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    context_id: Optional[str] = None


class DocumentContext(BaseModel):
    content: str
    metadata: Dict[str, str]
    context_id: str


def create_collection(content: str, doc_type: str) -> str:
    """Create a collection for the document content and return collection ID"""
    collection_id = str(uuid.uuid4())
    collection = chroma_client.create_collection(name=collection_id)

    chunks = text_splitter.split_text(content)

    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"doc_type": doc_type, "chunk_index": i}],
            ids=[f"chunk_{i}"],
        )

    return collection_id


def process_pdf_file(file_content: bytes) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def process_markdown_file(content: bytes) -> str:
    """Convert markdown to plain text"""
    md_content = content.decode("utf-8")
    html = markdown.markdown(md_content)
    # Simple HTML tag removal (you might want to use a proper HTML parser)
    text = html.replace("<p>", "\n").replace("</p>", "\n")
    return " ".join(text.split())


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        # Format conversation history
        conversation = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in chat_request.messages]
        )

        # If context_id is provided, retrieve relevant context
        context = ""
        if chat_request.context_id:
            collection = chroma_client.get_collection(name=chat_request.context_id)
            # Get last message for context search
            last_message = chat_request.messages[-1].content
            results = collection.query(
                query_embeddings=[embeddings.embed_query(last_message)], n_results=2
            )
            context = " ".join(results["documents"][0])

        prompt = f"""Context: {context}

Conversation:
{conversation}
Please proceed with the response, considering both the context and conversation history."""

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )

        return ChatResponse(response=response, context_id=chat_request.context_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/document/upload", response_model=DocumentContext)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "pdf":
            text_content = process_pdf_file(content)
            doc_type = "pdf"
        elif file_extension in ["md", "markdown"]:
            text_content = process_markdown_file(content)
            doc_type = "markdown"
        elif file_extension in ["txt", "text"]:
            text_content = content.decode("utf-8")
            doc_type = "text"
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF, MD, or TXT files.",
            )

        context_id = create_collection(text_content, doc_type)

        return DocumentContext(
            content=(
                text_content[:1000] + "..."
                if len(text_content) > 1000
                else text_content
            ),
            metadata={
                "filename": file.filename,
                "type": doc_type,
                "size": len(content),
            },
            context_id=context_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Chat Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Chat Assistant API",
    }
