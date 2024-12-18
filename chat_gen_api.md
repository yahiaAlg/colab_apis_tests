```python
%%writefile app.py
import os
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
# At the start of your FastAPI app
import requests
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Optional, Dict
import io
from collections import defaultdict
import json

app = FastAPI(
    title="Chat Assistant API",
    description="API for document discussions and chat interactions",
    version="1.0.0",
)

# CORS and rate limiting setup remains the same
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

MODEL_NAME = "llama3"
# Initialize embeddings model

# Add global storage for chat histories and document contexts
chat_histories = defaultdict(list)
document_contexts = {}

embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize ChromaDB
PERSIST_DIRECTORY = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Pydantic models remain the same
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    context_id: Optional[str] = None



def create_collection(content: str, doc_type: str) -> str:
    collection_id = str(uuid.uuid4())
    print(f"Creating collection with ID: {collection_id}")
    collection = chroma_client.create_collection(name=collection_id)

    # Split content into chunks (using your existing text_splitter)
    chunks = text_splitter.split_text(content)
    print(f"Split content into {len(chunks)} chunks")

    # Get embeddings for all chunks at once (more efficient)
    embeddings = embeddings_model.encode(chunks).tolist()

    # Add to collection
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"doc_type": doc_type, "chunk_index": i} for i in range(len(chunks))],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
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

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False
    return False

@app.on_event("startup")
async def startup_event():
    if not check_ollama_connection():
        raise RuntimeError("Cannot connect to Ollama server. Please ensure it's running on http://localhost:11434")



@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        # Get or create chat history
        session_id = chat_request.context_id or str(uuid.uuid4())
        
        # Add new message to history
        chat_histories[session_id].extend([{"role": msg.role, "content": msg.content} 
                                         for msg in chat_request.messages])
        
        # Format all messages for Ollama
        messages = chat_histories[session_id]
        
        print(f"Received messages: {messages}")
        
        # If context_id is provided, retrieve relevant context
        context = ""
        if chat_request.context_id and chat_request.context_id in document_contexts:
            try:
                collection = chroma_client.get_collection(name=chat_request.context_id)
                last_message = chat_request.messages[-1].content
                query_embedding = embeddings_model.encode(last_message).tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2
                )
                context = " ".join(results["documents"][0])
                print(f"Retrieved context: {context}")
                if context:
                    messages.insert(0, {"role": "system", "content": f"Context: {context}"})
            except Exception as e:
                print(f"Error retrieving context: {str(e)}")

        print(f"Sending messages to Ollama: {messages}")
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ollama.chat(
                model=MODEL_NAME,
                messages=messages
            )
        )
        
        response_content = response['message']['content']
        print(f"Ollama response: {response_content}")

        if not response_content:
            raise ValueError("Empty response from Ollama")

        # Add assistant's response to history
        chat_histories[session_id].append({"role": "assistant", "content": response_content})

        return ChatResponse(response=response_content, context_id=session_id)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/document/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text_content = process_pdf_file(content)
        elif file_extension in ['txt', 'md']:
            text_content = process_markdown_file(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        context_id = create_collection(text_content, file_extension)
        
        # Store document context
        document_contexts[context_id] = {
            "filename": file.filename,
            "type": file_extension,
            "content": text_content
        }

        return {
            "status": "success",
            "context_id": context_id,
            "metadata": {
                "type": file_extension,
                "size": len(content),
                "filename": file.filename
            },
            "content": text_content[:1000] + "..." if len(text_content) > 1000 else text_content
        }

    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear/{context_id}")
async def clear_history(context_id: str):
    try:
        if context_id in chat_histories:
            del chat_histories[context_id]
            if context_id in document_contexts:
                del document_contexts[context_id]
                try:
                    collection = chroma_client.get_collection(name=context_id)
                    chroma_client.delete_collection(name=context_id)
                except:
                    pass
            return {"status": "success", "message": "History cleared"}
        return {"status": "error", "message": "Context ID not found"}
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
