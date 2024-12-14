import os
from typing import List, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
from datetime import datetime
import uuid
from fastapi import Request, Depends

# Initialize FastAPI app
app = FastAPI(
    title="Code Assistant API",
    description="API for code generation, debugging, and documentation using WizardCoder",
    version="1.0.0"
)

# Configure CORS
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


# Initialize Ollama with WizardCoder and embeddings
llm = Ollama(model="wizardcoder", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="wizardcoder", base_url="http://localhost:11434")

COLLECTION_NAME = "documents"
PERSIST_DIRECTORY = "./chroma_db"
# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Text splitter for code
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
)

class CodeRequest(BaseModel):
    code: str
    task: str  # 'generate', 'debug', or 'document'
    language: Optional[str] = None
    description: Optional[str] = None

class CodeResponse(BaseModel):
    result: str
    task_type: str
    collection_id: Optional[str] = None

class DebugRequest(BaseModel):
    code: str
    language: str
    include_performance_analysis: bool = False
    include_security_analysis: bool = False

class Issue(BaseModel):
    severity: str
    message: str
    line_number: Optional[int]
    suggested_fix: str

class DebugResponse(BaseModel):
    issues: List[Issue]
    fixed_code: str
    performance_analysis: Optional[str] = None
    security_analysis: Optional[str] = None

def create_collection(code: str, language: str) -> str:
    """Create a collection for the code and return collection ID"""
    collection_id = str(uuid.uuid4())
    collection = chroma_client.create_collection(name=collection_id)

    # Split code into chunks
    chunks = code_splitter.split_text(code)

    # Generate embeddings and add to collection
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"language": language, "chunk_index": i}],
            ids=[f"chunk_{i}"]
        )

    return collection_id

def process_code_file(file: bytes) -> str:
    """Process code file content"""
    return file.decode("utf-8")

def analyze_code(code: str, language: str) -> List[Issue]:
    """Analyze code for common issues"""
    try:
        prompt = f"""Analyze the following {language} code for issues:
        {code}
        Provide a list of issues found, including severity (error/warning/info),
        line numbers when applicable, and suggested fixes.
        Format: SEVERITY|LINE|MESSAGE|FIX"""

        analysis = llm.invoke(prompt)
        issues = []

        for line in analysis.split('\n'):
            try:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) == 4:
                        severity, line_num, message, fix = parts
                        issues.append(Issue(
                            severity=severity.lower().strip(),
                            line_number=int(line_num) if line_num.strip().isdigit() else None,
                            message=message.strip(),
                            suggested_fix=fix.strip()
                        ))
            except Exception as line_error:
                print(f"Error processing analysis line: {str(line_error)}")
                continue

        return issues

    except Exception as e:
        print(f"Code analysis error: {str(e)}")
        return []

def analyze_performance(code: str, language: str) -> str:
    """Analyze code for performance issues"""
    prompt = f"""Analyze the following {language} code for performance optimization opportunities:
    {code}
    Provide detailed performance analysis and optimization suggestions."""

    return llm.invoke(prompt)

def analyze_security(code: str, language: str) -> str:
    """Analyze code for security vulnerabilities"""
    prompt = f"""Analyze the following {language} code for security vulnerabilities:
    {code}
    Provide detailed security analysis and remediation suggestions."""

    return llm.invoke(prompt)

@app.post("/api/debug", response_model=DebugResponse)
@limiter.limit("10/minute")
async def debug_code(
    request: Request,
    debug_request: DebugRequest
):
    try:
        # Log incoming request for debugging
        print(f"Received debug request: {debug_request}")

        # Validate input code
        if not debug_request.code or not debug_request.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")

        # Test LLM connection
        try:
            llm.invoke("Test connection")
        except Exception as llm_error:
            print(f"LLM connection error: {str(llm_error)}")
            raise HTTPException(status_code=500, detail="LLM service unavailable")

        # Enhanced code analysis with error handling
        try:
            issues = []
            analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analyze_code(debug_request.code, debug_request.language)
            )
            issues.extend(analysis)
        except Exception as analysis_error:
            print(f"Analysis error: {str(analysis_error)}")
            issues = []  # Continue with empty issues list

        # Generate fixed code with error handling
        try:
            fix_prompt = f"""Fix the following {debug_request.language} code addressing all identified issues:
            {debug_request.code}
            Provide the complete fixed code."""

            fixed_code = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.invoke(fix_prompt)
            )
        except Exception as fix_error:
            print(f"Fix generation error: {str(fix_error)}")
            fixed_code = debug_request.code  # Return original code if fix fails

        response = DebugResponse(
            issues=issues,
            fixed_code=fixed_code
        )

        # Optional performance analysis with error handling
        if debug_request.include_performance_analysis:
            try:
                performance_analysis = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: analyze_performance(debug_request.code, debug_request.language)
                )
                response.performance_analysis = performance_analysis
            except Exception as perf_error:
                print(f"Performance analysis error: {str(perf_error)}")
                response.performance_analysis = "Performance analysis failed"

        # Optional security analysis with error handling
        if debug_request.include_security_analysis:
            try:
                security_analysis = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: analyze_security(debug_request.code, debug_request.language)
                )
                response.security_analysis = security_analysis
            except Exception as sec_error:
                print(f"Security analysis error: {str(sec_error)}")
                response.security_analysis = "Security analysis failed"

        return response

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        print(f"Debug endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/debug/file")
@limiter.limit("10/minute")
async def debug_code_file(
    request: Request,
    file: UploadFile = File(...),
    include_performance_analysis: bool = False,
    include_security_analysis: bool = False
):
    try:
        content = await file.read()
        code = process_code_file(content)
        language = file.filename.split('.')[-1]

        debug_request = DebugRequest(
            code=code,
            language=language,
            include_performance_analysis=include_performance_analysis,
            include_security_analysis=include_security_analysis
        )

        return await debug_code(request, debug_request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/code/generate", response_model=CodeResponse)
@limiter.limit("10/minute")
async def generate_code(
    request: Request,
    code_request: CodeRequest
):
    try:
        prompt = ""
        if code_request.task == "generate":
            prompt = f"""Generate code in {code_request.language} for the following requirement:
            {code_request.description}
            Provide well-structured, efficient, and documented code."""

        elif code_request.task == "debug":
            prompt = f"""Debug the following code and explain the issues found:
            {code_request.code}
            Provide the corrected code and explanation of fixes."""

        elif code_request.task == "document":
            prompt = f"""Add comprehensive documentation to the following code:
            {code_request.code}
            Include function/class documentation, parameter descriptions, and usage examples."""
        else:
            raise HTTPException(status_code=400, detail="Invalid task type")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: llm.invoke(prompt)
        )

        # Create collection for the generated/processed code
        collection_id = create_collection(
            code=result,
            language=code_request.language or "unknown"
        )

        return CodeResponse(
            result=result,
            task_type=code_request.task,
            collection_id=collection_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/code/file", response_model=CodeResponse)
@app.post("/api/document/file", response_model=CodeResponse)
@limiter.limit("10/minute")
async def process_code_file_endpoint(
    request: Request,
    file: UploadFile = File(...),
    task: str = "document"
):
    try:
        content = await file.read()
        code = process_code_file(content)

        code_request = CodeRequest(
            code=code,
            task=task,
            language=file.filename.split('.')[-1]
        )

        return await generate_code(request, code_request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Code Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Code Assistant API"
    }