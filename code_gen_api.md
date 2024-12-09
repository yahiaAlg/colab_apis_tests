Cell 1 - Install Required Packages:
```python
!pip install fastapi uvicorn pydantic httpx slowapi torch ollama pyngrok python-multipart torch
```

Cell 2 - Import Dependencies:
```python
import subprocess
import os
import time
import logging
from datetime import datetime
import threading
import asyncio
import nest_asyncio
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from functools import wraps
import atexit

# Apply nest_asyncio for Jupyter
nest_asyncio.apply()
```

Cell 3 - Logging Setup:
```python
def setup_logging() -> Dict[str, str]:
    """Setup logging configuration and return log file paths"""
    log_dir = "ollama_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files = {
        'server': os.path.join(log_dir, f"ollama_server_{timestamp}.log"),
        'install': os.path.join(log_dir, f"ollama_install_{timestamp}.log"),
        'api': os.path.join(log_dir, f"api_{timestamp}.log")
    }
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_files['api']),
            logging.StreamHandler()
        ]
    )
    
    return log_files

# Initialize logging
log_files = setup_logging()
logger = logging.getLogger(__name__)
```

Cell 4 - Ollama Installation Function:
```python
def install_ollama(install_log_file: str) -> bool:
    """Install Ollama and return success status"""
    logger.info("Starting Ollama installation...")
    
    def download_install_script() -> Optional[str]:
        try:
            process = subprocess.run(
                ["curl", "-L", "https://ollama.ai/install.sh"],
                capture_output=True,
                text=True,
                check=True
            )
            return process.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download install script: {e}")
            return None
    
    def execute_install_script(script_content: str) -> bool:
        script_path = "install_ollama.sh"
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            with open(install_log_file, "w") as log_file:
                subprocess.run(
                    ["bash", script_path],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            return True
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)
    
    script_content = download_install_script()
    if script_content:
        return execute_install_script(script_content)
    return False
```

Cell 5 - Server Management Functions:
```python
@dataclass
class ServerProcess:
    process: Optional[subprocess.Popen] = None
    log_file: Optional[str] = None

def start_server(server_log_file: str) -> ServerProcess:
    """Start Ollama server and return process handle"""
    logger.info("Starting Ollama server...")
    try:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open(server_log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        time.sleep(5)  # Allow server initialization
        
        if process.poll() is None:
            logger.info("Ollama server started successfully")
            return ServerProcess(process=process, log_file=server_log_file)
        
        logger.error("Server failed to start")
        return ServerProcess()
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return ServerProcess()

def stop_server(server: ServerProcess) -> None:
    """Stop Ollama server gracefully"""
    if server.process:
        try:
            server.process.terminate()
            server.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.process.kill()
        logger.info("Server stopped")

def read_server_logs(server: ServerProcess) -> str:
    """Read server logs"""
    if server.log_file and os.path.exists(server.log_file):
        try:
            with open(server.log_file, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading logs: {e}"
    return "Log file not available"
```

Cell 6 - Server Monitor:
```python
def create_server_monitor(server: ServerProcess) -> threading.Thread:
    """Create a monitoring thread for the server"""
    def monitor_function():
        while True:
            if server.process and server.process.poll() is None:
                logger.info("Server status check:")
                logger.info(read_server_logs(server))
            time.sleep(300)  # Check every 5 minutes
    
    monitor_thread = threading.Thread(target=monitor_function, daemon=True)
    monitor_thread.start()
    return monitor_thread
```

Cell 7 - Server Initialization:
```python
def initialize_ollama() -> ServerProcess:
    """Initialize Ollama server and return process handle"""
    if not os.path.exists("/usr/local/bin/ollama"):
        if not install_ollama(log_files['install']):
            raise RuntimeError("Ollama installation failed")
    
    server = start_server(log_files['server'])
    if not server.process:
        raise RuntimeError("Failed to start Ollama server")
    
    return server

# Initialize server
try:
    ollama_server = initialize_ollama()
    monitor_thread = create_server_monitor(ollama_server)
    
    def cleanup():
        logger.info("Initiating cleanup...")
        stop_server(ollama_server)
    
    atexit.register(cleanup)
    
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise
```

Cell 8 - FastAPI Setup:
```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# API Models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field(default="python", description="Target language")
    temperature: float = Field(default=0.7, ge=0, le=1)

class ApiResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None

# Initialize FastAPI
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ollama import AsyncClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded



# API Models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field(default="python", description="Target language")
    temperature: float = Field(default=0.7, ge=0, le=1)


class ApiResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


# Initialize FastAPI
def create_app() -> FastAPI:
    app = FastAPI(title="Ollama Code Generation API")
    limiter = Limiter(key_func=get_remote_address) # Define limiter here

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app, limiter # Return limiter from create_app()


app, limiter = create_app() # Unpack app and limiter
```

Cell 9 - API Endpoints:
```python
from ollama import AsyncClient


@app.post("/generate", response_model=ApiResponse)
@limiter.limit("10/minute") # Now limiter is accessible here
async def generate_code(request: CodeRequest, req: Request = None):
    try:
        response = await AsyncClient().chat(
            model="wizardcoder",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate {request.language} code: {request.prompt}",
                }
            ],
        )

        return ApiResponse(
            success=True, data={"generated_code": response.message.content}
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return ApiResponse(success=False, error=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "server_logs": read_server_logs(ollama_server),
    }
```

Cell 10 - Server Start with Ngrok:
```python
from pyngrok import ngrok
import uvicorn
import subprocess
import shutil
import os
import requests

def install_ollama():
    """Install Ollama using the official install script"""
    try:
        # Download the install script
        install_script = requests.get("https://ollama.ai/install.sh").text

        # Save the script
        with open("install_ollama.sh", "w") as f:
            f.write(install_script)

        # Make the script executable
        os.chmod("install_ollama.sh", 0o755)

        # Run the install script
        subprocess.run(["sudo", "./install_ollama.sh"], check=True)

        # Clean up
        os.remove("install_ollama.sh")

        # Verify installation
        subprocess.run(["ollama", "--version"], check=True)
        return True
    except Exception as e:
        logger.error(f"Failed to install Ollama: {e}")
        return False

def start_api_server(port: int = 8000):
    """Start FastAPI server with ngrok tunnel"""

    # Check if ollama is installed and install if needed
    if not shutil.which('ollama'):
        logger.info("Ollama not found. Installing...")
        if not install_ollama():
            raise RuntimeError("Failed to install Ollama")
        logger.info("Ollama installed successfully")

    # Start Ollama service
    try:
        subprocess.run(["sudo", "systemctl", "start", "ollama"], check=True)
    except subprocess.CalledProcessError:
        logger.warning("Failed to start Ollama service via systemctl, trying alternative method")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError(f"Failed to start Ollama service: {e}")

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=port)

    # Start server in thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Setup ngrok
    ngrok.set_auth_token("2DqArVFoOn6ptBy4F8by2rV7eVl_HnmvlCipjgjzuxMiRCwb")  # Your token
    public_url = ngrok.connect(port)
    logger.info(f"Public URL: {public_url}")

    # Verify ollama service is running
    try:
        subprocess.run(["ollama", "list"], check=True)
    except subprocess.CalledProcessError:
        logger.error("Ollama service is not running properly")
        raise RuntimeError("Ollama service check failed")

    return public_url, server_thread

try:
    # Start API server
    public_url, api_thread = start_api_server()
    print(f"API is accessible at: {public_url}")
except Exception as e:
    logger.error(f"Failed to start API server: {e}")
    raise
```

Cell 11 - Test Request:
```python
import requests
import json

def test_api(url: str):
    test_request = {
        "prompt": "Create a function that calculates fibonacci numbers",
        "language": "python",
        "temperature": 0.7
    }
    
    response = requests.post(f"{url}/generate", json=test_request)
    print(json.dumps(response.json(), indent=2))

# Run test
test_api(public_url)
```