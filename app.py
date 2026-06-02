import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

import os
import asyncio
import uuid
from typing import Optional, List
import time
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Cookie
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from werkzeug.utils import secure_filename
from utils.config import Config
from agents.agent_decision import process_query, create_agent_graph
from utils.proxy_setting import *
from utils.streaming import current_stream_callback
from agents.patient_db_agent import PatientQueryEngine
from langchain_core.callbacks import BaseCallbackHandler

config = Config()

#Set proxy 
set_proxy()

patient_query_engine = PatientQueryEngine(config)

graph = create_agent_graph(patient_query_engine)


# Initialize FastAPI app with increased limits for large form data
app = FastAPI(
    title="Multi-Agent Medical Chatbot", 
    version="2.0"
)

# Increase payload size limits
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

class LargeRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Increase the max request size for this endpoint
        if request.url.path == "/validate":
            # Set a larger body limit for this specific endpoint
            request._body_size_limit = 50 * 1024 * 1024  # 50MB
        return await call_next(request)

app.add_middleware(LargeRequestMiddleware)

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
POLYP_SEGMENTATION_OUTPUT = "uploads/polyp_seg_output"
LOGS_DIR = "logs"
LOGS_IMAGES_DIR = "logs/images"
LOGS_REVIEWS_DIR = "logs/reviews"

# Create all required directories if they don't exist
for directory in [UPLOAD_FOLDER, POLYP_SEGMENTATION_OUTPUT, LOGS_DIR, LOGS_IMAGES_DIR, LOGS_REVIEWS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    conversation_history: List = []

def extract_chat_response(response_data: dict) -> tuple[str, str]:
    response_text = ""
    agent_name = response_data.get('agent_name', 'ASSISTANT')

    if 'messages' in response_data and len(response_data['messages']) > 0:
        last_message = response_data['messages'][-1]
        if hasattr(last_message, 'content'):
            response_text = last_message.content

    if not response_text and 'output' in response_data:
        output = response_data['output']
        if hasattr(output, 'content'):
            response_text = output.content
        elif isinstance(output, str):
            response_text = output

    if not response_text:
        response_text = "I'm sorry, I couldn't process your request properly."

    return response_text, agent_name

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

class QueueStreamingCallback(BaseCallbackHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs):
        if token:
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait,
                ("token", token),
            )

    def emit_event(self, event: str, payload: dict):
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait,
            (event, payload),
        )


@app.get("/")
async def root():
    """Redirect unused dashboard entrypoint to the chat interface."""
    return RedirectResponse(url="/chat")

@app.get("/chat")
async def chat_interface(request: Request):
    """Render the AI chat interface"""
    return templates.TemplateResponse(request, "index.html")

@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """Process a chat message and stream LLM tokens to the browser."""
    async def generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        callback = QueueStreamingCallback(loop, queue)
        token_count = 0

        def run_query():
            token = current_stream_callback.set(callback)
            try:
                response_data = process_query(request.message, graph=graph)
                response_text, agent_name = extract_chat_response(response_data)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("done", {
                        "status": "success",
                        "agent": agent_name,
                        "response": response_text,
                    }),
                )
            except Exception as e:
                print(f"Error processing streamed chat: {str(e)}")
                import traceback
                traceback.print_exc()
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("error", {
                        "status": "error",
                        "agent": "System",
                        "response": f"An error occurred: {str(e)}",
                    }),
                )
            finally:
                current_stream_callback.reset(token)

        try:
            yield sse_event("status", {"message": "Processing your medical query..."})
            task = asyncio.create_task(asyncio.to_thread(run_query))

            while True:
                event, payload = await queue.get()
                if event == "token":
                    token_count += 1
                    yield sse_event("token", {"text": payload})
                    continue

                if event == "agent":
                    yield sse_event("agent", payload)
                    continue

                if event == "status":
                    yield sse_event("status", payload)
                    continue

                if event == "done":
                    await task
                    yield sse_event("metadata", {
                        "status": payload["status"],
                        "agent": payload["agent"],
                        "streamed_tokens": token_count,
                    })
                    if token_count == 0:
                        yield sse_event("token", {"text": payload["response"]})
                    yield sse_event("done", payload)
                    break

                if event == "error":
                    await task
                    yield sse_event("error", payload)
                    break
        except asyncio.CancelledError:
            raise

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
   
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
        print(f"File saved to {file_path}")
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query, graph=graph)
        response_text = response_data['messages'][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is polyp segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        if response_data["agent_name"].startswith("POLYP_SEGMENTATION_AGENT"):
            segmentation_path = response_data.get("polyp_segmentation_path") or os.path.join(POLYP_SEGMENTATION_OUTPUT, "polyp_seg_image_output.jpg")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/polyp_seg_output/{os.path.basename(segmentation_path)}"
            else:
                print("Polyp Segmentation Output path does not exist.")
        elif response_data["agent_name"].startswith("POLYP_VQA_AGENT"):
            segmentation_path = response_data.get("polyp_segmentation_path")
            if segmentation_path and os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/polyp_seg_output/{os.path.basename(segmentation_path)}"
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate_medical_output(
    request: Request,
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    bot_response: Optional[str] = Form(None),
    user_question: Optional[str] = Form(None),
    image_data: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)
        
        # Log reviews for further analysis
        if validation_result.lower() not in ['yes', 'true', '1', 'confirm']:
            # Safely parse JSON data
            parsed_bot_response = None
            parsed_user_question = None
            
            try:
                if bot_response:
                    parsed_bot_response = json.loads(bot_response)
            except Exception as e:
                print(f"Error parsing bot_response JSON: {str(e)}")
                try:
                    # Try to fix common JSON errors
                    if bot_response:
                        # Store the raw data in case parsing fails
                        parsed_bot_response = {"error": "Failed to parse JSON", "raw_content": bot_response[:500] + "..." if len(bot_response) > 500 else bot_response}
                except Exception as inner_e:
                    print(f"Failed to create error object for bot_response: {str(inner_e)}")
                    parsed_bot_response = {"error": "Failed completely to handle response"}
                
            try:
                if user_question:
                    parsed_user_question = json.loads(user_question)
            except Exception as e:
                print(f"Error parsing user_question JSON: {str(e)}")
                try:
                    # Try to fix common JSON errors
                    if user_question:
                        # Store the raw data in case parsing fails
                        parsed_user_question = {"error": "Failed to parse JSON", "raw_content": user_question[:500] + "..." if len(user_question) > 500 else user_question}
                except Exception as inner_e:
                    print(f"Failed to create error object for user_question: {str(inner_e)}")
                    parsed_user_question = {"error": "Failed completely to handle question"}
            
            # Save image if provided
            image_path = None
            if image_data and image_data.startswith('data:image'):
                # Extract image data from base64 string
                try:
                    # Generate a timestamp for the filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    file_id = uuid.uuid4().hex[:6]
                    
                    # Get image format from data URL
                    format_type = "png"
                    if "data:image/jpeg" in image_data:
                        format_type = "jpg"
                    elif "data:image/png" in image_data:
                        format_type = "png"
                    
                    
                    # Create the filename with correct extension
                    image_filename = f"{LOGS_IMAGES_DIR}/review_image_{timestamp}_{file_id}.{format_type}"
                    
                    # Direct approach to save the image
                    success = False
                    
                    # Method 1: Using regex and standard base64 decode (with padding if needed)
                    try:
                        import re
                        image_data_cleaned = re.sub(r'^data:image/[^;]+;base64,', '', image_data)
                        
                        # Add padding if needed
                        padding_needed = 4 - len(image_data_cleaned) % 4
                        if padding_needed < 4:
                            image_data_cleaned += "=" * padding_needed
                            
                        image_binary = base64.b64decode(image_data_cleaned)
                        
                        # Write to file
                        with open(image_filename, "wb") as f:
                            f.write(image_binary)
                            
                        success = True
                    except Exception as method1_error:
                        print(f"Method 1 failed: {str(method1_error)}")
                    
                    # Method 2: Use a third-party library or direct file download
                    if not success:
                        try:
                            # For this method, we'll use PIL if available
                            from PIL import Image
                            import io
                            
                            # Try to extract the base64 part
                            try:
                                _, encoded = image_data.split(",", 1)
                            except:
                                encoded = image_data
                                
                            # Decode the base64 data
                            binary_data = base64.b64decode(encoded)
                            
                            # Create an image from the binary data
                            image = Image.open(io.BytesIO(binary_data))
                            
                            # Save the image
                            image.save(image_filename)
                            success = True
                        except Exception as method2_error:
                            print(f"Method 2 failed: {str(method2_error)}")
                    
                except Exception as e:
                    print(f"Failed to save image: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    image_path = None
            
            # Create a log entry
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session_id,
                "validation_result": validation_result,
                "comments": comments,
                "bot_response": parsed_bot_response,
                "user_question": parsed_user_question,
                "image_path": image_path,
            }
            
            # Generate a unique log filename
            log_filename = f"{LOGS_REVIEWS_DIR}/review_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
            
            # Save the log
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=4)
            
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query, graph=graph)
        
        # Get response from output or messages
        response_text = ""
        if 'output' in response_data and response_data['output']:
            if hasattr(response_data['output'], 'content'):
                response_text = response_data['output'].content
            elif isinstance(response_data['output'], str):
                response_text = response_data['output']
        elif 'messages' in response_data and len(response_data['messages']) > 0:
            last_message = response_data['messages'][-1]
            if hasattr(last_message, 'content'):
                response_text = last_message.content
            
        # If we still don't have a response, provide a fallback
        if not response_text:
            response_text = "Thank you for your validation."

        # Check validation result (case insensitive)
        if validation_result.lower() in ['yes', 'true', '1', 'confirm']:
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": "Cảm ơn bạn đã phản hồi!" if not response_text else response_text
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_text
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "**Error processing validation:**",
                "response": f"An error occurred: {str(e)}"
            }
        )

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
