import os
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO
import json
import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename

from config import Config
from agents.agent_decision import process_query
from proxy_setting import *
# Load configuration
config = Config()

#Set proxy 
set_proxy()

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
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
LOGS_DIR = "logs"
LOGS_IMAGES_DIR = "logs/images"
LOGS_REVIEWS_DIR = "logs/reviews"

# Create all required directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, 
                 LOGS_DIR, LOGS_IMAGES_DIR, LOGS_REVIEWS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    message: str
    image_data: Optional[str] = None
    conversation_history: List = []

@app.get("/")
async def root(request: Request):
    """Render the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """Process a chat message"""
    try:
        print('Processing chat request')
        # Get the message from the request
        message = request.message
        
        # Process the query
        response_data = process_query(message)
        
        # Extract the response from the last AI message
        response_text = ""
        agent_name = response_data.get('agent_name', 'ASSISTANT')
        
        # Try to get the response from the messages
        if 'messages' in response_data and len(response_data['messages']) > 0:
            last_message = response_data['messages'][-1]
            if hasattr(last_message, 'content'):
                response_text = last_message.content
        
        # If we couldn't get the response from messages, check if there's an output field
        if not response_text and 'output' in response_data:
            output = response_data['output']
            if hasattr(output, 'content'):
                response_text = output.content
            elif isinstance(output, str):
                response_text = output
        
        # If we still don't have a response, provide a fallback
        if not response_text:
            response_text = "I'm sorry, I couldn't process your request properly."
        
        return {
            "status": "success",
            "agent": agent_name,
            "response": response_text
        }
    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "agent": "System",
                "response": f"An error occurred: {str(e)}"
            }
        )

@app.post("/get-agent-status")
async def get_agent_status(request: QueryRequest):
    """Get the current agent that would process this query"""
    try:
        print('Getting agent status for query')
        # Get the message from the request
        message = request.message
        
        # Import the decision chain to get agent info without full processing
        from agents.agent_decision import AgentConfig, create_agent_graph
        from langchain_core.messages import HumanMessage
        from agents.agent_decision import init_agent_state
        
        # Initialize minimal state
        state = init_agent_state()
        state["current_input"] = message
        state["messages"] = [HumanMessage(content=message)]
        
        # Get the graph to access the decision chain
        graph = create_agent_graph()
        print('create successfully')
        # Run just the analyze_input and route_to_agent nodes to get the agent decision
        # without running the full agent processing
        try:
            # First run analyze_input
            state = graph.get_node("analyze_input").invoke(state)
            print('analyze_input successfully')
            # Then run route_to_agent to get the decision
            routing_result = graph.get_node("route_to_agent").invoke(state)
            print(routing_result)            
            # Extract the agent name from the routing result
            if isinstance(routing_result, dict) and "agent_state" in routing_result:
                agent_name = routing_result["agent_state"].get("agent_name", "Analyzing...")
                next_step = routing_result.get("next", "processing")
                
                # Map agent names to user-friendly messages
                agent_messages = {
                    "CONVERSATION_AGENT": "Processing your conversation request...",
                    "RAG_AGENT": "Searching medical knowledge database...",
                    "WEB_SEARCH_PROCESSOR_AGENT": "Searching the web for latest medical information...",
                    "SKIN_LESION_AGENT": "Analyzing skin lesion image...",
                    "GENERAL_MEDICAL_IMAGE_AGENT": "Analyzing medical image...",
                    "needs_validation": "Preparing response for validation..."
                }
                
                message = agent_messages.get(agent_name, f"Processing with {agent_name}...")
                
                return {
                    "status": "success",
                    "agent": agent_name,
                    "message": message
                }
            else:
                return {
                    "status": "success",
                    "agent": "Analyzing...",
                    "message": "Determining the best agent for your query..."
                }
                
        except Exception as decision_error:
            print(f"Error in decision logic: {decision_error}")
            # Fallback to basic analysis
            if "image" in message.lower() or "photo" in message.lower() or "picture" in message.lower():
                return {
                    "status": "success",
                    "agent": "Image Analysis Agent",
                    "message": "Preparing to analyze medical image..."
                }
            elif any(word in message.lower() for word in ["disease", "symptom", "treatment", "medicine", "drug"]):
                return {
                    "status": "success",
                    "agent": "Medical RAG Agent",
                    "message": "Searching medical knowledge database..."
                }
            else:
                return {
                    "status": "success",
                    "agent": "Conversation Agent",
                    "message": "Processing your conversation request..."
                }
        
    except Exception as e:
        print(f"Error getting agent status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "agent": "System",
                "message": "Error determining agent status"
            }
        )

@app.post("/get-processing-status")
async def get_processing_status(request: QueryRequest):
    """Get real-time processing status including agent transitions"""
    try:
        # This endpoint will be used to track agent transitions
        # For now, return a basic status that can be enhanced later
        return {
            "status": "processing",
            "current_agent": "Determining...",
            "message": "Processing your request...",
            "progress": 0
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Error getting processing status"
            }
        )

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
    # file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_path = f"./uploads/backend/{filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        print(f"File saved to {file_path}")
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['messages'][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")
        
        # Remove temporary file after sending
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")
        
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
        
        # Debug: Print received form data size
        form_data = await request.form()
        print(f"Received validation form data with {len(form_data)} fields")
        
        # Log reviews for further analysis
        if validation_result.lower() not in ['yes', 'true', '1', 'confirm']:
            # Safely parse JSON data
            parsed_bot_response = None
            parsed_user_question = None
            
            try:
                if bot_response:
                    print(f"Bot response length: {len(bot_response)}")
                    parsed_bot_response = json.loads(bot_response)
                    print("Successfully parsed bot_response JSON")
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
                    print(f"User question length: {len(user_question)}")
                    parsed_user_question = json.loads(user_question)
                    print("Successfully parsed user_question JSON")
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
            base64_path = None
            if image_data and image_data.startswith('data:image'):
                # Extract image data from base64 string
                try:
                    print("Attempting to save image from validation data...")
                    
                    # Generate a timestamp for the filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    file_id = uuid.uuid4().hex[:6]
                    
                    # Get image format from data URL
                    format_type = "png"
                    if "data:image/jpeg" in image_data:
                        format_type = "jpg"
                    elif "data:image/png" in image_data:
                        format_type = "png"
                    elif "data:image/gif" in image_data:
                        format_type = "gif"
                    
                    # Create the filename with correct extension
                    image_filename = f"{LOGS_IMAGES_DIR}/review_image_{timestamp}_{file_id}.{format_type}"
                    
                    # # Save raw base64 data as text first (as a backup)
                    # raw_backup_filename = f"{LOGS_IMAGES_DIR}/raw_base64_{timestamp}_{file_id}.txt"
                    # with open(raw_backup_filename, "w", encoding="utf-8") as f:
                    #     f.write(image_data)
                    # print(f"Saved raw base64 data as backup to {raw_backup_filename}")
                    
                    # Direct approach to save the image
                    success = False
                    
                    # Method 1: Using regex and standard base64 decode
                    try:
                        import re
                        header_pattern = r'^data:image/[^;]+;base64,'
                        image_data_cleaned = re.sub(header_pattern, '', image_data)
                        
                        # Debug info
                        print(f"Image data length after cleaning: {len(image_data_cleaned)}")
                        
                        # Decode base64 data
                        image_binary = base64.b64decode(image_data_cleaned)
                        print(f"Successfully decoded base64 data, length: {len(image_binary)}")
                        
                        # Write to file
                        with open(image_filename, "wb") as f:
                            f.write(image_binary)
                            
                        print(f"Successfully saved image to {image_filename}")
                        success = True
                    except Exception as method1_error:
                        print(f"Method 1 failed: {str(method1_error)}")
                        
                    # Method 2: Alternative approach with padding
                    if not success:
                        try:
                            print("Trying alternative method with padding...")
                            import re
                            image_data_cleaned = re.sub(r'^data:image/[^;]+;base64,', '', image_data)
                            
                            # Add padding if needed
                            padding_needed = 4 - len(image_data_cleaned) % 4
                            if padding_needed < 4:
                                image_data_cleaned += "=" * padding_needed
                                print(f"Added {padding_needed} padding characters")
                                
                            image_binary = base64.b64decode(image_data_cleaned)
                            
                            # Write to file
                            with open(image_filename, "wb") as f:
                                f.write(image_binary)
                                
                            print(f"Method 2: Successfully saved image to {image_filename}")
                            success = True
                        except Exception as method2_error:
                            print(f"Method 2 failed: {str(method2_error)}")
                    
                    # Method 3: Use a third-party library or direct file download
                    if not success:
                        try:
                            print("Trying method 3: Direct file save...")
                            # For this method, we'll use PIL if available
                            from PIL import Image
                            import io
                            
                            # Try to extract the base64 part
                            try:
                                header, encoded = image_data.split(",", 1)
                            except:
                                encoded = image_data
                                
                            # Decode the base64 data
                            binary_data = base64.b64decode(encoded)
                            
                            # Create an image from the binary data
                            image = Image.open(io.BytesIO(binary_data))
                            
                            # Save the image
                            image.save(image_filename)
                            print(f"Method 3: Successfully saved image using PIL to {image_filename}")
                            success = True
                        except Exception as method3_error:
                            print(f"Method 3 failed: {str(method3_error)}")
                    
                    # Method 4: Last resort - just save the base64 string
                    if not success:
                        # The original base64 is already saved, so we'll use that as our path
                        # image_filename = raw_backup_filename
                        # print(f"All decode methods failed. Using raw base64 file as image path: {image_filename}")
                        # success = True
                        pass
                    # Make the image path relative for storage in JSON
                    image_path = image_filename
                    relative_image_path = image_path.replace("\\", "/")
                    if relative_image_path.startswith("./"):
                        relative_image_path = relative_image_path[2:]
                    
                    print(f"Final image path: {image_path}")
                    
                    # Update the user_question with the image path if it's a parsed object
                    if parsed_user_question and isinstance(parsed_user_question, dict):
                        parsed_user_question["saved_image_path"] = relative_image_path
                        print(f"Added image path to user_question JSON: {relative_image_path}")
                    
                except Exception as e:
                    print(f"Failed to save image: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    image_path = None
                    base64_path = None
            
            # Create a log entry
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session_id,
                "validation_result": validation_result,
                "comments": comments,
                "bot_response": parsed_bot_response,
                "user_question": parsed_user_question,
                "image_path": image_path,
                "base64_path": base64_path
            }
            
            # Generate a unique log filename
            log_filename = f"{LOGS_REVIEWS_DIR}/review_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
            
            # Save the log
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=4)
            
            print(f"Review logged to {log_filename}")
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query)
        
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