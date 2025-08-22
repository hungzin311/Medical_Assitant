import os
import uuid
from typing import Optional, List
import time
import json
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from werkzeug.utils import secure_filename
from config import Config
from agents.agent_decision import process_query, create_agent_graph
from proxy_setting import *
from agents.patient_db_agent import PatientQueryEngine
from agents.patient_db_agent.patient_intake_form import PatientIntakeForm
# Load configuration
config = Config()

#Set proxy 
set_proxy()

graph = create_agent_graph()

patient_query_engine = PatientQueryEngine(config)

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
POLYP_SEGMENTATION_OUTPUT = "uploads/polyp_seg_output"
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
    """Render the dashboard interface"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/chat")
async def chat_interface(request: Request):
    """Render the AI chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/patient-intake")
async def patient_intake(request: Request):
    print("=== Patient Intake Endpoint Called ===")
    try:
        # Get raw JSON data first
        raw_data = await request.json()
        # Try to validate with Pydantic model
        try:
            patient_data = PatientIntakeForm(**raw_data)
        except Exception as validation_error:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "validation_error",
                    "message": "Dữ liệu không hợp lệ",
                    "error": str(validation_error),
                    "received_data": raw_data
                }
            )
        
        # Ingest patient form
        patient_query_engine.ingest_patient_form(patient_data)
        
        # Generate patient_id if not provided
        if not patient_data.patient_id:
            patient_data.patient_id = f"patient_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        red_flags_present = {}
        if patient_data.red_flags and isinstance(patient_data.red_flags, list):
            for flag in patient_data.red_flags:
                red_flags_present[flag] = True
        
        if red_flags_present:
            print(f"⚠️ RED FLAGS DETECTED: {', '.join(red_flags_present)}")
        
        return {
            "status": "success",
            "message": "Thông tin bệnh nhân đã được ghi nhận thành công",
            "patient_id": patient_data.patient_id,
            "red_flags_detected": list(red_flags_present.keys()),
        }
        
    except Exception as e:
        print(f"Error processing patient intake: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Có lỗi xảy ra khi xử lý thông tin bệnh nhân",
                "error": str(e)
            }
        )

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """Process a chat message"""
    try:
        # Get the message from the request
        message = request.message
        
        # Process the query
        response_data = process_query(message, graph=graph)
        
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
        if response_data["agent_name"] == "POLYP_SEGMENTATION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(POLYP_SEGMENTATION_OUTPUT, "polyp_seg_image_output.jpg")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/polyp_seg_output/polyp_seg_image_output.jpg"
            else:
                print("Polyp Segmentation Output path does not exist.")
                
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
            base64_path = None
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
                    
                    # Method 3: Last resort - just save the base64 string
                    if not success:
                        pass
                    # Make the image path relative for storage in JSON
                    image_path = image_filename
                    relative_image_path = image_path.replace("\\", "/")
                    if relative_image_path.startswith("./"):
                        relative_image_path = relative_image_path[2:]
                    
                    # Update the user_question with the image path if it's a parsed object
                    if parsed_user_question and isinstance(parsed_user_question, dict):
                        parsed_user_question["saved_image_path"] = relative_image_path
                    
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