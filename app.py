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
from agents.patient_db_agent.patient_form import PatientForm
import logging

logging.getLogger("httpx").disabled = True

# Load configuration
config = Config()

# Default patient when no login/session available
DEFAULT_PATIENT_ID = "PAT_001"

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


@app.get("/api/profile")
async def get_profile(patient_id: Optional[str] = None):
    pid = patient_id or DEFAULT_PATIENT_ID
    try:
        profile = patient_query_engine.get_patient_profile(pid)
        return {"status": "success", "patient_id": pid, "profile": profile or {}}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/patient/diseases")
async def list_patient_diseases(patient_id: Optional[str] = None):
    pid = patient_id or DEFAULT_PATIENT_ID
    try:
        diseases = patient_query_engine.get_patient_diseases(pid)
        return {"status": "success", "patient_id": pid, "diseases": diseases}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


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

        # Default patient id when not provided
        if not raw_data.get('patient_id'):
            raw_data['patient_id'] = DEFAULT_PATIENT_ID
        # Try to validate with Pydantic model
        try:
            patient_data = PatientForm(**raw_data)
            print(f"✅ Disease tracking form validated successfully for patient: {patient_data.patient_id}")
            
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
        # Update profile diseases
        if patient_data.primary_disease:
            try:
                patient_query_engine.add_diseases_to_profile(patient_data.patient_id, patient_data.primary_disease)
            except Exception as e:
                print(f"Failed to update profile diseases: {str(e)}")
        
        # Ensure patient_id present
        if not patient_data.patient_id:
            patient_data.patient_id = DEFAULT_PATIENT_ID

        red_flags_present = {}
        if patient_data.red_flags and isinstance(patient_data.red_flags, list):
            for flag in patient_data.red_flags:
                red_flags_present[flag] = True
        
        if red_flags_present:
            print(f"⚠️ RED FLAGS DETECTED: {', '.join(red_flags_present)}")
        
        return {
            "status": "success",
            "message": "Thông tin theo dõi bệnh đã được ghi nhận thành công",
            "patient_id": patient_data.patient_id,
            "visit_type": patient_data.visit_type,
            "disease_status": patient_data.disease_status,
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

@app.get("/api/blood-pressure-report")
async def get_blood_pressure_report(patient_id: str = "6"):
    """Get blood pressure report data for visualization"""
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Load appropriate CSV file
        if patient_id == "6":
            csv_file = "blood_pressure_patient6_7days.csv"
        else:
            csv_file = "blood_pressure_5patients_circadian_style.csv"
        
        # Read CSV
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        
        # Filter by patient_id
        df_patient = df[df['patient_id'] == int(patient_id)].copy()
        
        if len(df_patient) == 0:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Patient data not found"}
            )
        
        # Calculate overall summary
        summary = {
            "mean_systolic": round(df_patient['systolic'].mean(), 1),
            "mean_diastolic": round(df_patient['diastolic'].mean(), 1),
            "max_systolic": int(df_patient['systolic'].max()),
            "max_diastolic": int(df_patient['diastolic'].max()),
            "min_systolic": int(df_patient['systolic'].min()),
            "min_diastolic": int(df_patient['diastolic'].min()),
            "alerts_count": len(df_patient[(df_patient['systolic'] >= 140) | (df_patient['diastolic'] >= 90)]),
            "classification": classify_bp(df_patient['systolic'].mean(), df_patient['diastolic'].mean())
        }
        
        # Prepare chart data
        chart_data = {}
        
        # Check if 7-day data (Patient 6)
        if patient_id == "6" and len(df_patient) > 2880:
            # Prepare daily charts
            df_patient['date'] = df_patient['timestamp'].dt.date
            daily_charts = []
            
            for date, group in df_patient.groupby('date'):
                # Sample points for this day (max ~150 points per day)
                sample_rate = max(1, len(group) // 150)
                df_day_sampled = group.iloc[::sample_rate].copy()
                
                daily_charts.append({
                    "date": str(date),
                    "labels": df_day_sampled['timestamp'].dt.strftime('%H:%M').tolist(),
                    "systolic": df_day_sampled['systolic'].tolist(),
                    "diastolic": df_day_sampled['diastolic'].tolist(),
                    "mean_sys": round(group['systolic'].mean(), 1),
                    "mean_dia": round(group['diastolic'].mean(), 1)
                })
            
            chart_data = {
                "daily_charts": daily_charts
            }
        else:
            # Single day - one chart
            sample_rate = max(1, len(df_patient) // 500)
            df_sampled = df_patient.iloc[::sample_rate].copy()
            
            chart_data = {
                "labels": df_sampled['timestamp'].dt.strftime('%H:%M').tolist(),
                "systolic": df_sampled['systolic'].tolist(),
                "diastolic": df_sampled['diastolic'].tolist()
            }
        
        # Daily analysis (only for patient 6 with 7 days data)
        daily_analysis = []
        if patient_id == "6" and len(df_patient) > 2880:  # More than 1 day
            df_patient['date'] = df_patient['timestamp'].dt.date
            for day_idx, (date, group) in enumerate(df_patient.groupby('date')):
                alerts = len(group[(group['systolic'] >= 140) | (group['diastolic'] >= 90)])
                mean_sys = round(group['systolic'].mean(), 1)
                mean_dia = round(group['diastolic'].mean(), 1)
                
                status = 'high' if mean_sys >= 140 or mean_dia >= 90 else \
                        'elevated' if mean_sys >= 120 or mean_dia >= 80 else 'normal'
                
                assessment = f"{'Cao huyết áp' if status == 'high' else 'Bình thường' if status == 'normal' else 'Hơi cao'}"
                if day_idx == 0:
                    assessment += " - Trước điều trị"
                elif day_idx > 0:
                    assessment += " - Sau điều trị"
                
                daily_analysis.append({
                    "date": str(date),
                    "mean_sys": mean_sys,
                    "mean_dia": mean_dia,
                    "min_sys": int(group['systolic'].min()),
                    "max_sys": int(group['systolic'].max()),
                    "min_dia": int(group['diastolic'].min()),
                    "max_dia": int(group['diastolic'].max()),
                    "alerts": alerts,
                    "status": status,
                    "assessment": assessment
                })
        else:
            # Single day data
            daily_analysis.append({
                "date": str(df_patient['timestamp'].iloc[0].date()),
                "mean_sys": summary["mean_systolic"],
                "mean_dia": summary["mean_diastolic"],
                "min_sys": summary["min_systolic"],
                "max_sys": summary["max_systolic"],
                "min_dia": summary["min_diastolic"],
                "max_dia": summary["max_diastolic"],
                "alerts": summary["alerts_count"],
                "status": "high" if summary["mean_systolic"] >= 140 else "normal",
                "assessment": summary["classification"]
            })
        
        # Generate recommendations
        recommendations = generate_bp_recommendations(summary, daily_analysis, patient_id)
        
        return {
            "status": "success",
            "data": {
                "patient_id": patient_id,
                "summary": summary,
                "chart_data": chart_data,
                "daily_analysis": daily_analysis,
                "recommendations": recommendations
            }
        }
        
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Blood pressure data file not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def classify_bp(systolic, diastolic):
    """Classify blood pressure according to medical guidelines"""
    if systolic >= 180 or diastolic >= 120:
        return "Khủng hoảng tăng huyết áp"
    elif systolic >= 140 or diastolic >= 90:
        return "Tăng huyết áp độ 2"
    elif systolic >= 130 or diastolic >= 80:
        return "Tăng huyết áp độ 1"
    elif systolic >= 120 and diastolic < 80:
        return "Huyết áp tăng nhẹ"
    else:
        return "Huyết áp bình thường"

def generate_bp_recommendations(summary, daily_analysis, patient_id):
    """Generate medical recommendations based on BP data"""
    recommendations = []
    
    mean_sys = summary["mean_systolic"]
    mean_dia = summary["mean_diastolic"]
    alerts = summary["alerts_count"]
    
    # High BP recommendations
    if mean_sys >= 140 or mean_dia >= 90:
        recommendations.append({
            "priority": "high",
            "title": "Cần khám bác sĩ tim mạch",
            "description": "Huyết áp trung bình vượt ngưỡng cao (≥140/90 mmHg). Nên đặt lịch khám với bác sĩ chuyên khoa tim mạch trong tuần này."
        })
        recommendations.append({
            "priority": "high",
            "title": "Điều chỉnh chế độ ăn",
            "description": "Giảm lượng muối trong bữa ăn xuống dưới 5g/ngày. Tăng cường rau xanh, trái cây và ngũ cốc nguyên hạt."
        })
    
    # Treatment response (for patient 6)
    if patient_id == "6" and len(daily_analysis) > 1:
        day1_mean = daily_analysis[0]["mean_sys"]
        recent_mean = sum([d["mean_sys"] for d in daily_analysis[-3:]]) / 3
        
        if day1_mean >= 140 and recent_mean < 130:
            recommendations.append({
                "priority": "medium",
                "title": "Điều trị hiệu quả",
                "description": f"Huyết áp đã giảm từ {day1_mean} xuống {recent_mean:.1f} mmHg. Tiếp tục duy trì thuốc và lối sống hiện tại."
            })
            recommendations.append({
                "priority": "low",
                "title": "Theo dõi định kỳ",
                "description": "Đo huyết áp hàng ngày vào cùng thời điểm. Ghi nhận kết quả để báo cáo với bác sĩ."
            })
        else:
            recommendations.append({
                "priority": "medium",
                "title": "Tiếp tục theo dõi",
                "description": "Huyết áp đang ổn định. Duy trì lối sống lành mạnh và tuân thủ điều trị."
            })
    
    # Activity spike handling
    if alerts > 5 and mean_sys < 140:
        recommendations.append({
            "priority": "low",
            "title": "Quản lý hoạt động thể chất",
            "description": "Phát hiện nhiều đợt tăng huyết áp ngắn (có thể do vận động). Đây là bình thường nhưng nên tránh vận động quá sức."
        })
    
    # General advice
    recommendations.append({
        "priority": "low",
        "title": "Lối sống lành mạnh",
        "description": "Duy trì cân nặng hợp lý, tập thể dục đều đặn 30 phút/ngày, hạn chế rượu bia và không hút thuốc."
    })
    
    if len(recommendations) == 0:
        recommendations.append({
            "priority": "low",
            "title": "Huyết áp tốt",
            "description": "Huyết áp của bạn đang trong giới hạn bình thường. Tiếp tục duy trì lối sống hiện tại."
        })
    
    return recommendations

@app.post("/api/generate-bp-medical-report")
async def generate_bp_medical_report(request: Request):
    """Generate comprehensive medical report using LLM + medical guidelines"""
    try:
        import pandas as pd
        from llm_config import get_gemini_llm
        
        data = await request.json()
        patient_id = data.get('patient_id', '6')
        daily_analysis = data.get('daily_analysis', [])
        summary = data.get('summary', {})
        
        # Load CSV for detailed analysis
        if patient_id == "6":
            csv_file = "blood_pressure_patient6_7days.csv"
        else:
            csv_file = "blood_pressure_5patients_circadian_style.csv"
        
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        df_patient = df[df['patient_id'] == int(patient_id)].copy()
        
        # Prepare detailed context for LLM
        report_context = prepare_medical_context(df_patient, daily_analysis, summary, patient_id)
        
        # Generate report using LLM
        llm = get_gemini_llm(temperature=0.3)
        
        medical_report_prompt = f"""
Bạn là một hệ thống AI phân tích y khoa được huấn luyện trên các guidelines và tài liệu y khoa. Hãy tạo báo cáo phân tích dữ liệu huyết áp của bệnh nhân.

**QUAN TRỌNG - Định dạng báo cáo**:
- Đây là **BÁO CÁO PHÂN TÍCH TỰ ĐỘNG** được tạo bởi AI
- Phân tích dựa trên medical guidelines và clinical evidence
- Sử dụng ngôn ngữ khách quan, không tự nhận là bác sĩ
- KHÔNG ký tên bác sĩ hoặc thông tin cá nhân
- Sử dụng format: "Hệ thống phân tích...", "Dựa trên guidelines...", "AI analysis cho thấy..."

**NGUỒN THAM KHẢO** (để trích dẫn trong báo cáo):
- Guidelines từ American Heart Association (AHA) 2024
- European Society of Cardiology (ESC) Guidelines
- Tài liệu y khoa về quản lý tăng huyết áp
- Các nghiên cứu lâm sàng về treatment response

==================== THÔNG TIN BỆNH NHÂN ====================
{report_context}

==================== YÊU CẦU BÁO CÁO ====================

Hãy viết báo cáo theo cấu trúc sau (sử dụng Markdown format):

---
**📊 BÁO CÁO PHÂN TÍCH HUYẾT ÁP TỰ ĐỘNG**  
**Được tạo bởi: AI Medical Analysis System**  
**Dựa trên: AHA/ESC Guidelines & Medical Literature**  
**Thời gian tạo: [datetime now]**

---

## 1. TỔNG QUAN TÌNH TRẠNG

Hệ thống đã phân tích [số lượng] lần đo huyết áp trong 7 ngày. Dựa trên dữ liệu quan sát:
- Mô tả pattern huyết áp tổng quan
- Phân loại theo AHA/ESC guidelines (Normal/Elevated/Stage 1/Stage 2/Crisis)
- Đánh giá risk stratification

*Lưu ý: Đây là phân tích tự động, cần được bác sĩ xác nhận trước khi áp dụng lâm sàng.*

## 2. PHÂN TÍCH CHI TIẾT THEO GIAI ĐOẠN

### 2.1. Giai Đoạn Đầu (Ngày 1)
Hệ thống ghi nhận:
- Mức huyết áp và pattern
- So sánh với threshold từ medical guidelines
- Đánh giá nguy cơ

**Tham chiếu y khoa**: Theo AHA 2024 guidelines, huyết áp ≥140/90 mmHg được phân loại là Stage 2 Hypertension, đòi hỏi can thiệp y tế.

### 2.2. Giai Đoạn Theo Dõi (Ngày 2-7)
AI analysis cho thấy:
- Xu hướng thay đổi huyết áp
- Treatment response assessment
- Phân biệt activity-related spikes vs sustained elevation

**Tham chiếu y khoa**: Dựa trên ESC guidelines, giảm huyết áp >10 mmHg trong vòng 1 tuần cho thấy response tốt với can thiệp điều trị.

## 3. ĐÁNH GIÁ DỰA TRÊN TÀI LIỆU Y KHOA

### 3.1. Phân Tích Nhịp Sinh Học (Circadian Rhythm)
Hệ thống phân tích pattern 24 giờ:
- Đánh giá night-time dipping pattern
- So sánh với normal circadian variation

**Tham chiếu y khoa**: Nghiên cứu về circadian BP patterns cho thấy giảm 10-20% huyết áp vào ban đêm là bình thường. Non-dippers có nguy cơ cardiovascular cao hơn.

### 3.2. Đánh Giá Độ Biến Thiên Huyết Áp
AI analysis về variability:
- Phân tích standard deviation
- Phân biệt activity-related spikes
- Đánh giá consistency của measurements

**Tham chiếu y khoa**: Medical literature chỉ ra rằng BP variability cao (SD >15 mmHg) là yếu tố nguy cơ độc lập cho biến cố tim mạch.

### 3.3. Đánh Giá Đáp Ứng Điều Trị
Hệ thống so sánh:
- Baseline (Ngày 1) vs Current status (Ngày 7)
- Tốc độ và mức độ cải thiện
- Pattern ổn định của BP

**Tham chiếu y khoa**: Dựa trên clinical trials, bệnh nhân có response tốt trong tuần đầu điều trị có khả năng kiểm soát BP dài hạn tốt hơn.

## 4. NGƯỠNG THEO DÕI MỚI - DỰA TRÊN MEDICAL GUIDELINES

### 4.1. Personalized Thresholds
Hệ thống đề xuất ngưỡng cá nhân hóa dựa trên:
- Baseline BP của bệnh nhân
- Treatment response pattern quan sát được
- Standard guidelines (AHA/ESC 2024)
- Risk factors cá nhân

**Huyết Áp Tâm Thu (Systolic):**
- Alert Level 1 (Elevated): ≥[số cụ thể] mmHg
- Alert Level 2 (High): ≥[số cụ thể] mmHg  
- Alert Level 3 (Critical): ≥[số cụ thể] mmHg

**Huyết Áp Tâm Trương (Diastolic):**
- Alert Level 1: ≥[số cụ thể] mmHg
- Alert Level 2: ≥[số cụ thể] mmHg
- Alert Level 3: ≥[số cụ thể] mmHg

**Cơ sở khoa học**: Các ngưỡng được điều chỉnh dựa trên mức BP hiện tại của bệnh nhân (+10 mmHg cho elevated, +20 mmHg cho high) kết hợp với absolute thresholds từ guidelines.

## 5. KHUYẾN NGHỊ TỪ HỆ THỐNG

### 5.1. Quản Lý Thuốc
Hệ thống phân tích:
- Hiệu quả của phác đồ điều trị hiện tại
- Khuyến nghị về timing và dosage (cần bác sĩ xác nhận)

**Tham chiếu y khoa**: Dựa trên pharmacotherapy guidelines, điều chỉnh thuốc nên dựa trên home BP monitoring và target goals cá nhân.

### 5.2. Thay Đổi Lối Sống
AI khuyến nghị các can thiệp không dùng thuốc:
- Chế độ ăn DASH diet (giàu rau quả, ít muối)
- Giảm natri xuống <2g/ngày
- Vận động 150 phút/tuần (moderate intensity)
- Quản lý stress (yoga, meditation)

**Tham chiếu y khoa**: Clinical studies cho thấy lifestyle modifications có thể giảm SBP 10-15 mmHg, tương đương 1 loại thuốc hạ áp.

### 5.3. Lịch Trình Theo Dõi
Hệ thống đề xuất:
- Tái khám: [frequency based on risk]
- Xét nghiệm: Lipid panel, kidney function, ECG
- Monitoring targets: [specific goals]

## 6. CẢNH BÁO VÀ ĐIỀU KIỆN ALERT

Hệ thống sẽ cảnh báo khi:
- 🔴 SBP ≥ [threshold] mmHg (sustained >2 hours)
- 🔴 DBP ≥ [threshold] mmHg (sustained >2 hours)
- ⚠️ BP tăng đột ngột >30 mmHg trong 1 giờ
- 🚨 Triệu chứng: đau ngực, đau đầu dữ dội, khó thở, lú lẫn

**Khuyến nghị**: Các tình huống trên cần liên hệ bác sĩ hoặc cơ sở y tế ngay lập tức.

## 7. KẾT LUẬN

### Tóm Tắt Phân Tích
- Tình trạng hiện tại: [summary]
- Treatment response: [assessment]

### Tiên Lượng
Dựa trên pattern quan sát và evidence-based medicine, hệ thống đánh giá tiên lượng là [tốt/trung bình/cần theo dõi sát].

### Lưu Ý Quan Trọng
⚠️ **Disclaimer**: Đây là báo cáo phân tích tự động được tạo bởi AI system. Tất cả khuyến nghị cần được bác sĩ điều trị xem xét và xác nhận trước khi áp dụng. Không tự ý thay đổi thuốc hoặc liệu pháp điều trị.

---

**Thông Tin Báo Cáo:**
- Generated by: AI Medical Analysis System
- Data source: 7-day continuous BP monitoring
- Guidelines reference: AHA 2024, ESC 2024
- Analysis method: Pattern recognition + Clinical guidelines

---

**LƯU Ý KHI VIẾT**:
- Sử dụng số liệu cụ thể từ data context
- Trích dẫn guidelines một cách tự nhiên
- Threshold values phải reasonable và evidence-based  
- Language: Chuyên nghiệp, khách quan, dễ hiểu
- Format: Clean Markdown
- KHÔNG tự nhận là bác sĩ hoặc ký tên cá nhân
"""

        # Generate report
        response = llm.invoke(medical_report_prompt)
        medical_report = response.content
        
        # Extract threshold values from report for structured data
        thresholds = extract_thresholds_from_report(medical_report, daily_analysis)
        
        return {
            "status": "success",
            "data": {
                "report": medical_report,
                "thresholds": thresholds,
                "generated_at": pd.Timestamp.now().isoformat(),
                "patient_id": patient_id
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def prepare_medical_context(df_patient, daily_analysis, summary, patient_id):
    """Prepare detailed context for LLM"""
    
    context = f"""
Patient ID: {patient_id}
Monitoring Period: 7 days
Total Readings: {len(df_patient):,}
Sampling Frequency: Every 30 seconds

=== OVERALL STATISTICS ===
Mean Systolic: {summary.get('mean_systolic', 'N/A')} mmHg
Mean Diastolic: {summary.get('mean_diastolic', 'N/A')} mmHg
Max Systolic: {summary.get('max_systolic', 'N/A')} mmHg
Max Diastolic: {summary.get('max_diastolic', 'N/A')} mmHg
Min Systolic: {summary.get('min_systolic', 'N/A')} mmHg
Min Diastolic: {summary.get('min_diastolic', 'N/A')} mmHg
Threshold Crossings: {summary.get('alerts_count', 'N/A')} times (≥140/90)
Classification: {summary.get('classification', 'N/A')}

=== DAILY BREAKDOWN ===
"""
    
    for idx, day in enumerate(daily_analysis):
        context += f"""
Day {idx+1} ({day.get('date', 'N/A')}):
  - Mean BP: {day.get('mean_sys', 'N/A')}/{day.get('mean_dia', 'N/A')} mmHg
  - Range: {day.get('min_sys', 'N/A')}-{day.get('max_sys', 'N/A')}/{day.get('min_dia', 'N/A')}-{day.get('max_dia', 'N/A')} mmHg
  - Alert Events: {day.get('alerts', 'N/A')}
  - Status: {day.get('status', 'N/A')}
  - Assessment: {day.get('assessment', 'N/A')}
"""
    
    # Calculate variability
    df_patient['date'] = df_patient['timestamp'].dt.date
    if len(df_patient) > 2880:  # Multiple days
        context += "\n=== TREND ANALYSIS ===\n"
        for date, group in df_patient.groupby('date'):
            std_sys = group['systolic'].std()
            std_dia = group['diastolic'].std()
            context += f"Day {date}: Variability (SD) = Systolic {std_sys:.1f}, Diastolic {std_dia:.1f} mmHg\n"
    
    # Treatment response calculation
    if len(daily_analysis) > 1:
        day1_sys = daily_analysis[0].get('mean_sys', 0)
        day7_sys = daily_analysis[-1].get('mean_sys', 0)
        reduction = day1_sys - day7_sys
        context += f"\n=== TREATMENT RESPONSE ===\n"
        context += f"Systolic BP Reduction: {reduction:.1f} mmHg (from Day 1 to Day 7)\n"
        context += f"Percentage Improvement: {(reduction/day1_sys*100):.1f}%\n"
    
    return context

def extract_thresholds_from_report(report, daily_analysis):
    """Extract structured threshold values from generated report"""
    
    # For demo, create reasonable thresholds based on patient data
    # In production, this could use NLP to extract from report
    
    if len(daily_analysis) > 1:
        recent_days = daily_analysis[-3:]  # Last 3 days
        avg_sys = sum([d['mean_sys'] for d in recent_days]) / len(recent_days)
        avg_dia = sum([d['mean_dia'] for d in recent_days]) / len(recent_days)
    else:
        avg_sys = daily_analysis[0]['mean_sys']
        avg_dia = daily_analysis[0]['mean_dia']
    
    # Derive personalized thresholds
    thresholds = {
        "systolic": {
            "elevated": int(avg_sys + 10),
            "high": int(avg_sys + 20),
            "critical": 180
        },
        "diastolic": {
            "elevated": int(avg_dia + 8),
            "high": int(avg_dia + 15),
            "critical": 120
        },
        "monitoring_frequency": {
            "week_1_2": "2 times per day (morning & evening)",
            "week_3_4": "1 time per day",
            "after_1_month": "3-4 times per week"
        },
        "critical_times": [
            "6:00-9:00 AM (morning surge)",
            "6:00-9:00 PM (evening)",
            "Before taking medication"
        ]
    }
    
    return thresholds

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)