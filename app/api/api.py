import os
import json
import uuid
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Get project root and add to path
PROJECT_ROOT = Path(__file__).parents[2].absolute()
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Import functions - try both import paths
try:
    from src.inference.predict import load_model, load_class_mapping, run_inference
    from src.midi_converter import convert_detections_to_midi
except ImportError:
    from inference.predict import load_model, load_class_mapping, run_inference
    from midi_converter import convert_detections_to_midi

# Set up paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "models", "best.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "best.pt")  # fallback path

CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "raw_data", "class_mapping.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "results")
TEMP_DIR = os.path.join(tempfile.gettempdir(), "sheet_midi")

# Create needed directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="OMR to MIDI API",
    description="Convert sheet music images to MIDI files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for jobs
JOBS = {}

# Global model variables
model = None
class_mapping = None

# Function to get or load model
def get_model():
    global model, class_mapping

    if model is None:
        # Load model
        print(f"Loading model from {MODEL_PATH}")
        try:
            model = load_model(MODEL_PATH)
            print(f"Model loaded: {model is not None}")

            # Load class mapping
            class_mapping = load_class_mapping(CLASS_MAPPING_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")

    return model, class_mapping

# Try to load model at startup
try:
    model, class_mapping = get_model()
except Exception as e:
    print(f"Failed to load model at startup: {e}")

def visualize_detection_with_labels(image_path, detection_data, output_path):
    if not detection_data:
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    vis_img = img.copy()

    detection_data.sort(key=lambda x: x['x_center'])

    for det in detection_data:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = det['class_name']
        conf = f"{det['confidence']:.2f}"
        display_label = f"{label} ({conf})"

        if len(display_label) > 20:
            display_label = display_label[:17] + "..."

        text_x = max(5, min(x1, w-100))
        text_y = max(25, min(y1-10, h-10))

        text_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(vis_img, (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
        cv2.putText(vis_img, display_label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    return True

def process_sheet_music(image_path, job_id):
    try:
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        JOBS[job_id]["status"] = "processing"

        # Get model
        model, class_mapping = get_model()

        # Check if model loaded
        if model is None:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "Model not loaded"
            return

        # Run detection
        detection_path = os.path.join(job_dir, f"{job_id}_detection.json")
        detections, _ = run_inference(model, image_path, conf_threshold=0.1, class_mapping=class_mapping)

        if not detections:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "No music notation detected"
            return

        # Simple structure for output
        detection_result = {
            "metadata": {"image": os.path.basename(image_path)},
            "staves": [{"elements": detections}]
        }

        # Save detection results
        with open(detection_path, 'w') as f:
            json.dump(detection_result, f, indent=2)

        # Create visualization
        vis_path = os.path.join(job_dir, f"{job_id}_visualization.png")
        visualize_detection_with_labels(image_path, detections, vis_path)

        # Convert to MIDI
        midi_path = os.path.join(job_dir, f"{job_id}.mid")
        midi_file = convert_detections_to_midi(detection_path, midi_path)

        if not midi_file:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = "MIDI conversion failed"
            return

        # Update job status
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["detection_file"] = detection_path
        JOBS[job_id]["visualization_file"] = vis_path
        JOBS[job_id]["midi_file"] = midi_path

        # Count elements
        element_count = len(detections)
        JOBS[job_id]["element_count"] = element_count

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        print(f"Error processing job {job_id}: {e}")

@app.post("/detect-visualize")
async def detect_visualize(file: UploadFile = File(...)):
    # Get model
    model, class_mapping = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")
    vis_output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.png")

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Run detection
        detections, _ = run_inference(model, temp_file_path, conf_threshold=0.1, class_mapping=class_mapping)

        if not detections:
            raise HTTPException(status_code=400, detail="No music notation detected")

        # Create visualization
        success = visualize_detection_with_labels(temp_file_path, detections, vis_output_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create visualization")

        # Return image
        return FileResponse(
            vis_output_path,
            media_type="image/png",
            filename=f"detection_{os.path.basename(file.filename)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/detect-json")
async def detect_json(file: UploadFile = File(...)):
    # Get model
    model, class_mapping = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}")

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Run detection
        detections, _ = run_inference(model, temp_file_path, conf_threshold=0.1, class_mapping=class_mapping)

        if not detections:
            return JSONResponse(content={"message": "No music notation detected", "detections": []})

        # Return detection data
        return JSONResponse(content=detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/process")
async def process_sheet(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create job directory
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploaded file
    image_path = os.path.join(job_dir, f"input{os.path.splitext(file.filename)[1]}")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize job status
    JOBS[job_id] = {
        "status": "queued",
        "image_path": image_path,
        "filename": file.filename
    }

    # Process in background or synchronously
    if background_tasks:
        background_tasks.add_task(process_sheet_music, image_path, job_id)
    else:
        process_sheet_music(image_path, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Processing started"
    }

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found")

    job = JOBS[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename", "unknown")
    }

    if job["status"] == "completed":
        response["element_count"] = job.get("element_count", 0)
        response["download_url"] = f"/download/{job_id}"
        response["visualization_url"] = f"/visualization/{job_id}"
        response["preview_url"] = f"/preview/{job_id}"
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return response

@app.get("/download/{job_id}")
async def download_midi(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed")

    if "midi_file" not in job or not os.path.exists(job["midi_file"]):
        raise HTTPException(status_code=404, detail=f"MIDI file not found")

    return FileResponse(
        path=job["midi_file"],
        filename=f"sheet_music_{job_id}.mid",
        media_type="audio/midi"
    )

@app.get("/visualization/{job_id}")
async def get_visualization(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed")

    if "visualization_file" not in job or not os.path.exists(job["visualization_file"]):
        raise HTTPException(status_code=404, detail=f"Visualization not found")

    return FileResponse(
        path=job["visualization_file"],
        media_type="image/png",
        filename=f"detection_{job_id}.png"
    )

@app.get("/preview/{job_id}")
async def preview_detection(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found")

    job = JOBS[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed")

    if "detection_file" not in job or not os.path.exists(job["detection_file"]):
        raise HTTPException(status_code=404, detail=f"Detection file not found")

    with open(job["detection_file"], "r") as f:
        detection_data = json.load(f)

    return detection_data

@app.get("/health")
def health_check():
    model_loaded = model is not None
    # Try loading if not loaded yet
    if not model_loaded:
        get_model()
        model_loaded = model is not None

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "jobs_count": len(JOBS)
    }

@app.get("/")
def read_root():
    return {
        "status": "online",
        "endpoints": [
            {"path": "/", "method": "GET"},
            {"path": "/health", "method": "GET"},
            {"path": "/detect-visualize", "method": "POST"},
            {"path": "/detect-json", "method": "POST"},
            {"path": "/process", "method": "POST"},
            {"path": "/status/{job_id}", "method": "GET"},
            {"path": "/download/{job_id}", "method": "GET"},
            {"path": "/visualization/{job_id}", "method": "GET"},
            {"path": "/preview/{job_id}", "method": "GET"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
