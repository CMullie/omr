# Optical Music Recognition (OMR) System
A complete system for detecting musical notation in sheet music images and converting them to MIDI files. This project uses YOLOv5 for object detection and converts the detected notation into playable MIDI files.
## Table of Contents
[Overview](#overview)
[System Architecture](#system-architecture)
[Installation](#installation)
[Usage](#usage)
[API Documentation](#api-documentation)
[Deployment](#deployment)
[Troubleshooting](#troubleshooting)
## Overview
This Optical Music Recognition (OMR) system processes sheet music images to identify musical notation elements and converts them into MIDI files. The system consists of:
**Detection Model**: YOLOv5-based neural network trained to recognize musical symbols
**API Service**: FastAPI backend that processes images and serves results
**Streamlit Frontend**: User-friendly interface for uploading images and playing generated MIDI files
## System Architecture
```
├── app/                  # Application code
│   └── api/              # FastAPI service
│       └── api.py        # API endpoints
├── configs/              # Configuration files
│   ├── inference_config.yaml  # Inference settings
│   └── training_config.yaml   # Training settings
├── raw_data/             # Data directory (not in repo)
│   ├── images/           # Sheet music images
│   └── labels/           # Training annotations
├── src/                  # Core code modules
│   ├── inference/        # Prediction code
│   │   └── predict.py    # Inference pipeline
│   ├── training/         # Training code
│   │   └── train.py      # Training pipeline
│   └── midi_converter.py # MIDI conversion logic
├── output/               # Model outputs (not in repo)
│   └── models/           # Trained models
├── streamlit_app.py      # Streamlit frontend
├── setup_directories.py  # Directory setup utility
├── requirements.txt      # Dependencies
└── README.md             # This file
```
## Installation
### Prerequisites
Python 3.8+
Git
Google Cloud SDK (for deployment)
### Setup
Clone the repository:
   ```bash
   git clone https://github.com/yourusername/omr-system.git
   cd omr-system
   ```
Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Set up directories:
   ```bash
   python setup_directories.py
   ```
Download or train a model:
For training: Follow instructions in the [Training](#training) section
For using a pre-trained model: Place the model weights at `output/models/best.pt`
## Usage
### Running the API Locally
```bash
uvicorn app.api.api:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at http://localhost:8000 ### Running the Streamlit Frontend Locally
```bash
streamlit run streamlit_app.py
```
The Streamlit app will be available at http://localhost:8501 ### Training
To train the model on Google Cloud Platform:
Set up a VM with GPU:
   ```bash
   gcloud compute instances create omr-training \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --boot-disk-size=100GB \
     --image-family=debian-11 \
     --image-project=debian-cloud
   ```
SSH into the VM:
   ```bash
   gcloud compute ssh omr-training
   ```
Clone the repository and setup:
   ```bash
   git clone https://github.com/yourusername/omr-system.git cd omr-system
   pip install -r requirements.txt
   ```
Prepare your dataset in the `raw_data` directory with the following structure:
   ```
   raw_data/
   ├── images/
   │   ├── train/  # Training images (PNG)
   │   └── val/    # Validation images (PNG)
   ├── labels/
   │   ├── train/  # Training labels (YOLO format text files)
   │   └── val/    # Validation labels (YOLO format text files)
   └── class_mapping.csv  # Maps between class IDs and names
   ```
Run training:
   ```bash
   python src/training/train.py
   ```
## API Documentation
### Endpoints
`POST /detect-visualize`: Detect and visualize musical notation in an image
Input: Form data with `file` containing the image
Output: PNG image with detection boxes
`POST /detect-json`: Detect musical notation and return raw JSON data
Input: Form data with `file` containing the image
Output: JSON with detection data
`POST /process`: Process a sheet music image and convert to MIDI
Input: Form data with `file` containing the image
Output: JSON with job ID and status
`GET /status/{job_id}`: Check processing status
Output: JSON with job status and results when complete
`GET /download/{job_id}`: Download generated MIDI file
Output: MIDI file
`GET /visualization/{job_id}`: Get visualization image
Output: PNG image with detection boxes
`GET /preview/{job_id}`: Get JSON detection results
Output: JSON with detection data
## Deployment
### Deploying the API to Google Cloud Run
Build and push the Docker image:
   ```bash
   # Build image
   docker build -t gcr.io/your-project-id/omr-api . # Push to Google Container Registry
   docker push gcr.io/your-project-id/omr-api ```
Deploy to Cloud Run:
   ```bash
   gcloud run deploy omr-api-service \
     --image gcr.io/your-project-id/omr-api \ --platform managed \
     --region europe-west1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 5m
   ```
### Deploying the Streamlit App to Streamlit Cloud
Push your code to GitHub.
Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud).
Add environment variables:
`API_URL`: The URL of your deployed API (e.g., `https://omr-api-service-xxxxx.run.app`)
## Troubleshooting ### Common Issues
**Model not found**: Ensure you have the model weights file at `output/models/best.pt`
**Cloud Run deployment failures**: Make sure your API properly handles the `PORT` environment variable
**MIDI player not working**: Try downloading the MIDI file and playing it with a local player
**Images not processing**: Check that your images are in a supported format (PNG, JPEG)
### Debug Tips
Check the API health endpoint:
   ```bash
   curl https://your-api-url/health
  ```
Look at the logs for more information:
   ```bash
   # For Cloud Run
   gcloud logs read --limit=50 --service=omr-api-service
   # For local API
   uvicorn app.api.api:app --log-level=debug
   ```
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgements
[YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model
[FastAPI](https://fastapi.tiangolo.com/) for the API framework
[Streamlit](https://streamlit.io/) for the frontend framework
