import streamlit as st
import requests
import time
import json
import base64
import os
from PIL import Image
import io
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL with more robust fallback
API_URL = os.environ.get("API_URL", "https://omr-api-service-510610499515.europe-west1.run.app")
logger.info(f"Using API URL: {API_URL}")

st.set_page_config(
    page_title="Sheet Music to MIDI Converter",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "visualization_image" not in st.session_state:
    st.session_state.visualization_image = None
if "detection_data" not in st.session_state:
    st.session_state.detection_data = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None
if "selected_tempo" not in st.session_state:
    st.session_state.selected_tempo = 79


def create_download_link(content, filename, link_text):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:audio/midi;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def process_sheet_music(file, tempo=79):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        logger.info(f"Created temporary file at {temp_file_path}")

        # Visualization request
        try:
            files = {"file": (file.name, open(temp_file_path, "rb"), "image/png")}
            logger.info("Sending visualization request to API")
            visualization_response = requests.post(f"{API_URL}/detect-visualize", files=files)
            if visualization_response.status_code != 200:
                visualization_image = None
                logger.error(f"Visualization API error: {visualization_response.status_code}")
            else:
                visualization_image = visualization_response.content
                logger.info("Successfully received visualization image")
        except Exception as e:
            logger.error(f"Visualization request failed: {str(e)}")
            visualization_image = None

        # Process request
        try:
            files = {"file": (file.name, open(temp_file_path, "rb"), "image/png")}
            logger.info(f"Sending process request to API with tempo {tempo}")
            # Add tempo as a parameter
            response = requests.post(f"{API_URL}/process?tempo={tempo}", files=files)

            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}")

                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    logger.error(f"Raw response: {response.text[:500]}")

                os.unlink(temp_file_path)
                return None, visualization_image

            logger.info("Successfully received process response")

        except Exception as e:
            logger.error(f"Process request failed: {str(e)}")
            os.unlink(temp_file_path)
            return None, visualization_image

        # Clean up and return
        os.unlink(temp_file_path)
        result = response.json()
        return result, visualization_image

    except Exception as e:
        logger.error(f"Error in process_sheet_music: {str(e)}")
        return None, None


def poll_job_status(job_id):
    status = "queued"
    try:
        while status in ["queued", "processing"]:
            logger.info(f"Checking status of job {job_id}")
            response = requests.get(f"{API_URL}/status/{job_id}")

            if response.status_code != 200:
                logger.error(f"Failed to check job status: {response.status_code}")
                return None

            data = response.json()
            status = data.get("status", "unknown")
            logger.info(f"Current job status: {status}")

            if status == "completed":
                return data
            elif status == "failed":
                error_msg = data.get('error', 'Unknown error')
                logger.error(f"Processing failed: {error_msg}")
                return None

            time.sleep(2)
    except Exception as e:
        logger.error(f"Error in poll_job_status: {str(e)}")
        return None

    return None


def visualize_detections(detection_data):
    try:
        # If detection_data is a list (raw format from API), convert it to expected dictionary format
        if isinstance(detection_data, list):
            # Simple conversion from list to expected dictionary structure
            structured_data = {
                "staves": [{
                    "type": "Staff",
                    "elements": detection_data
                }]
            }
            detection_data = structured_data

        # Continue with original code
        staves = detection_data.get("staves", [])

        if st.session_state.visualization_image:
            with st.expander("Detection Visualization", expanded=True):
                st.image(st.session_state.visualization_image, caption="Sheet Music with Detection Boxes", use_column_width=True)

        st.subheader("Detected Staves")
        for i, staff in enumerate(staves):
            staff_type = staff.get("type", f"Staff {i+1}")
            elements = staff.get("elements", [])

            with st.expander(f"{staff_type.title()}: {len(elements)} elements"):
                element_data = []
                for j, elem in enumerate(elements):
                    element_data.append({
                        "Class": elem.get("class_name", "unknown"),
                        "Position": f"({elem.get('x_center', elem.get('x', 0)):.1f}, {elem.get('y_center', elem.get('y', 0)):.1f})",
                        "Confidence": f"{elem.get('confidence', 0):.2f}"
                    })

                if element_data:
                    st.table(element_data)
    except Exception as e:
        logger.error(f"Error in visualize_detections: {str(e)}")
        st.error(f"Error visualizing detections: {str(e)}")


def embed_midi_player(midi_url):
    try:
        st.subheader("MIDI Playback")

        logger.info(f"Fetching MIDI from {midi_url}")
        response = requests.get(midi_url)
        if response.status_code != 200:
            logger.error(f"Failed to play MIDI: {response.status_code}")
            st.error(f"Failed to play MIDI: {response.status_code}")
            st.warning("The MIDI file was created but couldn't be played directly. Please download it and use a MIDI player.")
            return

        midi_content = response.content
        logger.info(f"Successfully retrieved MIDI content of size {len(midi_content)} bytes")

        download_link = create_download_link(
            midi_content,
            "sheet_music.mid",
            "⬇️ Download MIDI File"
        )
        st.markdown(download_link, unsafe_allow_html=True)

        st.write("MIDI Preview:")

        midi_b64 = base64.b64encode(midi_content).decode()

        # Check if the CDN resources load correctly
        st.info("Loading MIDI player... If playback doesn't work, you can still download the MIDI file.")

        midi_player_html = f"""
        <script
            src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.77,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"
        ></script>

        <midi-player
            src="data:audio/midi;base64,{midi_b64}"
            sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
            visualizer="#myVisualizer"
            style="width:100%;">
        </midi-player>

        <midi-visualizer
            id="myVisualizer"
            type="piano-roll"
            style="width:100%;height:200px">
        </midi-visualizer>
        """

        st.components.v1.html(midi_player_html, height=350)
    except Exception as e:
        logger.error(f"Error in embed_midi_player: {str(e)}")
        st.error(f"Error embedding MIDI player: {str(e)}")


# Main app
logger.info("Starting app")

st.title("🎵 Sheet Music to MIDI Converter")
st.write("Upload a sheet music image to convert it to a playable MIDI file.")

# File uploader
uploaded_file = st.file_uploader("Choose a sheet music image", type=["png", "jpg", "jpeg"])

# Check if a new file was uploaded
if uploaded_file is not None:
    # Reset state when a new file is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.file_processed = False
        st.session_state.job_id = None
        st.session_state.visualization_image = None
        st.session_state.detection_data = None
        st.session_state.processing_error = None
        logger.info(f"New file uploaded: {uploaded_file.name}")

    # Display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sheet Music", use_column_width=True)
    except Exception as e:
        logger.error(f"Error opening uploaded image: {str(e)}")
        st.error(f"Error opening image: {str(e)}")

    # Tempo slider
    st.session_state.selected_tempo = st.slider(
        "Select Tempo (BPM)",
        min_value=40,
        max_value=200,
        value=st.session_state.selected_tempo,
        step=1
    )

    # Convert button with clear processing steps
    if st.button("Convert to MIDI"):
        # Create empty containers for progress display
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Status: Starting conversion process...")
        progress_bar.progress(10)

        try:
            # Process the sheet music
            status_text.text("Status: Processing image...")
            progress_bar.progress(20)

            # Only process if not already processed
            if not st.session_state.file_processed:
                result, visualization_image = process_sheet_music(uploaded_file, tempo=st.session_state.selected_tempo)

                if result is None:
                    st.error("Failed to process sheet music. Please try again.")
                    status_text.text("Status: Failed to process image ❌")
                    progress_bar.progress(100)
                else:
                    # Store results in session state
                    st.session_state.job_id = result.get("job_id")
                    st.session_state.visualization_image = visualization_image
                    st.session_state.file_processed = True

            if st.session_state.file_processed and st.session_state.job_id:
                job_id = st.session_state.job_id

                # Poll for job completion
                status_text.text(f"Status: Analyzing music elements (Job ID: {job_id})...")
                progress_bar.progress(40)

                job_result = poll_job_status(job_id)

                if job_result:
                    progress_bar.progress(60)
                    status_text.text("Status: Fetching detection data...")

                    # Fetch detection data
                    detection_response = requests.get(f"{API_URL}/preview/{job_id}")
                    if detection_response.status_code == 200:
                        st.session_state.detection_data = detection_response.json()
                        logger.info(f"Successfully fetched detection data for job {job_id}")

                        # Display detection visualization
                        progress_bar.progress(80)
                        status_text.text("Status: Generating MIDI...")

                        if st.session_state.detection_data:
                            # Display the results
                            progress_bar.progress(100)
                            status_text.text("Status: Completed! ✅")

                            st.success(f"Successfully converted your sheet music! Found {job_result.get('element_count', 0)} musical elements.")

                            # Show detection visualization
                            visualize_detections(st.session_state.detection_data)

                            # Show MIDI player
                            midi_url = f"{API_URL}/download/{job_id}"
                            embed_midi_player(midi_url)
                    else:
                        logger.error(f"Failed to get detection preview: {detection_response.status_code}")
                        st.error(f"Failed to get detection preview: {detection_response.status_code}")
                        status_text.text("Status: Failed to get detection data ❌")
                        progress_bar.progress(100)
                else:
                    logger.error(f"Job {job_id} failed or timed out")
                    st.error(f"Job {job_id} failed or timed out. Please try again.")
                    status_text.text("Status: Processing failed ❌")
                    progress_bar.progress(100)
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            status_text.text("Status: Error occurred ❌")
            progress_bar.progress(100)

st.markdown("---")
st.markdown("Made with ❤️ by the team behind | [Optical Music Recognition](https://github.com/cmullie/omr)")

logger.info("App rendered successfully")
