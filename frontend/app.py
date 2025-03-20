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
API_URL = os.environ.get("API_URL", "https://omr-api-service-yolo-510610499515.europe-west1.run.app/")
logger.info(f"Using API URL: {API_URL}")

st.set_page_config(
    page_title="Sheet Music to MIDI Converter",
    page_icon="🎵",
    layout="wide"
)

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
                st.error(f"Visualization failed: {visualization_response.status_code}")
                logger.error(f"Visualization API error: {visualization_response.status_code}")
            else:
                visualization_image = visualization_response.content
                logger.info("Successfully received visualization image")
        except Exception as e:
            logger.error(f"Visualization request failed: {str(e)}")
            visualization_image = None
            st.error(f"Visualization request failed: {str(e)}")

        # Process request
        try:
            files = {"file": (file.name, open(temp_file_path, "rb"), "image/png")}
            logger.info(f"Sending process request to API with tempo {tempo}")
            # Add tempo as a parameter
            response = requests.post(f"{API_URL}/process?tempo={tempo}", files=files)

            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}")
                st.error(f"API request failed: {response.status_code}")

                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    logger.error(f"Raw response: {response.text[:500]}")

                os.unlink(temp_file_path)
                return None

            logger.info("Successfully received process response")

        except Exception as e:
            logger.error(f"Process request failed: {str(e)}")
            st.error(f"Process request failed: {str(e)}")
            os.unlink(temp_file_path)
            return None

        # Clean up and return
        os.unlink(temp_file_path)
        result = response.json()
        result["visualization_image"] = visualization_image
        return result

    except Exception as e:
        logger.error(f"Error in process_sheet_music: {str(e)}")
        st.error(f"An error occurred during processing: {str(e)}")
        return None

def poll_job_status(job_id):
    status = "queued"
    try:
        while status in ["queued", "processing"]:
            logger.info(f"Checking status of job {job_id}")
            response = requests.get(f"{API_URL}/status/{job_id}")

            if response.status_code != 200:
                logger.error(f"Failed to check job status: {response.status_code}")
                st.error(f"Failed to check job status: {response.status_code}")
                return None

            data = response.json()
            status = data.get("status", "unknown")
            logger.info(f"Current job status: {status}")

            if status == "completed":
                return data
            elif status == "failed":
                error_msg = data.get('error', 'Unknown error')
                logger.error(f"Processing failed: {error_msg}")
                st.error(f"Processing failed: {error_msg}")
                return None

            progress_bar.progress(0.5)
            status_text.text(f"Status: {status.title()}...")

            time.sleep(2)
    except Exception as e:
        logger.error(f"Error in poll_job_status: {str(e)}")
        st.error(f"An error occurred while checking job status: {str(e)}")
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

        if "visualization_image" in st.session_state and st.session_state.visualization_image:
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

logger.info("Starting app")

if "visualization_image" not in st.session_state:
    st.session_state.visualization_image = None

st.title("🎵 Sheet Music to MIDI Converter")
st.write("Upload a sheet music image to convert it to a playable MIDI file.")

uploaded_file = st.file_uploader("Choose a sheet music image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sheet Music", use_column_width=True)
    except Exception as e:
        logger.error(f"Error opening uploaded image: {str(e)}")
        st.error(f"Error opening image: {str(e)}")

    # Add this tempo slider before the Convert button
    selected_tempo = st.slider(
        "Select Tempo (BPM)",
        min_value=40,
        max_value=200,
        value=79,  # Default matches the hardcoded value
        step=1
    )

    if st.button("Convert to MIDI"):
        with st.spinner("Processing your sheet music..."):
            progress_bar = st.progress(0.2)
            status_text = st.empty()
            status_text.text("Status: Uploading image...")

            try:
                job_response = process_sheet_music(uploaded_file, tempo=selected_tempo)


                if job_response:
                    job_id = job_response.get("job_id")
                    logger.info(f"Job created with ID: {job_id}")

                    if "visualization_image" in job_response and job_response["visualization_image"]:
                        st.session_state.visualization_image = job_response["visualization_image"]
                        logger.info("Stored visualization image in session state")

                    status_text.text(f"Status: Processing job {job_id}...")

                    result = poll_job_status(job_id)

                    if result:
                        progress_bar.progress(1.0)
                        status_text.text("Status: Completed! ✅")
                        logger.info("Job completed successfully")

                        st.success(f"Successfully converted your sheet music! Found {result.get('element_count', 0)} musical elements.")

                        try:
                            logger.info(f"Requesting preview for job {job_id}")
                            detection_response = requests.get(f"{API_URL}/preview/{job_id}")
                            if detection_response.status_code == 200:
                                detection_data = detection_response.json()
                                visualize_detections(detection_data)
                                midi_url = f"{API_URL}/download/{job_id}"
                                embed_midi_player(midi_url)
                            else:
                                logger.error(f"Failed to get detection preview: {detection_response.status_code}")
                                st.error(f"Failed to get detection preview: {detection_response.status_code}")
                        except Exception as e:
                            logger.error(f"Error getting detection preview: {str(e)}")
                            st.error(f"Error getting detection preview: {str(e)}")
                else:
                    progress_bar.progress(1.0)
                    status_text.text("Status: Failed ❌")
                    logger.error("Job response was None")
            except Exception as e:
                progress_bar.progress(1.0)
                status_text.text("Status: Failed ❌")
                logger.error(f"Error during job processing: {str(e)}")
                st.error(f"Error during job processing: {str(e)}")

st.markdown("---")
st.markdown("Made with ❤️ by the team behind | [Optical Music Recognition](https://github.com/cmullie/omr)")

logger.info("App rendered successfully")
