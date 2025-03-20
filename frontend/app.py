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
import threading
from queue import Queue

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

# Initialize session state variables
if "job_result" not in st.session_state:
    st.session_state.job_result = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "visualization_image" not in st.session_state:
    st.session_state.visualization_image = None
if "detection_data" not in st.session_state:
    st.session_state.detection_data = None
if "midi_content" not in st.session_state:
    st.session_state.midi_content = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None


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
                return None

            logger.info("Successfully received process response")

        except Exception as e:
            logger.error(f"Process request failed: {str(e)}")
            os.unlink(temp_file_path)
            return None

        # Clean up and return
        os.unlink(temp_file_path)
        result = response.json()
        result["visualization_image"] = visualization_image
        return result

    except Exception as e:
        logger.error(f"Error in process_sheet_music: {str(e)}")
        return None


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

        # Store midi content in session state for future use
        st.session_state.midi_content = midi_content

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


def fetch_detection_and_midi(job_id):
    """Background function to fetch detection data and MIDI content"""
    try:
        # 1. Fetch detection data
        detection_response = requests.get(f"{API_URL}/preview/{job_id}")
        if detection_response.status_code == 200:
            st.session_state.detection_data = detection_response.json()
            logger.info(f"Successfully fetched detection data for job {job_id}")
        else:
            logger.error(f"Failed to get detection preview: {detection_response.status_code}")
            st.session_state.processing_error = f"Failed to get detection preview: {detection_response.status_code}"

        # 2. Fetch MIDI content
        midi_url = f"{API_URL}/download/{job_id}"
        midi_response = requests.get(midi_url)
        if midi_response.status_code == 200:
            st.session_state.midi_content = midi_response.content
            logger.info(f"Successfully fetched MIDI content for job {job_id}")
        else:
            logger.error(f"Failed to get MIDI content: {midi_response.status_code}")
            st.session_state.processing_error = f"Failed to get MIDI content: {midi_response.status_code}"

        # Mark processing as complete
        st.session_state.processing_complete = True
        st.session_state.is_processing = False

    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        st.session_state.processing_error = f"Error in background processing: {str(e)}"
        st.session_state.is_processing = False


def background_process(uploaded_file, tempo):
    """Function to handle background processing when image is uploaded"""
    try:
        if st.session_state.is_processing:
            logger.info("Already processing, skipping duplicate request")
            return

        # Set processing flag
        st.session_state.is_processing = True
        st.session_state.processing_complete = False
        st.session_state.processing_error = None

        # Process the sheet music
        logger.info(f"Starting background processing with tempo {tempo}")
        job_response = process_sheet_music(uploaded_file, tempo=tempo)

        if not job_response:
            logger.error("Failed to process sheet music")
            st.session_state.processing_error = "Failed to process sheet music"
            st.session_state.is_processing = False
            return

        # Store job ID and visualization image
        job_id = job_response.get("job_id")
        st.session_state.job_id = job_id
        logger.info(f"Job created with ID: {job_id}")

        if "visualization_image" in job_response and job_response["visualization_image"]:
            st.session_state.visualization_image = job_response["visualization_image"]
            logger.info("Stored visualization image in session state")

        # Poll for job completion
        logger.info(f"Polling for completion of job {job_id}")
        result = poll_job_status(job_id)

        if not result:
            logger.error(f"Job {job_id} failed or timed out")
            st.session_state.processing_error = f"Job {job_id} failed or timed out"
            st.session_state.is_processing = False
            return

        # Store job result
        st.session_state.job_result = result
        logger.info(f"Job {job_id} completed successfully with {result.get('element_count', 0)} elements")

        # Fetch detection data and MIDI content
        logger.info(f"Fetching final results for job {job_id}")
        fetch_detection_and_midi(job_id)

    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        st.session_state.processing_error = f"Error in background processing: {str(e)}"
        st.session_state.is_processing = False


# Main app
logger.info("Starting app")

st.title("🎵 Sheet Music to MIDI Converter")
st.write("Upload a sheet music image to convert it to a playable MIDI file.")

uploaded_file = st.file_uploader("Choose a sheet music image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sheet Music", use_column_width=True)

        # Add tempo slider
        selected_tempo = st.slider(
            "Select Tempo (BPM)",
            min_value=40,
            max_value=200,
            value=79,
            step=1
        )

        # Start background processing when image is uploaded
        if not st.session_state.is_processing and not st.session_state.processing_complete:
            # Start processing in background
            bg_thread = threading.Thread(
                target=background_process,
                args=(uploaded_file, selected_tempo)
            )
            bg_thread.daemon = True
            bg_thread.start()

            # Show more noticeable processing indicator
            if st.session_state.is_processing:
                processing_info = st.info("🔄 Preparing sheet music analysis in background... Click 'Convert to MIDI' when ready.")
                progress_placeholder = st.empty()

                # Show a small indicator that work is happening
                import itertools
                indicators = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

                def update_spinner():
                    for _ in range(10):  # Show a few iterations of the spinner
                        if not st.session_state.is_processing:
                            break
                        progress_placeholder.text(f"Processing: {next(indicators)}")
                        time.sleep(0.2)

                # Start spinner in a non-blocking way
                update_thread = threading.Thread(target=update_spinner)
                update_thread.daemon = True
                update_thread.start()

        # Show convert button
        if st.button("Convert to MIDI"):
            # Always show progress indicator when button is clicked
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                if st.session_state.processing_error:
                    # If there was an error in background processing
                    progress_bar.progress(1.0)
                    status_text.text("Status: Failed ❌")
                    st.error(f"Error during processing: {st.session_state.processing_error}")
                elif st.session_state.processing_complete:
                    # If processing is already complete, show 100% progress
                    progress_bar.progress(1.0)
                    status_text.text("Status: Completed! ✅")
                else:
                    # If still processing, show animated progress
                    status_text.text("Status: Processing your sheet music...")

                    for percent_complete in [0.2, 0.4, 0.6, 0.8]:
                        progress_bar.progress(percent_complete)
                        time.sleep(0.5)

                    # Wait for processing to complete (with timeout)
                    timeout = 30  # seconds
                    start_time = time.time()
                    while st.session_state.is_processing and (time.time() - start_time < timeout):
                        # Show pulsing progress bar while waiting
                        for pulse in [0.8, 0.9, 0.95, 0.9, 0.8]:
                            progress_bar.progress(pulse)
                            time.sleep(0.5)
                            if not st.session_state.is_processing:
                                break

                    if st.session_state.processing_complete:
                        progress_bar.progress(1.0)
                        status_text.text("Status: Completed! ✅")
                    else:
                        progress_bar.progress(0.9)
                        status_text.text("Status: Still processing in background... ⏳")

            # If processing is complete, show results
            if st.session_state.processing_complete:
                st.success(f"Successfully converted your sheet music! Found {st.session_state.job_result.get('element_count', 0)} musical elements.")

                # Display detection visualization
                if st.session_state.detection_data:
                    visualize_detections(st.session_state.detection_data)

                # Display MIDI player
                if st.session_state.midi_content:
                    st.subheader("MIDI Playback")

                    download_link = create_download_link(
                        st.session_state.midi_content,
                        "sheet_music.mid",
                        "⬇️ Download MIDI File"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)

                    st.write("MIDI Preview:")

                    midi_b64 = base64.b64encode(st.session_state.midi_content).decode()

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
                else:
                    st.error("MIDI content could not be retrieved. Please try again.")

    except Exception as e:
        logger.error(f"Error opening uploaded image: {str(e)}")
        st.error(f"Error opening image: {str(e)}")


st.markdown("---")
st.markdown("Made with ❤️ by the team behind | [Optical Music Recognition](https://github.com/cmullie/omr)")

logger.info("App rendered successfully")
