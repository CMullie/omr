import streamlit as st
import requests
import time
import json
import base64
import os
from PIL import Image
import io
import tempfile

API_URL = os.environ.get("API_URL", "https://omr-api-service-510610499515.europe-west1.run.app")

st.set_page_config(
    page_title="Sheet Music to MIDI Converter",
    page_icon="🎵",
    layout="wide"
)

def create_download_link(content, filename, link_text):
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:audio/midi;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def process_sheet_music(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    files = {"file": (file.name, open(temp_file_path, "rb"), "image/png")}
    visualization_response = requests.post(f"{API_URL}/detect-visualize", files=files)
    if visualization_response.status_code != 200:
        visualization_image = None
        st.error(f"Visualization failed: {visualization_response.status_code}")
    else:
        visualization_image = visualization_response.content

    files = {"file": (file.name, open(temp_file_path, "rb"), "image/png")}
    response = requests.post(f"{API_URL}/process", files=files)

    if response.status_code != 200:
        st.error(f"API request failed: {response.status_code}")
        os.unlink(temp_file_path)
        return None

    os.unlink(temp_file_path)
    result = response.json()
    result["visualization_image"] = visualization_image
    return result

def poll_job_status(job_id):
    status = "queued"
    while status in ["queued", "processing"]:
        response = requests.get(f"{API_URL}/status/{job_id}")

        if response.status_code != 200:
            st.error(f"Failed to check job status: {response.status_code}")
            return None

        data = response.json()
        status = data.get("status", "unknown")

        if status == "completed":
            return data
        elif status == "failed":
            st.error(f"Processing failed: {data.get('error', 'Unknown error')}")
            return None

        progress_bar.progress(0.5)
        status_text.text(f"Status: {status.title()}...")

        time.sleep(2)

    return None

def visualize_detections(detection_data):
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
                    "Position": f"({elem.get('x', 0)}, {elem.get('y', 0)})",
                    "Confidence": f"{elem.get('confidence', 0):.2f}"
                })

            if element_data:
                st.table(element_data)

def embed_midi_player(midi_url):
    st.subheader("MIDI Playback")

    response = requests.get(midi_url)
    if response.status_code != 200:
        st.error(f"Failed to play MIDI: {response.status_code}")
        st.warning("The MIDI file was created but couldn't be played directly. Please download it and use a MIDI player.")
        return

    midi_content = response.content

    download_link = create_download_link(
        midi_content,
        "sheet_music.mid",
        "⬇️ Download MIDI File"
    )
    st.markdown(download_link, unsafe_allow_html=True)

    st.write("MIDI Preview:")

    midi_b64 = base64.b64encode(midi_content).decode()

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

if "visualization_image" not in st.session_state:
    st.session_state.visualization_image = None

st.title("🎵 Sheet Music to MIDI Converter")
st.write("Upload a sheet music image to convert it to a playable MIDI file.")

uploaded_file = st.file_uploader("Choose a sheet music image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sheet Music", use_column_width=True)

    if st.button("Convert to MIDI"):
        with st.spinner("Processing your sheet music..."):
            progress_bar = st.progress(0.2)
            status_text = st.empty()
            status_text.text("Status: Uploading image...")

            job_response = process_sheet_music(uploaded_file)

            if job_response:
                job_id = job_response.get("job_id")

                if "visualization_image" in job_response and job_response["visualization_image"]:
                    st.session_state.visualization_image = job_response["visualization_image"]

                status_text.text(f"Status: Processing job {job_id}...")

                result = poll_job_status(job_id)

                if result:
                    progress_bar.progress(1.0)
                    status_text.text("Status: Completed! ✅")

                    st.success(f"Successfully converted your sheet music! Found {result.get('element_count', 0)} musical elements.")

                    detection_response = requests.get(f"{API_URL}/preview/{job_id}")
                    if detection_response.status_code == 200:
                        detection_data = detection_response.json()
                        visualize_detections(detection_data)
                        midi_url = f"{API_URL}/download/{job_id}"
                        embed_midi_player(midi_url)
                    else:
                        st.error(f"Failed to get detection preview: {detection_response.status_code}")
            else:
                progress_bar.progress(1.0)
                status_text.text("Status: Failed ❌")

st.markdown("---")
st.markdown("Made with ❤️ by the team behind | [Optical Music Recognition](https://github.com/cmullie/omr)")
