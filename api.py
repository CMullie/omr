import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(tempfile.gettempdir(), "sheet_midi")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/convert")
async def convert_form(file: UploadFile = File(...)):
    content = await file.read()


    file_id = uuid.uuid4()
    download_url = f"https://storage.googleapis.com/sheet-music-midi/{file_id}/output.mid"

    return {"download_url": download_url}

@app.post("/convert-binary")
async def convert_binary(request: Request):

    content = await request.body()

    file_id = uuid.uuid4()
    download_url = f"https://storage.googleapis.com/sheet-music-midi/{file_id}/output.mid"

    return {"download_url": download_url}

@app.get("/")
def read_root():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
