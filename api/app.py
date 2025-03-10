import os
import shutil
import uuid
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import logging
from pydantic import BaseModel
import io
import torch

from .processor import process_video

# Enable TF32 precision for better performance on NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="MMAudio API", version="1.0.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp directory for file storage
TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)


class TaskStatus(BaseModel):
    task_id: str
    status: str
    position: Optional[int] = None
    message: Optional[str] = None


@app.post("/process")
async def process_video_endpoint(
    video: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    duration: float = Form(8.0),
    cfg_strength: float = Form(4.5),
    num_steps: int = Form(25),
    variant: str = Form("large_44k_v2"),
):
    """
    Process a video with MMAudio to generate synchronized audio.
    Returns the processed video directly in the response.
    """
    # Create a unique task ID for this processing job
    task_id = str(uuid.uuid4())
    
    # Create a temporary directory for this task
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    try:
        # Save the uploaded video
        video_path = os.path.join(task_dir, f"input_{video.filename}")
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        logger.info(f"Processing video {video_path}")
        
        # Process the video synchronously
        result = process_video(
            task_id=task_id,
            video_path=video_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            cfg_strength=cfg_strength,
            num_steps=num_steps,
            variant=variant,
            output_dir=task_dir,
        )
        
        # Check that the video was generated
        result_file = result.get("video_path")
        if not result_file or not os.path.exists(result_file):
            raise HTTPException(status_code=500, detail="Failed to generate video")
        
        # Read the file into memory
        with open(result_file, "rb") as f:
            video_bytes = f.read()
        
        # Clean up temporary files
        shutil.rmtree(task_dir)
        
        # Return the video file as a streaming response
        return StreamingResponse(
            io.BytesIO(video_bytes), 
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(result_file)}"}
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        logger.exception(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting MMAudio API")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down MMAudio API")
    # Clean up any remaining temp files
    for item in os.listdir(TEMP_DIR):
        if os.path.isdir(os.path.join(TEMP_DIR, item)):
            try:
                shutil.rmtree(os.path.join(TEMP_DIR, item))
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {item}: {str(e)}")
