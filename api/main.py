from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn
import os
from src.gan.transform import StyleTransferPipeline
from src.utils.image_utils import image_to_base64
from src.utils.logger import setup_logger
logger = setup_logger("api")
app = FastAPI(
    title="STYLE-SYNTH API",
    description="Face Style Transfer with LLM Quality Checker",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = StyleTransferPipeline()
@app.get("/")
def root():
    return {
        "app": "STYLE-SYNTH",
        "version": "1.0.0",
        "status": "running",
        "styles": ["anime", "sketch", "oil_painting", "watercolor", "cartoon"]
    }
@app.get("/health")
def health():
    return {"status": "healthy"}
@app.get("/styles")
def get_styles():
    return {
        "styles": [
            {"id": "anime", "name": "Anime", "description": "Japanese anime style"},
            {"id": "sketch", "name": "Pencil Sketch", "description": "Black and white sketch"},
            {"id": "watercolor", "name": "Watercolor", "description": "Soft watercolor painting"},
            {"id": "cartoon", "name": "Cartoon", "description": "Cartoon style effect"},
            {"id": "oil_painting", "name": "Oil Painting", "description": "Classic oil painting"}
        ]
    }
@app.post("/transform")
async def transform_image(
    file: UploadFile = File(...),
    style: str = Form(...),
    check_quality: bool = Form(True)
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image!")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image!")
        image = cv2.resize(image, (256, 256))
        result = pipeline.transform(image, style, check_quality)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))
        styled_base64 = image_to_base64(result["styled_image"])
        original_base64 = image_to_base64(image)
        return JSONResponse({
            "success": True,
            "style": style,
            "original_image": original_base64,
            "styled_image": styled_base64,
            "processing_time": result["processing_time"],
            "style_metrics": result["style_metrics"],
            "quality_report": result["quality_report"]
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
