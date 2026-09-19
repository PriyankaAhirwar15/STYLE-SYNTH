"""
STYLE-SYNTH FastAPI Server
RESTful endpoints for Face Style Transfer GAN, Latent Interpolation,
Batch Processing, and Deep Neural Architecture Inspection.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn
import os
import io
import zipfile
import base64
from typing import List, Optional

from src.gan.transform import StyleTransferPipeline
from src.utils.image_utils import image_to_base64
from src.utils.logger import setup_logger
from config import STYLES, STYLE_METADATA, APP_NAME, APP_VERSION, APP_DESCRIPTION, API_HOST, API_PORT

logger = setup_logger("api")

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StyleTransferPipeline()


@app.get("/")
def root():
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "status": "running",
        "styles": STYLES,
        "endpoints": [
            "/health",
            "/styles",
            "/transform",
            "/blend",
            "/batch-transform",
            "/gan/architecture"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "pipeline_ready": True,
        "supported_styles_count": len(STYLES)
    }


@app.get("/styles")
def get_styles():
    """Returns all 8 available GAN styles with rich metadata."""
    styles_list = []
    for s_id in STYLES:
        meta = STYLE_METADATA.get(s_id, {})
        styles_list.append({
            "id": s_id,
            "name": meta.get("name", s_id.title()),
            "category": meta.get("category", "Artistic"),
            "description": meta.get("description", ""),
            "badge": meta.get("badge", "GAN")
        })
    return {"styles": styles_list, "total": len(styles_list)}


@app.post("/transform")
async def transform_image(
    file: UploadFile = File(...),
    style: str = Form("anime"),
    style_strength: float = Form(1.0),
    smoothness: float = Form(0.0),
    vibrancy: float = Form(1.0),
    sharpness: float = Form(0.0),
    auto_face_crop: bool = Form(False),
    target_size: int = Form(512),
    check_quality: bool = Form(True)
):
    """
    Transforms single face portrait with specified GAN style and facial attribute modulations.
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a valid image!")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode image. Please check file format.")

        result = pipeline.transform(
            image=image,
            style=style,
            style_strength=style_strength,
            smoothness=smoothness,
            vibrancy=vibrancy,
            sharpness=sharpness,
            auto_face_crop=auto_face_crop,
            target_size=target_size,
            check_quality=check_quality
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Transformation failed"))

        styled_base64 = image_to_base64(result["styled_image"])
        original_base64 = image_to_base64(result["original_image"])

        return JSONResponse({
            "success": True,
            "style": style,
            "face_detected": result.get("face_detected", False),
            "original_image": original_base64,
            "styled_image": styled_base64,
            "processing_time": result["processing_time"],
            "style_metrics": result["style_metrics"],
            "quality_report": result["quality_report"]
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API Error in /transform: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blend")
async def blend_styles(
    file: UploadFile = File(...),
    style1: str = Form("anime"),
    style2: str = Form("cyberpunk"),
    alpha: float = Form(0.5),
    style_strength: float = Form(1.0),
    smoothness: float = Form(0.0),
    vibrancy: float = Form(1.0),
    sharpness: float = Form(0.0),
    auto_face_crop: bool = Form(False),
    target_size: int = Form(512),
    check_quality: bool = Form(True)
):
    """
    Interpolates between two distinct GAN styles with continuous alpha ratio [0.0 - 1.0].
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot decode image.")

        result = pipeline.blend(
            image=image,
            style1=style1,
            style2=style2,
            alpha=alpha,
            style_strength=style_strength,
            smoothness=smoothness,
            vibrancy=vibrancy,
            sharpness=sharpness,
            auto_face_crop=auto_face_crop,
            target_size=target_size,
            check_quality=check_quality
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Blending failed"))

        styled_base64 = image_to_base64(result["styled_image"])
        original_base64 = image_to_base64(result["original_image"])

        return JSONResponse({
            "success": True,
            "style1": style1,
            "style2": style2,
            "alpha": alpha,
            "face_detected": result.get("face_detected", False),
            "original_image": original_base64,
            "styled_image": styled_base64,
            "processing_time": result["processing_time"],
            "style_metrics": result["style_metrics"],
            "quality_report": result["quality_report"]
        })

    except Exception as e:
        logger.error(f"API Error in /blend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-transform")
async def batch_transform(
    files: List[UploadFile] = File(...),
    style: str = Form("anime"),
    style_strength: float = Form(1.0),
    auto_face_crop: bool = Form(False)
):
    """
    Batch transforms multiple face images simultaneously.
    """
    try:
        results = []
        for f in files:
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                res = pipeline.transform(
                    image=image,
                    style=style,
                    style_strength=style_strength,
                    auto_face_crop=auto_face_crop,
                    check_quality=False
                )
                if res["success"]:
                    results.append({
                        "filename": f.filename,
                        "styled_image": image_to_base64(res["styled_image"]),
                        "processing_time": res["processing_time"]
                    })

        return JSONResponse({
            "success": True,
            "total_processed": len(results),
            "style": style,
            "items": results
        })

    except Exception as e:
        logger.error(f"API Error in /batch-transform: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gan/architecture")
def gan_architecture():
    """Returns PyTorch GAN architecture specification and layer breakdown."""
    return pipeline.get_gan_architecture_info()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
