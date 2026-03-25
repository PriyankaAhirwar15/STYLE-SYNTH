import os
from dotenv import load_dotenv
load_dotenv()
# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"
# App
APP_NAME = "STYLE-SYNTH"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Face Style Transfer GAN with LLM Quality Checker"
# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
# Paths
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
# Style options
STYLES = [
    "anime",
    "sketch",
    "oil_painting",
    "watercolor",
    "cartoon"
]
# Image settings
IMAGE_SIZE = 256
MAX_IMAGE_SIZE_MB = 5
