import os
from dotenv import load_dotenv

load_dotenv()

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"

# Application Metadata
APP_NAME = "STYLE-SYNTH"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Next-Gen Face Style Transfer GAN Studio with LLM Quality Auditor"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8002"))

# Paths
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# GAN Style Options
STYLES = [
    "anime",
    "cartoon_3d",
    "cyberpunk",
    "oil_painting",
    "sketch",
    "watercolor",
    "pop_art",
    "gothic_ink"
]

STYLE_METADATA = {
    "anime": {
        "name": "🎌 Anime Studio GAN",
        "category": "Animation",
        "description": "Japanese animation aesthetic with clean ink lines, soft shading, and expressive vibrant tones.",
        "badge": "AnimeGANv2"
    },
    "cartoon_3d": {
        "name": "🦸 3D Cartoon / Pixar GAN",
        "category": "3D Animation",
        "description": "Pixar/Disney-like stylized 3D animated character appearance with smooth skin and volumetric lighting.",
        "badge": "Toon3D-GAN"
    },
    "cyberpunk": {
        "name": "🏙️ Cyberpunk Synthwave GAN",
        "category": "Futuristic Sci-Fi",
        "description": "High-tech dystopian aesthetic with neon cyan/magenta rim lighting, glow diffusion, and futuristic vibes.",
        "badge": "NeonSynth-GAN"
    },
    "oil_painting": {
        "name": "🎨 Renaissance Oil GAN",
        "category": "Fine Art",
        "description": "Classical portraiture with rich impasto textures, subtle brushstroke contours, and warm canvas palette.",
        "badge": "ImpastoGAN"
    },
    "sketch": {
        "name": "✏️ Fine Graphite Sketch GAN",
        "category": "Drafting & Line Art",
        "description": "Delicate pencil sketching with graphite cross-hatching, stippling, and paper grain emulation.",
        "badge": "SketchGAN"
    },
    "watercolor": {
        "name": "💧 Dreamy Watercolor GAN",
        "category": "Aquarelle Art",
        "description": "Soft watercolor wash with authentic pigment bleeding, translucent color layering, and wet edge pooling.",
        "badge": "AquaGAN"
    },
    "pop_art": {
        "name": "💥 Pop-Art Comic GAN",
        "category": "Graphic & Retro",
        "description": "Roy Lichtenstein comic book styling featuring dynamic halftone Ben-Day dots and heavy black ink outlines.",
        "badge": "PopComicGAN"
    },
    "gothic_ink": {
        "name": "🖤 Gothic Noir Ink GAN",
        "category": "Dark & Expressive",
        "description": "High-contrast dramatic chiaroscuro with expressive sumi-e ink wash and moody shadowy portrait contours.",
        "badge": "GothicInkGAN"
    }
}

# Image Processing Defaults
IMAGE_SIZE = 512
DEFAULT_PREVIEW_SIZE = 256
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
