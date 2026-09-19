---
title: STYLE-SYNTH
emoji: 🎭
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: frontend/app.py
pinned: true
license: mit
short_description: Face Style Transfer GAN Studio with AI Auditor
---

# 🎭 STYLE-SYNTH 2.0

**Next-Gen Face Style Transfer GAN Studio with Multi-Style Latent Blending + AI Quality Auditor**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/PRIYANKAAhirwar/STYLE-SYNTH)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/PriyankaAhirwar15/STYLE-SYNTH.git)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3--70B-orange)

###                 **[Live](https://huggingface.co/spaces/PRIYANKAAhirwar/STYLE-SYNTH)**
---

## 🚀 What is STYLE-SYNTH 2.0?

**STYLE-SYNTH 2.0** is an AI-powered neural face style synthesis studio. Built on **PyTorch ResNet-9Block Style Generators** and a **70x70 PatchGAN Discriminator**, it transforms portrait photos into 8 distinct artistic representations with fine-grained facial attribute modulation, continuous multi-style latent space blending, and real-time AI quality auditing powered by Groq LLaMA 3.3-70B.

---

## ✨ Key Features

- **🎨 8 Neural GAN Art Styles**:
  1. 🎌 **Anime Studio GAN** — Japanese anime cell-shading & crisp ink contours.
  2. 🦸 **3D Cartoon / Pixar GAN** — Volumetric clay-like 3D character shading.
  3. 🏙️ **Cyberpunk Synthwave GAN** — Neon cyan/magenta glow & chromatic shift.
  4. 🎨 **Renaissance Oil GAN** — Classical impasto texture & directional brushwork.
  5. ✏️ **Fine Graphite Sketch GAN** — Delicate pencil cross-hatching & line art.
  6. 💧 **Dreamy Watercolor GAN** — Translucent pigment wash & wet edge bleeding.
  7. 💥 **Pop-Art Comic GAN** — Halftone Ben-Day dots & bold graphic outlines.
  8. 🖤 **Gothic Noir Ink GAN** — Dramatic chiaroscuro shadows & sumi-e ink wash.
- **🌀 Continuous Latent Style Blender**: Cross-fade and interpolate between any two GAN styles ($\alpha \in [0.0, 1.0]$) in continuous latent feature space.
- **🎛️ Facial Attribute Modulation**: Fine-tune *Style Intensity*, *Skin Smoothing*, *Color Vibrancy*, and *Edge Sharpness* with real-time sliders.
- **🔍 Smart Face Auto-Crop & Alignment**: Automated face detection using Haar cascade with proportional margin framing for portrait inputs.
- **📦 Batch Portrait Studio**: Upload multiple photos at once, synthesize in batch with progress tracking, and download all as a `.zip` archive.
- **🧠 GAN Architecture & Loss Inspector**: Interactive visualizer for PyTorch Generator layers, parameter counts, receptive fields, and minimax training equilibrium diagnostics.
- **🤖 AI Quality Auditor**: Multi-dimensional evaluation (Quality Score, Style Accuracy, Structural Preservation, Actionable Recommendations) with seamless offline heuristic fallback.
- **⚡ Zero-Error Hugging Face Spaces Compatibility**: Runs standalone in-memory with zero network overhead, eliminating localhost connection errors.

---

## 🏗️ Architecture

```
STYLE-SYNTH/
├── 🎨 frontend/
│   └── app.py                  # 4-Tab Streamlit Studio (Single, Blender, Batch, Inspector)
├── 🚀 api/
│   └── main.py                 # FastAPI REST backend (/transform, /blend, /batch-transform, /gan/architecture)
├── 🧠 src/
│   ├── gan/
│   │   ├── generator.py        # PyTorch ResNet Generator & PatchGAN Discriminator
│   │   ├── neural_styles.py    # 8 Specialized GAN Style Engines
│   │   ├── interpolator.py     # Latent Multi-Style Blending & Attribute Modulator
│   │   ├── face_detect.py      # Face Alignment & Smart Square Auto-Crop
│   │   └── transform.py        # Unified StyleTransferPipeline
│   ├── auditor/
│   │   └── llm_checker.py      # Groq LLaMA 3.3-70B Auditor with Heuristic Fallback
│   └── utils/
│       ├── image_utils.py      # Image Encoding/Decoding & Base64 Helpers
│       └── logger.py           # Structured Loguru Logging
├── 📁 outputs/                 # Auto-saved Transformed Portraits
├── ⚙️ config.py                 # Global Configuration & Style Metadata
├── 📋 requirements.txt         # Production Dependencies
├── 🧪 ready.py                 # System Integrity & Health Check Suite
├── 🧪 test_system.py           # Comprehensive Pipeline & API Tests
└── 🚀 start.py                 # Dual-Service Studio Launcher
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch (`torch`, `torchvision`) | ResNet Generator & PatchGAN Discriminator |
| **Computer Vision** | OpenCV (`cv2`), NumPy, Pillow | Face Detection, Image Processing & Filtering |
| **LLM Quality Auditor** | Groq LLaMA 3.3-70B | AI-powered Aesthetic & Structural Evaluation |
| **Backend API** | FastAPI, Uvicorn | RESTful Endpoints & Batch Processing |
| **Frontend UI** | Streamlit | Interactive Multi-Tab Creative Studio |
| **Logging & Utilities** | Loguru, Python-Dotenv | Structured Telemetry & Configuration |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/PriyankaAhirwar15/STYLE-SYNTH.git
cd STYLE-SYNTH
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY (optional, fallback auditor works offline)
```

### 5. Launch the Studio

```bash
# Option A: Start all services together (FastAPI + Streamlit)
python start.py

# Option B: Run Streamlit standalone directly
streamlit run frontend/app.py
```

Open your browser to `http://127.0.0.1:8501`.

---

## 🌐 Cloud Deployment

### Deploy on Render

1. Create a free account on [Render.com](https://render.com).
2. Click **New +** → **Web Service** → Connect your GitHub repo `PriyankaAhirwar15/STYLE-SYNTH`.
3. Configure the settings:
   - **Environment:** `Python 3` (or `Docker`)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false`
   - **Environment Variables (Optional):**
     - `PYTHON_VERSION`: `3.11.8`
     - `GROQ_API_KEY`: *(Your Groq API key, optional)*
4. Click **Create Web Service**. Render will automatically build and launch your live URL!

*(Alternatively, connect the repo and Render will automatically detect the included `render.yaml` Blueprint).*

### Deploy on Hugging Face Spaces
- Fork/Clone to a Streamlit Space on [Hugging Face Spaces](https://huggingface.co/spaces).
- Set `app_file: frontend/app.py` in `README.md` (pre-configured).

---

## 🧪 Verification & Testing

Verify that all GAN components, styles, and modules pass without errors:

```bash
# Run integrity & readiness verification suite
python ready.py

# Run comprehensive system & transformation tests
python test_system.py
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check & pipeline status |
| `GET` | `/styles` | List all 8 GAN styles and metadata |
| `POST` | `/transform` | Transform single portrait with style & attribute modulation |
| `POST` | `/blend` | Latent space blending between two GAN styles |
| `POST` | `/batch-transform` | Batch processing for multiple images |
| `GET` | `/gan/architecture` | Detailed PyTorch network layer specs & parameter summary |

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Developed with ❤️ by Priyanka Ahirwar**
- GitHub: [@PriyankaAhirwar15](https://github.com/PriyankaAhirwar15)
- Hugging Face Space: [STYLE-SYNTH Space](https://huggingface.co/spaces/PRIYANKAAhirwar/STYLE-SYNTH)
