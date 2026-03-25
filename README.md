# 🎭 STYLE-SYNTH

Face Style Transfer GAN with LLM Quality Checker + Real-time Streamlit Demo

![STYLE-SYNTH Banner](https://img.shields.io/badge/AI-Art%20Generation-blue?style=for-the-badge&logo=artstation)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3--70B-orange?style=for-the-badge)

## 🚀 What is STYLE-SYNTH?

STYLE-SYNTH is an advanced AI-powered face style transfer system that transforms your photos into various artistic styles using GAN technology and provides intelligent quality assessment through LLM analysis.

### ✨ Key Features

- **🎨 Multiple Art Styles**: Anime, Sketch, Oil Painting, Watercolor, Cartoon
- **🤖 LLM Quality Checker**: AI-powered assessment of transformation quality
- **⚡ Real-time Processing**: FastAPI backend with Streamlit frontend
- **📊 Technical Metrics**: Brightness, contrast, edge density analysis
- **🔍 Quality Scoring**: 1-10 rating with detailed feedback
- **💾 Output Management**: Automatic saving of transformed images

## 🏗️ Architecture

```
STYLE-SYNTH/
├── 🎨 frontend/          # Streamlit web interface
├── 🚀 api/              # FastAPI backend server
├── 🧠 src/
│   ├── gan/             # Style transfer pipeline
│   ├── auditor/         # LLM quality checker
│   └── utils/           # Image processing & logging
├── 📁 models/           # Pre-trained model weights
├── 🖼️ outputs/          # Transformed images
└── ⚙️ config.py         # Application configuration
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Style Transfer** | OpenCV + PIL | Image processing & effects |
| **LLM Quality Check** | Groq LLaMA 3.3-70B | AI-powered quality assessment |
| **Backend API** | FastAPI | RESTful API server |
| **Frontend** | Streamlit | Interactive web interface |
| **Image Processing** | OpenCV, NumPy | Computer vision operations |
| **Logging** | Loguru | Structured logging |
| **Environment** | Python 3.8+ | Runtime environment |

## 🚀 Quick Start (Always Works!)

### One-Command Startup
```bash
python start.py
```
That's it! This reliable launcher ensures STYLE-SYNTH always works.

### What the Launcher Does:
- ✅ Checks all dependencies
- ✅ Starts FastAPI backend (port 8002)
- ✅ Starts Streamlit frontend (port 8501)
- ✅ Waits for services to be ready
- ✅ Provides clear status updates
- ✅ Handles graceful shutdown

### Quick Test (Anyone Can Run)
```bash
python test.py
```
This verifies everything is working properly.

### Manual Startup (If Needed)
```bash
# Terminal 1: Start API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8002 --reload

# Terminal 2: Start Web UI
python -m streamlit run frontend/app.py
```

## 🎨 Usage

1. **Open your browser** to `http://localhost:8501`
2. **Upload a face photo** (JPG, PNG supported)
3. **Select an art style** from the sidebar
4. **Click "Transform Image"** to apply the style
5. **View the result** with AI quality assessment

### API Usage

The FastAPI backend provides REST endpoints:

```bash
# Health check
GET http://localhost:8000/health

# Get available styles
GET http://localhost:8000/styles

# Transform image
POST http://localhost:8000/transform
Content-Type: multipart/form-data
Body: file=<image>, style=<style_name>, check_quality=<true/false>
```

## 🤖 LLM Quality Checker

The system uses Groq's LLaMA 3.3-70B model to analyze transformation quality:

- **Quality Score**: 1-10 rating
- **Style Accuracy**: Low/Medium/High assessment
- **Detailed Feedback**: Strengths and improvement suggestions
- **Technical Analysis**: Processing time and style metrics

## 📊 Style Options

| Style | Description | Best For |
|-------|-------------|----------|
| **Anime** | Japanese animation style | Portraits, character art |
| **Sketch** | Pencil drawing effect | Artistic, minimalist |
| **Oil Painting** | Classical painting technique | Realistic, textured |
| **Watercolor** | Soft watercolor effect | Dreamy, artistic |
| **Cartoon** | Exaggerated cartoon style | Fun, illustrative |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Style options
STYLES = ["anime", "sketch", "oil_painting", "watercolor", "cartoon"]

# Image settings
IMAGE_SIZE = 256
MAX_IMAGE_SIZE_MB = 5
```

## 🐳 Docker Deployment

### Build and run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

### Manual Docker build

```bash
# Build the image
docker build -t style-synth .

# Run the container
docker run -p 8000:8000 -p 8501:8501 style-synth
```

## 📈 Performance

- **Processing Time**: ~1-3 seconds per image
- **Image Size**: 256x256 pixels (optimized)
- **Memory Usage**: ~500MB RAM
- **Storage**: Outputs saved to `outputs/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast LLM inference
- **FastAPI** for the excellent web framework
- **Streamlit** for the beautiful UI components
- **OpenCV** for computer vision capabilities

---

**Made with ❤️ by Priyanka Ahirwar**

🎭 *Transforming faces, one style at a time*
