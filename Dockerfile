FROM python:3.11-slim

WORKDIR /app

# Install system graphics libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models outputs logs

ENV PORT=8501
EXPOSE 8501

# Start Streamlit application binding to Render's $PORT
CMD streamlit run frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
