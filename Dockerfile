FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY scout ./scout
COPY app ./app
COPY weights.yaml ./

RUN pip install --no-cache-dir -e ".[perception,ui,llm]"

EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
