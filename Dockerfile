FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DATA_DIR=/tmp/medrag_data
ENV HF_HOME=/tmp/hf_cache
ENV PREFETCH_MODEL=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements-space.txt

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
