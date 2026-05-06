# MultiWorld inference + Gradio UI image.
#
# Mirrors the pip recipe from readme.md:
#   - torch==2.7.1 + cu128
#   - the rest of requirements.txt
#   - extra UI deps from app/requirements.txt
#
# Built from a slim CUDA 12.8 cuDNN runtime image. We install Python 3.11 from
# the deadsnakes PPA so we don't have to track Ubuntu's default Python.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/.cache/modelscope \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_HOST=0.0.0.0 \
    GRADIO_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg git libgl1 libglib2.0-0 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install torch first (matches the readme).
RUN pip install --upgrade pip \
    && pip install \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Install the repo's other Python deps.
COPY requirements.txt /tmp/requirements.txt
COPY app/requirements.txt /tmp/app-requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install -r /tmp/app-requirements.txt

# Copy the source last so iteration on code doesn't bust the deps cache.
COPY . /workspace

EXPOSE 7860

# At runtime ./checkpoints and ./models are expected to be mounted in
# (see docker-compose.yml). They're empty in the image to keep size small.
CMD ["python", "-m", "app.app"]
