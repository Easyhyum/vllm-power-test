# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    tmux build-essential wget curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

# RUN pip install -r /workspace/requirements.txt

# RUN cd /workspace/vllm && MAX_JOBS=16 pip install -e .

RUN chmod +x .

CMD ["bash"]
