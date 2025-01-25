FROM nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install torch==2.5.1 --extra-index-url  https://download.pytorch.org/whl/cu124
RUN pip install transformers==4.48.1 memory_profiler psutil matplotlib

COPY download.py .
COPY utils.py .

RUN python download.py
