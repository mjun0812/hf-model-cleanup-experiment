FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv -p 3.11 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install torch==2.5.1 --extra-index-url  https://download.pytorch.org/whl/cu124
RUN uv pip install transformers==4.48.1 matplotlib sentence-transformers pynvml

COPY download.py .
COPY utils.py .

RUN python download.py
