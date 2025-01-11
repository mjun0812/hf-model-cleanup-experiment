FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install torch transformers memory_profiler psutil matplotlib sentence-transformers

COPY download.py .
COPY utils.py .

RUN python download.py
