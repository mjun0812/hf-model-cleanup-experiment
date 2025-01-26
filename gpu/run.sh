#!/bin/bash -x

mkdir -p figs csv

docker build -t memory-profiler-gpu .

# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/1.py
# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/2.py
# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/3.py
# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/4.py
# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/5.py
docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/6.py
# docker run -it --rm -v $(pwd):/app --gpus device=0 --pid=host memory-profiler-gpu python /app/7.py

docker run -it --rm -v $(pwd):/app memory-profiler-gpu python /app/plot_all.py csv
