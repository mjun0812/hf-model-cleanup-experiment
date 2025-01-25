#!/bin/bash -xe

mkdir -p figs csv

docker build -t memory-profiler-gpu .

docker run -it --rm -v $(pwd):/app memory-profiler python /app/1.py

docker run -it --rm -v $(pwd):/app memory-profiler python /app/plot_all.py csv
