#!/bin/bash -xe

docker build -t memory-profiler-cpu .

MEMORY_SIZES=(2048m 4096m 8192m 16384m)
for MEMORY_SIZE in ${MEMORY_SIZES[@]}; do
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app memory-profiler python /app/1.py
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app memory-profiler python /app/2.py
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app memory-profiler python /app/3.py
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app memory-profiler python /app/4.py
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app memory-profiler python /app/5.py
    docker run -it --rm --memory=$MEMORY_SIZE -v $(pwd):/app -e MALLOC_TRIM_THRESHOLD_=-1 memory-profiler  python /app/6.py

    mkdir -p figs/$MEMORY_SIZE csv/$MEMORY_SIZE
    mv -f figs/*.png figs/$MEMORY_SIZE
    mv -f csv/*.csv csv/$MEMORY_SIZE

    docker run -it --rm -v $(pwd):/app memory-profiler python /app/plot_all.py csv/$MEMORY_SIZE
done
