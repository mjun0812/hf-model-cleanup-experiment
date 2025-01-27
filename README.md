# HuggingFace Model Memory Cleanup Experiment

HuggingFaceのモデルをメモリから解放する方法の調査。  
このリポジトリはHuggingFaceモデルを対象としていますが、PyTorchモデルでも同じ方法が使えます。

This repository is for HuggingFace models, but you can also use PyTorch models.

## CPU

```bash
cd cpu
./run.sh
```

### 1. `del model; gc.collect();`で削除

![1](cpu/figs/2048m/memory_usage_1.png)

### 2. モデルのパラメータを手動で削除 + `del model; gc.collect();`

![2](cpu/figs/2048m/memory_usage_2.png)

### 3. `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![3](cpu/figs/2048m/memory_usage_3.png)

### 4. モデルのパラメータを手動で削除 + `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![4](cpu/figs/2048m/memory_usage_4.png)

### 5. `os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"` + `del model; gc.collect();`で削除

![5](cpu/figs/2048m/memory_usage_5.png)

### 6. `export MALLOC_TRIM_THRESHOLD_=-1` + `del model; gc.collect();`で削除

![6](cpu/figs/2048m/memory_usage_6.png)

### All

![all](cpu/figs/2048m/memory_usage_all.png)

## GPU

```
cd gpu
./run.sh
```

C++ CUDA

```bash
cd gpu
nvcc main.cu
./a.out
```

### 1. `del model; gc.collect();`

![1-gpu](gpu/figs/memory_usage_1.png)

### 2. `del model; gc.collect();` + `torch.cuda.empty_cache()`

![2-gpu](gpu/figs/memory_usage_2.png)

### 3. `del model; gc.collect();` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`

![3-gpu](gpu/figs/memory_usage_3.png)

### 4. `del model; gc.collect();` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` + `torch._C._cuda_clearCublasWorkspaces()`

![4-gpu](gpu/figs/memory_usage_4.png)

### 5. `del model; gc.collect();` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` + `torch._C._cuda_clearCublasWorkspaces()` + `torch.backends.cuda.cufft_plan_cache.clear()`

![5-gpu](gpu/figs/memory_usage_5.png)

### 6. `model.to("cpu")` + `del model; gc.collect();` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` + `torch._C._cuda_clearCublasWorkspaces()` + `torch.backends.cuda.cufft_plan_cache.clear()`

![6-gpu](gpu/figs/memory_usage_6.png)

### 7. `model.to("cpu")` + `del model; gc.collect();` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` + `torch._C._cuda_clearCublasWorkspaces()` + `torch.backends.cuda.cufft_plan_cache.clear()` + `cudaDeviceReset()`

Error

### All

![all-gpu](gpu/figs/memory_usage_all.png)
