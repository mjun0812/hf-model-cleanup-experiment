# HuggingFace Model Release Memory

HuggingFaceのモデルをメモリから解放する方法の調査。

```bash
docker build -t memory-profiler .
```

## 1. `del model; gc.collect();`で削除

![1](figs/memory_usage_1.png)

## 2. モデルのパラメータを手動で削除 + `del model; gc.collect();`

![2](figs/memory_usage_2.png)

## 3. `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![3](figs/memory_usage_3.png)

## 4. モデルのパラメータを手動で削除 + `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![4](figs/memory_usage_4.png)
