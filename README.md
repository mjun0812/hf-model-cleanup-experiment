# HuggingFace Model Memory Cleanup Experiment

HuggingFaceのモデルをメモリから解放する方法の調査。

## CPU

```bash
cd cpu
./run.sh
```

### 1. `del model; gc.collect();`で削除

![1](cpu/figs/2048m/memory_usage_1.png)

#### 2. モデルのパラメータを手動で削除 + `del model; gc.collect();`

![2](cpu/figs/2048m/memory_usage_2.png)

#### 3. `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![3](cpu/figs/2048m/memory_usage_3.png)

#### 4. モデルのパラメータを手動で削除 + `del model; gc.collect();` + `ctypes.CDLL("libc.so.6")`で削除

![4](cpu/figs/2048m/memory_usage_4.png)

#### 5. `os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"` + `del model; gc.collect();`で削除

![5](cpu/figs/2048m/memory_usage_5.png)

#### 6. `export MALLOC_TRIM_THRESHOLD_=-1` + `del model; gc.collect();`で削除

![6](cpu/figs/2048m/memory_usage_6.png)

#### All

![all](cpu/figs/2048m/memory_usage_all.png)
