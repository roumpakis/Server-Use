# Shared GPU Server --- User Monitoring Manual (Jupyter / ML)

## Overview

This guide explains how users can monitor **CPU, RAM, and GPU usage** on
a shared server environment.
# System Resources Overview

This server provides the following shared hardware resources.
All resources are shared among active users.

---

## 🧠 CPU

- **Model:** AMD EPYC 7513 32-Core Processor
- **Sockets:** 2
- **Cores per socket:** 32
- **Threads per core:** 2
- **Total Logical CPUs:** 128
- **Architecture:** x86_64
- **NUMA Nodes:** 8

⚠️ Total available compute threads: **128**
All CPU cores are shared across users.

---

## 💾 RAM (System Memory)

- **Total RAM:** 1.0 TiB (~1024 GB)
- **Swap:** Disabled (0B)

⚠️ There is NO swap space.
If RAM is exhausted, processes may be killed by the system.

---

## 🎮 GPU

- **GPU Model:** NVIDIA A100-SXM4-40GB
- **Number of GPUs:** 1
- **Total GPU Memory (VRAM):** 40 GB (40960 MiB)

⚠️ The GPU is fully shared.
All users compete for:
- 40 GB VRAM
- GPU compute (SM utilization)
- Memory bandwidth

If one job uses all 40GB VRAM, other users cannot allocate GPU memory.

---

## 💽 Storage

- **Total Storage:** 27 TB
- **Used:** 19 TB
- **Available:** 7.5 TB
- **Usage:** 72%

⚠️ Disk space is shared across users.

---

# Resource Sharing Summary

This is a high-performance shared ML server with:

- 128 CPU threads
- 1 TB RAM
- 1× NVIDIA A100 (40GB VRAM)
- 27 TB storage

All resources are shared among active users.
Please monitor your usage responsibly.

------------------------------------------------------------------------
# 0. Light GPU Utilization Demo (For Screenshots)

Run inside Jupyter Notebook:

``` python
import time
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"
n = 4096

a = torch.randn((n, n), device=device)
b = torch.randn((n, n), device=device)

print("Running light GPU workload...")

for _ in range(100):
    c = a @ b
    torch.cuda.synchronize()
    time.sleep(0.05)

print("Finished.")
```

📸 **Screenshot Placeholder:**\
*Add screenshot showing GPU utilization during execution*

------------------------------------------------------------------------

# 1. CPU & RAM Monitoring

## 1.1 Live CPU View

``` bash
top -u $USER
```


## CPU Usage Comparison

<p align="center">
  <img src="simages/top_before.png" width="48%">
  <img src="simages/top_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>




------------------------------------------------------------------------

## 1.2 Total RAM Usage (Your Processes Only)

``` bash
ps -u $USER -o rss= | awk '{sum+=$1} END {printf "My RAM usage: %.2f GB\n", sum/1024/1024}'
```

## RAM Usage Comparison

<p align="center">
  <img src="simages/ram_before.png" width="48%">
  <img src="simages/ram_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

## 1.3 Top RAM% & CPU% Consuming Processes

``` bash
ps -u $USER -o pid,cmd,%cpu,%mem,rss --sort=-rss | head
```

<p align="center">
  <img src="simages/ps_before.png" width="48%">
  <img src="simages/ps_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

# 2. GPU Monitoring

## 2.1 Snapshot View

``` bash
nvidia-smi
```


<p align="center">
  <img src="simages/sm1_before.png" width="48%">
  <img src="simages/smi1_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

## 2.2 Live Refresh View

``` bash
watch -n 1 nvidia-smi
```

<p align="center">
  <img src="simages/sm2_before.png" width="48%">
  <img src="simages/smi2_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

## 2.3 Real-Time GPU Metrics

``` bash
nvidia-smi dmon -d 1
```

### Column Description

-   `gpu / Idx` → GPU index
-   `pwr (W)` → Power consumption
-   `gtemp (C)` → Core temperature
-   `mtemp (C)` → Memory temperature (if supported)
-   `sm (%)` → GPU compute utilization
-   `mem (%)` → Memory controller utilization
-   `mclk (MHz)` → Memory clock
-   `pclk (MHz)` → GPU core clock


<p align="center">
  <img src="simages/smi3_before.png" width="48%">
  <img src="simages/smi3_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

# 3. Identify Your GPU Processes

``` bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

<p align="center">
  <img src="simages/smi4_before.png" width="48%">
  <img src="simages/smi4_after.png" width="48%">
</p>

<p align="center">
  <b>Before</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>After</b>
</p>


------------------------------------------------------------------------

# 4) Releasing RAM / VRAM (PyTorch & TensorFlow)

In a shared GPU environment, users should release memory after finishing
training or experiments. The following methods affect **only your own
Jupyter kernel / Python process**.

------------------------------------------------------------------------

## ✅ PyTorch: Free VRAM/RAM (within your notebook)

### 1) Delete large tensors / models and run garbage collection

``` python
import gc
del model
del optimizer
del loss
del batch
gc.collect()
```

### 2) Clear CUDA cache (VRAM cache)

``` python
import torch
torch.cuda.empty_cache()
```

### 3) (Optional) Clean IPC handles (useful with DataLoader multiprocessing)

``` python
import torch
torch.cuda.ipc_collect()
```

### 4) Synchronize before cleanup (ensure GPU kernels finish)

``` python
import torch
torch.cuda.synchronize()
```

### 🔹 Recommended Cleanup Combination

``` python
import gc, torch

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
```

📌 Note: `empty_cache()` does not always return all VRAM to the system,
but it releases cached memory so new allocations can succeed.

------------------------------------------------------------------------

## ✅ TensorFlow / Keras: Free VRAM/RAM (within your notebook)

### 1) Clear Keras session

``` python
import tensorflow as tf
tf.keras.backend.clear_session()
```

### 2) Run garbage collection

``` python
import gc
gc.collect()
```

### 3) Prevent full VRAM pre-allocation (recommended at the TOP of your notebook)

``` python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

This prevents TensorFlow from reserving all GPU memory at startup.

------------------------------------------------------------------------

# PyTorch Training Template + Cleanup

``` python
# ===== PyTorch Training Template (shared GPU friendly) =====
import gc
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ...            # your model
optimizer = ...        # your optimizer
criterion = ...        # your loss

model.to(device)
model.train()

for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

# ===== Cleanup (ONLY affects your kernel / process) =====
del out, loss, x, y, batch
del model, optimizer, criterion

gc.collect()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("PyTorch cleanup done. If VRAM still appears high, restart the kernel.")
```

------------------------------------------------------------------------

# TensorFlow / Keras Training Template + Cleanup

``` python
# ===== TensorFlow/Keras Training Template (shared GPU friendly) =====
import gc
import tensorflow as tf

# Put this at the TOP of your notebook
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

model = ...  # tf.keras.Model(...)
model.compile(...)
model.fit(...)

# ===== Cleanup (ONLY affects your kernel / process) =====
tf.keras.backend.clear_session()
del model
gc.collect()

print("TensorFlow cleanup done. If VRAM still appears high, restart the kernel.")
```

------------------------------------------------------------------------

## ⚠ Important

If memory does not fully drop after cleanup, **restart the Jupyter
kernel**.

This is the only 100% reliable way to release all RAM/VRAM from your
session.


------------------------------------------------------------------------

# Best Practices

-   Close unused notebooks.
-   Avoid running multiple heavy jobs simultaneously.
-   Monitor your usage before starting training.
-   Use reasonable batch sizes.
-   Restart kernels after heavy workloads.
