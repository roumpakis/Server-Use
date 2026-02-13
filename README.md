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



# Best Practices

-   Close unused notebooks.
-   Avoid running multiple heavy jobs simultaneously.
-   Monitor your usage before starting training.
-   Use reasonable batch sizes.
-   Restart kernels after heavy workloads.
