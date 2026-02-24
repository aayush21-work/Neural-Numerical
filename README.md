# Neural-Numerical: Hybrid ML-Accelerated Root Finder
**Accelerating Numerical Convergence with Ultra-Low Latency Neural Warm-Starts**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview
This project demonstrates a hybrid approach to solving non-linear systems. By utilising a Neural Network to provide an initial guess (Warm Start) for a Newton-Raphson solver, we significantly reduce the iteration count and eliminate convergence failures common in "cold start" numerical methods. The idea is to expand this into a full blown setup for a large pool of commonly used numerical methods espicially which depends on user given priors. The benchmark shows the prototype with the simple benchmark, the target is a polynomial so the gap is not that much but for a stiff function a large speedup is expected which will be added in due time.

---

## Benchmark Results (2000 Samples)
The following data compares a standard **Newton-Raphson** implementation against our **Hybrid ML + Newton** approach.

| Metric | Plain Newton-Raphson | ML + Newton (Hybrid) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Time** | 27.12 ms | **20.33 ms** | **1.33x Speedup** |
| **Avg Iterations** | 7.25 | **5.00** | **1.45x Reduction** |
| **Convergence Failures** | 9 | **0** | **100% Reliability** |

### **GPU Inference Performance**
The neural network component is optimized for ultra-low latency, ensuring that the "guess" phase does not bottleneck the numerical refinement.
* **Total GPU Inference (2000 samples):** 0.171 ms
* **Per-Sample Latency:** **0.085 μs**



---

## Qualitative Analysis
Five random samples illustrating the Hybrid solver's ability to stay within the basin of attraction:

```text
Sample 1:
Plain Newton  : steps=6
Hybrid Newton : steps=4 (Guess Error: 0.0129)

Sample 2:
Plain Newton  : steps=6
Hybrid Newton : steps=8 (Guess Error: 0.1190)

Sample 3:
Plain Newton  : steps=5
Hybrid Newton : steps=4 (Guess Error: 0.0299)

Sample 4:
Plain Newton  : steps=8
Hybrid Newton : steps=5 (Guess Error: 0.0994)

Sample 5:
Plain Newton  : steps=7
Hybrid Newton : x=-2.167243
Hybrid Newton : x=1.528670 (Converged to alternate root via ML)
```

## Usage
Clone the repo and simply run ```make```. The default is the cuda accelrated version but if that does not work try ```make cpu``` which should work albeit slow. The overall result will not change but the training part may take longer depending upon the cpu power and model.
