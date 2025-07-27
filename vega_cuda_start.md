# Learn CUDA Programming using Vega.

This repository aims to curate exercises for learning CUDA-based GPU programming. It contains code snippets, notes, and benchmarks designed to run on a High-Performance Computing (HPC) cluster.

## HPC Environment Setup

Follow the steps below to configure the environment on the HPC system (Vega).

### Step 1: Request a GPU Node

Use `msub` to request an interactive session on the GPU node:

```bash
msub -I -l nodes=gpu01:ppn=192
```

> `gpu01` is the GPU node, and `ppn=192` requests 192 processing cores.

### Step 2: Load Required NVIDIA Modules

Purge current modules and load the CUDA environment:

```bash
module purge
module use /apps/spack/share/spack/modules/linux-rocky8-zen4
module load cuda/12.2.0-gcc-13.2.0-nwhgfor
```

This ensures access to the correct CUDA toolkit and compatible compiler toolchain.

---

## Tutorials Covered

This project follows several tutorials and reference guides for CUDA development. Below is the list of tutorials currently covered:

* [CUDA Tutorial by ReadTheDocs](https://cuda-tutorial.readthedocs.io/en/latest/): A comprehensive beginner-friendly guide introducing CUDA basics, memory management, and kernel execution.

More references will be added as the learning progresses.

---

## Next Steps

Coming soon:

* CUDA kernel development
* Performance profiling
* Parallel algorithm implementation
* Benchmarks and documentation

