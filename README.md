<div align="center">

<h1>🔴⚫🌳 Red-Black Tree Search Project 🌳 🔴⚫</h1>

![MPI](https://img.shields.io/badge/MPI-0071C5?style=for-the-badge&logo=openmpi&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-7CB342?style=for-the-badge&logo=openmp&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C](https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

*Parallelized Red-Black Tree search with MPI, OpenMP, and CUDA - Performance analysis on various hardware configurations*

[📖 Full Report](#-full-report) • [🛠️ Makefile Commands](#-makefile-commands) • [📊 Performance](#-performance-analysis)

</div>

## 📚 Project Overview
This project investigates the parallelisation of the Red-Black Tree Search Algorithm, a self-balancing binary search tree widely used for efficient data retrieval. The study examines how different computational paradigms can be employed to improve performance and scalability. In particular, this project implements 3 different computational approaches:

- **Sequential implementation** 🏛️ (baseline).
- **MPI + OpenMP implementation** ⚡ (distributed parallelism).
- **CUDA + OpenMP implementation** 🚀 (GPU acceleration).

In particular, the sequential version of the algorithm is considered as a baseline and is compared against two parallel approaches: an implementation based on the Message Passing Interface (MPI) combined with OpenMP, and a GPU-accelerated solution that integrates CUDA with OpenMP.

## 📖 Theoretical Background
The algorithm under investigation relies on the **Red-Black AVL Tree**, which combines properties of both **Red-Black Tree** and **AVL Tree** to maintain efficient balance for search operations.

Parallelisation is explored through three distinct paradigms:
  - **OpenMP** enables shared-memory multi-threading, exploiting modern CPUs with multiple cores.
  - **MPI** introduces distributed parallelism through message passing, which makes it suitable for cluster environments but also exposes the system to communication overhead.
  - **CUDA** leverages thousands of lightweight GPU cores, enabling highly parallel execution when the problem size is sufficiently large to offset the cost of data transfer between host and device.

## ⚙️ Implementation
The implementation is organised across a set of source files located in the `src/` directory, each designed to separate concerns related to algorithmic logic, computational kernel, and matrix operations. Below is a concise description of the main files and their roles:

`RBSequential.c` — Contains the sequential (baseline) implementation of the Red-Black tree search algorithm. It implements the tree operations (insertion, search, balancing) without any form of parallelism. This file serves as a reference to measure speed-ups obtained by the parallel versions.

`RBParallel_MPI_OpenMP.c` — Implements the Red-Black tree search using a hybrid approach: inter-process parallelism provided by MPI, and intra-process parallelism using OpenMP. It orchestrates division of work among MPI processes, handles thread creation within each process, and manages coordination and synchronization overhead.

`RBParallel_CUDA_OpenMP.cu` — Implements the version leveraging CUDA for GPU acceleration, combined with OpenMP for parallelism on the CPU side. It contains kernel definitions for subtree search, routines for memory transfer between host and device, and optimisations such as shared memory use and atomic operations to maintain correctness under parallel execution.

`RBMatrix.c` — Provides utility functions for matrix operations needed by the test harness or algorithmic support (for example, generating input data, transforming test sets). It decouples auxiliary data handling and input matrix setup from the core tree search logic.

`CreateTables.py` — Post-processing script written in Python that reads the CSV output of tests, aggregates the results, and produces tables summarising performance metrics (such as elapsed time, speed-up, scaling behaviour).

`Performance.py` — Another Python script aimed at generating visual representations (plots) of performance trends across different implementations, input sizes, and parallel configurations; it also computes statistical metrics needed for analysis (e.g. mean, variance of runs).

These files together form the full pipeline: from compilation, execution under different configurations, generation of raw result files, to analysis via tables and plots.

## 🖥️ Experimental Setup
### 💻 Hardware
- **CPU:** AMD Ryzen 5 7640HS (6 cores, 12 threads)
- **GPU:** NVIDIA RTX 4060 (8GB VRAM, 24 CUDA cores)
- **RAM:** 16GB DDR5

### 🛠️ Software
- **OS:** Windows 10 Pro
- **Compilers:** GCC, NVCC (CUDA 12.3), MPI Compiler
- **Tools:** Python 3.11 for analysis, CMake for builds

## 📊 Performance Analysis
Performance was evaluated by measuring:
- **Execution time** under different parallel configurations.
- **Speed-up** compared to the sequential baseline.
- **Scalability** across varying input sizes.

**Key Findings:**
- **MPI+OpenMP** is effective for **multi-node CPU clusters**, but performance is limited by communication overhead.
- **CUDA+OpenMP** achieves better speed-up on **large datasets**, benefiting from **GPU parallelism**.
- **Sequential execution** is efficient for small inputs but does not scale well.

---

## 🔧 Prerequisites

Before running the Makefile commands, ensure you have the following installed:

- **GCC compiler** 🖥️ for C code compilation.
- **MPICC compiler** 🔀 for MPI parallel execution.
- **NVCC compiler** 🎮 from the CUDA toolkit for GPU-based computation.
- **OpenMP** 🏎️ for multi-threading support.
- **Python 3** 🐍 with:
  - `matplotlib` 📊 (for plots)
  - `pandas` 📑 (for data processing)
  - `numpy` 🔢 (for numerical computations)

To install Python dependencies, run:
```sh
make install_dependencies
```

---

## 🛠️ Makefile Commands

The **Makefile** automates compilation, testing, cleaning, and performance analysis.

### ⚙️ Compilation Commands
| Command | Description |
|---------|------------|
| `make clean` | 🗑️ Removes compiled binaries and output data from previous builds. |
| `make all` | 🔄 Compiles **all versions** (with different optimisation levels). |
| `make compile0` | 🏗️ Compiles with `-O0` (no optimisation). |
| `make compile1` | 🔹 Compiles with `-O1` (basic optimisations). |
| `make compile2` | 🔷 Compiles with `-O2` (further optimisations). |
| `make compile3` | 🚀 Compiles with `-O3` (full optimisation). |

---

### 🧪 Testing Commands
| Command | Description |
|---------|------------|
| `make test0` / `make test1` / `make test2` / `make test3` | ✅ Runs tests for different compiled versions. |
| `make test00` / `make test01` / `make test02` / `make test03` | 🔄 Runs tests with predefined **values, threads, and MPI processes**. |
| `make all_test1` / `make all_test2` | 🏆 Combines multiple test targets for **convenience**. |

---

### 📊 Performance Analysis
| Command | Description |
|---------|------------|
| `make tables` | 📑 Runs a Python script to create **tables** from CSV results. |
| `make performance` | 📊 Runs a Python script to **analyse performance** and generate plots & tables. |

---

## 🚀 Usage Guide

To **compile and test** the project with all optimisation levels and generate performance tables and plots, run:

```sh
make help
make clean
make all
make all_test1
make tables
make performance
```

Each command should be run in the terminal from the **root directory** of the project.

---

## 📖 Full Report
For detailed **performance results** and **analysis**, see the full **[Project Report](ProjectReport.pdf)** 📑.

---

## 🔗 Related Resources

- [OpenMP Official Documentation](https://www.openmp.org/specifications/)
- [MPI Forum Standards](https://www.mpi-forum.org/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 🤝 Contributors
- **Agostino Cardamone** 🎓 (Student & Creator)
- **Supervisor:** Francesco Moscato 🏫
