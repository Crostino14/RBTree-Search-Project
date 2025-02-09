# 🔴⚫ Red-Black Tree Search Project

## 📌 Overview
This project focuses on the **parallelisation** of the **Red-Black Tree Search Algorithm**, a self-balancing binary search tree, using different computational approaches:
- **Sequential implementation** 🏛️ (baseline).
- **MPI + OpenMP implementation** ⚡ (distributed parallelism).
- **CUDA + OpenMP implementation** 🚀 (GPU acceleration).

## 🎯 Objectives
- Implement and analyse the **Red-Black AVL Tree Search** algorithm in **parallel computing environments**.
- Evaluate performance differences between **MPI, OpenMP, and CUDA** implementations.
- Conduct extensive benchmarking with **different input sizes** and **hardware configurations**.

## 📖 Theoretical Background
- **Red-Black AVL Tree** combines **Red-Black Tree** and **AVL Tree** properties to maintain efficient balance for search operations.
- **Parallel Computing Approaches:**
  - **OpenMP**: Multi-threading on shared-memory architectures.
  - **MPI**: Distributed computing with message-passing.
  - **CUDA**: GPU acceleration using thousands of cores.

## ⚙️ Implementation
The project implements **three versions** of the search algorithm:
1. **Sequential version** (baseline).
2. **MPI + OpenMP** version: 
   - Distributes work across **multiple processes** using **MPI**.
   - Utilises **OpenMP threads** for intra-process parallelism.
3. **CUDA + OpenMP** version:
   - **CUDA Kernels** handle subtree search in **parallel**.
   - **Optimised memory management** with shared memory and atomic operations.

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

## 🤝 Contributors
- **Agostino Cardamone** 🎓 (Student & Creator)
- **Supervisor:** Francesco Moscato 🏫



