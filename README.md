# 🔴⚫ Red-Black Tree Search Project

## 📌 Overview
This project focuses on the **parallelisation** of the **Red-Black Tree Search Algorithm**, a self-balancing binary search tree, using different computational approaches:
- **Sequential implementation** 🏛️ (baseline).
- **MPI + OpenMP implementation** ⚡ (distributed parallelism).
- **CUDA + OpenMP implementation** 🚀 (GPU acceleration).

The goal is to compare the efficiency of **parallel computing techniques** and **optimise** search performance.

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
For detailed **performance results** and **analysis**, see the full **[Project Report](Project Report HPC.pdf)** 📑.

---

## 🤝 Contributors
- **Agostino Cardamone** 🎓 (Student & Creator)
- **Supervisor:** Francesco Moscato 🏫



