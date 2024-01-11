# Red-Black Tree Search Project Makefile Guide

This Makefile is part of the RB Tree Search Project, which includes a sequential version of Red-Black Tree search and parallel versions using MPI and OpenMP, and CUDA and OpenMP.

## Prerequisites

Before running the Makefile commands, ensure you have the following installed:
- GCC compiler for C code compilation.
- MPICC compiler for MPI code compilation.
- NVCC compiler from the CUDA toolkit for CUDA code compilation.
- Python 3 and required Python packages (matplotlib, pandas, numpy).
- OpenMP for parallel programming support.
- matplotlib, pandas and numpy libraries available with the make command of the makefile (see make install_dependencies)

## Makefile Commands

The Makefile includes several targets that automate the compilation, testing, cleaning, and performance evaluation of the project.

## make clean

Removes all compiled binaries and output data from previous builds and tests.

## make all

Compiles all versions (with different optimization levels) of the Red-Black Tree search project.

## make install_dependencies

Installs the required Python packages of matplotlib, pandas and numpy.

## make compile0 make compile1 make compile2 make compile3

Compiles the project with different optimization levels:

compile0 for O0 (no optimization)
compile1 for O1 (basic optimizations)
compile2 for O2 (further optimizations)
compile3 for O3 (full optimization)

## make test0 make test1 make test2 make test3 make test00 make test01 make test02 make test03

Runs tests for different compiled versions of the project with predefined numbers of values, threads, and MPI processes using a fixed seed.

## make all_test1 make all_test2

Combines multiple test targets for convenience.

## make tables

Runs a Python script to create tables from the CSV results.

## make performance

Runs a Python script to analyze the performance and generate plots and tables.

## Usage
To compile and test the project with all optimization levels and generate performance tables and plots, you can run:

make clean
make all
make all_test1
make tables
make performance

Each command should be run in the terminal from the root directory of the project. Ensure all dependencies are installed, and the environment is set up correctly for MPI, OpenMP and CUDA before running the commands.

This Makefile is designed to facilitate the development and testing workflow of the RB Tree Search Project. By using these commands, you can easily manage the build process and evaluate the performance of the sequential and parallel implementations of the Red-Black Tree search algorithm.