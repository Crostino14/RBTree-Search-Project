/*
 * Course: High Performance Computing 2023/2024
 *
 * Lecturer: Francesco Moscato      fmoscato@unisa.it
 *
 * Student and Creator:
 * Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
 *
 * Source Code for parallel version of Red-Black Tree search using CUDA and OpenMP.
 *
 * Copyright (C) 2023 - All Rights Reserved
 *
 * This file is part of RB Tree Search Project.
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of
 * the GNU General Public License as published by the Free Software Foundation, either version
 * 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with RB Tree Search Project.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file RBParallel_CUDA_OpenMP.c
 *
 * @brief Source file providing a parallel implementation of Red-Black Tree search using CUDA and OpenMP.
 *
 *                                    REQUIREMENTS OF THE ASSIGNMENT
 * 
 * This file contains the implementation of a parallel Red-Black Tree (RBTree) search, utilizing a hybrid approach
 * with CUDA for GPU parallelization and OpenMP for intra-GPU thread parallelism. It aims to evaluate and compare
 * the performance of parallel solutions against the known sequential implementation on a single-processing node.
 * The assignment requires the parallelization of the RBTree search algorithm using both CUDA + OpenMP and MPI + OpenMP, 
 * and students are tasked with comparing and analyzing the results and differences observed for various input types 
 * and sizes. The parallel algorithms utilized in the CUDA + OpenMP and MPI + OpenMP solutions may differ from each other.
 *
 * This code specifically addresses the CUDA + OpenMP version, leveraging the GPU's parallel processing capabilities
 * through CUDA and the multi-threading features of OpenMP. The program calculates optimal block and grid sizes for
 * CUDA kernels based on device properties and array sizes to maximize parallelism and performance. Furthermore,
 * OpenMP is employed to manage threads, distributing the workload across multiple CPU cores, enhancing overall
 * computation speed.
 *
 * The RBParallel_CUDA_OpenMP code synergistically combines CUDA's GPU parallelism and OpenMP's multi-threading
 * to efficiently search for values within a Red-Black Tree, enabling high-performance parallel computation and search operations.
 *
 * @copyright Copyright (C) 2023 - All Rights Reserved
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../include/RBMatrix.h"

/**
 * @brief CUDA kernel designed for parallel search within subtrees of a Red-Black Matrix
 *
 * This CUDA kernel performs a parallel search operation within subtrees of a Red-Black Matrix.
 * Its main objective is to distribute the workload across multiple CUDA threads efficiently
 * to search for a specific value within the specified subtrees. Each thread within a block
 * is responsible for searching a subset of rows within the matrix, aimed at optimizing the search
 * process for the target value.
 *
 * The kernel initializes by obtaining thread and block IDs, then iterates through the assigned
 * rows of each subtree to locate the specified search value. During this process, the minimum
 * index of the value within each subset is calculated and stored in shared memory. Utilizing
 * shared memory, the kernel performs a reduction operation to find the global minimum index
 * across the threads in a block.
 *
 * Furthermore, this kernel employs atomic operations to ensure consistency when updating the
 * global minimum index. By using the `atomicMin` function, the kernel accurately determines
 * and writes the overall minimum index of the target value within the entire matrix to global memory.
 *
 * This CUDA kernel leverages shared memory for intra-block synchronization and reduction,
 * optimizing memory access and computation efficiency. It allows for efficient parallel
 * processing of Red-Black Matrix subtrees, enhancing the search functionality for specific
 * values within large-scale matrices through the use of GPU parallelization techniques.
 *
 * @param rbMatrixNodes Pointer to the Red-Black Matrix nodes residing in device memory
 * @param subtreeRoots Pointer to the array containing the starting row indices of subtrees
 * @param subtreeSizes Pointer to the array containing the sizes of subtrees
 * @param numSubtrees The number of subtrees to be searched
 * @param searchValue The value to be searched in the matrix
 * @param foundIndex Pointer to the variable storing the index of the found value
 */
__global__ void searchSubtreeKernel(RBNode *rbMatrixNodes, int *subtreeRoots, int *subtreeSizes, int numSubtrees, int searchValue, int *foundIndex)
{
    extern __shared__ int s_minIndices[]; // Questo array sarà allocato dinamicamente
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int localMinIndex = INT_MAX;
    int startRow = subtreeRoots[blockID];
    int subtreeSize = subtreeSizes[blockID];

    // Search in the subtree
    for (int i = threadID; i < subtreeSize; i += blockDim.x)
    {
        int globalIndex = startRow + i;
        if (rbMatrixNodes[globalIndex].value == searchValue)
        {
            localMinIndex = min(localMinIndex, globalIndex);
        }
    }

    // Write the local minimum index to shared memory
    s_minIndices[threadID] = localMinIndex;
    __syncthreads();

    // Dynamically allocate shared memory size
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadID < stride)
        {
            s_minIndices[threadID] = min(s_minIndices[threadID], s_minIndices[threadID + stride]);
        }
        __syncthreads();
    }

    // The first thread in the block writes the result to global memory
    if (threadID == 0)
    {
        atomicMin(foundIndex, s_minIndices[0]);
    }
}

/**
 * @brief CUDA kernel employing a balanced reduction strategy for parallel subtree search within a Red-Black Matrix
 *
 * The `balancedSearchSubtreeKernel` presents an innovative approach to parallel subtree search within a Red-Black Matrix,
 * focusing on optimal reduction strategies for efficient GPU utilization. It significantly differs from the traditional
 * `searchSubtreeKernel` by introducing a hybrid reduction methodology, combining warp-level and block-level reductions.
 * 
 * Key Functionalities:
 * - Independent Subtree Search: Threads conduct autonomous searches within assigned subtrees to locate the specified value.
 * 
 * Implementation Contrast with `searchSubtreeKernel`:
 * - Warp-Level Reduction: In contrast to the traditional shared memory-based reduction in `searchSubtreeKernel`, this kernel employs
 *   `__shfl_down_sync` for warp-level reduction, allowing threads within a warp to communicate and compute the minimum index without
 *   relying extensively on shared memory. This approach minimizes shared memory usage and potentially reduces contention.
 * - Block-Level Strategy: The kernel implements a combination of warp and block-level strategies, where each warp's minimum index is
 *   stored in shared memory by the first warp thread. Subsequently, the first thread in each block conducts the final reduction,
 *   aggregating warp-level results from shared memory to determine the global minimum index. This hybrid approach reduces synchronization
 *   and minimizes the number of accesses to shared memory.
 * - Reduced Synchronization Overhead: By adopting warp-level communication and a streamlined shared memory access pattern, this kernel aims
 *   to minimize synchronization overhead compared to the conventional shared memory-based approach in `searchSubtreeKernel`, potentially
 *   enhancing performance particularly on modern GPU architectures.
 * 
 * The `balancedSearchSubtreeKernel` encapsulates an advanced reduction methodology, showcasing the effective utilization of warp-level
 * communication and block-level aggregation for optimizing parallel subtree searches. This hybrid approach endeavors to leverage modern
 * GPU architectures, potentially leading to improved performance in search operations within Red-Black Matrices.
 *
 * @param rbMatrixNodes Pointer to the Red-Black Matrix nodes residing in device memory
 * @param subtreeRoots Pointer to the array containing the starting row indices of subtrees
 * @param subtreeSizes Pointer to the array containing the sizes of subtrees
 * @param numSubtrees The number of subtrees to be searched
 * @param searchValue The value to be searched in the matrix
 * @param foundIndex Pointer to the variable storing the index of the found value
 */
__global__ void balancedSearchSubtreeKernel(RBNode *rbMatrixNodes, int *subtreeRoots, int *subtreeSizes, int numSubtrees, int searchValue, int *foundIndex)
{
    extern __shared__ int s_minIndices[]; // Array for storing minimum indices of subtrees in the kernel shared memory
    // Extracting thread, warp, and block IDs for thread management
    int threadID = threadIdx.x;                      // Thread ID within the block
    int warpID = threadID / warpSize;                // ID of the warp (group of threads)
    int blockID = blockIdx.x;                        // Block ID within the grid
    int localMinIndex = INT_MAX;                     // Local minimum index found within the subtree
    int startRow = subtreeRoots[blockID];            // Starting row index of the subtree for the block
    int subtreeSize = subtreeSizes[blockID];         // Size of the subtree associated with the block


    // Search in the subtree
    for (int i = threadID; i < subtreeSize; i += blockDim.x)
    {
        int globalIndex = startRow + i;
        if (rbMatrixNodes[globalIndex].value == searchValue)
        {
            localMinIndex = min(localMinIndex, globalIndex);
        }
    }

    // Initial reduction in shared memory
    s_minIndices[threadID] = localMinIndex;
    __syncthreads();

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        localMinIndex = min(localMinIndex, __shfl_down_sync(0xFFFFFFFF, localMinIndex, offset));
    }

    // Each first warp thread writes the minimum of its warp
    if (threadID % warpSize == 0)
    {
        s_minIndices[warpID] = localMinIndex;
    }
    __syncthreads();

    // Final reduction in the first thread of the block
    if (threadID == 0)
    {
        int blockMin = INT_MAX;
        for (int i = 0; i < blockDim.x / warpSize; ++i)
        {
            blockMin = min(blockMin, s_minIndices[i]);
        }
        atomicMin(foundIndex, blockMin);
    }
}

/**
 * @brief Parallel Search in Red-Black Matrix using CUDA and OpenMP
 *
 * The main function orchestrates a parallel search operation within a Red-Black Matrix by combining
 * CUDA and OpenMP parallelization techniques. This program initializes necessary variables,
 * allocates memory, creates a Red-Black Matrix, populates it with random values, and performs
 * a parallel search for a randomly chosen value within the matrix using CUDA and OpenMP.
 *
 * The primary objective is to leverage the parallel processing capabilities of both CUDA GPU
 * and OpenMP multi-threading to efficiently execute the search operation on the matrix.
 * The program calculates optimal block and grid sizes for CUDA kernels based on device
 * properties and array sizes to maximize parallelism and performance.
 *
 * Furthermore, this program utilizes OpenMP to handle threads, distributing the workload
 * across multiple CPU cores, enhancing overall computation speed.
 *
 * After executing the parallel search, the program measures and records the execution time
 * of the CUDA kernel. It then writes the matrix and the search result (whether the target
 * value was found or not) alongside the execution time to an output file named "CUDA_OpenMP_result.txt".
 *
 * This approach synergistically combines CUDA's GPU parallelism and OpenMP's multi-threading
 * to efficiently search for values within a Red-Black Matrix, enabling high-performance
 * parallel computation and search operations.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Exit code (0 for successful execution, 1 for failure)
 */
int main(int argc, char **argv)
{
    // Start measuring the execution time of the program
    double programStartTime = getCurrentTime();

    // Checking if the required number of arguments is provided
    if (argc != 6) {
        fprintf(stderr, "Usage: %s [ompNumThreads] [numValues] [seed] [outputFile] [csvFilePath]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Retrieving OpenMP thread count, number of values, and seed from command-line arguments
    int ompNumThreads = atoi(argv[1]); // Number of OpenMP threads
    int numValues = atoi(argv[2]); // Number of values in the Red-Black matrix
    unsigned int seed = atoi(argv[3]); // Seed for random value generation
    char *outputFileName = argv[4]; // Output txt file name
    char *csvFilePath = argv[5]; // Output csv file name

    srand(seed); // Setting the random seed for generating values

    // Creating a Red-Black Matrix and populating it with random values
    RBMatrix *rbMatrix = createMatrix(numValues); // Pointer to the Red-Black Matrix
    randomPopulateRBMatrix(rbMatrix, numValues);

    // Generating a random search value
    int searchValue = rand() % 100; // Random value for search
    // Setting OpenMP thread count
    omp_set_num_threads(ompNumThreads); 

    cudaEvent_t startEvent, stopEvent; // CUDA events for kernel execution timing
    // Creating CUDA events for timing
    cudaEventCreate(&startEvent); 
    cudaEventCreate(&stopEvent);

    // Retrieving CUDA device properties to calculate optimal block and grid sizes
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;

    // Calculating maximum number of blocks that can run concurrently
    int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor; // Maximum blocks per Streaming Multiprocessor
    int numSMs = deviceProp.multiProcessorCount; // Total number of Streaming Multiprocessors
    int totalBlocks = numSMs * maxBlocksPerSM; // Total blocks that can run concurrently on the entire GPU

    int blockSize;   // The size of each block, obtained from the CUDA launch configurator
    int minGridSize; // The minimum grid size required to achieve maximum occupancy on the GPU
    int gridSize;    // The actual grid size calculated based on the input size and block size

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, searchSubtreeKernel, 0, 0);

    // Rounding up the grid size according to the array size and total blocks
    gridSize = (numValues + blockSize - 1) / blockSize;

    // Adjusting the grid size to fully utilize the GPU if necessary
    if (gridSize < totalBlocks)
    {
        gridSize = totalBlocks;
    }
    // Calculate the number of blocks required to cover all values with the given block size
    int numBlocks = (numValues + blockSize - 1) / blockSize; 

    // Calculating rows per subtree for distributing the workload
    int rowsPerSubtree = rbMatrix->totalNodes / numBlocks;

    // Allocating memory for arrays storing subtree roots and sizes
    int *h_subtreeRoots;
    int *h_subtreeSizes;

    cudaMallocHost((void **)&h_subtreeRoots, sizeof(int) * numBlocks);
    cudaMallocHost((void **)&h_subtreeSizes, sizeof(int) * numBlocks);

    // Generating subtree information using OpenMP parallelization
    #pragma omp parallel for
    for (int i = 0; i < numBlocks; i++)
    {
        int startRow = i * rowsPerSubtree;
        int endRow = (i + 1) * rowsPerSubtree;
        h_subtreeRoots[i] = startRow;
        h_subtreeSizes[i] = endRow - startRow;
    }

    RBNode *d_rbMatrixNodes; // Pointer to device memory for Red-Black Matrix nodes
    cudaError_t cudaStatus; // Variable to capture CUDA API function statuses

    // Allocating and transferring Red-Black Matrix nodes to device memory
    cudaStatus = cudaMalloc(&d_rbMatrixNodes, sizeof(RBNode) * rbMatrix->totalNodes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_rbMatrixNodes! Error: %s\n", cudaGetErrorString(cudaStatus));
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMemcpy(d_rbMatrixNodes, rbMatrix->nodes, sizeof(RBNode) * rbMatrix->totalNodes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for copying rbMatrix->nodes to d_rbMatrixNodes! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_rbMatrixNodes);
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }


    int *subtreeRoots; // Pointer to the array of subtree starting indices
    int *subtreeSizes; // Pointer to the array of subtree sizes

    // Allocating and transferring subtree information to device memory
    cudaStatus = cudaMalloc(&subtreeRoots, sizeof(int) * numBlocks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for subtreeRoots! Error: %s\n", cudaGetErrorString(cudaStatus));
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMalloc(&subtreeSizes, sizeof(int) * numBlocks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for subtreeSizes! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(subtreeRoots);
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMemcpy(subtreeRoots, h_subtreeRoots, sizeof(int) * numBlocks, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for copying h_subtreeRoots to subtreeRoots! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(subtreeSizes);
        cudaFree(subtreeRoots);
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMemcpy(subtreeSizes, h_subtreeSizes, sizeof(int) * numBlocks, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for copying h_subtreeSizes to subtreeSizes! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(subtreeRoots);
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    // Initializing variables for storing and transferring search results
    int foundIndex = INT_MAX;
    int *d_foundIndex; // Pointer to the found value index

    cudaStatus = cudaMalloc(&d_foundIndex, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_foundIndex! Error: %s\n", cudaGetErrorString(cudaStatus));
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    cudaStatus = cudaMemcpy(d_foundIndex, &foundIndex, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for copying foundIndex to d_foundIndex! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_foundIndex);
        destroyMatrix(rbMatrix);
        free(h_subtreeRoots);
        free(h_subtreeSizes);
        return EXIT_FAILURE;
    }

    // Calculating and limiting dynamic shared memory size for the kernel
    size_t dynamicSharedMemSize = sizeof(int) * blockSize;
    if (dynamicSharedMemSize > sharedMemPerBlock)
    {
        dynamicSharedMemSize = sharedMemPerBlock;
    }

    // Timing the kernel execution using CUDA events
    cudaEventRecord(startEvent);
    balancedSearchSubtreeKernel<<<gridSize, blockSize, dynamicSharedMemSize>>>(d_rbMatrixNodes, subtreeRoots, subtreeSizes, numBlocks, searchValue, d_foundIndex);
    cudaEventRecord(stopEvent);
    
    cudaDeviceSynchronize();

    // Calculating and retrieving elapsed time for kernel execution
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    double executionTime = elapsedTime / 1000.0;

    // Copying the search result from device memory to host memory
    cudaMemcpy(&foundIndex, d_foundIndex, sizeof(int), cudaMemcpyDeviceToHost);

    // End measuring the execution time of the program
    double programEndTime = getCurrentTime();
    double programExecutionTime = programEndTime - programStartTime;

    // Writing the search result and execution time to an output file
    FILE *fp = fopen(outputFileName, "w");
    if (fp != NULL)
    {
        fprintf(fp, "Input Parameters:\n");
        fprintf(fp, "Number of OpenMP threads: %d\n", ompNumThreads);
        fprintf(fp, "Number of CUDA threads per block: %d\n", blockSize);
        fprintf(fp, "Number of values (nodes in RBTree) to insert: %d\n", numValues);
        fprintf(fp, "\n");
        fprintf(fp, "Parallel search with CUDA for value %d.\n", searchValue);
        if (foundIndex < INT_MAX)
        {
            fprintf(fp, "Value found at node %d\n", foundIndex);
        }
        else
        {
            fprintf(fp, "Value not found.\n");
        }
        fprintf(fp, "Execution time of parallel search: %.9f seconds\n", executionTime);
        fprintf(fp, "Execution time of total program: %.9f seconds.\n", programExecutionTime);
        fprintf(fp, "\n");
        printMatrix(rbMatrix, fp);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "Failed to open file for output.\n");
    }
    writeResultsToCSV(csvFilePath, ompNumThreads, 0, numValues, blockSize, executionTime, programExecutionTime, foundIndex);
    // Freeing allocated memory and cleaning up resources
    cudaFree(d_rbMatrixNodes);
    cudaFree(subtreeRoots);
    cudaFree(subtreeSizes);
    cudaFree(d_foundIndex);

    cudaFreeHost(h_subtreeRoots);
    cudaFreeHost(h_subtreeSizes);
    destroyMatrix(rbMatrix);

    return 0; // Exiting the program
}