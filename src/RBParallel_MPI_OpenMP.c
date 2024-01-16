/*
 * Course: High Performance Computing 2023/2024
 *
 * Lecturer: Francesco Moscato      fmoscato@unisa.it
 *
 * Student and Creator:
 * Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
 *
 * Source Code for parallel version of Red-Black Tree search using MPI and OpenMP.
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
 * @file RBParallel_MPI_OpenMP.c
 *
 * @brief Source file providing a parallel implementation of Red-Black Tree search using MPI and OpenMP.
 *
 *                                    REQUIREMENTS OF THE ASSIGNMENT
 * 
 * The file contains the implementation of a parallel Red-Black Tree (RBTree) search, utilizing a hybrid approach
 * with MPI for inter-process communication and OpenMP for intra-process parallelism. This code aims to compare the
 * performance of parallel solutions against the sequential implementation on a single-processing node.
 * The assignment requires parallelizing the RBTree search algorithm using both OpenMP + MPI and OpenMP + CUDA, 
 * comparing their performance against the known single-processing node solution. Students need to discuss and analyze 
 * the results and differences observed for various input types and sizes. The parallel algorithms utilized 
 * in the OpenMP + MPI and OpenMP + CUDA solutions may differ from each other.
 *
 * @copyright Copyright (C) 2023 - All Rights Reserved
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "../include/RBMatrix.h"
#include "../include/RBNode.h"


/**
 * @struct Task
 * @brief Structure representing a task for parallel search within RBMatrix.
 */
typedef struct
{
    int startIndex; // Starting index in the RBNode array
    int endIndex;   // Ending index
} Task;

/**
 * @brief Initialize and populate an RBMatrix with random integer values.
 *
 * The function is designed to create a Red-Black Tree represented as a matrix (RBMatrix)
 * and populate it with random integer values. The RBMatrix is a key data structure in this
 * parallel algorithm, which allows for efficient data manipulation and search operations
 * suitable for distributed and multi-threaded processing. The `numValues` parameter
 * dictates the size of the tree, which in turn affects the complexity and performance of the
 * parallel search operation. Proper initialization and random population of the matrix are
 * crucial for a fair and unbiased performance analysis of the parallel search algorithm.
 * Upon successful creation, the function returns a pointer to the populated RBMatrix.
 * If the matrix cannot be created, it aborts the MPI environment to prevent any undefined
 * behavior in the subsequent parallel search steps. This is a critical step as it sets up the
 * data before the MPI processes and OpenMP threads undertake the search task.
 *
 * @param numValues The total number of nodes/values to insert into the RBMatrix.
 * @return RBMatrix* A pointer to the newly created and populated RBMatrix.
 */
RBMatrix *createAndPopulateRBMatrix(int numValues)
{
    RBMatrix *rbMatrix = createMatrix(numValues);
    if (!rbMatrix) {
        fprintf(stderr, "Errore nella creazione della matrice RBMatrix.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    randomPopulateRBMatrix(rbMatrix, numValues);
    return rbMatrix;
}


/**
 * @brief Distribute work segments of the RBMatrix to MPI processes as discrete tasks.
 *
 * This function is responsible for the crucial step of dividing the entire search domain, 
 * represented by the RBMatrix, into smaller, manageable tasks that can be processed in parallel 
 * by different MPI processes. The division of labor is based on the total number of nodes 
 * within the RBMatrix and the number of available MPI processes. Each task encompasses a range 
 * of indices, defining which portion of the RBMatrix a given MPI process will search. The 
 * distribution strategy ensures that each process receives a nearly equal share of the workload, 
 * with any remainder nodes being distributed one by one to the initial processes. This helps 
 * maintain load balancing across processes, which is vital for achieving optimal parallel 
 * performance. The function returns an array of Task structures, with each Task corresponding 
 * to the search range assigned to a particular MPI process. This organized approach to task 
 * distribution sets the stage for an efficient parallel search, as it aligns with the data 
 * parallelism model where each process operates on a different subset of the data.
 *
 * @param rbMatrix Pointer to the RBMatrix containing the nodes to be searched.
 * @param numProcesses The total number of MPI processes available for the search.
 * @return Task* An array of Task structures, with each Task defining the search range for an MPI process.
 */
Task *createTasksForAllProcesses(RBMatrix *rbMatrix, int numProcesses)
{
    Task *tasks = malloc(numProcesses * sizeof(Task));
    int totalNodes = rbMatrix->totalNodes;
    int nodesPerProcess = totalNodes / numProcesses;
    int remainingNodes = totalNodes % numProcesses;

    for (int i = 0; i < numProcesses; ++i)
    {
        int start = i * nodesPerProcess;
        int extraNode = (i < remainingNodes) ? 1 : 0;
        int end = start + nodesPerProcess + extraNode;

        if (end > totalNodes)
        {
            end = totalNodes;
        }

        tasks[i].startIndex = start;
        tasks[i].endIndex = end;
    }

    return tasks;
}

/**
 * @brief Perform a parallelized search for a value within a segment of the RBMatrix.
 *
 * This function is at the heart of the parallel search algorithm. It utilizes OpenMP to
 * distribute the search workload across multiple threads, each operating on a subset of nodes
 * within the given task's range. The task structure includes start and end indices, delineating
 * a segment of the RBMatrix to be searched. The OpenMP `parallel for` directive, along with
 * dynamic scheduling, ensures a balanced distribution of work, where threads dynamically claim
 * work segments as they become available. The `reduction` clause is critical for identifying
 * the smallest index at which the desired value is found, as it performs a parallel reduction
 * across the threads to find the minimum of all indices where the search value matches.
 * The search operation's parallel nature aims to leverage multi-core processors to expedite
 * the search process, crucial for large data sets where sequential search would be
 * prohibitively slow. The function returns the minimum index where the value is found
 * or INT_MAX if the value is not present in the provided RBMatrix segment.
 *
 * @param task The task containing the range indices to define the search boundaries within the RBMatrix.
 * @param rbMatrix The RBMatrix instance to search within.
 * @param searchValue The integer value to find within the RBMatrix.
 * @return int The minimum index where the search value is found, or INT_MAX if not found.
 */
int parallelSearchInSubtree(Task task, RBMatrix *rbMatrix, int searchValue) {
    if (!rbMatrix) {
        fprintf(stderr, "Matrice RB non inizializzata nel processo di ricerca parallela.\n");
        return INT_MAX;
    }

    int globalFoundIndex = INT_MAX;

    // OpenMP parallelization directive with dynamic scheduling and reduction clause
#pragma omp parallel for schedule(dynamic) reduction(min : globalFoundIndex)
    for (int i = task.startIndex; i < task.endIndex; ++i) {
        // Each thread checks a subset of RBMatrix nodes for the search value
        if (rbMatrix->nodes[i].value == searchValue) {
            // The 'reduction' clause ensures the minimum index where the value is found among all threads
            globalFoundIndex = i;
        }
    }

    return globalFoundIndex;
}

/**
 * @brief Main function for parallel Red-Black Tree search using MPI and OpenMP.
 *
 * The main function coordinates the parallel search of a Red-Black Tree using a hybrid approach
 * with MPI for communication among multiple processes and OpenMP for intra-process parallelism.
 * It initializes MPI, sets the number of OpenMP threads, and handles the distribution of RBMatrix data
 * among MPI processes. The RBMatrix is searched in parallel using OpenMP directives.
 * MPI communication functions like MPI_Bcast, MPI_Scatter, and MPI_Reduce are employed
 * for data sharing and result collection among processes. Additionally, MPI_Barrier ensures synchronization
 * between process stages. The results of the search are collected and saved in an output file.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments (expected format: [ompNumThreads] [numValues] [seed]).
 * @return int Exit status.
 */
int main(int argc, char **argv) {

    // Start measuring the execution time of the program
    double programStartTime = getCurrentTime();

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Check command-line arguments
    if (argc != 6) {
        if (mpi_rank == 0) {
            fprintf(stderr, "Usage: %s [ompNumThreads] [numValues] [seed] [outputFileName] [csvFilePath]\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Set number of OpenMP threads and extract command-line arguments
    int ompNumThreads = atoi(argv[1]);
    omp_set_num_threads(ompNumThreads);
    int numValues = atoi(argv[2]);
    unsigned int seed = atoi(argv[3]);
    char *csvFilePath = argv[5];
    
    // Seed the random number generator
    srand(seed);
    int searchValue = rand() % 100;
    double searchTime;
    int foundIndex = -1;

    RBMatrix *rbMatrix = NULL;

    // Process 0 creates and populates the RBMatrix
    if (mpi_rank == 0) {
        rbMatrix = createAndPopulateRBMatrix(numValues);

        // Broadcast RBMatrix metadata and node data to all processes
        MPI_Bcast(&rbMatrix->totalNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rbMatrix->maxNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(rbMatrix->nodes, rbMatrix->totalNodes * sizeof(RBNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        // Receive RBMatrix metadata and node data from process 0
        int totalNodes, maxNodes;
        MPI_Bcast(&totalNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&maxNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        rbMatrix = createMatrix(numValues);
        rbMatrix->totalNodes = totalNodes;
        rbMatrix->maxNodes = maxNodes;
        rbMatrix->nodes = malloc(totalNodes * sizeof(RBNode));
        MPI_Bcast(rbMatrix->nodes, totalNodes * sizeof(RBNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter tasks among processes
    Task *tasks = NULL;
    Task myTask;
    if (mpi_rank == 0) {
        tasks = createTasksForAllProcesses(rbMatrix, mpi_size);
    }

    double startTime = MPI_Wtime();

    // Scatter tasks to each process
    MPI_Scatter(tasks, sizeof(Task), MPI_BYTE, &myTask, sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Perform parallel search within each process' task using OpenMP
    foundIndex = parallelSearchInSubtree(myTask, rbMatrix, searchValue);

    // Barrier to ensure all processes complete their search
    MPI_Barrier(MPI_COMM_WORLD);

    double endTime = MPI_Wtime();

    // Reduce the found indices across all processes to get the global result
    int globalFound;
    MPI_Reduce(&foundIndex, &globalFound, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Process 0 writes results to an output file
    if (mpi_rank == 0) {
        double elapsedTimeSeconds = (endTime - startTime);

        // End measuring the execution time of the program
        double programEndTime = getCurrentTime();
        double programExecutionTime = programEndTime - programStartTime;

        // Write results to an output file
        char *outputFileName = argv[4];
        FILE *fp = fopen(outputFileName, "w");
        if (fp == NULL) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(fp, "Input Parameters:\n");
        fprintf(fp, "Number of OpenMP threads: %d\n", ompNumThreads);
        fprintf(fp, "Number of MPI processes: %d\n", mpi_size);
        fprintf(fp, "Number of values (nodes in RBTree) to insert: %d\n", numValues);
        fprintf(fp, "Random seed: %u\n", seed);
        fprintf(fp, "\n");
        fprintf(fp, "Parallel search with OpenMP+MPI for value %d.\n", searchValue);
        if (globalFound < INT_MAX) {
            fprintf(fp, "Value found at node %d.\n", globalFound);
        } else {
            fprintf(fp, "Value not found.\n");
        }
        char elapsedTimeStr[50];
        sprintf(elapsedTimeStr, "%.9f", elapsedTimeSeconds);
        fprintf(fp, "Execution time of parallel search: %s seconds\n", elapsedTimeStr);
        fprintf(fp, "Execution time of total program: %.9f seconds.\n", programExecutionTime);
        fprintf(fp, "\n");
        printMatrix(rbMatrix, fp);
        fclose(fp);
        writeResultsToCSV(csvFilePath, ompNumThreads, mpi_size, numValues, 0, elapsedTimeSeconds, programExecutionTime, foundIndex);
    }

    // Cleanup allocated memory and finalize MPI
    if (rbMatrix != NULL) {
        destroyMatrix(rbMatrix);
    }

    if (tasks != NULL) {
        free(tasks);
    }

    MPI_Finalize();
    return 0;
}