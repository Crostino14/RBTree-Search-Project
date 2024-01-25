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
 * @brief Structure representing a sub-matrix for parallel search within RBMatrix.
 */
typedef struct {
    RBNode *nodes;   // Array of RBMatrix's node
    int count;       // Number of nodes of the sub-matrix
} RBSubMatrix;

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
 * @brief Creates tasks for all MPI processes in the parallel Red-Black Tree search.
 *
 * This function divides the entire search domain of the Red-Black Tree, represented by the RBMatrix,
 * into discrete tasks for processing by different MPI processes. It calculates the distribution
 * of nodes from the RBMatrix to each process, ensuring a balanced workload. The function returns
 * an array of Task structures, with each Task encompassing a range of indices (start and end index) 
 * in the RBNode array that a particular MPI process will handle. This division of tasks is crucial 
 * for achieving load balancing across MPI processes and thereby enhancing the efficiency of the parallel 
 * search operation.
 *
 * @param rbMatrix Pointer to the RBMatrix containing the nodes to be searched.
 * @param numProcesses The total number of MPI processes available for executing the search.
 * @return Task* An array of Task structures, each defining the search range (start and end index) 
 *         for an MPI process. If memory allocation fails, the program will abort with an MPI_Abort call.
 *
 * @return
 * An array of Task structures, each indicating the portion of the RBMatrix to be handled by an MPI process.
 */
Task *createTasksForAllProcesses(RBMatrix *rbMatrix, int numProcesses) {
    Task *tasks = malloc(numProcesses * sizeof(Task));
    if (!tasks) {
        fprintf(stderr, "Failed to allocate memory for tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int totalNodes = rbMatrix->totalNodes;
    int nodesPerProcess = totalNodes / numProcesses;
    int remainingNodes = totalNodes % numProcesses;
    int currentStartIndex = 0;

    for (int i = 0; i < numProcesses; ++i) {
        tasks[i].startIndex = currentStartIndex;
        tasks[i].endIndex = currentStartIndex + nodesPerProcess + (i < remainingNodes);
        currentStartIndex = tasks[i].endIndex;
    }

    return tasks;
}


/**
 * @brief Creates a custom MPI datatype for the RBSubMatrix structure.
 *
 * This function defines and commits a custom MPI datatype that mirrors the layout of the
 * RBSubMatrix structure. RBSubMatrix is designed to hold a segment of the Red-Black Tree
 * nodes along with their count, facilitating efficient parallel search operations across
 * multiple MPI processes. The custom datatype is essential for correctly interpreting
 * the bytes received during MPI communications, ensuring that the data structure's integrity
 * is maintained across different computing environments. The function handles the intricacies
 * of memory layout such as padding and alignment, which can vary between compilers and platforms.
 * By creating this custom datatype, it simplifies the MPI communication code, making it
 * more readable and maintainable. Furthermore, it enhances portability and scalability of
 * the application by abstracting the data transmission details, making the code agnostic
 * to the underlying architecture. The function calculates the relative offsets of each
 * element within the structure and constructs the MPI datatype using MPI_Type_create_struct,
 * followed by committing it with MPI_Type_commit for later use in MPI communication routines.
 *
 * @return MPI_Datatype - The newly created and committed MPI datatype that represents the RBSubMatrix structure.
 */
MPI_Datatype createRBSubMatrixType() {
    RBSubMatrix sampleSubMatrix;

    // Numero di elementi nella struttura
    int count = 2;

    // Array delle lunghezze degli elementi
    int array_of_blocklengths[2] = {1, 1};

    // Calcola gli offset all'interno della struttura
    MPI_Aint array_of_displacements[2];
    MPI_Get_address(&sampleSubMatrix.nodes, &array_of_displacements[0]);
    MPI_Get_address(&sampleSubMatrix.count, &array_of_displacements[1]);
    array_of_displacements[1] -= array_of_displacements[0];
    array_of_displacements[0] = 0;

    // Tipi dei blocchi
    MPI_Datatype array_of_types[2] = {MPI_INT, MPI_INT};

    // Creazione del tipo di dato MPI
    MPI_Datatype customType;
    MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, array_of_types, &customType);
    MPI_Type_commit(&customType);

    return customType;
}

/**
 * @brief Parallel search in a subtree of a Red-Black Tree using OpenMP.
 *
 * This function performs a parallel search within a specified subtree of a Red-Black Tree
 * using OpenMP for intra-process parallelization. The search is conducted for a given value,
 * and the function returns the global index of the first occurrence of the value within the
 * RBMatrix. The parallelization is achieved using OpenMP directives for loop parallelization,
 * and a reduction operation is applied to find the minimum global index across all threads.
 *
 * @param task A Task structure defining the search range within the RBNode array.
 * @param subMatrix An RBSubMatrix structure representing the subtree to search within.
 * @param searchValue The value to search for within the RBNode array.
 * @return int The global index of the first occurrence of the searchValue, or INT_MAX if not found.
 *
 * @details
 * The function uses OpenMP to parallelize the search process, with each thread handling
 * a subset of RBNode elements within the specified subtree. The global index of the first
 * occurrence of the searchValue is determined, and a reduction operation is employed to find
 * the minimum index across all threads. The global index is then returned as the result.
 * If the searchValue is not found within the specified subtree, the function returns INT_MAX.
 *
 * @return
 * The global index of the first occurrence of the searchValue, or INT_MAX if not found.
 */
int parallelSearchInSubtree(Task task, RBSubMatrix subMatrix, int searchValue) {
    int globalFoundIndex = INT_MAX;

    #pragma omp parallel for schedule(dynamic) reduction(min:globalFoundIndex)
    for (int i = 0; i < subMatrix.count; ++i) {
        if (subMatrix.nodes[i].value == searchValue) {
            int globalIndex = task.startIndex + i;
            globalFoundIndex = (globalFoundIndex < globalIndex) ? globalFoundIndex : globalIndex;
        }
    }

    return globalFoundIndex;
}


/**
 * @brief Main function for parallel Red-Black Tree search using MPI and OpenMP.
 *
 * The main function coordinates the parallel search of a Red-Black Tree (RBTree) using a hybrid approach
 * with MPI for inter-process communication and OpenMP for intra-process parallelism. 
 * It initializes the MPI environment, sets the number of OpenMP threads, and handles the distribution 
 * of RBMatrix data among MPI processes using MPI_Scatterv. The search within each submatrix is then
 * performed in parallel using OpenMP. The function includes proper memory management for dynamic 
 * allocation and deallocation of resources. It also ensures synchronization among processes and threads 
 * for a coherent parallel execution. The final search results are collected and can be saved in an output file.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments: expected format includes number of OpenMP threads, 
 *        number of values in RBTree, random seed, output file name, and CSV file path.
 * @return int Exit status of the program.
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
    Task *tasks = NULL;
    int *sendCounts = NULL, *displacements = NULL;

    if (mpi_rank == 0) {
        rbMatrix = createAndPopulateRBMatrix(numValues);
        tasks = createTasksForAllProcesses(rbMatrix, mpi_size);

        sendCounts = (int *)malloc(mpi_size * sizeof(int));
        displacements = (int *)malloc(mpi_size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < mpi_size; i++) {
            sendCounts[i] = (tasks[i].endIndex - tasks[i].startIndex) * sizeof(RBNode);
            displacements[i] = offset;
            offset += sendCounts[i];
        }
    }

    // Ricezione del task
    Task myTask;
    MPI_Scatter(tasks, sizeof(Task), MPI_BYTE, &myTask, sizeof(Task), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Preparazione della sotto-matrice locale
    RBSubMatrix localSubMatrix;
    localSubMatrix.count = myTask.endIndex - myTask.startIndex;
    localSubMatrix.nodes = (RBNode *)malloc(localSubMatrix.count * sizeof(RBNode));

    // Distribuzione delle sotto-matrici utilizzando MPI_Scatterv
    MPI_Scatterv(rbMatrix ? rbMatrix->nodes : NULL, sendCounts, displacements, MPI_BYTE, localSubMatrix.nodes, localSubMatrix.count * sizeof(RBNode), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    // Perform parallel search within each process' task using OpenMP
    foundIndex = parallelSearchInSubtree(myTask, localSubMatrix, searchValue);

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

    if (localSubMatrix.nodes != NULL) {
        free(localSubMatrix.nodes);
    }

    if (mpi_rank == 0) {
        free(sendCounts);
        free(displacements);
    }

    MPI_Finalize();

    return 0;
}