/*
 * Course: High Performance Computing 2023/2024
 *
 * Lecturer: Francesco Moscato      fmoscato@unisa.it
 *
 * Student and Creator:
 * Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
 *
 * Source Code for sequential version of Red-Black Tree search.
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
 * @file RBSequential.c
 * 
 * @brief Source file providing a sequential implementation of the RBTree search.
 * 
 *                                    REQUIREMENTS OF THE ASSIGNMENT
 * 
 * The file contains the implementation of a sequential Red-Black Tree (RBTree) search, forming a basis for comparative
 * parallel implementations using OpenMP + MPI and OpenMP + CUDA. The assignment requires parallelizing the RBTree search
 * algorithm using both OpenMP + MPI and OpenMP + CUDA, comparing their performance against the known single-processing
 * node solution. Students need to discuss and analyze the results and differences observed for various input types and sizes.
 * The parallel algorithms utilized in the OpenMP + MPI and OpenMP + CUDA solutions may differ from each other.
 * 
 * @copyright Copyright (C) 2023 - All Rights Reserved
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../include/RBMatrix.h"

/**
 * @brief Executes a linear search on a Red-Black Tree for a given value.
 *
 * The searchRBTree function is a fundamental part of the RBTree Search Project, encapsulating the core logic 
 * behind the sequential search algorithm. It applies a simple yet efficient approach to traverse the nodes of 
 * the Red-Black Tree, which is internally represented as a matrix (RBMatrix). Starting from the root node, it 
 * iteratively compares the search value against the node's value, following the tree's branches according to 
 * the binary search tree property. The function's efficiency stems from this property, which significantly reduces 
 * the search space at each step. If the search value matches a node's value, the index of this node within the 
 * RBMatrix is returned, signifying a successful search. If the traversal reaches a leaf without finding the value, 
 * the function concludes that the value does not exist within the tree and returns -1. This search mechanism serves 
 * as a baseline for comparing the performance of subsequent parallel implementations.
 *
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param value The value to search for in the Red-Black Tree.
 * @return The index of the node containing the specified value if found; otherwise, returns -1.
 */
int searchRBTree(RBMatrix *matrix, int value)
{
    int currentIndex = 0; // Inizia dalla radice

    while (currentIndex != -1 && matrix->nodes[currentIndex].value != -1)
    {
        RBNode currentNode = matrix->nodes[currentIndex];

        if (value < currentNode.value)
        {
            currentIndex = currentNode.left;
        }
        else if (value > currentNode.value)
        {
            currentIndex = currentNode.right;
        }
        else
        {
            return currentIndex; // Valore trovato
        }
    }
    return -1; // Valore non trovato
}


/**
 * @brief Main entry point for the sequential search in a Red-Black Tree.
 *
 * This main function orchestrates the sequential search process in a Red-Black Tree. It begins by parsing the 
 * command-line arguments to configure the search parameters, such as the number of nodes and the random seed. 
 * It proceeds to initialize the RBMatrix and populate it with randomly generated values, ensuring that the 
 * search operations are conducted on a valid and filled data structure. Following initialization, a search value 
 * is randomly chosen, and the searchRBTree function is called to locate this value within the RBMatrix. The 
 * function measures the time taken to execute the search, providing insights into the efficiency of the sequential 
 * algorithm. Finally, the results, including whether the value was found and the execution time, are logged to 
 * an output file and a CSV file. This sequential search serves as a comparative benchmark for the parallel search 
 * methods implemented using OpenMP + MPI and OpenMP + CUDA, fulfilling the project's requirements to analyze 
 * performance across different search algorithms and configurations.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments (expected format: <numValues> <seed> <outputFileName> <csvFilePath>).
 * @return int Exit status.
 */
int main(int argc, char **argv) {

    // Start measuring the execution time of the program
    double programStartTime = getCurrentTime();

    // Check if the correct number of command-line arguments is provided
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <numValues> <seed> <outputFileName> <csvFilePath>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Retrieve the number of values, seed, and output file name from command-line arguments
    int numValues = atoi(argv[1]);
    if (numValues <= 0) {
        fprintf(stderr, "Invalid number of values!\n");
        exit(EXIT_FAILURE);
    }

    const char *outputFileName = argv[3]; // Get the output file name
    char *csvFilePath = argv[4]; // Get the CSV file path
    
    // Open an output file for writing
    FILE *fp = fopen(outputFileName, "w");
    if (fp == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    int seed = atoi(argv[2]);
    srand(seed);

    // Create RBMatrix and populate it randomly
    RBMatrix *rbMatrix = createMatrix(numValues);
    randomPopulateRBMatrix(rbMatrix, numValues);

    // Generate a random value for searching in the RBTree
    int searchValue = rand() % 100;

    // Start measuring the execution time of the search
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Perform sequential search on the RBTree
    int foundIndex = searchRBTree(rbMatrix, searchValue);

    // End measuring the execution time of the search
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate the search execution time
    long searchTimeSec = end.tv_sec - start.tv_sec;
    long searchTimeNano = end.tv_nsec - start.tv_nsec;
    if (searchTimeNano < 0) {
        searchTimeSec--;
        searchTimeNano += 1000000000L;
    }

    // End measuring the execution time of the program
    double programEndTime = getCurrentTime();
    double programExecutionTime = programEndTime - programStartTime;

    // Write search results and execution time to the output file
    fprintf(fp, "Input Parameters:\n");
    fprintf(fp, "Number of values (nodes in RBTree) to insert: %d\n", numValues);
    fprintf(fp, "\n");
    fprintf(fp, "Sequential search for value %d.\n", searchValue);
    if (foundIndex != -1) {
        fprintf(fp, "Value found at node %d.\n", foundIndex);
    } else {
        fprintf(fp, "Value not found.\n");
    }
    double totalSearchTimeSeconds = searchTimeSec + (searchTimeNano / 1.0e9);
    fprintf(fp, "Execution time of sequential search: %.9f seconds.\n", totalSearchTimeSeconds);
    fprintf(fp, "Execution time of total program: %.9f seconds.\n", programExecutionTime);
    fprintf(fp, "\n");

    // Print RBMatrix data to the output file
    printMatrix(rbMatrix, fp);
    writeResultsToCSV(csvFilePath, 0, 0, numValues, 0, totalSearchTimeSeconds, programExecutionTime, foundIndex);

    // Close the output file and clean up allocated memory
    fclose(fp);
    destroyMatrix(rbMatrix);

    return 0;
}