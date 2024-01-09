/*
 * Course: High Performance Computing 2023/2024
 *
 * Lecturer: Francesco Moscato      fmoscato@unisa.it
 *
 * Student and Creator:
 * Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
 *
 * Source Code for sparse matrix implementation of of Red-Black Tree.
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
 * @file RBMatrix.c
 *
 * @brief Source file providing a sparse matrix of a Red-Black Tree (RBTree).
 *
 *                                    REQUIREMENTS OF THE ASSIGNMENT
 * 
 * The file contains the implementation of a sparse matrix Red-Black Tree (RBTree). This code aims to compare the
 * performance of parallel solutions against the sequential implementation on a single-processing node.
 * The assignment requires parallelizing the RBTree search algorithm using both OpenMP + MPI and OpenMP + CUDA, 
 * comparing their performance against the known single-processing node solution. Students need to discuss and analyze 
 * the results and differences observed for various input types and sizes. The parallel algorithms utilized 
 * in the OpenMP + MPI and OpenMP + CUDA solutions may differ from each other.
 *
 * @copyright Copyright (C) 2023 - All Rights Reserved
 * 
 */
#include "../include/RBMatrix.h"
#include <stdlib.h>
#include <stdbool.h>

/**
 * @brief Creates and initializes a new RBMatrix object with the specified maximum number of nodes.
 * 
 * The createMatrix function allocates memory for the RBMatrix structure and its nodes array.
 * It initializes the fields within the structure and sets default values for RBNodes to indicate
 * unused elements in the matrix. This function returns a pointer to the newly allocated RBMatrix.
 * 
 * @param maxNodes The maximum capacity of the Red-Black Tree.
 * @return A pointer to the newly allocated RBMatrix structure.
 */
RBMatrix *createMatrix(int maxNodes)
{
    // Allocate memory for the RBMatrix structure
    RBMatrix *matrix = (RBMatrix *)malloc(sizeof(RBMatrix));
    if (!matrix)
    {
        return NULL; // Memory allocation failed
    }

    // Initialize the matrix fields
    matrix->maxNodes = maxNodes;
    matrix->totalNodes = 0;

    // Allocate memory for the nodes array
    matrix->nodes = (RBNode *)calloc(maxNodes, sizeof(RBNode));
    if (!matrix->nodes)
    {
        free(matrix); // Free the matrix structure if node allocation fails
        return NULL;
    }

    // Initialize each node in the matrix
    for (int i = 0; i < maxNodes; i++)
    {
        matrix->nodes[i].value = -1; // Indicate an unused node
        matrix->nodes[i].color = BLACK;
        matrix->nodes[i].left = -1;
        matrix->nodes[i].right = -1;
        matrix->nodes[i].parent = -1;
    }

    return matrix;
}

/**
 * @brief Resizes the RBMatrix by allocating more memory for RBNode elements when nearing capacity.
 *
 * This function checks if the RBMatrix is approaching its maximum capacity. If so, it reallocates
 * memory to accommodate more RBNodes, increasing the matrix size to ensure efficient operations.
 *
 * @param matrix The RBMatrix to be resized.
 * @return 0 on successful reallocation, -1 on failure.
 */
int resizeMatrix(RBMatrix *matrix) {
    if (!matrix) {
        fprintf(stderr, "Matrice non inizializzata.\n");
        return -1;
    }

    int newMaxNodes = (int)(matrix->maxNodes * 1.5);
    if (newMaxNodes <= matrix->maxNodes) {
        newMaxNodes = matrix->maxNodes + 1;
    }

    RBNode *newNodes = (RBNode *)realloc(matrix->nodes, newMaxNodes * sizeof(RBNode));
    
    if (!newNodes) {
        fprintf(stderr, "Errore di riallocazione della memoria.\n");
        return -1;
    }

    for (int i = matrix->maxNodes; i < newMaxNodes; i++) {
        newNodes[i].value = -1;
        newNodes[i].color = BLACK;
        newNodes[i].left = -1;
        newNodes[i].right = -1;
        newNodes[i].parent = -1;
    }

    matrix->nodes = newNodes;
    matrix->maxNodes = newMaxNodes;

    return 0;
}

/**
 * @brief Sets a specific RBNode element at the given index in the RBMatrix.
 * 
 * The matrixSetElement function assigns an RBNode element to a specified position in the matrix,
 * updating the RBMatrix with the provided node at the given row index.
 * 
 * @param matrix The matrix to set the element in.
 * @param row The row index of the element.
 * @param node The RBNode element to insert into the matrix.
 */
void matrixSetElement(RBMatrix *matrix, int row, RBNode node)
{
    if (!matrix || row < 0 || row >= matrix->maxNodes) {
        fprintf(stderr, "Indice fuori dai limiti o matrice non inizializzata.\n");
        return;
    }

    matrix->nodes[row] = node;
    if (node.value != -1 && matrix->totalNodes < matrix->maxNodes) {
        matrix->totalNodes++;
    }
}

/**
 * @brief Retrieves a specific RBNode element from the RBMatrix based on the given index.
 * 
 * The matrixGetElement function fetches an RBNode element located at the specified index in the matrix
 * and returns it to the caller. If the provided row index is out of bounds, a default RBNode is returned.
 * 
 * @param matrix The matrix to retrieve the element from.
 * @param row The row index of the element to retrieve.
 * @return The RBNode element at the specified row index.
 */
RBNode matrixGetElement(RBMatrix *matrix, int row)
{
    if (!matrix || row < 0 || row >= matrix->maxNodes) {
        fprintf(stderr, "Indice fuori dai limiti o matrice non inizializzata.\n");
        return (RBNode){-1, BLACK, -1, -1, -1};
    }

    return matrix->nodes[row];
}

/**
 * @brief Checks if a specific value is already present in the Red-Black Tree.
 *
 * The isValuePresent function examines whether a given value exists in the Red-Black Tree represented
 * by the matrix. It traverses the tree structure to determine the presence of the specified value.
 * 
 * @param matrix The RBMatrix to search for the value.
 * @param value The value to check for existence in the tree.
 * @return 1 if the value exists, 0 if not.
 */
int isValuePresent(RBMatrix *matrix, int value)
{
    if (matrix == NULL || matrix->totalNodes == 0)
    {
        return 0;
    }

    int currentIndex = 0;

    while (currentIndex != -1)
    {
        RBNode currentNode = matrix->nodes[currentIndex];

        if (currentNode.value == value)
        {
            return 1; // Valore trovato
        }

        if (value < currentNode.value)
        {
            currentIndex = currentNode.left;
        }
        else
        {
            currentIndex = currentNode.right;
        }
    }
    return 0;
}

/**
 * @brief Inserts a new value into the Red-Black Tree represented by the RBMatrix.
 * 
 * The insertValue function adds a new value to the Red-Black Tree within the RBMatrix.
 * It ensures the value is not already present in the tree before insertion, and if necessary,
 * it performs memory reallocation to accommodate additional nodes in the RBMatrix.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param value The value to be inserted into the tree.
 */
void insertValue(RBMatrix *matrix, int value)
{
    if (!matrix) {
        fprintf(stderr, "Matrice non inizializzata.\n");
        return;
    }

    // Controlla se la matrice è vicina alla sua capacità massima e rialloca la memoria
    if (matrix->totalNodes >= matrix->maxNodes * 0.90) { // Soglia del 90%
        if (resizeMatrix(matrix) == -1) {
            fprintf(stderr, "Impossibile riallocare la memoria per la matrice.\n");
            return;
        }
    }

    RBNode newNode = {value, RED, -1, -1, -1};

    if (isValuePresent(matrix, value))
    {
        return;
    }

    int current = 0;
    int parent = -1;

    while (current != -1 && matrix->nodes[current].value != -1)
    {
        parent = current;
        current = (value < matrix->nodes[current].value) ? matrix->nodes[current].left : matrix->nodes[current].right;
    }

    newNode.parent = parent;
    if (parent == -1)
    {
        matrix->nodes[0] = newNode;
        matrix->totalNodes = 1;
    }
    else if (value < matrix->nodes[parent].value)
    {
        matrix->nodes[parent].left = matrix->totalNodes;
    }
    else
    {
        matrix->nodes[parent].right = matrix->totalNodes;
    }

    if (current == -1)
    {
        matrixSetElement(matrix, matrix->totalNodes, newNode);
        matrix->totalNodes++;
    }

    rebalanceInsert(matrix, matrix->totalNodes - 1);
}

/**
 * @brief Rebalances the Red-Black Tree after insertion of a new node.
 * 
 * Upon insertion of a new node into the Red-Black Tree, the rebalanceInsert function ensures
 * that the tree maintains its properties. It performs necessary rotations and color adjustments
 * to restore the balance of the Red-Black Tree according to the established rules.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param index The index of the newly inserted node that requires rebalancing.
 */
void rebalanceInsert(RBMatrix *matrix, int index)
{
    if (!matrix) {
        fprintf(stderr, "Matrice non inizializzata.\n");
        return;
    }

    while (index != 0 && matrix->nodes[matrix->nodes[index].parent].color == RED)
    {
        int parent = matrix->nodes[index].parent;
        int grandparent = matrix->nodes[parent].parent;

        if (parent == matrix->nodes[grandparent].left)
        {
            int uncle = matrix->nodes[grandparent].right;

            if (uncle != -1 && matrix->nodes[uncle].color == RED)
            {
                matrix->nodes[parent].color = BLACK;
                matrix->nodes[uncle].color = BLACK;
                matrix->nodes[grandparent].color = RED;
                index = grandparent;
            }
            else
            {
                if (index == matrix->nodes[parent].right)
                {
                    index = parent;
                    rotateLeft(matrix, index);
                }
                matrix->nodes[parent].color = BLACK;
                matrix->nodes[grandparent].color = RED;
                rotateRight(matrix, grandparent);
            }
        }
        else
        {
            int uncle = matrix->nodes[grandparent].left;

            if (uncle != -1 && matrix->nodes[uncle].color == RED)
            {
                matrix->nodes[parent].color = BLACK;
                matrix->nodes[uncle].color = BLACK;
                matrix->nodes[grandparent].color = RED;
                index = grandparent;
            }
            else
            {
                if (index == matrix->nodes[parent].left)
                {
                    index = parent;
                    rotateRight(matrix, index);
                }
                matrix->nodes[parent].color = BLACK;
                matrix->nodes[grandparent].color = RED;
                rotateLeft(matrix, grandparent);
            }
        }
    }

    matrix->nodes[0].color = BLACK;
}

/**
 * @brief Rotates a specified node in the Red-Black Tree to the left.
 * 
 * The rotateLeft function performs a left rotation on the Red-Black Tree around the specified node.
 * It adjusts the tree structure to maintain the properties of the Red-Black Tree after insertion.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param xIndex The index of the node around which the left rotation occurs.
 */
void rotateLeft(RBMatrix *matrix, int xIndex)
{
    if (!matrix || xIndex < 0 || xIndex >= matrix->maxNodes) {
        fprintf(stderr, "Indice fuori dai limiti o matrice non inizializzata.\n");
        return;
    }

    int yIndex = matrix->nodes[xIndex].right;
    if (yIndex == -1)
        return; 

    matrix->nodes[xIndex].right = matrix->nodes[yIndex].left;
    if (matrix->nodes[yIndex].left != -1)
    {
        matrix->nodes[matrix->nodes[yIndex].left].parent = xIndex;
    }

    matrix->nodes[yIndex].parent = matrix->nodes[xIndex].parent;
    if (matrix->nodes[xIndex].parent == -1)
    {
        matrix->totalNodes = yIndex;
    }
    else if (xIndex == matrix->nodes[matrix->nodes[xIndex].parent].left)
    {
        matrix->nodes[matrix->nodes[xIndex].parent].left = yIndex;
    }
    else
    {
        matrix->nodes[matrix->nodes[xIndex].parent].right = yIndex;
    }

    matrix->nodes[yIndex].left = xIndex;
    matrix->nodes[xIndex].parent = yIndex;
}

/**
 * @brief Rotates a specified node in the Red-Black Tree to the right.
 * 
 * The rotateRight function performs a right rotation on the Red-Black Tree around the specified node.
 * It restructures the tree to ensure the preservation of Red-Black Tree properties after insertion.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param yIndex The index of the node around which the right rotation occurs.
 */
void rotateRight(RBMatrix *matrix, int yIndex)
{
    if (!matrix || yIndex < 0 || yIndex >= matrix->maxNodes) {
        fprintf(stderr, "Indice fuori dai limiti o matrice non inizializzata.\n");
        return;
    }

    int xIndex = matrix->nodes[yIndex].left;
    if (xIndex == -1)
        return;

    matrix->nodes[yIndex].left = matrix->nodes[xIndex].right;
    if (matrix->nodes[xIndex].right != -1)
    {
        matrix->nodes[matrix->nodes[xIndex].right].parent = yIndex;
    }

    matrix->nodes[xIndex].parent = matrix->nodes[yIndex].parent;
    if (matrix->nodes[yIndex].parent == -1)
    {
        matrix->totalNodes = xIndex;
    }
    else if (yIndex == matrix->nodes[matrix->nodes[yIndex].parent].right)
    {
        matrix->nodes[matrix->nodes[yIndex].parent].right = xIndex;
    }
    else
    {
        matrix->nodes[matrix->nodes[yIndex].parent].left = xIndex;
    }

    matrix->nodes[xIndex].right = yIndex;
    matrix->nodes[yIndex].parent = xIndex;
}

/**
 * @brief Generates a random integer within a specified range.
 * 
 * The getRandomNumber function generates a pseudo-random integer within the range [min, max].
 * It utilizes the rand() function from the C standard library to produce a random number.
 * The generated number is inclusive of both the minimum and maximum values provided.
 * 
 * @param min The minimum value of the range.
 * @param max The maximum value of the range.
 * @return A random integer within the specified range [min, max].
 */
int getRandomNumber(int min, int max)
{
    return min + rand() % (max - min + 1);
}

/**
 * @brief Populates the Red-Black Tree represented by the RBMatrix with random values.
 * 
 * The randomPopulateRBMatrix function fills the Red-Black Tree within the RBMatrix with a specified
 * number of randomly generated unique values. It uses a random number generator to create values and
 * ensures each value inserted into the tree is unique.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param numValues The number of random values to insert into the tree.
 */
void randomPopulateRBMatrix(RBMatrix *matrix, int numValues)
{
    int maxValue = numValues * 2;
    int count = 0;

    while (count < numValues)
    {
        int randomValue = getRandomNumber(0, numValues);

        if (!isValuePresent(matrix, randomValue))
        {
            insertValue(matrix, randomValue);
            count++;
        }
    }
}

/**
 * @brief Prints a visual representation of the Red-Black Tree contained within the RBMatrix.
 * 
 * The printMatrix function outputs a visual representation of the Red-Black Tree stored in the RBMatrix
 * to a specified file stream. It displays details such as the total number of nodes, maximum tree capacity,
 * and information about each node in the tree, including its value, color, parent, left, and right pointers.
 * 
 * @param matrix The RBMatrix containing the Red-Black Tree.
 * @param fp The file pointer to write the tree representation.
 */
void printMatrix(RBMatrix *matrix, FILE *fp)
{
    if (!fp || !matrix) {
        fprintf(stderr, "File o matrice non inizializzata.\n");
        return;
    }

    fprintf(fp, "Red-Black Tree Representation:\n\n");
    fprintf(fp, "Real Total Number of Nodes: %d\n", matrix->totalNodes);
    fprintf(fp, "Maximum Tree Capacity: %d\n\n", matrix->maxNodes);


    for (int i = 0; i < matrix->maxNodes; i++)
    {
        RBNode node = matrix->nodes[i];
        if (node.value != -1)
        {
            fprintf(fp, "8========================================8\nNodo %d: Valore = %d, Colore = %s\n",
                    i, node.value, (node.color == RED) ? "Rosso" : "Nero");
            fprintf(fp, "Genitore = %d, Sinistro = %d, Destro = %d\n8========================================8\n",
                    node.parent, node.left, node.right);
        }
    }
}

/**
 * @brief Deallocates the memory occupied by the RBMatrix and its associated elements.
 * 
 * The destroyMatrix function frees the memory allocated for the RBMatrix structure and its nodes array.
 * It ensures the proper release of memory resources to prevent memory leaks after using the RBMatrix.
 * 
 * @param matrix The RBMatrix to be destroyed.
 */
void destroyMatrix(RBMatrix *matrix)
{
    if (!matrix) return;

    if (matrix->nodes) {
        free(matrix->nodes);
    }

    free(matrix);
}

/**
 * @brief Get the current time in seconds with microsecond precision.
 *
 * @return The current time in seconds as a double
 */
double getCurrentTime() {
    double timeInMicroseconds = 0.0;

#ifdef _WIN32
    LARGE_INTEGER frequency, start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    timeInMicroseconds = (double)start.QuadPart / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timeInMicroseconds = (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
#endif

    return timeInMicroseconds;
}

/**
 * @brief Writes the performance results of the Red-Black tree search to a CSV file.
 *
 * This function appends the performance data of a Red-Black tree search to a specified CSV file.
 * If the file does not exist or is empty, it creates the file and writes a header row.
 * The function records various parameters and timings of the search process, allowing for
 * easy aggregation and analysis of data across multiple runs.
 *
 * @param filePath The path to the CSV file where the data will be written.
 * @param ompThreads The number of OpenMP threads used in the search.
 * @param mpiProcesses The number of MPI processes used in the search (for MPI implementations).
 * @param numValues The number of values (nodes) in the Red-Black tree.
 * @param blockSize The number of CUDA threads per block (for CUDA implementations), or 0 if not applicable.
 * @param searchTime The time taken for the search process (in seconds).
 * @param totalProgramTime The total execution time of the program (in seconds).
 * @param foundIndex The index where the value was found; if the value is not found, it should be INT_MAX.
 */
void writeResultsToCSV(const char *filePath, int ompThreads, int mpiProcesses, int numValues, int blockSize, double searchTime, double totalProgramTime, int foundIndex) {
    FILE *file = fopen(filePath, "a"); // Open the file in append mode
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the header if the file is newly created or empty
    fseek(file, 0, SEEK_END); // Move to the end of the file
    if (ftell(file) == 0) {
        // If the file is empty, write the header
        fprintf(file, "OMP Threads,MPI Processes,Block Size,Num Values,Search Time (s),"
                      "Total Program Time (s),Value Found (1=yes 0=no)\n");
    }

    // Write the data in CSV format
    fprintf(file, "%d,%d,%d,%d,%.9f,%.9f,%d\n", ompThreads, mpiProcesses, blockSize, numValues,
                                                searchTime, totalProgramTime, (foundIndex != INT_MAX));

    fclose(file); // Close the file
}
