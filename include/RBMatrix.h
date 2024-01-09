/*
 * Course: High Performance Computing 2023/2024
 *
 * Lecturer: Francesco Moscato      fmoscato@unisa.it
 *
 * Student and Creator:
 * Agostino Cardamone       0622702276      a.cardamone7@studenti.unisa.it
 *
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
 * @file RBMatrix.h
 * 
 * @brief Header file defining the RBMatrix structure for managing Red-Black Tree nodes.
 * 
 *                                    REQUIREMENTS OF THE ASSIGNMENT
 * 
 * The assignment requires parallelizing the RBTree search algorithm using both OpenMP + MPI and OpenMP + CUDA, 
 * comparing their performance against the known single-processing node solution. Students need to discuss and analyze 
 * the results and differences observed for various input types and sizes. The parallel algorithms utilized 
 * in the OpenMP + MPI and OpenMP + CUDA solutions may differ from each other.
 *
 * @copyright Copyright (C) 2023 - All Rights Reserved
 * 
 */
#ifndef RB_MATRIX_H
#define RB_MATRIX_H

#include "RBNode.h"
#include <stdlib.h>
#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct RBMatrix
 * 
 * @brief A sparse matrix designed specifically for efficiently handling Red-Black Tree nodes.
 * 
 * The RBMatrix is a specialized structure optimized for managing Red-Black Tree nodes in a sparse manner.
 * It minimizes memory consumption while facilitating efficient access to RBNode elements.
 * Additionally, it tracks the total number of nodes to assist parallel processing.
 */
typedef struct {
    RBNode *nodes;   // Array of RBNode elements
    int totalNodes;  // Total number of nodes in the tree
    int maxNodes;    // Maximum capacity of the tree
} RBMatrix;

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
RBMatrix *createMatrix(int maxNodes);

/**
 * @brief Resizes the RBMatrix by allocating more memory for RBNode elements when nearing capacity.
 *
 * This function checks if the RBMatrix is approaching its maximum capacity. If so, it reallocates
 * memory to accommodate more RBNodes, increasing the matrix size to ensure efficient operations.
 *
 * @param matrix The RBMatrix to be resized.
 * @return 0 on successful reallocation, -1 on failure.
 */
int resizeMatrix(RBMatrix *matrix);

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
void matrixSetElement(RBMatrix *matrix, int row, RBNode node);

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
RBNode matrixGetElement(RBMatrix *matrix, int row);

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
int isValuePresent(RBMatrix *matrix, int value);

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
void insertValue(RBMatrix *matrix, int value);

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
void rebalanceInsert(RBMatrix *matrix, int index);

/**
 * @brief Rotates a specified node in the Red-Black Tree to the left.
 * 
 * The rotateLeft function performs a left rotation on the Red-Black Tree around the specified node.
 * It adjusts the tree structure to maintain the properties of the Red-Black Tree after insertion.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param xIndex The index of the node around which the left rotation occurs.
 */
void rotateLeft(RBMatrix *matrix, int xIndex);

/**
 * @brief Rotates a specified node in the Red-Black Tree to the right.
 * 
 * The rotateRight function performs a right rotation on the Red-Black Tree around the specified node.
 * It restructures the tree to ensure the preservation of Red-Black Tree properties after insertion.
 * 
 * @param matrix The RBMatrix representing the Red-Black Tree.
 * @param yIndex The index of the node around which the right rotation occurs.
 */
void rotateRight(RBMatrix *matrix, int yIndex);

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
int getRandomNumber(int min, int max);

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
void randomPopulateRBMatrix(RBMatrix *matrix, int numValues);

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
void printMatrix(RBMatrix *matrix, FILE *fp);

/**
 * @brief Deallocates the memory occupied by the RBMatrix and its associated elements.
 * 
 * The destroyMatrix function frees the memory allocated for the RBMatrix structure and its nodes array.
 * It ensures the proper release of memory resources to prevent memory leaks after using the RBMatrix.
 * 
 * @param matrix The RBMatrix to be destroyed.
 */
void destroyMatrix(RBMatrix *matrix);

/**
 * @brief Get the current time in seconds with microsecond precision.
 *
 * @return The current time in seconds as a double
 */
double getCurrentTime();

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
void writeResultsToCSV(const char *filePath, int ompThreads, int mpiProcesses, int numValues, int blockSize, double searchTime, double totalProgramTime, int foundIndex);

#ifdef __cplusplus
}
#endif

#endif // RB_MATRIX_H
