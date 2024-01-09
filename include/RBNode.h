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
 * @file RBNode.h
 * 
 * @brief Header file containing the structure definition for Red-Black Tree node (RBNode).
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

#ifndef RB_NODE_H
#define RB_NODE_H

#define RED 1       // Red color indicator for RBNode
#define BLACK 0     // Black color indicator for RBNode

/**
 * @struct RBNode
 * 
 * @brief Structure representing a node in a Red-Black Tree (RBT).
 * 
 * The RBNode structure represents a node in a Red-Black Tree (RBT). Each node holds information
 * about its value, color (RED or BLACK), and the indices of its left child, right child, and parent nodes.
 * This header file defines the RBNode structure and associated constants used to specify node colors.
 */
typedef struct {
    int value;      // Value stored in the node
    int color;      // Color of the node: RED or BLACK
    int left;       // Index of the left child in the RBMatrix
    int right;      // Index of the right child in the RBMatrix
    int parent;     // Index of the parent in the RBMatrix
} RBNode;

#endif // RB_NODE_H