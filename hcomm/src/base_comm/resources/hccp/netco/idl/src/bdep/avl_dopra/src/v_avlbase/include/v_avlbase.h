/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef V_AVLBASE_H
#define V_AVLBASE_H

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

#define AVL_NULL_PTR 0L

typedef struct AVLBaseNode {
    struct AVLBaseNode *pstParent;
    struct AVLBaseNode *pstLeft;
    struct AVLBaseNode *pstRight;
    short int sLHeight;
    short int sRHeight;
} AVLBASE_NODE_S;

typedef struct AVLBaseTree {
    AVLBASE_NODE_S *pstRoot;
    AVLBASE_NODE_S *pstFirst;
    AVLBASE_NODE_S *pstLast;
} AVLBASE_TREE_S;

/* Macro to determine the largest value of two variables (internal use) */
#define VOS_V2_AVL_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define FIND_LEFTMOST_NODE(pstNode)                                                                                    \
    do {                                                                                                               \
        while ((pstNode)->pstLeft != AVL_NULL_PTR) {                                                                   \
            (pstNode) = (pstNode)->pstLeft;                                                                            \
        }                                                                                                              \
    } while (0)

#define FIND_RIGHTMOST_NODE(pstNode)                                                                                   \
    do {                                                                                                               \
        while ((pstNode)->pstRight != AVL_NULL_PTR) {                                                                  \
            (pstNode) = (pstNode)->pstRight;                                                                           \
        }                                                                                                              \
    } while (0)

static inline void VosAvlNodeRightInsert(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstParentNode,
    AVLBASE_NODE_S *pstNode)
{
    pstNode->pstParent = pstParentNode;
    pstParentNode->pstRight = pstNode;
    pstParentNode->sRHeight = 1;
    if (pstParentNode == pstTree->pstLast) {
        /* parent was the right-most pstNode in the pstTree, */
        /* so new pstNode is now right-most               */
        pstTree->pstLast = pstNode;
    }
}

static inline void VosAvlNodeLeftInsert(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstParentNode, AVLBASE_NODE_S *pstNode)
{
    pstNode->pstParent = pstParentNode;
    pstParentNode->pstLeft = pstNode;
    pstParentNode->sLHeight = 1;
    if (pstParentNode == pstTree->pstFirst) {
        /* parent was the left-most pstNode in the pstTree, */
        /* so new pstNode is now left-most               */
        pstTree->pstFirst = pstNode;
    }
}

extern void VosAvlRotateRight(AVLBASE_NODE_S **ppstSubTree);
extern void VosAvlRotateLeft(AVLBASE_NODE_S **ppstSubTree);
extern void VosAvlSwapRightMost(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstSubTree, AVLBASE_NODE_S *pstNode);
extern void VosAvlSwapLeftMost(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstSubTree, AVLBASE_NODE_S *pstNode);
extern void VosAvlRebalance(AVLBASE_NODE_S **ppstSubTree);
extern void VosAvlBalanceTree(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode);
extern AVLBASE_NODE_S *VosAvlDeleteCheck(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode);
extern void VosAvlDelete(AVLBASE_NODE_S *pstBaseNode, AVLBASE_TREE_S *pstBaseTree);

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif
