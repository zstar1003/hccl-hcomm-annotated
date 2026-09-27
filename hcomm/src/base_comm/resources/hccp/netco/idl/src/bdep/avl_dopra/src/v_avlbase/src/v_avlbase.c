/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v_avlbase.h"

/*******************************************************************************
 Description  : AVL_Rotate_Right will Rotate a subtree of the AVL tree right.
 Input        : ppstSubTree - pointer to the subtree to rotate
 Output       : ppstSubTree - pointer to the subtree
 Return       : None
*******************************************************************************/
void VosAvlRotateRight(AVLBASE_NODE_S **ppstSubTree)
{
    AVLBASE_NODE_S *pstLeftSon = (*ppstSubTree)->pstLeft;

    (*ppstSubTree)->pstLeft = pstLeftSon->pstRight;
    if ((*ppstSubTree)->pstLeft != AVL_NULL_PTR) {
        (*ppstSubTree)->pstLeft->pstParent = (*ppstSubTree);
    }

    (*ppstSubTree)->sLHeight = pstLeftSon->sRHeight;
    pstLeftSon->pstParent = (*ppstSubTree)->pstParent;
    pstLeftSon->pstRight = *ppstSubTree;
    pstLeftSon->pstRight->pstParent = pstLeftSon;
    pstLeftSon->sRHeight = (1 + VOS_V2_AVL_MAX((*ppstSubTree)->sRHeight, (*ppstSubTree)->sLHeight));

    *ppstSubTree = pstLeftSon;

    return;
}

/*******************************************************************************
 Description  : AVL_Rotate_Left will Rotate a subtree of the AVL tree left
 Input        : ppstSubTree  - pointer to the subtree to rotate
 Output       : ppstSubTree  - pointer to the subtree
 Return       : None
*******************************************************************************/
void VosAvlRotateLeft(AVLBASE_NODE_S **ppstSubTree)
{
    AVLBASE_NODE_S *pstRightSon = (*ppstSubTree)->pstRight;

    (*ppstSubTree)->pstRight = pstRightSon->pstLeft;
    if ((*ppstSubTree)->pstRight != AVL_NULL_PTR) {
        (*ppstSubTree)->pstRight->pstParent = (*ppstSubTree);
    }

    (*ppstSubTree)->sRHeight = pstRightSon->sLHeight;
    pstRightSon->pstParent = (*ppstSubTree)->pstParent;
    pstRightSon->pstLeft = *ppstSubTree;
    pstRightSon->pstLeft->pstParent = pstRightSon;
    pstRightSon->sLHeight = (1 + VOS_V2_AVL_MAX((*ppstSubTree)->sRHeight, (*ppstSubTree)->sLHeight));

    *ppstSubTree = pstRightSon;

    return;
}

void VosAvlUpdateSwapNode(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstSwapNode, const AVLBASE_NODE_S *pstBaseNode)
{
    /* move swap pstNode to its new position                                  */
    pstSwapNode->pstParent = pstBaseNode->pstParent;
    pstSwapNode->pstRight = pstBaseNode->pstRight;
    pstSwapNode->pstLeft = pstBaseNode->pstLeft;
    pstSwapNode->sRHeight = pstBaseNode->sRHeight;
    pstSwapNode->sLHeight = pstBaseNode->sLHeight;
    pstSwapNode->pstRight->pstParent = pstSwapNode;
    pstSwapNode->pstLeft->pstParent = pstSwapNode;

    if (pstBaseNode->pstParent == AVL_NULL_PTR) {
        /* pstNode is at root of pstTree                                      */
        pstTree->pstRoot = pstSwapNode;
    } else if (pstBaseNode->pstParent->pstRight == pstBaseNode) {
        /* pstNode is right-son of parent                                     */
        pstSwapNode->pstParent->pstRight = pstSwapNode;
    } else {
        /* pstNode is left-son of parent                                      */
        pstSwapNode->pstParent->pstLeft = pstSwapNode;
    }
}

/*******************************************************************************
 Description  : AVL_MoveNodeToNewPos will move node to the new pos. The new pos
                must be only one child or no child;
 Input        : pstNode      - the node which need move
                pstNewParent - the pstNode's new parent
                pstNewLeftSon    - the pstNode's new left son
                pstNewRightSon   - the pstNode's new right son
 Output       : None
 Return       : None
*******************************************************************************/
void VosAvlMoveNodeToNewPos(AVLBASE_NODE_S *pstNode, AVLBASE_NODE_S *pstNewParent, AVLBASE_NODE_S *pstNewLeftSon,
    AVLBASE_NODE_S *pstNewRightSon)
{
    pstNode->pstParent = pstNewParent;
    pstNode->pstLeft = pstNewLeftSon;
    pstNode->pstRight = pstNewRightSon;
    pstNode->sLHeight = 0;
    pstNode->sRHeight = 0;

    if (pstNewLeftSon != AVL_NULL_PTR) {
        pstNode->pstLeft->pstParent = pstNode;
        pstNode->sLHeight = 1;
    }

    if (pstNewRightSon != AVL_NULL_PTR) {
        pstNode->pstRight->pstParent = pstNode;
        pstNode->sRHeight = 1;
    }
}

/*******************************************************************************
 Description  : AVL_Swap_Right_Most will Swap node with right-most descendent
                of subtree
 Input        : pstTree    - pointer to the tree
                pstSubTree - pointer to the subtree
                pstNode    - pointer to the node to swap
 Output       : None
 Return       : None
*******************************************************************************/
void VosAvlSwapRightMost(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstSubTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstSwapNode = pstSubTree;
    AVLBASE_NODE_S *pstSwapParent;
    AVLBASE_NODE_S *pstSwapLeft;

    /* find right-most descendent of pstSubTree                               */
    FIND_RIGHTMOST_NODE(pstSwapNode);

    if ((pstSwapNode->sRHeight != 0) || (pstSwapNode->sLHeight > 1)) {
        return;
    }

    /* save parent and left-son of right-most descendent                      */
    pstSwapParent = pstSwapNode->pstParent;
    pstSwapLeft = pstSwapNode->pstLeft;

    VosAvlUpdateSwapNode(pstTree, pstSwapNode, pstNode);
    VosAvlMoveNodeToNewPos(pstNode, pstSwapParent, pstSwapLeft, AVL_NULL_PTR);

    pstNode->pstParent->pstRight = pstNode;

    return;
}

/*******************************************************************************
 Description  : AVL_Swap_Left_Most will Swap node with left-most descendent
                of subtree.
 Input        : pstTree    - pointer to the tree
                pstSubTree - pointer to the subtree
                pstNode    - pointer to the node to swap
 Output       : None
 Return       : None
*******************************************************************************/
void VosAvlSwapLeftMost(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstSubTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstSwapNode = pstSubTree;
    AVLBASE_NODE_S *pstSwapParent;
    AVLBASE_NODE_S *pstSwapRight;

    /* find left-most descendent of pstSubTree                                */
    FIND_LEFTMOST_NODE(pstSwapNode);

    if ((pstSwapNode->sLHeight != 0) || (pstSwapNode->sRHeight > 1)) {
        return;
    }

    /* save parent and right-son of left-most descendent                      */
    pstSwapParent = pstSwapNode->pstParent;
    pstSwapRight = pstSwapNode->pstRight;

    VosAvlUpdateSwapNode(pstTree, pstSwapNode, pstNode);
    VosAvlMoveNodeToNewPos(pstNode, pstSwapParent, AVL_NULL_PTR, pstSwapRight);

    pstNode->pstParent->pstLeft = pstNode;

    return;
}

/* ******************************************************************************
 Description  : AVL_Rebalance will Rebalance a subtree of the AVL tree
                (if necessary).
 Input        : ppstSubTree - pointer to the subtree to rebalance
 Output       : ppstSubTree - pointer to the subtree
 Return       : None
****************************************************************************** */
void VosAvlRebalance(AVLBASE_NODE_S **ppstSubTree)
{
    int iMoment;

    iMoment = (*ppstSubTree)->sRHeight - (*ppstSubTree)->sLHeight;

    if (iMoment > 1) {
        /* ppstSubTree is heavy on the right side                             */
        if ((*ppstSubTree)->pstRight->sLHeight > (*ppstSubTree)->pstRight->sRHeight) {
            /* right ppstSubTree is heavier on left side, so must perform     */
            /* right rotation on this ppstSubTree to make it heavier on the   */
            /* right side                                                     */
            VosAvlRotateRight(&(*ppstSubTree)->pstRight);
        }

        /* now rotate the ppstSubTree left                                    */
        VosAvlRotateLeft(ppstSubTree);
    } else if (iMoment < -1) {
        /* ppstSubTree is heavy on the left side                              */
        if ((*ppstSubTree)->pstLeft->sRHeight > (*ppstSubTree)->pstLeft->sLHeight) {
            /* left ppstSubTree is heavier on right side, so must perform     */
            /* left rotation on this ppstSubTree to make it heavier on the    */
            /* left side                                                      */
            VosAvlRotateLeft(&(*ppstSubTree)->pstLeft);
        }

        /* now rotate the ppstSubTree right                                   */
        VosAvlRotateRight(ppstSubTree);
    }

    return;
}

/*******************************************************************************
 Description  : AVL_Balance_Tree will Rebalance the tree starting at the
                supplied node and ending at the root of the tree.
 Input        : pstTree    - pointer to the AVL tree
                pstNode    - pointer to the node to start from where to
                             start balancing.
 Output       : None
 Return       : None
*******************************************************************************/
void VosAvlBalanceTree(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode)
{
    /* balance the pstTree starting at the supplied pstNode, and ending       */
    /* at the root of the pstTree                                             */
    AVLBASE_NODE_S *pstNodeTmp = pstNode;
    while (pstNodeTmp->pstParent != AVL_NULL_PTR) {
        /* pstNode has uneven balance, so may need to rebalance it            */
        if (pstNodeTmp->pstParent->pstRight == pstNodeTmp) {
            /* pstNode is right-son of its parent                             */
            pstNodeTmp = pstNodeTmp->pstParent;
            VosAvlRebalance(&pstNodeTmp->pstRight);

            /* now update the right height of the parent                      */
            pstNodeTmp->sRHeight = (1 + VOS_V2_AVL_MAX(pstNodeTmp->pstRight->sRHeight, pstNodeTmp->pstRight->sLHeight));
        } else {
            /* pstNode is left-son of its parent                              */
            pstNodeTmp = pstNodeTmp->pstParent;
            VosAvlRebalance(&pstNodeTmp->pstLeft);

            /* now update the left height of the parent                       */
            pstNodeTmp->sLHeight = (1 + VOS_V2_AVL_MAX(pstNodeTmp->pstLeft->sRHeight, pstNodeTmp->pstLeft->sLHeight));
        }
    }

    if (pstNodeTmp->sLHeight != pstNodeTmp->sRHeight) {
        /* rebalance root pstNode                                             */
        VosAvlRebalance(&pstTree->pstRoot);
    }

    return;
}

AVLBASE_NODE_S *VosAVLSearchReplaceNodeInRTree(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstReplaceNode;

    if (pstNode->pstRight->pstLeft == AVL_NULL_PTR) {
        /* can replace pstNode with right-son(since it has no left son) */
        pstReplaceNode = pstNode->pstRight;
        pstReplaceNode->pstLeft = pstNode->pstLeft;
        pstReplaceNode->pstLeft->pstParent = pstReplaceNode;
        pstReplaceNode->sLHeight = pstNode->sLHeight;
    } else {
        /* swap with left-most descendent of right subtree            */
        VosAvlSwapLeftMost(pstTree, pstNode->pstRight, pstNode);
        pstReplaceNode = pstNode->pstRight;
    }

    return pstReplaceNode;
}

AVLBASE_NODE_S *VosAvlSearchReplaceNodeInLTree(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstReplaceNode;

    if (pstNode->pstLeft->pstRight == AVL_NULL_PTR) {
        /* can replace pstNode with left-son(since it has no right son) */
        pstReplaceNode = pstNode->pstLeft;
        pstReplaceNode->pstRight = pstNode->pstRight;
        pstReplaceNode->pstRight->pstParent = pstReplaceNode;
        pstReplaceNode->sRHeight = pstNode->sRHeight;
    } else {
        /* swap with right-most descendent of left subtree            */
        VosAvlSwapRightMost(pstTree, pstNode->pstLeft, pstNode);
        pstReplaceNode = pstNode->pstLeft;
    }

    return pstReplaceNode;
}

AVLBASE_NODE_S *VosAvlSearchReplaceNode(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstReplaceNode;

    if (pstNode->sRHeight > pstNode->sLHeight) {
        /* right subtree is higher than left subtree                      */
        pstReplaceNode = VosAVLSearchReplaceNodeInRTree(pstTree, pstNode);
    } else {
        /* left subtree is higher (or subtrees are of same height)        */
        pstReplaceNode = VosAvlSearchReplaceNodeInLTree(pstTree, pstNode);
    }

    return pstReplaceNode;
}

AVLBASE_NODE_S *VosAvlDeleteCheck(AVLBASE_TREE_S *pstTree, AVLBASE_NODE_S *pstNode)
{
    AVLBASE_NODE_S *pstReplaceNode;

    if ((pstNode->pstLeft == AVL_NULL_PTR) && (pstNode->pstRight == AVL_NULL_PTR)) {
        /* barren pstNode (no children), so just delete it                    */
        pstReplaceNode = AVL_NULL_PTR;

        if (pstTree->pstFirst == pstNode) {
            /* pstNode was first in pstTree, so replace it                    */
            pstTree->pstFirst = pstNode->pstParent;
        }

        if (pstTree->pstLast == pstNode) {
            /* pstNode was last in pstTree, so replace it                     */
            pstTree->pstLast = pstNode->pstParent;
        }
    } else if (pstNode->pstLeft == AVL_NULL_PTR) {
        /* pstNode has no left son, so replace with right son                 */
        pstReplaceNode = pstNode->pstRight;

        if (pstTree->pstFirst == pstNode) {
            /* pstNode was first in pstTree, so replace it                    */
            pstTree->pstFirst = pstReplaceNode;
        }
    } else if (pstNode->pstRight == AVL_NULL_PTR) {
        /* pstNode has no right son, so replace with left son                 */
        pstReplaceNode = pstNode->pstLeft;

        if (pstTree->pstLast == pstNode) {
            /* pstNode was last in pstTree, so replace it                     */
            pstTree->pstLast = pstReplaceNode;
        }
    } else {
        /* pstNode has both left and right-sons                               */
        pstReplaceNode = VosAvlSearchReplaceNode(pstTree, pstNode);
    }
    return pstReplaceNode;
}

void VosAvlDelete(AVLBASE_NODE_S *pstBaseNode, AVLBASE_TREE_S *pstBaseTree)
{
    AVLBASE_NODE_S *pstReplaceNode;
    AVLBASE_NODE_S *pstParentNode;
    short int sNewHeight = 0;

    pstReplaceNode = VosAvlDeleteCheck(pstBaseTree, pstBaseNode);
    /* save parent pstNode of deleted pstNode                                 */
    pstParentNode = pstBaseNode->pstParent;

    /* reset deleted pstNode                                                  */
    pstBaseNode->pstParent = AVL_NULL_PTR;
    pstBaseNode->pstRight = AVL_NULL_PTR;
    pstBaseNode->pstLeft = AVL_NULL_PTR;
    pstBaseNode->sRHeight = -1;
    pstBaseNode->sLHeight = -1;

    if (pstReplaceNode != AVL_NULL_PTR) {
        /* fix-up parent pointer of replacement pstNode, and calculate new    */
        /* height of subtree                                                  */
        pstReplaceNode->pstParent = pstParentNode;
        sNewHeight = (1 + VOS_V2_AVL_MAX(pstReplaceNode->sLHeight, pstReplaceNode->sRHeight));
    }

    if (pstParentNode != AVL_NULL_PTR) {
        /* fixup parent pstNode                                               */
        if (pstParentNode->pstRight == pstBaseNode) {
            /* pstNode is right son of parent                                 */
            pstParentNode->pstRight = pstReplaceNode;
            pstParentNode->sRHeight = sNewHeight;
        } else {
            /* pstNode is left son of parent                                  */
            pstParentNode->pstLeft = pstReplaceNode;
            pstParentNode->sLHeight = sNewHeight;
        }

        /* now rebalance the pstTree (if necessary)                           */
        VosAvlBalanceTree(pstBaseTree, pstParentNode);
    } else {
        /* replacement pstNode is now root of pstTree                         */
        pstBaseTree->pstRoot = pstReplaceNode;
    }

    return;
}
