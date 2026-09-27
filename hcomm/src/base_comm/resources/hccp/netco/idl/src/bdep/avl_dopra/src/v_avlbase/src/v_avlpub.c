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
#include "v_avl_base.h"

/*******************************************************************************
 Description  : VOS_AVL_Insert_Or_Find will Insert the supplied node into the
                specified AVL tree if key does not already exist, otherwise
                returning the existing node
 Operation    : Scan down the tree looking for the insert point, going left
                if the insert key is less than the key in the tree and
                right if it is greater. When the insert point is found insert
                the new node and rebalance the tree if necessary. Return the
                existing entry instead, if found.
 Input        : pstTree    - pointer to the AVL tree
                pstNode    - pointer to the node to insert
 Output       : None
 Return       : Pointer to existing entry if found.
                AVL_NULL_PTR if no such entry (implies node successfully
                inserted)
*******************************************************************************/
void *VOS_AVL_Insert_Or_Find(AVL_TREE *pstTree, AVL_NODE *pstNode)
{
    AVL_NODE *pstParentNode;
    int iResult;

    if ((pstTree == AVL_NULL_PTR) || (pstNode == AVL_NULL_PTR) || (VOS_AVL_IN_TREE(*pstNode))) {
        return AVL_NULL_PTR;
    }

    pstNode->sRHeight = 0;
    pstNode->sLHeight = 0;

    if (pstTree->pstRoot == AVL_NULL_PTR) {
        pstTree->pstRoot = pstNode;
        pstTree->pstFirst = pstNode;
        pstTree->pstLast = pstNode;
        return AVL_NULL_PTR;
    }

    /* scan down the pstTree looking for the appropriate insert point         */
    for (pstParentNode = pstTree->pstRoot; pstParentNode != AVL_NULL_PTR;) {
        /* go left or right, depending on comparison                          */
        iResult = pstTree->pfnCompare(pstNode->pKey, pstParentNode->pKey);
        if (iResult > 0) {
            /* new key is greater than this pstNode's key, so move down right */
            /* subtree                                                        */
            if (pstParentNode->pstRight != AVL_NULL_PTR) {
                /* right subtree is not empty                                 */
                pstParentNode = pstParentNode->pstRight;
                continue;
            }
            /* right subtree is empty, so insert here                     */
            VosAvlNodeRightInsert((AVLBASE_TREE_S *)(void *)(&(pstTree->pstRoot)), (AVLBASE_NODE_S *)pstParentNode,
                (AVLBASE_NODE_S *)pstNode);

            break;
        } else if (iResult < 0) {
            /* new key is less than this pstNode's key, so move down left     */
            /* subtree                                                        */
            if (pstParentNode->pstLeft != AVL_NULL_PTR) {
                /* left subtree is not empty                                  */
                pstParentNode = pstParentNode->pstLeft;
                continue;
            }
            /* left subtree is empty, so insert here                      */
            VosAvlNodeLeftInsert((AVLBASE_TREE_S *)(void *)(&(pstTree->pstRoot)), (AVLBASE_NODE_S *)pstParentNode,
                (AVLBASE_NODE_S *)pstNode);

            break;
        }
        /* found a matching key, so get out now and return entry found    */
        pstNode->sRHeight = -1;
        pstNode->sLHeight = -1;
        return pstParentNode->pSelf;
    }

    /* now rebalance the pstTree if necessary                                 */
    if (pstParentNode != AVL_NULL_PTR) {
        VosAvlBalanceTree((AVLBASE_TREE_S *)(void *)(&(pstTree->pstRoot)), (AVLBASE_NODE_S *)pstParentNode);
    }

    return AVL_NULL_PTR;
}

void VOS_AVL_Delete(AVL_TREE *pstTree, AVL_NODE *pstNode)
{
    AVLBASE_NODE_S *pstBaseNode;
    AVLBASE_TREE_S *pstBaseTree;

    if ((pstTree == AVL_NULL_PTR) || (pstNode == AVL_NULL_PTR) || (!VOS_AVL_IN_TREE(*pstNode))) {
        return;
    }

    pstBaseNode = (AVLBASE_NODE_S *)pstNode;
    pstBaseTree = (AVLBASE_TREE_S *)(void *)(&(pstTree->pstRoot));
    VosAvlDelete(pstBaseNode, pstBaseTree);
    return;
}

/*******************************************************************************
 Description  : VOS_AVL_Find will Find the node in the AVL tree with the
                supplied key
 Operation    : Search down the tree going left if the search key is less than
                the node in the tree and right if the search key is greater.
                When we run out of tree to search through either we've found
                it or the node is not in the tree.
 Input        : pstTree    - pointer to the AVL tree
                pKey       - pointer to the key
 Output       : None
 Return       : A pointer to the node or
                AVL_NULL_PTR if no node is found with the specified key
*******************************************************************************/
void *VOS_AVL_Find(AVL_TREE *pstTree, const void *pKey)
{
    /* find node with specified pKey                                          */
    AVL_NODE *pstNode;
    int iResult;

    if (pstTree == AVL_NULL_PTR) {
        return AVL_NULL_PTR;
    }
    pstNode = pstTree->pstRoot;

    while (pstNode != AVL_NULL_PTR) {
        /* compare pKey of current pstNode with supplied pKey                 */
        iResult = pstTree->pfnCompare(pKey, pstNode->pKey);
        if (iResult > 0) {
            /* specified pKey is greater than pKey of this pstNode, so look   */
            /* in subtree                                                     */
            pstNode = pstNode->pstRight;
        } else if (iResult < 0) {
            /* specified pKey is less than pKey of this pstNode, so look in   */
            /* left subtree                                                   */
            pstNode = pstNode->pstLeft;
        } else {
            /* found the requested pstNode                                    */
            break;
        }
    }

    return ((pstNode != AVL_NULL_PTR) ? pstNode->pSelf : AVL_NULL_PTR);
}

/*******************************************************************************
 Description  : VOS_AVL_Next will Find next node in the AVL tree.
 Operation    : If the specified node has a right-son then return the left-
                most son of this. Otherwise search back up until we find a
                node of which we are in the left sub-tree and return that.
 Input        : pstNode    - pointer to the current node in the tree
 Output       : None
 Return       : A pointer to the next node in the tree
*******************************************************************************/
void *VOS_AVL_Next(AVL_NODE *pstNode)
{
    AVL_NODE *pstNodeTmp = pstNode;
    if ((pstNodeTmp == AVL_NULL_PTR) || (!VOS_AVL_IN_TREE(*pstNodeTmp))) {
        return AVL_NULL_PTR;
    }

    if (pstNodeTmp->pstRight != AVL_NULL_PTR) {
        /* next pstNode is left-most pstNode in right subtree                 */
        pstNodeTmp = pstNodeTmp->pstRight;
        FIND_LEFTMOST_NODE(pstNodeTmp);
    } else {
        /* no right-son, so find a pstNode of which we are in the left subtree */
        while (pstNodeTmp != AVL_NULL_PTR) {
            if ((pstNodeTmp->pstParent == AVL_NULL_PTR) || (pstNodeTmp->pstParent->pstLeft == pstNodeTmp)) {
                pstNodeTmp = pstNodeTmp->pstParent;
                break;
            }

            pstNodeTmp = pstNodeTmp->pstParent;
        }
    }

    return ((pstNodeTmp != AVL_NULL_PTR) ? pstNodeTmp->pSelf : AVL_NULL_PTR);
}

/*******************************************************************************
 Description  : VOS_AVL_Prev will Find previous node in the AVL tree.
 Operation    : If we have a left-son then the previous node is the right-most
                son of this. Otherwise, look for a node of whom we are in the
                left subtree and return that.
 Input        : pstNode    - pointer to the current node in the tree
 Output       : None
 Return       : A pointer to the previous node in the tree
*******************************************************************************/
void *VOS_AVL_Prev(AVL_NODE *pstNode)
{
    AVL_NODE *pstNodeTmp = pstNode;
    if ((pstNodeTmp == AVL_NULL_PTR) || (!VOS_AVL_IN_TREE(*pstNodeTmp))) {
        return AVL_NULL_PTR;
    }

    if (pstNodeTmp->pstLeft != AVL_NULL_PTR) {
        /* previous pstNode is right-most pstNode in left subtree             */
        pstNodeTmp = pstNodeTmp->pstLeft;
        FIND_RIGHTMOST_NODE(pstNodeTmp);
    } else {
        /* no left-son, so find a pstNode of which we are in the right subtree */
        while (pstNodeTmp != AVL_NULL_PTR) {
            if ((pstNodeTmp->pstParent == AVL_NULL_PTR) || (pstNodeTmp->pstParent->pstRight == pstNodeTmp)) {
                pstNodeTmp = pstNodeTmp->pstParent;
                break;
            }

            pstNodeTmp = pstNodeTmp->pstParent;
        }
    }

    return ((pstNodeTmp != AVL_NULL_PTR) ? pstNodeTmp->pSelf : AVL_NULL_PTR);
}

void *VOS_AVL_Find_Or_Find_Next(AVL_TREE *pstTree, const void *pKey, unsigned int bValue)
{
    AVL_NODE *pstNode;
    void *pFoundNode = AVL_NULL_PTR;
    int iResult;

    if (pstTree == AVL_NULL_PTR) {
        return AVL_NULL_PTR;
    }
    pstNode = pstTree->pstRoot;

    if (pstNode == AVL_NULL_PTR) {
        return (pFoundNode);
    }

    /* There is something in the pstTree                                  */
    for (;;) {
        /* compare pKey of current pstNode with supplied pKey             */
        iResult = pstTree->pfnCompare(pKey, pstNode->pKey);
        if (iResult > 0) {
            /* specified pKey is greater than pKey of this pstNode, so    */
            /* look in right subtree                                      */
            if (pstNode->pstRight == AVL_NULL_PTR) {
                /* We've found the previous pstNode - so we now need to   */
                /* find the successor to this one.                        */
                pFoundNode = VOS_AVL_Next(pstNode);
                break;
            }

            pstNode = pstNode->pstRight;
        } else if (iResult < 0) {
            /* specified pKey is less than pKey of this pstNode, so       */
            /* look in left subtree                                       */
            if (pstNode->pstLeft == AVL_NULL_PTR) {
                /* We've found the next pstNode so store and drop out     */
                pFoundNode = pstNode->pSelf;
                break;
            }

            pstNode = pstNode->pstLeft;
        } else {
            /* found the requested pstNode                                */
            if (bValue) {
                /* need to find the successor pstNode to this pstNode     */
                pFoundNode = VOS_AVL_Next(pstNode);
            } else {
                pFoundNode = pstNode->pSelf;
            }
            break;
        }
    }

    return (pFoundNode);
}
