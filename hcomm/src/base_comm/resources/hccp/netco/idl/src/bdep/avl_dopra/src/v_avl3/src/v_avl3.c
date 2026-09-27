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
#include "v_avl3_inner.h"
#include "v_avl3.h"

void *VosAvl3FindOrFindNext(AVL3_TREE *pstTree, const void *pKey, unsigned int bFlag, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstNode;
    void *pFoundNode = AVL_NULL_PTR;
    int iResult;
    int iKeyOffset;

    if (TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo)) {
        return AVL_NULL_PTR;
    }
    pstNode = pstTree->pstRoot;
    if (pstNode == AVL_NULL_PTR) {
        return AVL_NULL_PTR;
    }

    iKeyOffset = GET_KEYOFFSET(pstTreeInfo);
    /* The Tree not empty. So start traversing through the tree.          */
    for (;;) {
        /* compare pKey of current pstNode with supplied pKey             */
        iResult = pstTreeInfo->pfCompare(pKey, (void *)((unsigned char *)pstNode + iKeyOffset));
        if (iResult > 0) {
            /* specified pKey is greater than pKey of this pstNode,       */
            /* so look in right subtree                                   */
            if (pstNode->pstRight == AVL_NULL_PTR) {
                /* We've found the previous pstNode - so we now need to   */
                /* find the successor to this one.                        */
                pFoundNode = VosAvl3Next(pstNode, pstTreeInfo);
                break;
            }

            pstNode = pstNode->pstRight;
        } else if (iResult < 0) {
            /* specified pKey is less than pKey of this pstNode, so       */
            /* look in left subtree                                       */
            if (pstNode->pstLeft == AVL_NULL_PTR) {
                /* We've found the next pstNode so store and drop out     */
                pFoundNode = (void *)((unsigned char *)pstNode - pstTreeInfo->usNodeOffset);
                break;
            }

            pstNode = pstNode->pstLeft;
        } else {
            /* found the requested pstNode for supplied pKey               */
            if (bFlag) {
                /* need to find the successor pstNode to this pstNode     */
                pFoundNode = VosAvl3Next(pstNode, pstTreeInfo);
            } else {
                pFoundNode = (void *)((unsigned char *)pstNode - pstTreeInfo->usNodeOffset);
            }
            break;
        }
    }

    return pFoundNode;
}

void *VosAvl3InsertOrFind(AVL3_TREE *pstTree, AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstParentNode;
    int iResult;
    int iKeyOffset;

    if (TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo) || (pstNode == AVL_NULL_PTR)) {
        return AVL_NULL_PTR;
    }

    pstNode->sRHeight = 0;
    pstNode->sLHeight = 0;

    if (pstTree->pstRoot == AVL_NULL_PTR) {
        /* pstTree is empty, so insert at root                                */
        pstTree->pstRoot = pstNode;
        pstTree->pstFirst = pstNode;
        pstTree->pstLast = pstNode;
        return AVL_NULL_PTR;
    }

    /* scan down the pstTree looking for the appropriate insert point         */
    pstParentNode = pstTree->pstRoot;

    iKeyOffset = GET_KEYOFFSET(pstTreeInfo);
    while (pstParentNode != AVL_NULL_PTR) {
        /* go left or right, depending on comparison                          */
        iResult = pstTreeInfo->pfCompare((void *)((unsigned char *)pstNode + iKeyOffset),
            (void *)((unsigned char *)pstParentNode + iKeyOffset));
        if (iResult > 0) {
            /* new key is greater than this pstNode's key, so move down       */
            /* right subtree                                                  */
            if (pstParentNode->pstRight != AVL_NULL_PTR) {
                /* right subtree is not empty                                 */
                pstParentNode = pstParentNode->pstRight;
                continue;
            }

            /* right subtree is empty, so insert here                         */
            VosAvlNodeRightInsert((AVLBASE_TREE_S *)pstTree, (AVLBASE_NODE_S *)pstParentNode,
                (AVLBASE_NODE_S *)pstNode);
        } else if (iResult < 0) {
            /* new key is less than this pstNode's key, so move down left     */
            /* subtree                                                        */
            if (pstParentNode->pstLeft != AVL_NULL_PTR) {
                /* left subtree is not empty                                  */
                pstParentNode = pstParentNode->pstLeft;
                continue;
            }

            /* left subtree is empty, so insert here                          */
            VosAvlNodeLeftInsert((AVLBASE_TREE_S *)pstTree, (AVLBASE_NODE_S *)pstParentNode, (AVLBASE_NODE_S *)pstNode);
        } else {
            /* found a matching key, so get out now and return entry found    */
            pstNode->sRHeight = -1;
            pstNode->sLHeight = -1;
            return (void *)((unsigned char *)pstParentNode - pstTreeInfo->usNodeOffset);
        }

        break;
    }

    /* now rebalance the pstTree if necessary                                 */
    VosAvlBalanceTree((AVLBASE_TREE_S *)pstTree, (AVLBASE_NODE_S *)pstParentNode);

    return AVL_NULL_PTR;
}

void VosAvl3Delete(AVL3_TREE *pstTree, AVL3_NODE *pstNode)
{
    /* delete specified pstNode from pstTree                                  */
    AVLBASE_NODE_S *pstBaseNode;
    AVLBASE_TREE_S *pstBaseTree;
    if ((pstTree == AVL_NULL_PTR) || (pstNode == AVL_NULL_PTR)) {
        return;
    }

    pstBaseNode = (AVLBASE_NODE_S *)pstNode;
    pstBaseTree = (AVLBASE_TREE_S *)pstTree;
    VosAvlDelete(pstBaseNode, pstBaseTree);
}

/*******************************************************************************
 Description  : VOS_AVL3_Find will Find the node in the AVL3 tree with the
                supplied key
 Input        : pstTree     - pointer to the AVL3 tree
                pstKey      - pointer to the key
                pstTreeInfo - pointer to the AVL3 tree info
 Output       : None
 Return       : A pointer to the node if found else AVL_NULL_PTR if no node
                is found with the specified key
 Operation:     Search down the tree going left if the search key is less than
                the node in the tree and right if the search key is greater.
                When we run out of tree to search through either we've found
                it or the node is not in the tree.
*******************************************************************************/
void *VosAvl3Find(AVL3_TREE *pstTree, const void *pstKey, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstNode;
    int iResult;
    int iKeyOffset;

    if (TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo)) {
        return AVL_NULL_PTR;
    }

    pstNode = pstTree->pstRoot;
    iKeyOffset = GET_KEYOFFSET(pstTreeInfo);

    while (pstNode != AVL_NULL_PTR) {
        iResult = pstTreeInfo->pfCompare(pstKey, (void *)((unsigned char *)pstNode + iKeyOffset));
        if (iResult > 0) {
            /* specified pstKey is greater than pstKey of this pstNode, so    */
            /* look in right subtree                                          */
            pstNode = pstNode->pstRight;
        } else if (iResult < 0) {
            /* specified pstKey is less than pstKey of this pstNode, so look  */
            /* in left subtree                                                */
            pstNode = pstNode->pstLeft;
        } else {
            /* found the requested pstNode                                    */
            break;
        }
    }

    return GET_NODE_START_ADDRESS(pstNode, pstTreeInfo->usNodeOffset);
}

void *VosAvl3First(AVL3_TREE *pstTree, AVL3_TREE_INFO *pstTreeInfo)
{
    /* find first node in pstTree                                             */
    AVL3_NODE *pstNode;

    if (TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo)) {
        return AVL_NULL_PTR;
    }

    pstNode = pstTree->pstFirst;

    return GET_NODE_START_ADDRESS(pstNode, pstTreeInfo->usNodeOffset);
}

void *VosAvl3Last(AVL3_TREE *pstTree, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstNode;

    if (TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo)) {
        return AVL_NULL_PTR;
    }

    pstNode = pstTree->pstLast;

    return GET_NODE_START_ADDRESS(pstNode, pstTreeInfo->usNodeOffset);
}

/*******************************************************************************
 Description  : VOS_AVL3_Next will Find next node in the AVL3 tree
 Input        : pstNode     - pointer to the current node in the tree
                pstTreeInfo - pointer to the AVL3 tree info
 Output       : None
 Return       : A pointer to the next node in the tree
 Operation    : If the specified node has a right-son then return the left-
                most son of this.  Otherwise search back up until we find a
                node of which we are in the left sub-tree and return that node.
*******************************************************************************/
void *VosAvl3Next(AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstNodeTmp = pstNode;
    if ((pstNodeTmp == AVL_NULL_PTR) || (pstTreeInfo == AVL_NULL_PTR)) {
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

    return GET_NODE_START_ADDRESS(pstNodeTmp, pstTreeInfo->usNodeOffset);
}

/*******************************************************************************
 Description  : VOS_AVL3_Prev will Find previous node in the AVL3 tree
 Input        : pstNode     - pointer to the current node in the tree
                pstTreeInfo - pointer to the AVL3 tree info
 Output       : None
 Return       : A pointer to the previous node in the tree
 Operation    : If we have a left-son then the previous node is the right-most
                son of this.  Otherwise, look for a node of whom we are in the
                left subtree and return that node.
*******************************************************************************/
void *VosAvl3Prev(AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo)
{
    AVL3_NODE *pstNodeTmp = pstNode;
    if ((pstNodeTmp == AVL_NULL_PTR) || (pstTreeInfo == AVL_NULL_PTR)) {
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
                /* Node is found, break from the loop.                        */
                pstNodeTmp = pstNodeTmp->pstParent;
                break;
            }

            pstNodeTmp = pstNodeTmp->pstParent;
        }
    }

    return GET_NODE_START_ADDRESS(pstNodeTmp, pstTreeInfo->usNodeOffset);
}
