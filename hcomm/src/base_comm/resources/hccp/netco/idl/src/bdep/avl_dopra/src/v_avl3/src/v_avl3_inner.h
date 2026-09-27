/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef V_AVL3_INNER_H
#define V_AVL3_INNER_H

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

#define TREE_OR_TREEINFO_IS_NULL(pstTree, pstTreeInfo) (((pstTree) == AVL_NULL_PTR) || ((pstTreeInfo) == AVL_NULL_PTR))

#define GET_NODE_START_ADDRESS(pstNode, usOffset)                                                                      \
    (((pstNode) != AVL_NULL_PTR) ? (void *)((unsigned char *)(pstNode) - (usOffset)) : AVL_NULL_PTR)

#define GET_KEYOFFSET(pstTreeInfo) ((int)((pstTreeInfo)->usKeyOffset - (pstTreeInfo)->usNodeOffset))

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif /* V_AVL3_INNER_H */
