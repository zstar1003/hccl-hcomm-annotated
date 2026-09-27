/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @defgroup v_avl3 AVL3
 * @ingroup util
 */

#ifndef V_AVL3_H
#define V_AVL3_H

#include "avl_adapt.h"

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

/* AVL3_COMPARE compare function type. */
/**
 * @ingroup v_avl3
 * 本函数定义了树中节点的排序策略，任何需要访问树中元素的接口（包括添加、查找及删除树中节点）
 * 都要通过此函数来访问树中节点元素。
 */
#if defined(AVL_INT32_TO_LONG)
/* 兼容SSP VXWORKS */
typedef long (*AVL3_COMPARE)(const void *, const void *);
#else
typedef int (*AVL3_COMPARE)(const void *, const void *);
#endif

/* Structure  : AVL3_NODE */
/* Description: Node in an AVL3 tree. */
/**
 * @ingroup v_avl3
 * 本数据结构描述树中节点的位置。该数据结构在使用前必须调用接口#VOS_AVL3_INIT_NODE 对其进行初始化。
 */
#pragma pack(4)
typedef struct avl3_node {
    struct avl3_node *pstParent;
    struct avl3_node *pstLeft;
    struct avl3_node *pstRight;
    short int sLHeight;
    short int sRHeight;
} AVL3_NODE;
#if defined(AVL_PACKEND_ZERO)
/* 兼容SSP VXWORKS */
#pragma pack(0)
#else
#pragma pack()
#endif

/* Structure  : AVL3_TREE_INFO */
/* Description: AVL3 tree information. */
/**
 * @ingroup v_avl3
 * 本数据结构包含AVL3树的比较函数，树中节点的地址以及树中各元素的KEY。
 */
#pragma pack(4)
typedef struct avl3_tree_info {
    AVL3_COMPARE pfCompare;
    unsigned short int usKeyOffset;
    unsigned short int usNodeOffset;
} AVL3_TREE_INFO;
#if defined(AVL_PACKEND_ZERO)
/* 兼容SSP VXWORKS */
#pragma pack(0)
#else
#pragma pack()
#endif

/* Structure  : AVL3_TREE */
/* Description: AVL3 tree root. */
/**
 * @ingroup v_avl3
 * 本数据结构描述树的基础信息以及树中节点的信息。
 * 该数据结构在使用前必须调用接口#VOS_AVL3_INIT_TREE 对其进行初始化。
 */
typedef struct avl3_tree {
    AVL3_NODE *pstRoot;
    AVL3_NODE *pstFirst;
    AVL3_NODE *pstLast;
} AVL3_TREE;

/* EXTERN VARIABLES & FUNCTIONS DECLARATION */
/**
 * @ingroup v_avl3
 * @brief 本接口尝试插入一个节点到AVL3树中。
 *
 * @par 描述
 * 本接口尝试插入一个节点到AVL3树中。如果树中已存在具有相同KEY值的节点，则插入失败且本接口返回该节点的指针。
 * @attention
 * @li 树和节点必须已调用接口#VOS_AVL3_INIT_TREE 和 #VOS_AVL3_INIT_NODE 完成初始化。
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 不支持并发场景。
 *
 * @param *pstTree [IN]  表示待插入节点的树。
 * @param *pstNode [IN]  表示待插入树中的节点。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 如果插入成功，则返回AVL_NULL_PTR（0L）。
 * @retval 失败 返回树中具有相同KEY值的节点的指针。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_IN_TREE
 * @li VOS_AVL3_INIT_NODE
 * @li VOS_AVL3_FIND
 * @li VOS_AVL3_FIND_NEXT
 */
extern void *VosAvl3InsertOrFind(AVL3_TREE *pstTree, AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 删除指定的节点。
 *
 * @par 描述
 * 本接口将先前插入树中的指定节点移除。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 如果节点原本就不属于AVL树，则调用本接口的结果不可预料。
 * @li 不支持并发场景。
 *
 * @param *pstTree [IN]  表示待移除节点所在的树。
 * @param *pstNode [IN]  表示要从树中移除的节点。
 *
 * @retval 无。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see VOS_AVL3_INSERT
 */
extern void VosAvl3Delete(AVL3_TREE *pstTree, AVL3_NODE *pstNode);

/**
 * @ingroup v_avl3
 * @brief 查找指定节点。
 *
 * @par 描述
 * 根据输入的KEY返回树中元素的指针。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 用户需要保证树中存在待查找的节点。如果节点原本就不属于AVL3树，则调用本接口的结果不可预料。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pstKey [IN]  指向待搜索的KEY的指针。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回指定KEY值的节点指针。
 * @retval 失败 如果树中没有指针KEY值的节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_FIND_NEXT
 * @li VOS_AVL3_FIND_OR_FIND_NEXT
 */
extern void *VosAvl3Find(AVL3_TREE *pstTree, const void *pstKey, AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 根据输入的KEY值在树中查找，如果匹配成功则返回树中节点的指针。
 *
 * @par 描述
 * 根据输入的KEY返回树中具有相同KEY值的节点指针。如果树中没有相同KEY值的节点，
 * 则本接口根据比较函数定义的大小规则返回KEY值略大的下一个节点的指针。
 * 本接口相当于 #VOS_AVL3_FIND 和#VOS_AVL3_FIND_NEXT功能的组合。
 * @attention
 * @li 在一个按字母排序的树中，假设树中包含元素的KEY值分别为A，B，C，D，E，F。
 * 使用本接口时传入D或E可以返回KEY值为F的节点，使用本接口传入KEY值G则会返回NULL
 * (因为KEY值为G的节点在树中没有下一个节点)。
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pKey [IN]  指向待搜索的KEY的指针。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回树中具有相同KEY值的节点指针（如果匹配成功）。如果指定KEY值的元素在树中不存在，
 * 则返回树中KEY值略大的下一个节点指针。
 * @retval 失败 如果树中指定KEY值的节点不存在且无下一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_FIND
 * @li VOS_AVL3_FIND_NEXT
 */
extern void *VosAvl3FindOrFindNext(AVL3_TREE *pstTree, const void *pKey, unsigned int bFlag,
    AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 本接口返回树中第一个节点的指针。
 *
 * @par 描述
 * 本接口返回树中第一个节点的指针。
 * @attention AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回树中第一个元素的指针。
 * @retval 失败 如果树为空，返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_LAST
 * @li VOS_AVL3_NEXT
 * @li VOS_AVL3_PREV
 */
extern void *VosAvl3First(AVL3_TREE *pstTree, AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 本接口返回树中最后一个节点的指针。
 *
 * @par 描述
 * 本接口返回树中最后一个节点的指针。
 * @attention
 * AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回树中最后一个元素的指针。
 * @retval 失败 如果树为空，返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_FIRST
 * @li VOS_AVL3_NEXT
 * @li VOS_AVL3_PREV
 */
extern void *VosAvl3Last(AVL3_TREE *pstTree, AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 根据输入的KEY返回树中比KEY值大的下一个节点的指针。
 *
 * @par 描述
 * 根据输入的KEY返回树中比KEY值大的下一个节点的指针。即如果传入节点的右子树存在，
 * 则返回右子树中最左边的节点。如果传入节点的右子树不存在，则返回AVL3树的根节点。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 输入的节点必须属于待查找的AVL3树。
 *
 * @param *pstNode [IN]  表示输入的节点，本接口返回比该输入节点的KEY值大的下一个节点。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回树中下一个元素的指针。
 * @retval 失败 如果输入的节点是树中的最后一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_FIRST
 * @li VOS_AVL3_LAST
 * @li VOS_AVL3_PREV
 */
extern void *VosAvl3Next(AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo);

/**
 * @ingroup v_avl3
 * @brief 本接口返回树中指定节点的前一个节点指针。
 *
 * @par 描述
 * 本接口返回树中指定节点的前一个节点指针。即如果传入节点的左子树存在，
 * 则返回左子树中最右边的节点。如果传入节点的左子树不存在，则返回AVL3树的根节点。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 输入的节点必须属于待查找的AVL3树。
 *
 * @param *pstNode [IN]  表示输入的节点，本接口返回比该输入节点的KEY值小的前一个节点。
 * @param *pstTreeInfo [IN]  包含树的描述信息。
 *
 * @retval 成功 返回树中前一个元素的指针。
 * @retval 失败 如果输入的节点是树中的第一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl3.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL3_LAST
 * @li VOS_AVL3_NEXT
 */
extern void *VosAvl3Prev(AVL3_NODE *pstNode, AVL3_TREE_INFO *pstTreeInfo);

/* AVL3 EXPOSED MACROS */
#define VOS_AVL3_INIT_TREE(TREE, TREE_INFO)                                                                            \
    do {                                                                                                               \
        (TREE).pstFirst = (AVL3_NODE *)AVL_NULL_PTR;                                                                   \
        (TREE).pstLast = (AVL3_NODE *)AVL_NULL_PTR;                                                                    \
        (TREE).pstRoot = (AVL3_NODE *)AVL_NULL_PTR;                                                                    \
    } while (0)

#define VOS_AVL3_INIT_NODE(NODE)                                                                                       \
    do {                                                                                                               \
        (NODE).pstParent = (AVL3_NODE *)AVL_NULL_PTR;                                                                  \
        (NODE).pstLeft = (AVL3_NODE *)AVL_NULL_PTR;                                                                    \
        (NODE).pstRight = (AVL3_NODE *)AVL_NULL_PTR;                                                                   \
        (NODE).sLHeight = -1;                                                                                          \
        (NODE).sRHeight = -1;                                                                                          \
    } while (0)

#define VOS_AVL3_INSERT(TREE, NODE, TREE_INFO) (AVL_NULL_PTR == VosAvl3InsertOrFind(&(TREE), &(NODE), &(TREE_INFO)))

#define VOS_AVL3_INSERT_OR_FIND(TREE, NODE, TREE_INFO) VosAvl3InsertOrFind(&(TREE), &(NODE), &(TREE_INFO))

#define VOS_AVL3_DELETE(TREE, NODE) VosAvl3Delete(&(TREE), &(NODE))

#define VOS_AVL3_FIND(TREE, KEY, TREE_INFO) VosAvl3Find(&(TREE), (KEY), &(TREE_INFO))

#define VOS_AVL3_NEXT(NODE, TREE_INFO) VosAvl3Next(&(NODE), &(TREE_INFO))

#define VOS_AVL3_PREV(NODE, TREE_INFO) VosAvl3Prev(&(NODE), &(TREE_INFO))

#define VOS_AVL3_FIRST(TREE, TREE_INFO) VosAvl3First(&(TREE), &(TREE_INFO))

#define VOS_AVL3_LAST(TREE, TREE_INFO) VosAvl3Last(&(TREE), &(TREE_INFO))

#define VOS_AVL3_IN_TREE(NODE) (((NODE).sLHeight != -1) && ((NODE).sRHeight != -1))

#define VOS_AVL3_FIND_NEXT(TREE, KEY, TREE_INFO) VosAvl3FindOrFindNext(&(TREE), (KEY), AVL_TRUE, &(TREE_INFO))

#define VOS_AVL3_FIND_OR_FIND_NEXT(TREE, KEY, TREE_INFO) VosAvl3FindOrFindNext(&(TREE), (KEY), AVL_FALSE, &(TREE_INFO))

/* Macro to determine the largest value of two variables (internal use) */
#define VOS_AVL3_MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif
