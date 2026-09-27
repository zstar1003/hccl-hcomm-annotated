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
 * @defgroup util Util
 */
/**
 * @defgroup v_avl AVL
 * @ingroup util
 */
#ifndef V_AVL_BASE_H
#define V_AVL_BASE_H

#include "avl_adapt.h"

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

/* compare function */
/**
 * @ingroup v_avl
 * 本函数定义了树中节点的排序策略，任何需要访问树中元素的接口（包括添加、查找及
 * 删除树中节点）都要通过此函数来访问树中节点元素。
 */
#if defined(AVL_INT32_TO_LONG)
/* 兼容SSP VXWORKS */
typedef long (*AVL_V2_COMPARE_FUNC)(const void *, const void *);
#else
typedef int (*AVL_V2_COMPARE_FUNC)(const void *, const void *);
#endif

/* Structure  : AVL_NODE */
/* Description: Node in an AVL tree. */
/**
 * @ingroup v_avl
 * 本数据结构包含本节点在树中的位置信息，以及树结构体中KEY的地址。
 */
#pragma pack(4)
typedef struct avl_node {
    struct avl_node
        *pstParent; /**< 指向本节点在树中的父节点的指针。如果本节点是树的根节点或者本节点不在树中，则指针为空。 */
    struct avl_node *pstLeft;
    struct avl_node *pstRight;
    short int sLHeight;
    short int sRHeight;
    void *pSelf;
    void *pKey;
} AVL_NODE;
#if defined(AVL_PACKEND_ZERO)
/* 兼容SSP VXWORKS */
#pragma pack(0)
#else
#pragma pack()
#endif

/* Structure  : AVL_TREE */
/* Description: AVL tree root. */
/**
 * @ingroup v_avl
 * 本数据结构用来表示树的基本信息。同时还包含了树中节点的信息以及使用的比较函数。
 */
typedef struct avl_tree {
    AVL_V2_COMPARE_FUNC pfnCompare;
    AVL_NODE *pstRoot;
    AVL_NODE *pstFirst;
    AVL_NODE *pstLast;
} AVL_TREE;

/* EXTERN VARIABLES & FUNCTIONS DECLARATION */
/**
 * @ingroup v_avl
 * @brief 本接口尝试插入一个节点到AVL树中。
 *
 * @par 描述
 * 本接口尝试插入一个节点到AVL树中。如果树中已存在具有相同KEY值的节点，
 * 则插入失败且本接口返回该节点的指针 pSelf 。pSelf 指针是 #AVL_NODE 中的一个变量。
 * @attention
 * @li AVL树和待插入的节点需要预先调用接口 #VOS_AVL_INIT_TREE 和#VOS_AVL_INIT_NODE 完成初始化。
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待插入节点的树。
 * @param *pstNode [IN]  表示待插入树中的节点。
 *
 * @retval 成功 如果插入成功，则返回AVL_NULL_PTR（0L）。
 * @retval 失败 返回具有相同KEY值的节点指针（如果树中已有相同KEY值的节点）。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL_INSERT
 * @li VOS_AVL_FIND
 */
extern void *VOS_AVL_Insert_Or_Find(AVL_TREE *pstTree, AVL_NODE *pstNode);

/**
 * @ingroup v_avl
 * @brief 删除指定的节点。
 *
 * @par 描述
 * 本接口将先前插入树中的指定节点移除。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 如果节点原本就不属于AVL树，则调用本接口的结果不可预料。
 *
 * @param *pstTree [IN]  表示待移除节点所在的树。
 * @param *pstNode [IN]  表示要从树中移除的节点。
 *
 * @retval 无。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see 无。
 */
extern void VOS_AVL_Delete(AVL_TREE *pstTree, AVL_NODE *pstNode);

/**
 * @ingroup v_avl
 * @brief 查找指定的节点。
 *
 * @par 描述
 * 根据输入的KEY返回树中元素的指针。
 * @attention
 * AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pKey [IN]  指向待搜索的KEY的指针。
 *
 * @retval 成功 返回指定KEY值的节点指针。
 * @retval 失败 如果树中没有指针KEY值的节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see VOS_AVL_FIND_NEXT
 */
extern void *VOS_AVL_Find(AVL_TREE *pstTree, const void *pKey);

/**
 * @ingroup v_avl
 * @brief 本接口根据传入的KEY值在树中进行查找，如果匹配成功则返回对应节点指针，
 * 若匹配失败，则返回比KEY值大的下一个节点。
 *
 * @par 描述
 * 根据输入的KEY值在树中查找，如果匹配成功则根据bValue的值，返回相关节点的指针。
 * 如果树中没有指定KEY值的节点，则返回比KEY值大的下一个节点。该大小顺序是由比较函数定义好的。
 *
 * @attention
 * @li 在一个按字母排序的树中，假设树中包含元素的KEY值分别为A，B，C，D，E，F。
 * 使用本接口时传入D或E可以返回KEY值为F的节点，使用本接口传入KEY值G则会返回NULL（因
 * 为KEY值为G的节点在树中没有下一个节点）。
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 *
 * @param *pstTree [IN]  表示待查找的树。
 * @param *pKey [IN]  指向待搜索的KEY的指针。
 * @param *bValue [IN]  成功匹配到KEY值对应节点时，如果bValue为AVL_TRUE，则返回大于KEY的节点；
 * 如果bValue为AVL_FALSE，则返回大于等于KEY的节点。
 *
 * @retval 成功 返回如下值： \n
 * @li 根据输入KEY值匹配成功的节点指针（如果输入的KEY值在树中存在）。
 * @li 如果输入在KEY值在树中不存在，则根据比较函数定义的大小规则返回比KEY值大的下一个节点。
 * @retval 失败 如果树中指定KEY值的节点无下一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL_FIND_NEXT
 * @li VOS_AVL_FIND
 */
extern void *VOS_AVL_Find_Or_Find_Next(AVL_TREE *pstTree, const void *pKey, unsigned int bValue);

/**
 * @ingroup v_avl
 * @brief 本接口返回树中指定节点的下一个节点指针。
 *
 * @par 描述
 * 本接口返回树中指定节点pstNode的下一个节点指针。即如果传入节点的右子树存在，则返回右子树中最左边的节点。
 * 如果传入节点的右子树不存在，则返回AVL树的根节点。
 * @attention
 * @li 本接口并未对传入参数做空指针校验。
 * @li 输入的节点必须属于某个AVL树。
 *
 * @param *pstNode [IN]  输入的当前节点，本接口将返回其下一个节点。
 *
 * @retval 成功 返回树中下一个元素的指针。
 * @retval 失败 如果输入的节点是树中的最后一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL_FIRST
 * @li VOS_AVL_LAST
 * @li VOS_AVL_PREV
 */
extern void *VOS_AVL_Next(AVL_NODE *pstNode);
/**
 * @ingroup v_avl
 * @brief 本接口返回树中指定节点的前一个节点指针。
 *
 * @par 描述
 * 本接口返回指定节点pstNode的前一个节点的指针。即如果传入节点的左子树存在，则返回左子树中最右边的节点。
 * 如果传入节点的左子树不存在，则返回AVL树的根节点。
 * @attention
 * @li 本接口并未对传入参数做空指针校验。
 * @li 输入的节点必须属于待查找的AVL3树。
 *
 * @param *pstNode [IN]  输入的当前节点，本接口返回其前一个节点。
 *
 * @retval 成功 返回树中的前一个节点。
 * @retval 失败 如果输入的节点是树中的第一个节点，则返回AVL_NULL_PTR（0L）。
 * @par 依赖
 * @li v_avl.h：该接口声明所在的文件。
 * @see
 * @li VOS_AVL_FIRST
 * @li VOS_AVL_NEXT
 * @li VOS_AVL_LAST
 */
extern void *VOS_AVL_Prev(AVL_NODE *pstNode);

/* V2 AVL ACCESS MACROS */
/**
 * @ingroup  v_avl
 * 初始化AVL树
 */
#define VOS_AVL_INIT_TREE(TREE, COMPARE)                                                                               \
    do {                                                                                                               \
        (TREE).pfnCompare = (COMPARE);                                                                                 \
        (TREE).pstFirst = (AVL_NODE *)AVL_NULL_PTR;                                                                    \
        (TREE).pstLast = (AVL_NODE *)AVL_NULL_PTR;                                                                     \
        (TREE).pstRoot = (AVL_NODE *)AVL_NULL_PTR;                                                                     \
    } while (0)

/**
 * @ingroup  v_avl
 * 初始化AVL树的结点
 */
#define VOS_AVL_INIT_NODE(NODE, SELF, KEY)                                                                             \
    do {                                                                                                               \
        (NODE).pstParent = (AVL_NODE *)AVL_NULL_PTR;                                                                   \
        (NODE).pstLeft = (AVL_NODE *)AVL_NULL_PTR;                                                                     \
        (NODE).pstRight = (AVL_NODE *)AVL_NULL_PTR;                                                                    \
        (NODE).pSelf = (SELF);                                                                                         \
        (NODE).pKey = (KEY);                                                                                           \
        (NODE).sLHeight = -1;                                                                                          \
        (NODE).sRHeight = -1;                                                                                          \
    } while (0)

/* V2 AVL EXPOSED MACRO DEFINITIONS */
/**
 * @ingroup  v_avl
 * 尝试插入一个节点到AVL树中，返回值为[VOS_TRUE, VOS_FALSE]，详细说明可以参考接口#VOS_AVL_Insert_Or_Find.
 */
#define VOS_AVL_INSERT(TREE, NODE) (VOS_AVL_Insert_Or_Find(&(TREE), &(NODE)) == AVL_NULL_PTR)
/**
 * @ingroup  v_avl
 * 尝试插入一个节点到AVL树中，返回值以及详细说明可以参考接口#VOS_AVL_Insert_Or_Find.
 */
#define VOS_AVL_INSERT_OR_FIND(TREE, NODE) VOS_AVL_Insert_Or_Find(&(TREE), &(NODE))
/**
 * @ingroup  v_avl
 * 删除指定的节点，详细说明可以参考接口#VOS_AVL_Delete.
 */
#define VOS_AVL_DELETE(TREE, NODE) VOS_AVL_Delete(&(TREE), &(NODE))
/**
 * @ingroup  v_avl
 * 查找指定的节点，详细说明可以参考接口#VOS_AVL_Find.
 */
#define VOS_AVL_FIND(TREE, KEY) VOS_AVL_Find(&(TREE), (KEY))
/**
 * @ingroup  v_avl
 * 返回树中指定节点的下一个节点指针，详细说明可以参考接口#VOS_AVL_Next.
 */
#define VOS_AVL_NEXT(NODE) VOS_AVL_Next(&(NODE))
/**
 * @ingroup  v_avl
 * 返回树中指定节点的前一个节点指针，详细说明可以参考接口#VOS_AVL_Prev.
 */
#define VOS_AVL_PREV(NODE) VOS_AVL_Prev(&(NODE))
/**
 * @ingroup  v_avl
 * 返回树中首个节点的指针。
 */
#define VOS_AVL_FIRST(TREE)                                                                                            \
    (((&(TREE))->pstFirst != (AVL_NODE *)AVL_NULL_PTR) ? (&(TREE))->pstFirst->pSelf : AVL_NULL_PTR)
/**
 * @ingroup  v_avl
 * 返回树中最后一个节点的指针。
 */
#define VOS_AVL_LAST(TREE) (((&(TREE))->pstLast != (AVL_NODE *)AVL_NULL_PTR) ? (&(TREE))->pstLast->pSelf : AVL_NULL_PTR)
/**
 * @ingroup  v_avl
 * 判断给定节点是否在树中，返回值为[VOS_TRUE, VOS_FALSE]
 */
#define VOS_AVL_IN_TREE(NODE) (((NODE).sLHeight != -1) && ((NODE).sRHeight != -1))
/**
 * @ingroup  v_avl
 * 查找大于关键字KEY的节点，详细说明参考接口#VOS_AVL_Find_Or_Find_Next.
 */
#define VOS_AVL_FIND_NEXT(TREE, KEY) VOS_AVL_Find_Or_Find_Next(&(TREE), (KEY), AVL_TRUE)
/**
 * @ingroup  v_avl
 * 查找大于等于关键字KEY的节点，详细说明参考接口#VOS_AVL_Find_Or_Find_Next.
 */
#define VOS_AVL_FIND_OR_FIND_NEXT(TREE, KEY) VOS_AVL_Find_Or_Find_Next(&(TREE), (KEY), AVL_FALSE)

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif
