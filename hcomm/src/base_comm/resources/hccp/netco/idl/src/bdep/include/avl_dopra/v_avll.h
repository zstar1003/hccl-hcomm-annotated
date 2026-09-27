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
 * @defgroup v_avll AVLL
 * @ingroup util
 */
#ifndef V_AVLL_H
#define V_AVLL_H

#include "v_avl3.h"

#ifdef __cplusplus
extern "C" {
#endif /* --cplusplus */

/* AVLL compare function type. */
typedef AVL3_COMPARE AVLL_COMPARE;

/*
 * Structure  : AVLL_NODE
 * Description: Node in an AVLL tree (deprecated).
 */
/**
 * @ingroup v_avll
 * 本数据结构包含了树中元素的地址信息。#AVLL_NODE与#AVL3_NODE的内容完全相同。
 */
typedef AVL3_NODE AVLL_NODE;

/*
 * Structure  : AVLL_TREE
 * Description: AVLL tree root (deprecated).
 */
/**
 * @ingroup v_avll
 * 本数据结构包含了树中节点的信息以及使用的比较函数。同时还包含了树中节点的地址信
 * 息以及关键元素信息。本数据结构由AVL3_TREE和AVL3_TREE_INFO组成。
 */
typedef struct avll_tree {
    AVL3_TREE stTree;          /**< AVL3 tree structure. */
    AVL3_TREE_INFO stTreeInfo; /**< AVL3 tree information structure. */
} AVLL_TREE;

/* AVL3 EXPOSED MACROS (deprecated). */
/**
 * @ingroup v_avll
 * @brief 本接口初始化一个AVL树的基本信息。
 *
 * @par 描述
 * 这个API初始化AVLL树的数据结构。
 * @attention
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param pfnCompare [IN]  该参数表示表示比较函数。取值范围为#int类型的有效函数指针。
 * @param usKey_offset [IN]  该参数表示表示KEY偏移量。取值范围为该数据类型支持的有效范围。
 * @param usNode_offset [IN]  该参数表示表示节点偏移量。取值范围为该数据类型支持的有效范围。
 * @retval 无。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_IN_TREE
 * @li VOS_AVLL_INIT_NODE
 */
#define VOS_AVLL_INIT_TREE(TREE, COMPARE, KEY_OFF, NODE_OFF)                                                           \
    do {                                                                                                               \
        (TREE).stTreeInfo.pfCompare = (COMPARE);                                                                       \
        (TREE).stTreeInfo.usKeyOffset = (KEY_OFF);                                                                     \
        (TREE).stTreeInfo.usNodeOffset = (NODE_OFF);                                                                   \
        VOS_AVL3_INIT_TREE((TREE).stTree, (TREE).stTreeInfo);                                                          \
    } while (0)

/**
 * @ingroup v_avll
 * @brief 本接口对待插入树中的节点 #AVLL_NODE 进行初始化。
 *
 * @par 描述
 * 本接口对待插入树中的节点 #AVLL_NODE 进行初始化。
 * @attention
 * @param stNode [IN]  该参数表示表示待初始化的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 无。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_IN_TREE
 * @li VOS_AVLL_INIT_TREE
 */
#define VOS_AVLL_INIT_NODE(NODE) VOS_AVL3_INIT_NODE((NODE))

/**
 * @ingroup v_avll
 * @brief 本接口插入一个节点到AVLL树中。
 *
 * @par 描述
 * 本接口插入一个节点到AVLL树中。树和节点必须已调用接口 #VOS_AVLL_INIT_TREE 和 #VOS_AVLL_INIT_NODE 完成初始化。
 * @attention
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param stNode [IN]  该参数表示表示待插入树中的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 如果节点成功插入到树中，则返回#VOS_TRUE表示成功。
 * @retval 如果待插入树中的节点的KEY值与树中某一元素相同，则返回#VOS_FALSE表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_INSERT_OR_FIND
 * @li VOS_AVLL_INIT_NODE
 * @li VOS_AVLL_INIT_TREE
 */
#define VOS_AVLL_INSERT(TREE, NODE) VOS_AVL3_INSERT((TREE).stTree, (NODE), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口尝试插入一个节点到AVLL树中。
 *
 * @par 描述
 * 本接口尝试插入一个节点到AVLL树中。如果树中已存在具有相同KEY值的节点，
 * 则插入失败且本接口返回该节点的指针。
 * @attention
 * 树和节点必须已调用接口 #VOS_AVLL_INIT_TREE 和 #VOS_AVLL_INIT_NODE 完成初始化。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param stNode [IN]  该参数表示表示待插入树中的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 如果插入成功，则返回#AVL_NULL_PTR表示成功。
 * @retval 树中具有相同KEY值的节点的指针表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see VOS_AVLL_INSERT
 */
#define VOS_AVLL_INSERT_OR_FIND(TREE, NODE) VOS_AVL3_INSERT_OR_FIND((TREE).stTree, (NODE), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 删除指定的节点。
 *
 * @par 描述
 * 本接口将先前插入树中的指定节点移除。
 * @attention
 * @li AVL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 用户需要保证节点在树中存在。如果节点原本就不属于AVL树，则调用本接口的结果不可预料。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree 类型的有效数据结构。
 * @param stNode [IN]  该参数表示表示要操作的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 无。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_INSERT
 * @li VOS_AVLL_INSERT_OR_FIND
 */
#define VOS_AVLL_DELETE(TREE, NODE) VOS_AVL3_DELETE((TREE).stTree, (NODE))

/**
 * @ingroup v_avll
 * @brief 查找指定的节点。
 *
 * @par 描述
 * 根据输入的KEY返回树中元素的指针。
 * @attention
 * 本接口并未对传入参数做空指针校验。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param pKey [IN]  该参数表示指向待查找的KEY值的指针。取值范围为有效指针。
 *
 * @retval 指定KEY值的节点指针表示成功。
 * @retval 如果树中没有指针KEY值的节点，则返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_FIND_NEXT
 * @li VOS_AVLL_FIND_OR_FIND_NEXT
 */
#define VOS_AVLL_FIND(TREE, KEY) VOS_AVL3_FIND((TREE).stTree, (KEY), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口树据输入的节点返回树中下一个节点的指针。
 *
 * @par 描述
 * 本接口树据输入的节点返回树中下一个节点的指针。即如果传入节点的右子树存在，
 * 则返回右子树中最左边的节点。如果传入节点的右子树不存在，则返回AVLL树的根节点。
 * @attention
 * @li AVLL模块对用户传入的参数并未做空指针校验，需要由用户保证参数的有效性。
 * @li 输入的节点必须属于待查找的AVLL树。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param stNode [IN]  该参数表示表示要操作的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 树中下一个元素的指针表示成功。
 * @retval 如果输入的节点是树中的最后一个节点，则返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_FIRST
 * @li VOS_AVLL_LAST
 * @li VOS_AVLL_PREV
 */
#define VOS_AVLL_NEXT(TREE, NODE) VOS_AVL3_NEXT((NODE), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口返回树中指定节点的前一个节点指针。
 *
 * @par 描述
 * 本接口返回树中指定节点的前一个节点指针。即如果传入节点的左子树存在，
 * 则返回左子树中最右边的节点。如果传入节点的左子树不存在，
 * 则返回AVLL树的根节点。
 * @attention
 * @li 本接口并未对传入参数做空指针校验。
 * @li 输入的节点必须属于待查找的AVLL树。
 *
 * @param  [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param  [IN]  该参数表示表示要操作的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 树中前一个元素的指针表示成功。
 * @retval 如果输入的节点是树中的第一个节点，则返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_LAST
 * @li VOS_AVLL_NEXT
 */
#define VOS_AVLL_PREV(TREE, NODE) VOS_AVL3_PREV((NODE), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口返回树中第一个节点的指针。
 *
 * @par 描述
 * 本接口返回树中第一个节点的指针。
 * @attention
 * 无。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 *
 * @retval 树中第一个元素的指针表示成功。
 * @retval 如果树为空，返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_LAST
 * @li VOS_AVLL_NEXT
 * @li VOS_AVLL_PREV
 */
#define VOS_AVLL_FIRST(TREE) VOS_AVL3_FIRST((TREE).stTree, (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口返回树中最后一个节点的指针。
 *
 * @par 描述
 * 本接口返回树中最后一个节点的指针。
 * @attention
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 *
 * @retval 树中最后一个元素的指针表示成功。
 * @retval 如果树为空，返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_FIRST
 * @li VOS_AVLL_NEXT
 * @li VOS_AVLL_PREV
 */
#define VOS_AVLL_LAST(TREE) VOS_AVL3_LAST((TREE).stTree, (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 本接口检查当前元素是否属于某一个AVL树。
 *
 * @par 描述
 * 本接口检查当前元素是否属于某一个AVL树。
 * @attention
 * 节点必须已调用接口 #VOS_AVLL_INIT_NODE完成初始化。
 *
 * @param stNode [IN]  该参数表示表示要检查的节点。取值范围为#AVL3_NODE类型的有效数据结构。
 *
 * @retval 如果节点属于某个已初始化的AVLL树，则返回#VOS_TRUE表示成功。
 * @retval 如果节点不属于任何AVLL树，则返回#VOS_FALSE表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_INIT_NODE
 * @li VOS_AVLL_INIT_TREE
 */
#define VOS_AVLL_IN_TREE(NODE) VOS_AVL3_IN_TREE((NODE))

/**
 * @ingroup v_avll
 * @brief 本接口查找树中的下一个节点。
 *
 * @par 描述
 * 本接口查找树中的下一个节点。本接口根据输入的KEY值查找树中的下一个节点，
 * 树中节点已经根据比较函数定义的规则排序。
 * @attention
 * 在一个按字母排序的树中，假设树中包含元素的KEY值分别为A，B，C，D，E，F。
 * 使用本接口时传入D或E可以返回KEY值为F的节点，
 * 使用本接口传入KEY值G则会返回#AVL_NULL_PTR（因为KEY值为G的节点在树中没有下一个节点）。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param pKey [IN]  该参数表示指向待查找的KEY值的指针。取值范围为有效指针。
 *
 * @retval 树中某节点的指针，该节点的KEY值根据比较函数定义的规则略大于输入的KEY值表示成功。
 * @retval 如果树中指定KEY值的节点无下一个节点，则返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_FIND
 * @li VOS_AVLL_FIND_OR_FIND_NEXT
 * @li VOS_AVLL_LAST
 * @li VOS_AVLL_PREV
 */
#define VOS_AVLL_FIND_NEXT(TREE, KEY) VOS_AVL3_FIND_NEXT((TREE).stTree, (KEY), (TREE).stTreeInfo)

/**
 * @ingroup v_avll
 * @brief 根据输入的KEY值在树中查找，如果匹配成功则返回树中节点的指针。
 *
 * @par 描述
 * 本接口是 #VOS_AVLL_FIND 和
 * #VOS_AVLL_FIND_NEXT功能的组合。根据输入的KEY值在树中查找，如果匹配成功则返回树中节点的指针。如果树中没有指定KEY值的节点，
 * 则返回比KEY值大的下一个节点。该大小顺序是由比较函数定义好的。
 * @attention
 * 在一个按字母排序的树中，假设树中包含元素的KEY值分别为A，B，C，D，E，F。
 * 使用本接口时传入D或E可以返回KEY值为F的节点，使用本接口传入KEY值G则会返回#AVL_NULL_PTR（因为KEY值为G的节点在树中没有下一个节点）。
 *
 * @param stTree [IN]  该参数表示表示某个AVLL树。取值范围为#avll_tree类型的有效数据结构。
 * @param pKey [IN]  该参数表示指向待查找的KEY值的指针。取值范围为有效指针。
 *
 * @retval 树中具有相同KEY值的节点指针（如果匹配成功）。如果指定KEY值的元素在树中不存在，
 * 则返回树中KEY值略大的下一个节点指针表示成功。
 * @retval 如果树中指定KEY值的节点不存在且无下一个节点，则返回#AVL_NULL_PTR表示失败。
 * @par 依赖
 * @li v_avll：该接口所属的开发包。
 * @li v_avll.h：该接口声明所在的头文件。
 * @since V200R002C00
 * @see
 * @li VOS_AVLL_FIND
 * @li VOS_AVLL_FIND_NEXT
 */
#define VOS_AVLL_FIND_OR_FIND_NEXT(TREE, KEY) VOS_AVL3_FIND_OR_FIND_NEXT((TREE).stTree, (KEY), (TREE).stTreeInfo)

#ifdef __cplusplus
}
#endif /* --cplusplus */

#endif /* V_AVLL_H */
