/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DL_H
#define BKF_DL_H

#include "vos_list.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)

typedef VOS_LIST_HEAD_S BkfDl;
typedef VOS_LIST_HEAD_S BkfDlNode;

/**
 * @brief 初始化链表
 *
 * @param[in] dl 链表头
 * @return none
 */
#define BKF_DL_INIT(dl) VOS_ListInit(dl)

/**
 * @brief 初始化链表节点
 *
 * @param[in] node 链表节点
 * @return none
 */
#define BKF_DL_NODE_INIT(node)                                                                                         \
    do {                                                                                                               \
        (node)->next = VOS_NULL;                                                                                       \
        (node)->prev = VOS_NULL;                                                                                       \
    } while (0)

/**
 * @brief 在指定链表节点前面插入节点
 *
 * @param[in] node 指定的链表节点
 * @param[in] node2Add 待插入的节点
 * @return none
 */
#define BKF_DL_ADD_BEFORE(node, node2Add) VOS_ListAddBefore((node2Add), (node))

/**
 * @brief 在指定链表节点后面插入节点
 *
 * @param[in] node 指定的链表节点
 * @param[in] node2Add 待插入的节点
 * @return none
 */
#define BKF_DL_ADD_AFTER(node, node2Add) VOS_ListAdd((node2Add), (node))

/**
 * @brief 将节点插入链头
 *
 * @param[in] dl 链表
 * @param[in] node2Add 待插入的节点
 * @return none
 */
#define BKF_DL_ADD_FIRST(dl, node2Add) BKF_DL_ADD_AFTER((dl), (node2Add))

/**
 * @brief 将节点插入链尾
 *
 * @param[in] dl 链表
 * @param[in] node2Add 待插入的节点
 * @return none
 */
#define BKF_DL_ADD_LAST(dl, node2Add) BKF_DL_ADD_BEFORE((dl), (node2Add))

/**
 * @brief 移除链表节点
 *
 * @param[in] node 待移除的链表节点
 * @return none
 */
#define BKF_DL_REMOVE(node)                                                                                            \
    do {                                                                                                               \
        VOS_ListRemove(node);                                                                                          \
        BKF_DL_NODE_INIT(node);                                                                                        \
    } while (0)

/**
 * @brief 获取指定链表节点的前继节点
 *
 * @param[in] dl 链表
 * @param[in] node 指点的节点
 * @return 前继节点
 *   @retval 非空 正常的前继节点
 *   @retval 空 无前继节点
 */
#define BKF_DL_GET_PREV(dl, node) (((node)->prev != (dl)) ? (node)->prev : VOS_NULL)

/**
 * @brief 获取指定链表节点的后继节点
 *
 * @param[in] dl 链表
 * @param[in] node 指点的节点
 * @return 后继节点
 *   @retval 非空 正常的后继节点
 *   @retval 空 无前继节点
 */
#define BKF_DL_GET_NEXT(dl, node) (((node)->next != (dl)) ? (node)->next : VOS_NULL)

/**
 * @brief 获取指定链表头节点
 *
 * @param[in] dl 链表
 * @return 头节点
 *   @retval 非空 正常的头节点
 *   @retval 空 空链表，无头节点
 */
#define BKF_DL_GET_FIRST(dl) BKF_DL_GET_NEXT((dl), (dl))

/**
 * @brief 获取指定链表尾节点
 *
 * @param[in] dl 链表
 * @return 尾节点
 *   @retval 非空 正常的尾节点
 *   @retval 空 空链表，无尾节点
 */
#define BKF_DL_GET_LAST(dl) BKF_DL_GET_PREV((dl), (dl))

/**
 * @brief 判断是否为空链表
 *
 * @param[in] dl 链表
 * @return 是否为空链表
 *   @retval true 空链表
 *   @retval false 非空链表
 */
#define BKF_DL_IS_EMPTY(dl) VOS_ListIsEmpty(dl)

/**
 * @brief 判断节点是否在链表中
 *
 * @param[in] node 待判断的链表节点。
 * @return 是否在链表中
 *   @retval true 在链表
 *   @retval false 不在链表
 */
#define BKF_DL_NODE_IS_IN(node)                                                                                        \
    (((node)->prev != VOS_NULL) && (((node)->prev->next) == (node)) && ((node)->next != VOS_NULL) &&                   \
        (((node)->next->prev) == (node)))

/**
 * @brief 将链表中所有节点移到另一个链表的尾部，并且节点的相对位置不变
 *
 * @param[in] dlFrom 待移动节点的链表
 * @param[in] dlTo 移动到的链表
 * @return none
 */
#define BKF_DL_MOVE_ALL_2TAIL(dlFrom, dlTo)                                                                            \
    do {                                                                                                               \
        if (((dlFrom) != (dlTo)) && !BKF_DL_IS_EMPTY(dlFrom)) {                                                        \
            (dlFrom)->prev->next = (dlTo);                                                                             \
            (dlFrom)->next->prev = (dlTo)->prev;                                                                       \
            (dlTo)->prev->next = (dlFrom)->next;                                                                       \
            (dlTo)->prev = (dlFrom)->prev;                                                                             \
            BKF_DL_INIT(dlFrom);                                                                                       \
        }                                                                                                              \
    } while (0)

/**
 * @brief 将链表中所有节点移到另一个链表的头部，并且节点的相对位置不变
 *
 * @param[in] dlFrom 待移动节点的链表
 * @param[in] dlTo 移动到的链表
 * @return none
 */
#define BKF_DL_MOVE_ALL_2HEAD(dlFrom, dlTo)                                                                            \
    do {                                                                                                               \
        if (((dlFrom) != (dlTo)) && !BKF_DL_IS_EMPTY(dlFrom)) {                                                        \
            (dlFrom)->next->prev = (dlTo);                                                                             \
            (dlFrom)->prev->next = (dlTo)->next;                                                                       \
            (dlTo)->next->prev = (dlFrom)->prev;                                                                       \
            (dlTo)->next = (dlFrom)->next;                                                                             \
            BKF_DL_INIT(dlFrom);                                                                                       \
        }                                                                                                              \
    } while (0)

/**
 * @brief 获取内嵌链表节点结构的起始地址
 *
 * @param[in] node 内嵌的链表节点地址
 * @param[in] strucType 包含链表节点的结构定义
 * @param[in] nodeMember 结构中链表节点成员变量名
 * @return 结构的起始地址
 *   @retval 地址 结构的起始地址
 */
#define BKF_DL_GET_ENTRY(node, strucType, nodeMember) VOS_LIST_ENTRY((node), strucType, nodeMember)

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
