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
 * @defgroup vos_list 双向链表
 * @ingroup util
 */
#ifndef __VOS_LIST_H__
#define __VOS_LIST_H__

#include "vos_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The struct of a list item */
/**
 * @ingroup vos_list
 * 该结构被用来保存双向链表中节点的前向指针和后向指针。
 * 这个链表不包含实质的数据区，一般用于组织(串接)数据节点，参见 #VOS_LIST_ENTRY 的注释
 */
typedef struct tagVosListHead {
    struct tagVosListHead *next, *prev;
} VOS_LIST_HEAD_S;

/**
 * @ingroup vos_list
 * 初始化链表(用户不应该直接用这个宏，而应该使用封装后的宏#VOS_LIST_DECLARE_AND_INIT)。
 * 请参考#VOS_LIST_DECLARE_AND_INIT说明
 *
 * @param list [IN] 需要初始化的链表 (注意不要把链表地址传进来) (Note, not the address)
 */
#define VOS_LIST_INIT_VAL(list) {&(list), &(list)}

/**
 * @ingroup vos_list
 * 定义且初始化一个链表，把链表节点的前向指针与后向指针指向自己 (Declare a list and init it )
 * @param list [IN] 需要定义和初始化的链表变量名(注意不要把链表地址传进来)。
 */
#define VOS_LIST_DECLARE_AND_INIT(list) VOS_LIST_HEAD_S(list) = VOS_LIST_INIT_VAL(list)

/**
 * @ingroup vos_list
 * 初始化链表(链表重用时的初始化)
 *
 * @param head [IN] 链表头结点的地址(The address of the head of a list )
 */
#define VOS_ListInit(head)                                                                                             \
    do {                                                                                                               \
        (head)->next = (head)->prev = (head);                                                                          \
    } while (0)

/**
 * @ingroup vos_list
 * 在链表指定位置后边插入节点 (Add an item to a list after a special location)
 *
 * @param item  [IN] 节点地址(The address of the item)
 * @param where [IN] 该节点插入位置的前一个节点地址。(The address where the item will be inserted after)
 */
#define VOS_ListAdd(item, where)                                                                                       \
    do {                                                                                                               \
        (item)->next = (where)->next;                                                                                  \
        (item)->prev = (where);                                                                                        \
        (where)->next = (item);                                                                                        \
        (item)->next->prev = (item);                                                                                   \
    } while (0)

/**
 * @ingroup vos_list
 * 在链表指定位置前边插入节点 (Add an item to a list before a special location)
 *
 * @param item  [IN] 节点地址(The address of the item)
 * @param where [IN] 该节点插入位置的后一个节点地址。(The address where the item will be inserted before)
 */
#define VOS_ListAddBefore(item, where) VOS_ListAdd((item), (where)->prev)

/**
 * @ingroup vos_list
 * 从链表中删除一个节点(Remove an item from a list)
 *
 * @param item [IN] 待删除的节点(The address of the item to be removed)
 */
#define VOS_ListRemove(item)                                                                                           \
    do {                                                                                                               \
        (item)->prev->next = (item)->next;                                                                             \
        (item)->next->prev = (item)->prev;                                                                             \
    } while (0)

/**
 * @ingroup vos_list
 * 检查链表是否为空(Judge whether a list is empty)
 *
 * @param head [IN] 需要检查的链表(The address of the list to be judged)
 * @retval #VOS_TRUE 1，链表为空。
 * @retval #VOS_FALSE 0，链表不为空。
 */
#define VOS_ListIsEmpty(head) ((head)->next == (head))

/**
 * @ingroup vos_list
 * 遍历一个链表(Travel through a list)
 *
 * @param head [IN] 需要遍历的链表(The head of a list )
 * @param item [IN] 遍历链表所用的缓存节点(A temporary list item for travelling the list)
 */
#define VOS_LIST_FOR_EACH_ITEM(item, head)                                                                             \
    for ((item) = (head)->next; (item) != (head) && (item) != NULL; (item) = (item)->next)

/**
 * @ingroup vos_list
 * 安全遍历一个链表(Travel through a list safety)
 *
 * @param head [IN] 需要遍历的链表(The head of a list)
 * @param temp [IN] 指向当前节点以便安全删除当前节点(pointer used to save current item so you can free item safety)
 * @param item [IN] 遍历链表所用的缓存节点(A temporary list item for travelling the list)
 */
#define VOS_LIST_FOR_EACH_ITEM_SAFE(item, temp, head)                                                                  \
    for ((item) = (head)->next, (temp) = (item)->next; (item) != (head) && (item) != NULL;                             \
         (item) = (temp), (temp) = (item)->next)

/**
 * @ingroup vos_list
 * 倒序遍历一个链表(Traverse a list backwards)
 *
 * @param head [IN] 待遍历的链表头节点地址(The head of a list)
 * @param item [IN] 遍历链表所用的缓存节点(The loop index variable)
 */
#define VOS_LIST_FOR_EACH_ITEM_REV(item, head)                                                                         \
    for ((item) = (head)->prev; (item) != (head) && (item) != NULL; (item) = (item)->prev)

/**
 * @ingroup vos_list
 * 从指定节点开始倒序遍历一个链表(Traverse a list backwards from a certain node)
 *
 * @param start [IN] 开始遍历的节点(The node to start traversing from)
 * @param head  [IN] 待遍历的链表头节点地址(The head of a list)
 * @param item  [IN] 遍历链表所用的缓存节点(The loop index variable)
 */
#define VOS_LIST_FOR_EACH_ITEM_REV_FROM(item, start, head)                                                             \
    for ((item) = (start); (item) != (head) && (item) != NULL; (item) = (item)->prev)

/**
 * @ingroup vos_list
 * 通过链表某个节点(小结点)找到该节点所在结构(大节点)的起始地址(Find the entry of a struct through its member variable
 * whose type is list item)
 *
 * @param item   [IN] 特定节点变量(The address of a list item)
 * @param type   [IN] 包含链表节点的大节点类型(The type of a struct which includes the list item)
 * @param member [IN] 结构体内的list节点成员名称(The member variable of the struct whose type is list item)
 *
 * 说明:
 * 每个结构变量形成一个一个的大节点(包含数据和list小结点), 大节点是通过list 这个链表(小节点)串起来的。
 * ---------      ---------      ---------    --               ----
 * |  pre    |<---|  pre    |<---|  pre    |     |==>小结点         |
 * |  next   |--->|  next   |--->|  next   |     |                  |
 * ---------      ---------      ---------    --                   | ===> 大节点
 * |  data1  |    |  data1  |    |  data1  |                        |
 * |  data2  |    |  data2  |    |  data2  |                        |
 * ---------      ---------      ---------                     ----
 * 不直接用list作为大节点的原因是 list(VOS_LIST_HEAD_S 类型)只有头尾指针，不包含数据区。
 * 这样 list链表可以适用挂接任意个数据的场合，具有通用性。
 */
#define VOS_LIST_ENTRY(item, type, member) ((type *)((char *)(item) - (uintptr_t)(&((type *)0)->member)))

#ifdef __cplusplus
}
#endif

#endif /* __VOS_LIST_H__ */
