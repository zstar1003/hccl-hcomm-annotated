/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_LIST_H
#define RS_LIST_H
#include <ccan/list.h>

struct RsListHead {
    struct RsListHead *next, *prev;
};

static inline void RS_INIT_LIST_HEAD(struct RsListHead *list)
{
    list->next = list;
    list->prev = list;
}

#define RS_LIST_GET_HEAD_ENTRY(pos, n, head, member, type)                                                             \
    do {                                                                                                               \
        (pos) = list_entry((head)->next, type, member);                                                                \
        (n) = list_entry((pos)->member.next, type, member);                                                            \
    } while (0)

static inline bool RsListEmpty(struct RsListHead *head)
{
    return head->next == head;
}

static inline void __rs_list_add(struct RsListHead *xnew, struct RsListHead *prev, struct RsListHead *next)
{
    next->prev = xnew;
    xnew->next = next;
    xnew->prev = prev;
    prev->next = xnew;
}

static inline void RsListAddTail(struct RsListHead *xnew, struct RsListHead *head)
{
    __rs_list_add(xnew, head->prev, head);
}

static inline void __rs_list_del(struct RsListHead *prev, struct RsListHead *next)
{
    next->prev = prev;
    prev->next = next;
}

static inline void RsListDel(struct RsListHead *entry)
{
    __rs_list_del(entry->prev, entry->next);
}
#endif // RS_LIST_H
