/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_str.h"
#include "bkf_assert.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#define BKF_STR_SIGNL 0xAB000008
#define BKF_STR_SIGNH 0xAB000009
STATIC const uint32_t g_BkfStrGuard = 0xAB000010;
typedef struct tagBkfStrHead {
    uint32_t signL;
    BkfMemMng *memMng;
    int16_t strLen;
    uint8_t pad1[2];
    uint32_t signH;
    char strVal[0];
} BkfStrHead;

#define BKF_STR_SIGN_SET(head)                                                                                         \
    do {                                                                                                               \
        BKF_SIGN_SET((head)->signL, BKF_STR_SIGNL);                                                                    \
        BKF_SIGN_SET((head)->signH, BKF_STR_SIGNH);                                                                    \
    } while (0)
#define BKF_STR_SIGN_CLR(head)                                                                                         \
    do {                                                                                                               \
        BKF_SIGN_CLR((head)->signL);                                                                                   \
        BKF_SIGN_CLR((head)->signH);                                                                                   \
    } while (0)
#define BKF_STR_SIGN_IS_VALID(head)                                                                                    \
    (BKF_SIGN_IS_VALID((head)->signH, BKF_STR_SIGNH) && BKF_SIGN_IS_VALID((head)->signL, BKF_STR_SIGNL))

#define BKF_STR_LEN_IS_VALID(strLen) (((strLen) >= 0) && ((strLen) <= BKF_STR_LEN_MAX))

#define BKF_STR_GET_NEED_LEN(strLen) (sizeof(BkfStrHead) + (strLen) + 1 + sizeof(g_BkfStrGuard))
#define BKF_STR_2GUARD(head) ((uint8_t *)(head) + sizeof(BkfStrHead) + (head)->strLen + 1)

#define BKF_STR_GUARD_IS_VALID(head) (VOS_MemCmp(BKF_STR_2GUARD(head), &g_BkfStrGuard, sizeof(g_BkfStrGuard)) == 0)

char *BkfStrNew(BkfMemMng *memMng, const char *fmt, ...)
{
    BOOL paramIsInvalid = VOS_FALSE;
    int32_t strLen;
    va_list args;
    BkfStrHead *head = VOS_NULL;
    int32_t len;
    char buf[BKF_STR_LEN_MAX + 0x10];

    paramIsInvalid = (memMng == VOS_NULL) || (fmt == VOS_NULL);
    if (paramIsInvalid) {
        goto error;
    }

    buf[0] = '\0';
    va_start(args, fmt);
    strLen = vsnprintf_truncated_s(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (!BKF_STR_LEN_IS_VALID(strLen)) {
        goto error;
    }

    len = (int32_t)BKF_STR_GET_NEED_LEN((uint32_t)strLen);
    head = BKF_MALLOC(memMng, len);
    if (head == VOS_NULL) {
        goto error;
    }
    head->memMng = memMng;
    head->strLen = (int16_t)snprintf_truncated_s(head->strVal, strLen + 1, "%s", buf);
    if (head->strLen != strLen) {
        goto error;
    }
    BKF_STR_SIGN_SET(head);
    errno_t err = memcpy_s(BKF_STR_2GUARD(head), sizeof(g_BkfStrGuard), &g_BkfStrGuard, sizeof(g_BkfStrGuard));
    BKF_ASSERT(err == EOK);
    return head->strVal;

error:

    if (head != VOS_NULL) {
        BKF_FREE(memMng, head);
    }
    return VOS_NULL;
}

void BkfStrDel(char *str)
{
    BkfStrHead *head = VOS_NULL;

    if (str == VOS_NULL) {
        return;
    }

    head = BKF_MBR_PARENT(BkfStrHead, strVal[0], str);
    if (!BKF_STR_SIGN_IS_VALID(head)) {
        BKF_ASSERT(0);
        return;
    }
    /* 后面即使发生错误，也要free */
    if (BKF_STR_LEN_IS_VALID(head->strLen)) {
        BKF_ASSERT(head->strVal[head->strLen] == '\0');
        BKF_ASSERT(BKF_STR_GUARD_IS_VALID(head));
    } else {
        BKF_ASSERT(0);
    }

    (void)memset_s(BKF_STR_2GUARD(head), sizeof(g_BkfStrGuard), 0, sizeof(g_BkfStrGuard));
    BKF_STR_SIGN_CLR(head);
    BKF_FREE(head->memMng, head);
    return;
}

BOOL BkfStrIsValid(char *str)
{
    BkfStrHead *head = VOS_NULL;

    if (str == VOS_NULL) {
        return VOS_FALSE;
    }

    head = BKF_MBR_PARENT(BkfStrHead, strVal[0], str);
    if (!BKF_STR_SIGN_IS_VALID(head)) {
        return VOS_FALSE;
    }
    if (!BKF_STR_LEN_IS_VALID(head->strLen)) {
        return VOS_FALSE;
    }
    if ((head->strVal[head->strLen] != '\0') || !BKF_STR_GUARD_IS_VALID(head)) {
        return VOS_FALSE;
    }

    return VOS_TRUE;
}

int32_t BkfStrGetMemUsedLen(char *str)
{
    BkfStrHead *head = VOS_NULL;

    if (!BkfStrIsValid(str)) {
        return -1;
    }

    head = BKF_MBR_PARENT(BkfStrHead, strVal[0], str);
    return BKF_STR_GET_NEED_LEN(head->strLen);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
