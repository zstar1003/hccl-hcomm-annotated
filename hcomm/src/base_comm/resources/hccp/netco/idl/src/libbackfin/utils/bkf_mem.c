/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pthread.h>
#include "bkf_assert.h"
#include "v_stringlib.h"
#include "v_avll.h"
#include "vos_patch_nop.h"
#include "securec.h"
#include "bkf_mem.h"

/* 不能引入vrp_dbg，避免yxl编译失败 */

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define BKF_MALLOC_(IMem, len, str, num) (IMem)->malloc((IMem)->cookie, (len), (str), (num))
#define BKF_FREE_(IMem, ptr, str, num) (IMem)->free((IMem)->cookie, (ptr), (str), (num))

#define BKF_MEM_MNG_SIGN (0xaf88)
struct tagBkfMemMng {
    uint16_t sign;
    uint8_t mutexInitOk : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1;
    BkfIMem IMem;
    char name[BKF_NAME_LEN_MAX + 1];

#ifdef BKF_MEM_STAT
    pthread_mutex_t mutex;
    int64_t appReqMemLenLeft;
    int64_t appReqMemLenPeak;
    AVLL_TREE statSet;
#endif
};

typedef struct tagBkfMemStat {
    AVLL_NODE avlNode;
    uint16_t selfLen;
    uint8_t pad1[2];
    int64_t appReqMemLenLeft;
    int32_t appReqMemBlkCntLeft;
    int64_t appReqMemLenPeak;
    BkfMemStatKey key;
    char str[0];
} BkfMemStat;
BKF_STATIC_ASSERT(BKF_MBR_IS_LAST(BkfMemStat, str));

#define BKF_MEM_HEAD_SIGN (0xaf99)
typedef struct tagBkfMemHead {
    BkfMemMng *memMng; /* 业务释放内存，不输入句柄场景。 */
#ifdef BKF_MEM_STAT
    BkfMemStat *stat;
    uint32_t appReqLen;
    uint32_t sign;
#endif
} BkfMemHead;

/* proc */
STATIC uint32_t BkfMemInitStat(BkfMemMng *memMng);
STATIC void BkfMemUninitStat(BkfMemMng *memMng);
STATIC void BkfMemUpdStatAfterMalloc(BkfMemMng *memMng, BkfMemHead *head, uint32_t len, const char *str,
    const uint16_t num);
STATIC BOOL BkfMemUpdStatBeforeFree(BkfMemMng *memMng, BkfMemHead *head, const char *str, const uint16_t num);

/* data op */
#ifdef BKF_MEM_STAT
STATIC int32_t BkfMemStatKeyCmp(const BkfMemStatKey *key1Input, const BkfMemStatKey *key2InDs);
STATIC BkfMemStat *BkfMemAddStat(BkfMemMng *memMng, const char *str, const uint16_t num);
STATIC void BkfMemDelAllStat(BkfMemMng *memMng);
STATIC BkfMemStat *BkfMemFindStat(BkfMemMng *memMng, const char *str, const uint16_t num);
STATIC BkfMemStat *BkfMemFindNextStat(BkfMemMng *memMng, const char *str, const uint16_t num);
STATIC BkfMemStat *BkfMemGetFirstStat(BkfMemMng *memMng, void **itorOutOrNull);
STATIC BkfMemStat *BkfMemGetNextStat(BkfMemMng *memMng, void **itorInOut);

#endif

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
/* func */
BkfMemMng *BkfMemInit(BkfIMem *IMem)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfMemMng *memMng = VOS_NULL;
    uint16_t len;
    int32_t err;
    uint32_t ret;
    void *stubRet = VOS_NULL;

    paramIsInvalid = ((IMem == VOS_NULL) || (IMem->name == VOS_NULL) || (IMem->malloc == VOS_NULL) ||
                      (IMem->free == VOS_NULL));
    if (paramIsInvalid) {
        goto error;
    }
    stubRet = BkfMemBaseStubInit(IMem->name);
    if (stubRet != VOS_NULL) {
        IMem->cookie = stubRet;
    }

    len = sizeof(BkfMemMng);
    memMng = BKF_MALLOC_(IMem, len, "memInit", BKF_MEM_NUM);
    if (memMng == VOS_NULL) {
        goto error;
    }
    (void)memset_s(memMng, len, 0, len);
    memMng->IMem = *IMem;
    err = snprintf_truncated_s(memMng->name, sizeof(memMng->name), "%s", IMem->name);
    if (err < 0) {
        goto error;
    }
    memMng->IMem.name = memMng->name;
    ret = BkfMemInitStat(memMng);
    if (ret != BKF_OK) {
        goto error;
    }

    BKF_SIGN_SET(memMng->sign, BKF_MEM_MNG_SIGN);
    return memMng;

error:

    BkfMemUninit(memMng);
    return VOS_NULL;
}

void BkfMemUninit(BkfMemMng *memMng)
{
    void *cookie = VOS_NULL;

    if ((memMng == VOS_NULL) || (memMng->IMem.free == VOS_NULL)) {
        return;
    }
    cookie = memMng->IMem.cookie;

    BkfMemUninitStat(memMng);
    BKF_SIGN_CLR(memMng->sign);
    BkfIMem IMem = memMng->IMem;
    BKF_FREE_(&IMem, memMng, "memUninit", BKF_MEM_NUM);
    BkfMemBaseStubUninit(cookie);
    return;
}

void *BkfMalloc(BkfMemMng *memMng, uint32_t len, const char *str, const uint16_t num)
{
    BkfMemHead *head = VOS_NULL;

    if ((memMng == VOS_NULL) || (memMng->IMem.malloc == VOS_NULL)) {
        return VOS_NULL;
    }

    head = BKF_MALLOC_(&memMng->IMem, len + sizeof(BkfMemHead), str, num);
    if (head != VOS_NULL) {
        head->memMng = memMng;
        BkfMemUpdStatAfterMalloc(memMng, head, len, str, num);
        return (head + 1);
    } else {
        return VOS_NULL;
    }
}

void BkfFree(BkfMemMng *memMng, void *ptr, const char *str, const uint16_t num)
{
    BkfMemHead *head = VOS_NULL;

    if ((memMng == VOS_NULL) || (memMng->IMem.free == VOS_NULL) || ((uintptr_t)ptr <= sizeof(BkfMemHead))) {
        return;
    }

    head = (BkfMemHead *)((uint8_t *)ptr - sizeof(BkfMemHead));
    if (BkfMemUpdStatBeforeFree(memMng, head, str, num)) {
        BKF_FREE_(&memMng->IMem, head, str, num);
    }
    return;
}

void BkfFreeNHnd(void *ptr, const char *str, const uint16_t num)
{
    BkfMemHead *head = VOS_NULL;
    BkfMemMng *memMng = VOS_NULL;

    if ((uintptr_t)ptr <= sizeof(BkfMemHead)) {
        BKF_ASSERT(0);
        return;
    }
    head = (BkfMemHead *)((uint8_t *)ptr - sizeof(BkfMemHead));
    memMng = head->memMng;
    if ((memMng == VOS_NULL) || !BKF_SIGN_IS_VALID(memMng->sign, BKF_MEM_MNG_SIGN) || (memMng->IMem.free == VOS_NULL)) {
        BKF_ASSERT(0);
        return;
    }

    BkfMemUpdStatBeforeFree(memMng, head, str, num);
    BKF_FREE_(&memMng->IMem, head, str, num);
    return;
}

void *BkfMemGetAdpeeMemHnd(BkfMemMng *memMng)
{
    F_BKF_IMEM_GET_ADPEE_MEM_HND getAdpeeMemHdn = VOS_NULL;

    if (memMng == VOS_NULL) {
        return VOS_NULL;
    }
    if ((getAdpeeMemHdn = memMng->IMem.getAdpeeMemHndOrNull) == VOS_NULL) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }
    return getAdpeeMemHdn(memMng->IMem.cookie);
}

#ifdef BKF_MEM_STAT
STATIC uint32_t BkfMemGetSummaryStrFmt(BkfMemMng *memMng, uint8_t *buf, int32_t bufLen, uint32_t statCnt,
    uint32_t statUseMemLen, int32_t appReqMemLeftCnt, int64_t appReqMemLeftLen)
{
    int32_t ret;
    int32_t usedLen = 0;

    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), "\n%-15s: %-12u B,     (%#x, %s)",
        "memMng", sizeof(BkfMemMng), BKF_MASK_ADDR(memMng), memMng->name);
    if (ret < 0) {
        return BKF_ERR;
    }
    usedLen += ret;
    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen),
        "\n%-15s: %-12u B,     ((%u+x) * %u)", "memStatUse", statUseMemLen, sizeof(BkfMemStat), statCnt);
    if (ret < 0) {
        return BKF_ERR;
    }
    usedLen += ret;
    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), "\n%-15s: %-12u B,     (%u * %d)",
        "memHeadUse", sizeof(BkfMemHead) * appReqMemLeftCnt, sizeof(BkfMemHead), appReqMemLeftCnt);
    if (ret < 0) {
        return BKF_ERR;
    }
    usedLen += ret;
    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen),
        "\n%-15s: %-12" VOS_PRId64 " B,     (%" VOS_PRId64 "/%d)", "memAppReq", memMng->appReqMemLenLeft,
        appReqMemLeftLen, appReqMemLeftCnt);
    if (ret < 0) {
        return BKF_ERR;
    }
    usedLen += ret;
    ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), "\n%-15s: %-12" VOS_PRId64 " B",
        "memAppReqPeak", memMng->appReqMemLenPeak);
    if (ret < 0) {
        return BKF_ERR;
    }
    usedLen += ret;

    return BKF_OK;
}
char *BkfMemGetSummaryStr(BkfMemMng *memMng, uint8_t *buf, int32_t bufLen)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfMemStat *stat = VOS_NULL;
    void *itor = VOS_NULL;
    uint32_t statCnt = 0;
    uint32_t statUseMemLen = 0;
    int32_t appReqMemLeftCnt = 0;
    int64_t appReqMemLeftLen = 0;

    paramIsInvalid = (memMng == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return "mem_arg_error";
    }

    (void)pthread_mutex_lock(&memMng->mutex);
    for (stat = BkfMemGetFirstStat(memMng, &itor); stat != VOS_NULL; stat = BkfMemGetNextStat(memMng, &itor)) {
        statCnt++;
        statUseMemLen += stat->selfLen;
        appReqMemLeftCnt += stat->appReqMemBlkCntLeft;
        appReqMemLeftLen += stat->appReqMemLenLeft;
    }
    if (BkfMemGetSummaryStrFmt(memMng, buf, bufLen, statCnt, statUseMemLen, appReqMemLeftCnt, appReqMemLeftLen) !=
        BKF_OK) {
        goto error;
    }

    (void)pthread_mutex_unlock(&memMng->mutex);
    return (char *)buf;

error:

    (void)pthread_mutex_unlock(&memMng->mutex);
    return "mem_snprintf_ng";
}

STATIC char *BkfMemGetStatStr(BkfMemMng *memMng, BkfMemStat *stat, uint8_t *buf, int32_t bufLen)
{
    (void)memMng;
    int32_t ret;

    ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen,
        "stat(%06#x), appReqMemLenLeft(%12" VOS_PRId64 ")"
        "/appReqMemBlkCntLeft(%10d)/appReqMemLenPeak(%12" VOS_PRId64 ")",
        BKF_MASK_ADDR(stat), stat->appReqMemLenLeft, stat->appReqMemBlkCntLeft, stat->appReqMemLenPeak);
    if (ret < 0) {
        return VOS_NULL;
    }
    return (char *)buf;
}
char *BkfMemGetFirstStatStr(BkfMemMng *memMng, BkfMemStatKey *keyOut, uint8_t *buf, int32_t bufLen)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfMemStat *stat = VOS_NULL;
    char *str = VOS_NULL;

    paramIsInvalid = (memMng == VOS_NULL) || (keyOut == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    (void)pthread_mutex_lock(&memMng->mutex);
    stat = BkfMemGetFirstStat(memMng, VOS_NULL);
    if (stat != VOS_NULL) {
        *keyOut = stat->key;
        str = BkfMemGetStatStr(memMng, stat, buf, bufLen);
    }
    (void)pthread_mutex_unlock(&memMng->mutex);
    return str;
}

char *BkfMemGetNextStatStr(BkfMemMng *memMng, BkfMemStatKey *keyInOut, uint8_t *buf, int32_t bufLen)
{
    BOOL paramIsInvalid = VOS_FALSE;
    BkfMemStat *stat = VOS_NULL;
    char *str = VOS_NULL;

    paramIsInvalid = (memMng == VOS_NULL) || (keyInOut == VOS_NULL) || (keyInOut->str == VOS_NULL) ||
                     (buf == VOS_NULL) || (bufLen <= 0);
    if (paramIsInvalid) {
        return VOS_NULL;
    }

    (void)pthread_mutex_lock(&memMng->mutex);
    stat = BkfMemFindNextStat(memMng, keyInOut->str, keyInOut->num);
    if (stat != VOS_NULL) {
        *keyInOut = stat->key;
        str = BkfMemGetStatStr(memMng, stat, buf, bufLen);
    }
    (void)pthread_mutex_unlock(&memMng->mutex);
    return str;
}

#else

char *BkfMemGetSummaryStr(BkfMemMng *memMng, uint8_t *buf, int32_t bufLen)
{
    int32_t ret;

    if ((memMng == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "mem_arg_error";
    }

    ret = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "memMng(%#x, %s, %u *), stat off & no more info",
        BKF_MASK_ADDR(memMng), memMng->name, sizeof(BkfMemMng));
    if (ret < 0) {
        return "mem_snprintf_ng";
    }
    return (char *)buf;
}

char *BkfMemGetFirstStatStr(BkfMemMng *memMng, BkfMemStatKey *keyOut, uint8_t *buf, int32_t bufLen)
{
    return VOS_NULL;
}

char *BkfMemGetNextStatStr(BkfMemMng *memMng, BkfMemStatKey *keyInOut, uint8_t *buf, int32_t bufLen)
{
    return VOS_NULL;
}

#endif

void *BkfMemBaseStubInit(char *name)
{
    (void)name;
    VOS_NOP();
    return VOS_NULL;
}

void BkfMemBaseStubUninit(void *cookie)
{
    (void)cookie;
    VOS_NOP();
}

#endif

#if BKF_BLOCK("私有函数定义")
#ifdef BKF_MEM_STAT
/* proc */
STATIC uint32_t BkfMemInitStat(BkfMemMng *memMng)
{
    int ret;

    VOS_AVLL_INIT_TREE(memMng->statSet, (AVLL_COMPARE)BkfMemStatKeyCmp, BKF_OFFSET(BkfMemStat, key),
        BKF_OFFSET(BkfMemStat, avlNode));
    ret = pthread_mutex_init(&memMng->mutex, VOS_NULL);
    if (ret != 0) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    memMng->mutexInitOk = VOS_TRUE;
    return BKF_OK;
}

STATIC void BkfMemUninitChkLeak(BkfMemMng *memMng)
{
    BkfMemStat *stat = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t leakCnt = 0;
    int64_t leakLen = 0;
    BOOL hasLeak = VOS_FALSE;
    int strLen;
    int usedLen = 0;
    int32_t outCntMax = 3;
    int32_t i = 0;
    char buf[BKF_1K];
    uint32_t bufLen = sizeof(buf);

    for (stat = BkfMemGetFirstStat(memMng, &itor); stat != VOS_NULL; stat = BkfMemGetNextStat(memMng, &itor)) {
        leakCnt += stat->appReqMemBlkCntLeft;
        leakLen += stat->appReqMemLenLeft;
    }
    hasLeak = (memMng->appReqMemLenLeft != 0) || (leakCnt != 0);
    if (!hasLeak) {
        return;
    }

    strLen = snprintf_truncated_s(buf, bufLen,
        ">>>>>leakCnt(%d)/leakLen(%" VOS_PRId64 "/%" VOS_PRId64 "), "
        "memory leak & ng, top %d list:\n",
        leakCnt, leakLen, memMng->appReqMemLenLeft, BKF_GET_MIN(leakCnt, outCntMax));
    if (strLen < 0) {
        BKF_ASSERT(0);
        return;
    }
    usedLen += strLen;

    for (stat = BkfMemGetFirstStat(memMng, &itor); (stat != VOS_NULL) && (i < outCntMax);
         stat = BkfMemGetNextStat(memMng, &itor)) {
        if (stat->appReqMemBlkCntLeft <= 0) {
            continue;
        }

        strLen = snprintf_truncated_s(buf + usedLen, bufLen - usedLen,
            "[%2d]: [%s, %u]/leakCnt(%d)/leakLen(%" VOS_PRId64 ")\n", i, stat->key.str, stat->key.num,
            stat->appReqMemBlkCntLeft, stat->appReqMemLenLeft);
        if (strLen < 0) {
            BKF_ASSERT(0);
            return;
        }
        usedLen += strLen;
        i++;
    }
    BKF_ASSERT_WITHLOG(0, BKF_MEM_STR, BKF_MEM_NUM, buf);
}
STATIC void BkfMemUninitStat(BkfMemMng *memMng)
{
    BkfMemUninitChkLeak(memMng);
    if (memMng->mutexInitOk) {
        (void)pthread_mutex_lock(&memMng->mutex);
        BkfMemDelAllStat(memMng);
        (void)pthread_mutex_unlock(&memMng->mutex);
        (void)pthread_mutex_destroy(&memMng->mutex);
    }
}

STATIC void BkfMemUpdStatAfterMalloc(BkfMemMng *memMng, BkfMemHead *head, uint32_t len, const char *str,
    const uint16_t num)
{
    BkfMemStat *stat = VOS_NULL;

    (void)pthread_mutex_lock(&memMng->mutex);
    head->stat = VOS_NULL;
    head->appReqLen = len;
    BKF_SIGN_SET(head->sign, BKF_MEM_HEAD_SIGN);

    memMng->appReqMemLenLeft += (int64_t)len;
    memMng->appReqMemLenPeak = BKF_GET_MAX(memMng->appReqMemLenPeak, memMng->appReqMemLenLeft);

    stat = BkfMemFindStat(memMng, str, num);
    if (stat == VOS_NULL) {
        stat = BkfMemAddStat(memMng, str, num);
        if (str == VOS_NULL) {
            goto error;
        }
    }
    head->stat = stat;
    stat->appReqMemLenLeft += (int64_t)len;
    stat->appReqMemBlkCntLeft++;
    stat->appReqMemLenPeak = BKF_GET_MAX(stat->appReqMemLenPeak, stat->appReqMemLenLeft);

error:

    (void)pthread_mutex_unlock(&memMng->mutex);
}

STATIC BOOL BkfMemUpdStatBeforeFree(BkfMemMng *memMng, BkfMemHead *head, const char *str, const uint16_t num)
{
    (void)str;
    (void)num;
    BkfMemStat *stat = VOS_NULL;
    uint32_t len;

    if (!BKF_SIGN_IS_VALID(head->sign, BKF_MEM_HEAD_SIGN)) {
        BKF_ASSERT(0);
        return VOS_FALSE;
    }

    stat = head->stat;
    len = head->appReqLen;
    (void)pthread_mutex_lock(&memMng->mutex);
    memMng->appReqMemLenLeft -= (int64_t)len;
    if (stat != VOS_NULL) {
        stat->appReqMemLenLeft -= (int64_t)len;
        stat->appReqMemBlkCntLeft--;
    }
    (void)pthread_mutex_unlock(&memMng->mutex);
    return VOS_TRUE;
}

/* data op */
STATIC int32_t BkfMemStatKeyCmp(const BkfMemStatKey *key1Input, const BkfMemStatKey *key2InDs)
{
    int32_t ret;

    ret = VOS_StrCmp(key1Input->str, key2InDs->str);
    if (ret != 0) {
        return ret;
    } else if (key1Input->num > key2InDs->num) {
        return 1;
    } else if (key1Input->num < key2InDs->num) {
        return -1;
    }

    return 0;
}

STATIC BkfMemStat *BkfMemAddStat(BkfMemMng *memMng, const char *str, const uint16_t num)
{
    BkfMemStat *stat = VOS_NULL;
    uint32_t strLen;
    uint32_t len;
    int32_t ret;
    BOOL isInsOk = VOS_FALSE;

    strLen = VOS_StrLen(str);
    len = sizeof(BkfMemStat) + strLen + 1;
    stat = BKF_MALLOC_(&memMng->IMem, len, "memAddStat_", BKF_MEM_NUM);
    if (stat == VOS_NULL) {
        goto error;
    }
    (void)memset_s(stat, len, 0, len);
    stat->selfLen = len;
    stat->key.str = stat->str;
    stat->key.num = num;
    ret = snprintf_truncated_s(stat->str, strLen + 1, "%s", str);
    if (ret < 0) {
        goto error;
    }
    VOS_AVLL_INIT_NODE(stat->avlNode);
    isInsOk = VOS_AVLL_INSERT(memMng->statSet, stat->avlNode);
    if (!isInsOk) {
        goto error;
    }

    return stat;

error:

    if (stat != VOS_NULL) {
        /* isInsOk 一定为false */
        BKF_FREE_(&memMng->IMem, stat, "memAddStat_", BKF_MEM_NUM);
    }
    return VOS_NULL;
}

STATIC void BkfMemDelAllStat(BkfMemMng *memMng)
{
    BkfMemStat *stat = VOS_NULL;
    void *itor = VOS_NULL;

    for (stat = BkfMemGetFirstStat(memMng, &itor); stat != VOS_NULL; stat = BkfMemGetNextStat(memMng, &itor)) {
        VOS_AVLL_DELETE(memMng->statSet, stat->avlNode);
        BKF_FREE_(&memMng->IMem, stat, "memDelAllStat_", BKF_MEM_NUM);
    }
    return;
}

STATIC BkfMemStat *BkfMemFindStat(BkfMemMng *memMng, const char *str, const uint16_t num)
{
    BkfMemStat *stat = VOS_NULL;
    BkfMemStatKey key = {(char *)str, num, {}};

    stat = VOS_AVLL_FIND(memMng->statSet, &key);
    return stat;
}

STATIC BkfMemStat *BkfMemFindNextStat(BkfMemMng *memMng, const char *str, const uint16_t num)
{
    BkfMemStat *stat = VOS_NULL;
    BkfMemStatKey key = {(char *)str, num, {}};

    stat = VOS_AVLL_FIND_NEXT(memMng->statSet, &key);
    return stat;
}

STATIC BkfMemStat *BkfMemGetFirstStat(BkfMemMng *memMng, void **itorOutOrNull)
{
    BkfMemStat *stat = VOS_NULL;

    stat = VOS_AVLL_FIRST(memMng->statSet);
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = (stat != VOS_NULL) ? VOS_AVLL_NEXT(memMng->statSet, stat->avlNode) : VOS_NULL;
    }
    return stat;
}

STATIC BkfMemStat *BkfMemGetNextStat(BkfMemMng *memMng, void **itorInOut)
{
    BkfMemStat *stat = VOS_NULL;

    stat = (*itorInOut);
    *itorInOut = (stat != VOS_NULL) ? VOS_AVLL_NEXT(memMng->statSet, stat->avlNode) : VOS_NULL;
    return stat;
}

#else
/* proc */
STATIC uint32_t BkfMemInitStat(BkfMemMng *memMng)
{
    return BKF_OK;
}

STATIC void BkfMemUninitStat(BkfMemMng *memMng)
{
    return;
}

STATIC void BkfMemUpdStatAfterMalloc(BkfMemMng *memMng, BkfMemHead *head, uint32_t len, const char *str,
    const uint16_t num)
{
    return;
}

STATIC BOOL BkfMemUpdStatBeforeFree(BkfMemMng *memMng, BkfMemHead *head, const char *str, const uint16_t num)
{
    return VOS_TRUE;
}

#endif
#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
