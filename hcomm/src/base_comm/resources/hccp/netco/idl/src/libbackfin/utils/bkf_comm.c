/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_comm.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
BKF_STATIC_ASSERT(1);
BKF_STATIC_ASSERT(sizeof(0xffff) == 4);
BKF_STATIC_ASSERT(sizeof(0x7fffffffffffffff) == 8);

const int g_BkfHelloWorldNum10025 = 0;

uint32_t BkfModsInit(const BkfModVTbl *modsVTbl, void *doArg, int32_t cnt)
{
    int32_t i;
    uint32_t ret;

    if ((modsVTbl == VOS_NULL) || (cnt <= 0)) {
        return BKF_ERR;
    }

    for (i = 0; i < cnt; i++) {
        ret = modsVTbl[i].init(doArg);
        if (ret != BKF_OK) {
            return BKF_BUILD_RET(ret, i);
        }
    }

    return BKF_OK;
}

void BkfModsUninit(const BkfModVTbl *modsVTbl, void *doArg, int32_t cnt)
{
    int32_t i;

    if ((modsVTbl == VOS_NULL) || (cnt <= 0)) {
        return;
    }

    for (i = cnt - 1; i >= 0; i--) {
        if (modsVTbl[i].uninit) {
            modsVTbl[i].uninit(doArg);
        }
    }
}

STATIC char *BkfGetMemStrChkParam(uint8_t *mem, uint32_t memLen, uint32_t newLenPerByte, uint8_t *buf, int32_t bufLen)
{
    if (mem == VOS_NULL) {
        return "memNull";
    }
    if (memLen == 0) {
        return "memLenZero";
    }
    if (buf == VOS_NULL) {
        return "bufNull";
    }
    if (bufLen < 1) {
        return "bufLenNotEnough";
    }
    if (newLenPerByte == 0) {
        return "lenPerByteIsZero";
    }

    return VOS_NULL;
}
char *BkfGetMemStr(uint8_t *mem, uint32_t memLen, uint32_t newLenPerByte, uint8_t *buf, int32_t bufLen)
{
    char *chkRet = VOS_NULL;
    uint32_t i = 0;
    int32_t ret;
    int32_t usedLen = 0;

    if (newLenPerByte == 0) {
        return "lenPerByteIsZero";
    }
    chkRet = BkfGetMemStrChkParam(mem, memLen, newLenPerByte, buf, bufLen);
    if (chkRet != VOS_NULL) {
        return chkRet;
    }

    buf[0] = '\0';
    while (i < memLen) {
        if (i % newLenPerByte == 0) {
            ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), "%.2x", mem[i]);
        } else {
            ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), " %.2x", mem[i]);
        }
        if (ret <= 0) {
            break;
        }
        usedLen += ret;
        i++;

        if (i % newLenPerByte == 0) {
            ret = snprintf_truncated_s((char *)buf + usedLen, (uint32_t)(bufLen - usedLen), "\n");
            if (ret <= 0) {
                break;
            }
            usedLen += ret;
        }
    }

    return (char *)buf;
}

char *BkfTrimStrPath(char *str)
{
    char *temp = VOS_NULL;

    if (str == VOS_NULL) {
        return "__str_null__";
    }

    temp = VOS_StrRChr(str, '/');
    return (temp != VOS_NULL) ? (temp + 1) : str;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
