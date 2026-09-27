/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bifrost_cncoi_table.h"
#include "v_stringlib.h"

#if __cplusplus
extern "C" {
#endif

int32_t BifrostCncoiComminfoKeyCmp(const BifrostCncoiComminfoKeyT *key1Input, const BifrostCncoiComminfoKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiComminfoKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiComminfoKeyGetStr(const BifrostCncoiComminfoKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiComminfoGetStr_ng_param__";
    }

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    tempLen = BKF_GET_MIN(sizeof(key->commDesc), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->commInitTime);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->packetId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    tempLen = BKF_GET_MIN(sizeof(key->resv), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->resv, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

int32_t BifrostCncoiOperatorKeyCmp(const BifrostCncoiOperatorKeyT *key1Input, const BifrostCncoiOperatorKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiOperatorKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiOperatorKeyGetStr(const BifrostCncoiOperatorKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strTotalLen = 0;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiAlgoGetStr_ng_param__";
    }

    int32_t strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    int32_t tempLen = BKF_GET_MIN(sizeof(key->commDesc), 20);
    char *tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->commInitTime);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->operator);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->algorithm);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->rootRank);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    tempLen = BKF_GET_MIN(sizeof(key->resv), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->resv, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

int32_t BifrostCncoiAdjacencyKeyCmp(const BifrostCncoiAdjacencyKeyT *key1Input,
    const BifrostCncoiAdjacencyKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiAdjacencyKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiAdjacencyKeyGetStr(const BifrostCncoiAdjacencyKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiAdjacencyGetStr_ng_param__";
    }

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    tempLen = BKF_GET_MIN(sizeof(key->commDesc), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    tempLen = BKF_GET_MIN(sizeof(key->commMd5sum), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->commMd5sum, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->srcLocalRankId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->operator);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->algorithm);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->rootRank);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->resv);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

int32_t BifrostCncoiRankKeyCmp(const BifrostCncoiRankKeyT *key1Input, const BifrostCncoiRankKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiRankKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiRankKeyGetStr(const BifrostCncoiRankKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiRankGetStr_ng_param__";
    }

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    tempLen = BKF_GET_MIN(sizeof(key->commDesc), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->commInitTime);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->packetId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    tempLen = BKF_GET_MIN(sizeof(key->resv), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->resv, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

int32_t BifrostCncoiRankDistributeKeyCmp(const BifrostCncoiRankDistributeKeyT *key1Input,
    const BifrostCncoiRankDistributeKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiRankDistributeKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiRankDistributeKeyGetStr(const BifrostCncoiRankDistributeKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiRankDistributeGetStr_ng_param__";
    }

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->npuIp);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->resv);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

int32_t BifrostCncoiRootRankKeyCmp(const BifrostCncoiRootRankKeyT *key1Input, const BifrostCncoiRootRankKeyT *key2InDs)
{
    int32_t ret;

    ret = VOS_MemCmp(key1Input, key2InDs, sizeof(BifrostCncoiRootRankKeyT));
    if (ret != 0) {
        return ret;
    }

    return ret;
}

char *BifrostCncoiRootRankKeyGetStr(const BifrostCncoiRootRankKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__SEGR_SidiRootRankGetStr_ng_param__";
    }

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    tempLen = BKF_GET_MIN(sizeof(key->commDesc), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->commInitTime);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->operator);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->algorithm);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    tempLen = BKF_GET_MIN(sizeof(key->resv), 20);
    tempStr = BKF_GET_MEM_STD_STR(&key->resv, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    return (char *)buf;

error:

    buf[strTotalLen] = '\0';
    return (char *)buf;
}

#if __cplusplus
}
#endif
