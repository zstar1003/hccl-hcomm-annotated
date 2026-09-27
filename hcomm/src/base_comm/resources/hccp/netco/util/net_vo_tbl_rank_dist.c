/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_rank_dist.h"
#include "bkf_bas_type_mthd.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t NetTblRankDistributeInfoKeyCmp(NetTblRankDistributeInfoKey *key1Input, NetTblRankDistributeInfoKey *key2InDs)
{
    int32_t ret = BKF_CMP_X(key1Input->taskId, key2InDs->taskId);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->npuIp, key2InDs->npuIp);
    if (ret != 0) {
        return ret;
    }
    return 0;
}

uint32_t NetTblRankDistributeInfoKeyH2N(NetTblRankDistributeInfoKey *keyH, NetTblRankDistributeInfoKey *keyN)
{
    keyN->taskId = VOS_HTONLL(keyH->taskId);
    keyN->npuIp = VOS_HTONL(keyH->npuIp);
    return BKF_OK;
}

uint32_t NetTblRankDistributeInfoKeyN2H(NetTblRankDistributeInfoKey *keyN, NetTblRankDistributeInfoKey *keyH)
{
    keyH->taskId = VOS_NTOHLL(keyN->taskId);
    keyH->npuIp = VOS_NTOHL(keyN->npuIp);
    return BKF_OK;
}

char *NetTblRankDistributeInfoKeyGetStr(NetTblRankDistributeInfoKey *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRankDistributeInfoKeyGetStr";
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
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

uint32_t NetTblRankDistributeInfoH2N(NetTblRankDistributeInfo *kvH, NetTblRankDistributeInfo *kvN)
{
    (void)NetTblRankDistributeInfoKeyH2N(&kvH->key, &kvN->key);
    kvN->val.serverIp = VOS_HTONL(kvH->val.serverIp);
    kvN->val.nodeId = VOS_HTONL(kvH->val.nodeId);
    kvN->val.localRankNum = kvH->val.localRankNum;
    kvN->val.rankTotalNum = VOS_HTONL(kvH->val.rankTotalNum);
    return BKF_OK;
}

uint32_t NetTblRankDistributeInfoN2H(NetTblRankDistributeInfo *kvN, NetTblRankDistributeInfo *kvH)
{
    (void)NetTblRankDistributeInfoKeyN2H(&kvN->key, &kvH->key);
    kvH->val.serverIp = VOS_NTOHL(kvN->val.serverIp);
    kvH->val.nodeId = VOS_NTOHL(kvN->val.nodeId);
    kvH->val.localRankNum = kvN->val.localRankNum;
    kvH->val.rankTotalNum = VOS_NTOHL(kvN->val.rankTotalNum);
    return BKF_OK;
}

char *NetTblRankDistributeInfoGetStr(NetTblRankDistributeInfo *kv, uint8_t *buf, int32_t bufLen)
{
    if (kv == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRankDistributeInfoGetStr";
    }
    int32_t fillLen = 0;
    (void)NetTblRankDistributeInfoKeyGetStr(&kv->key, buf, bufLen);
    fillLen = (int32_t)VOS_StrLen((char *)buf);
    if (fillLen == bufLen) {
        return (char *)buf;
    }
    (void)NetTblRankDistributeInfoValGetStr(&kv->val, buf + fillLen, bufLen - fillLen);
    return (char *)buf;
}

char *NetTblRankDistributeInfoValGetStr(NetTblRankDistributeInfoVal *val, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;

    if (val == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRankDistributeInfoGetStr";
    }
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->serverIp);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->nodeId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u", val->localRankNum,
        val->rankTotalNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

#ifdef __cplusplus
}
#endif
