/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_adj.h"
#include "bkf_bas_type_mthd.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t NetTblAdjacencyKeyCmp(NetTblAdjacencyKey *key1Input, NetTblAdjacencyKey *key2InDs)
{
    int32_t ret = BKF_CMP_X(key1Input->taskId, key2InDs->taskId);
    if (ret != 0) {
        return ret;
    }
    ret = VOS_StrNCmp(key1Input->commDesc, key2InDs->commDesc, NET_COMMDESC_MAXLEN);
    if (ret != 0) {
        return ret;
    }
    ret = VOS_MemCmp(key1Input->commMd5Sum, key2InDs->commMd5Sum, NET_COMMMD5SUM_MAXLEN);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->srcLocalRankId, key2InDs->srcLocalRankId);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->op, key2InDs->op);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->algorithm, key2InDs->algorithm);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->rootRank, key2InDs->rootRank);
    if (ret != 0) {
        return ret;
    }
    return 0;
}

uint32_t NetTblAdjacencyKeyH2N(NetTblAdjacencyKey *keyH, NetTblAdjacencyKey *keyN)
{
    keyN->taskId = VOS_HTONLL(keyH->taskId);
    errno_t err1 = memmove_s(keyN->commDesc, NET_COMMDESC_MAXLEN, keyH->commDesc, NET_COMMDESC_MAXLEN);
    errno_t err2 = memmove_s(keyN->commMd5Sum, NET_COMMMD5SUM_MAXLEN, keyH->commMd5Sum, NET_COMMMD5SUM_MAXLEN);
    if (err1 != EOK || err2 != EOK) {
        return BKF_ERR;
    }
    keyN->srcLocalRankId = VOS_HTONS(keyH->srcLocalRankId);
    keyN->op = keyH->op;
    keyN->algorithm = keyH->algorithm;
    keyN->rootRank = VOS_HTONS(keyH->rootRank);
    return BKF_OK;
}

uint32_t NetTblAdjacencyKeyN2H(NetTblAdjacencyKey *keyN, NetTblAdjacencyKey *keyH)
{
    keyH->taskId = VOS_NTOHLL(keyN->taskId);
    errno_t err1 = memmove_s(keyH->commDesc, NET_COMMDESC_MAXLEN, keyN->commDesc, NET_COMMDESC_MAXLEN);
    errno_t err2 = memmove_s(keyH->commMd5Sum, NET_COMMMD5SUM_MAXLEN, keyN->commMd5Sum, NET_COMMMD5SUM_MAXLEN);
    if (err1 != EOK || err2 != EOK) {
        return BKF_ERR;
    }
    keyH->srcLocalRankId = VOS_NTOHS(keyN->srcLocalRankId);
    keyH->op = keyN->op;
    keyH->algorithm = keyN->algorithm;
    keyH->rootRank = VOS_NTOHS(keyN->rootRank);
    return BKF_OK;
}

char *NetTblAdjacencyKeyGetStr(NetTblAdjacencyKey *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblAdjacencyKeyGetStr";
    }
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", key->taskId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint8_t tempBuf[40] = {0};
    tempLen = BKF_GET_MIN((int32_t)sizeof(key->commDesc), NET_TBL_MEM_LEN_MAX);
    tempStr = BKF_GET_MEM_STD_STR(&key->commDesc, tempLen, tempBuf, sizeof(tempBuf));
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%s|", tempStr);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint32_t i;
    for (i = 0; i < NET_COMMMD5SUM_MAXLEN; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%02x", key->commMd5Sum[i]);
        if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
            goto error;
        }
        strTotalLen += strLen;
    }
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "|%u|", key->srcLocalRankId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u|%u|", key->op, key->algorithm,
        key->rootRank);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

uint32_t NetTblAdjacencyH2N(NetTblAdjacency *kvH, NetTblAdjacency *kvN)
{
    (void)NetTblAdjacencyKeyH2N(&kvH->key, &kvN->key);

    uint32_t i;
    for (i = 0; i < kvH->val.dstRankNum; i++) {
        kvN->val.adjinfo[i].dstLocalRankId = VOS_HTONS(kvH->val.adjinfo[i].dstLocalRankId);
        kvN->val.adjinfo[i].phaseId = kvH->val.adjinfo[i].phaseId;
    }
    kvN->val.dstRankNum = VOS_HTONS(kvH->val.dstRankNum);
    return BKF_OK;
}

uint32_t NetTblAdjacencyN2H(NetTblAdjacency *kvN, NetTblAdjacency *kvH)
{
    (void)NetTblAdjacencyKeyN2H(&kvN->key, &kvH->key);

    kvH->val.dstRankNum = VOS_NTOHS(kvN->val.dstRankNum);
    uint32_t i;
    for (i = 0; i < kvH->val.dstRankNum; i++) {
        kvH->val.adjinfo[i].dstLocalRankId = VOS_NTOHS(kvN->val.adjinfo[i].dstLocalRankId);
        kvH->val.adjinfo[i].phaseId = kvN->val.adjinfo[i].phaseId;
    }
    return BKF_OK;
}

char *NetTblAdjacencyGetStr(NetTblAdjacency *kv, uint8_t *buf, int32_t bufLen)
{
    if (kv == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblAdjacencyGetStr";
    }
    int32_t fillLen = 0;
    (void)NetTblAdjacencyKeyGetStr(&kv->key, buf, bufLen);
    fillLen = (int32_t)VOS_StrLen((char *)buf);
    if (fillLen == bufLen) {
        return (char *)buf;
    }
    (void)NetTblAdjacencyValGetStr(&kv->val, buf + fillLen, bufLen - fillLen);
    return (char *)buf;
}

char *NetTblAdjacencyValGetStr(NetTblAdjacencyVal *val, uint8_t *buf, int32_t bufLen)
{
    if (val == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblAdjacencyValGetStr";
    }
    int32_t strLen;
    int32_t strTotalLen = 0;
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->dstRankNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint32_t i;
    for (i = 0; i < val->dstRankNum; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u|",
            val->adjinfo[i].dstLocalRankId, val->adjinfo[i].phaseId);
        if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
            goto error;
        }
        strTotalLen += strLen;
    }
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

#ifdef __cplusplus
}
#endif
