/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_comm_info.h"
#include "bkf_bas_type_mthd.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
int32_t NetTblCommInfoKeyCmp(NetTblCommInfoKey *key1Input, NetTblCommInfoKey *key2InDs)
{
    int32_t ret = BKF_CMP_X(key1Input->taskId, key2InDs->taskId);
    if (ret != 0) {
        return ret;
    }
    ret = VOS_StrNCmp(key1Input->commDesc, key2InDs->commDesc, NET_COMMDESC_MAXLEN);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->commInitTime, key2InDs->commInitTime);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(key1Input->packetId, key2InDs->packetId);
    if (ret != 0) {
        return ret;
    }

    return 0;
}

uint32_t NetTblCommInfoKeyH2N(NetTblCommInfoKey *keyH, NetTblCommInfoKey *keyN)
{
    keyN->taskId = VOS_HTONLL(keyH->taskId);
    errno_t err = memmove_s(keyN->commDesc, NET_COMMDESC_MAXLEN, keyH->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyN->commInitTime = VOS_HTONLL(keyH->commInitTime);
    keyN->packetId = VOS_HTONS(keyH->packetId);
    return BKF_OK;
}

uint32_t NetTblCommInfoKeyN2H(NetTblCommInfoKey *keyN, NetTblCommInfoKey *keyH)
{
    keyH->taskId = VOS_NTOHLL(keyN->taskId);
    errno_t err = memmove_s(keyH->commDesc, NET_COMMDESC_MAXLEN, keyN->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyH->commInitTime = VOS_NTOHLL(keyN->commInitTime);
    keyH->packetId = VOS_NTOHS(keyN->packetId);
    return BKF_OK;
}

char *NetTblCommInfoKeyGetStr(NetTblCommInfoKey *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblCommInfoKeyGetStr";
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
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

uint32_t NetTblCommInfoH2N(NetTblCommInfo *kvH, NetTblCommInfo *kvN)
{
    (void)NetTblCommInfoKeyH2N(&kvH->key, &kvN->key);

    uint32_t i;
    for (i = 0; i < kvH->val.packetNum; i++) {
        kvN->val.sendRankInfo[i] = VOS_HTONL(kvH->val.sendRankInfo[i]);
    }
    kvN->val.packetNum = VOS_HTONS(kvH->val.packetNum);

    errno_t err = memmove_s(kvN->val.commMd5Sum, NET_COMMMD5SUM_MAXLEN, kvH->val.commMd5Sum, NET_COMMMD5SUM_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    kvN->val.rankTotalNum = VOS_HTONS(kvH->val.rankTotalNum);
    for (i = 0; i < kvH->val.rankNum; i++) {
        kvN->val.rankInfo[i].deviceIp = VOS_HTONL(kvH->val.rankInfo[i].deviceIp);
        kvN->val.rankInfo[i].serverIp = VOS_HTONL(kvH->val.rankInfo[i].serverIp);
        kvN->val.rankInfo[i].podId = VOS_HTONS(kvH->val.rankInfo[i].podId);
    }
    kvN->val.rankNum = VOS_HTONS(kvH->val.rankNum);
    return BKF_OK;
}

uint32_t NetTblCommInfoN2H(NetTblCommInfo *kvN, NetTblCommInfo *kvH)
{
    (void)NetTblCommInfoKeyN2H(&kvN->key, &kvH->key);
    kvH->val.packetNum = VOS_NTOHS(kvN->val.packetNum);
    uint32_t i;
    for (i = 0; i < kvH->val.packetNum; i++) {
        kvH->val.sendRankInfo[i] = VOS_NTOHL(kvN->val.sendRankInfo[i]);
    }
    errno_t err = memmove_s(kvH->val.commMd5Sum, NET_COMMMD5SUM_MAXLEN, kvN->val.commMd5Sum, NET_COMMMD5SUM_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    kvH->val.rankTotalNum = VOS_NTOHS(kvN->val.rankTotalNum);
    kvH->val.rankNum = VOS_NTOHS(kvN->val.rankNum);
    for (i = 0; i < kvH->val.rankNum; i++) {
        kvH->val.rankInfo[i].deviceIp = VOS_NTOHL(kvN->val.rankInfo[i].deviceIp);
        kvH->val.rankInfo[i].serverIp = VOS_NTOHL(kvN->val.rankInfo[i].serverIp);
        kvH->val.rankInfo[i].podId = VOS_NTOHS(kvN->val.rankInfo[i].podId);
    }
    return BKF_OK;
}

char *NetTblCommInfoGetStr(NetTblCommInfo *kv, uint8_t *buf, int32_t bufLen)
{
    if (kv == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblCommInfoGetStr";
    }
    int32_t fillLen = 0;
    (void)NetTblCommInfoKeyGetStr(&kv->key, buf, bufLen);
    fillLen = (int32_t)VOS_StrLen((char *)buf);
    if (fillLen == bufLen) {
        return (char *)buf;
    }
    (void)NetTblCommInfoValGetStr(&kv->val, buf + fillLen, bufLen - fillLen);
    return (char *)buf;
}

char *NetTblCommInfoValGetStr(NetTblCommInfoVal *val, uint8_t *buf, int32_t bufLen)
{
    if (val == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblCommInfoValGetStr";
    }
    int32_t strLen;
    int32_t strTotalLen = 0;
    uint32_t i;
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->packetNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;
    for (i = 0; i < val->packetNum; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->sendRankInfo[i]);
        if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
            goto error;
        }
        strTotalLen += strLen;
    }

    for (i = 0; i < NET_COMMMD5SUM_MAXLEN; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%02x", val->commMd5Sum[i]);
        if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
            goto error;
        }
        strTotalLen += strLen;
    }
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "|%u|", val->rankTotalNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->rankNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    for (i = 0; i < val->rankNum; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u|%u|",
            val->rankInfo[i].deviceIp, val->rankInfo[i].serverIp, val->rankInfo[i].podId);
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
