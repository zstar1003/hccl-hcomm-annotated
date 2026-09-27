/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_oper.h"
#include "bkf_bas_type_mthd.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t NetTblOperatorKeyCmp(NetTblOperatorKey *key1Input, NetTblOperatorKey *key2InDs)
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

uint32_t NetTblOperatorKeyH2N(NetTblOperatorKey *keyH, NetTblOperatorKey *keyN)
{
    keyN->taskId = VOS_HTONLL(keyH->taskId);
    errno_t err = memmove_s(keyN->commDesc, NET_COMMDESC_MAXLEN, keyH->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyN->commInitTime = VOS_HTONLL(keyH->commInitTime);
    keyN->op = keyH->op;
    keyN->algorithm = keyH->algorithm;
    keyN->rootRank = VOS_HTONS(keyH->rootRank);
    return BKF_OK;
}

uint32_t NetTblOperatorKeyN2H(NetTblOperatorKey *keyN, NetTblOperatorKey *keyH)
{
    keyH->taskId = VOS_NTOHLL(keyN->taskId);
    errno_t err = memmove_s(keyH->commDesc, NET_COMMDESC_MAXLEN, keyN->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyH->commInitTime = VOS_NTOHLL(keyN->commInitTime);
    keyH->op = keyN->op;
    keyH->algorithm = keyN->algorithm;
    keyH->rootRank = VOS_NTOHS(keyN->rootRank);
    return BKF_OK;
}

char *NetTblOperatorKeyGetStr(NetTblOperatorKey *key, uint8_t *buf, int32_t bufLen)
{
    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblOperatorKeyGetStr";
    }

    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;
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

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u|", key->op, key->algorithm);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", key->rootRank);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

uint32_t NetTblOperatorH2N(NetTblOperator *kvH, NetTblOperator *kvN)
{
    (void)NetTblOperatorKeyH2N(&kvH->key, &kvN->key);
    kvN->val.trafficCnt = VOS_HTONLL(kvH->val.trafficCnt);
    kvN->val.l4SPortId = VOS_HTONS(kvH->val.l4SPortId);
    kvN->val.maskLen = VOS_HTONS(kvH->val.maskLen);
    return BKF_OK;
}

uint32_t NetTblOperatorN2H(NetTblOperator *kvN, NetTblOperator *kvH)
{
    (void)NetTblOperatorKeyN2H(&kvN->key, &kvH->key);

    kvH->val.trafficCnt = VOS_NTOHLL(kvN->val.trafficCnt);
    kvH->val.l4SPortId = VOS_NTOHS(kvN->val.l4SPortId);
    kvH->val.maskLen = VOS_NTOHS(kvN->val.maskLen);
    return BKF_OK;
}

char *NetTblOperatorGetStr(NetTblOperator *kv, uint8_t *buf, int32_t bufLen)
{
    if (kv == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblOperatorGetStr";
    }
    int32_t fillLen = 0;
    (void)NetTblOperatorKeyGetStr(&kv->key, buf, bufLen);
    fillLen = (int32_t)VOS_StrLen((char *)buf);
    if (fillLen == bufLen) {
        return (char *)buf;
    }
    (void)NetTblOperatorValGetStr(&kv->val, buf + fillLen, bufLen - fillLen);
    return (char *)buf;
}

char *NetTblOperatorValGetStr(NetTblOperatorVal *val, uint8_t *buf, int32_t bufLen)
{
    if (val == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblOperatorValGetStr";
    }
    int32_t strLen;
    int32_t strTotalLen = 0;
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", val->trafficCnt);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->l4SPortId);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->maskLen);
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
