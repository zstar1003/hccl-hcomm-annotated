/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_root_rank.h"
#include "bkf_bas_type_mthd.h"
#include "v_stringlib.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t NetTblRootRankKeyCmp(NetTblRootRankKey *key1Input, NetTblRootRankKey *key2InDs)
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
    return 0;
}

uint32_t NetTblRootRankKeyH2N(NetTblRootRankKey *keyH, NetTblRootRankKey *keyN)
{
    keyN->taskId = VOS_HTONLL(keyH->taskId);
    errno_t err = memmove_s(keyN->commDesc, NET_COMMDESC_MAXLEN, keyH->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyN->commInitTime = VOS_HTONLL(keyH->commInitTime);
    keyN->op = keyH->op;
    keyN->algorithm = keyH->algorithm;
    return BKF_OK;
}

uint32_t NetTblRootRankKeyN2H(NetTblRootRankKey *keyN, NetTblRootRankKey *keyH)
{
    keyH->taskId = VOS_NTOHLL(keyN->taskId);
    errno_t err = memmove_s(keyH->commDesc, NET_COMMDESC_MAXLEN, keyN->commDesc, NET_COMMDESC_MAXLEN);
    if (err != EOK) {
        return BKF_ERR;
    }
    keyH->commInitTime = VOS_NTOHLL(keyN->commInitTime);
    keyH->op = keyN->op;
    keyH->algorithm = keyN->algorithm;
    return BKF_OK;
}

char *NetTblRootRankKeyGetStr(NetTblRootRankKey *key, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;
    int32_t tempLen = 0;
    char *tempStr = VOS_NULL;

    if (key == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRootRankKeyGetStr";
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

    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|%u", key->op, key->algorithm);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;
    return (char *)buf;
error:
    buf[strTotalLen] = '\0';
    return (char *)buf;
}

uint32_t NetTblRootRankH2N(NetTblRootRank *kvH, NetTblRootRank *kvN)
{
    (void)NetTblRootRankKeyH2N(&kvH->key, &kvN->key);

    uint32_t i;
    for (i = 0; i < kvH->val.rootRankNum; i++) {
        kvN->val.rootRankInfo[i] = VOS_HTONS(kvH->val.rootRankInfo[i]);
    }
    kvN->val.rootRankNum = VOS_HTONS(kvH->val.rootRankNum);
    return BKF_OK;
}

uint32_t NetTblRootRankN2H(NetTblRootRank *kvN, NetTblRootRank *kvH)
{
    (void)NetTblRootRankKeyN2H(&kvN->key, &kvH->key);
    kvH->val.rootRankNum = VOS_HTONS(kvN->val.rootRankNum);
    uint32_t i;
    for (i = 0; i < kvH->val.rootRankNum; i++) {
        kvH->val.rootRankInfo[i] = VOS_HTONS(kvN->val.rootRankInfo[i]);
    }
    return BKF_OK;
}

char *NetTblRootRankGetStr(NetTblRootRank *kv, uint8_t *buf, int32_t bufLen)
{
    if (kv == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRootRankGetStr";
    }
    int32_t fillLen = 0;
    (void)NetTblRootRankKeyGetStr(&kv->key, buf, bufLen);
    fillLen = (int32_t)VOS_StrLen((char *)buf);
    if (fillLen == bufLen) {
        return (char *)buf;
    }
    (void)NetTblRootRankValGetStr(&kv->val, buf + fillLen, bufLen - fillLen);
    return (char *)buf;
}

char *NetTblRootRankValGetStr(NetTblRootRankVal *val, uint8_t *buf, int32_t bufLen)
{
    int32_t strLen;
    int32_t strTotalLen = 0;

    if (val == VOS_NULL || buf == VOS_NULL || bufLen <= 0) {
        return "__NetTblRootRankValGetStr";
    }
    strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%u|", val->rootRankNum);
    if (strLen <= 0 || (strTotalLen + strLen) >= bufLen) {
        goto error;
    }
    strTotalLen += strLen;

    uint32_t i;
    for (i = 0; i < val->rootRankNum; i++) {
        strLen = snprintf_truncated_s((char *)buf + strTotalLen, bufLen - strTotalLen, "%llu|", val->rootRankInfo[i]);
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
