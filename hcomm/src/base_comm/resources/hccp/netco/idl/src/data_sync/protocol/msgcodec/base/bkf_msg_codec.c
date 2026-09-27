/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((codecer)->logOrNull)
#define BKF_MOD_NAME ((codecer)->name)

#include "bkf_msg_codec.h"
#include "bkf_dc.h"
#include "vos_base.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("tlv编解码表")
#pragma pack(4)
/* 不要访问转出来的tl字段 */
#define BKF_CODEC_VAL_2TLV(val) ((void *)((uint8_t *)(val) - sizeof(BkfTL)))

STATIC uint32_t BkfCodecTlvTransNum(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvTransNum *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "transNum(%" VOS_PRIu64 "/%" VOS_PRIu64 "), inValLen(%d)\n", temp->num,
        VOS_HTONLL(temp->num), inValLen);

    temp->num = VOS_HTONLL(temp->num);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvProjectName(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvProjectName *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "prjName(%s), inValLen(%d)\n", temp->name, inValLen);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvProjectVersion(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvProjectVersion *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "version(%u), inValLen(%d)\n", temp->version, inValLen);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvServiceName(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvServiceName *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "svcName(%s), inValLen(%d)\n", temp->name, inValLen);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvPuberName(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvPuberName *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "puberName(%s), inValLen(%d)\n", temp->name, inValLen);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvSuberName(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvSuberName *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "suberName(%s), inValLen(%d)\n", temp->name, inValLen);
    return BKF_OK;
}
STATIC uint32_t BkfCodecTlvIdlVersion(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvIdlVersion *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "major(%u/%u)/minor(%u/%u), inValLen(%d)\n", temp->major, VOS_HTONS(temp->major),
        temp->minor, VOS_HTONS(temp->minor), inValLen);

    temp->major = VOS_HTONS(temp->major);
    temp->minor = VOS_HTONS(temp->minor);
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvSuberLsnUrl(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    (void)inValLen;
    BkfTlvSuberLsnUrl *temp = BKF_CODEC_VAL_2TLV(val);

    if (temp->lsnUrl.type == BKF_URL_TYPE_DMS) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "BkfCodecTlvSuberLsnUrl:dmsId %#x\n", temp->lsnUrl.dmsId);

        temp->lsnUrl.dmsId = VOS_HTONL(temp->lsnUrl.dmsId);
    } else {
        temp->lsnUrl.ip.addrH = VOS_HTONL(temp->lsnUrl.ip.addrH);
        temp->lsnUrl.ip.port = VOS_HTONS(temp->lsnUrl.ip.port);
        BKF_LOG_DEBUG(BKF_LOG_HND, "BkfCodecTlvSuberLsnUrl:addr %#x, port %u\n", temp->lsnUrl.ip.addrH,
            temp->lsnUrl.ip.port);
    }
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvReasonCode(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvReasonCode *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "reasonCode(%u, %s), inValLen(%d)\n", temp->reasonCode,
        BkfReasonCodeGetStr(temp->reasonCode), inValLen);
    return BKF_OK;
}
#define BKF_CODEC_TLV_SLICE_KEY_2STR(codecer, val, buf, bufLen)                                                        \
    (((codecer)->sliceKeyGetStrOrNull != VOS_NULL) ? (codecer)->sliceKeyGetStrOrNull((val), (buf), (bufLen)) : "-")
STATIC uint32_t BkfCodecTlvSliceKey(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    uint8_t buf[BKF_1K / 4];
    uint32_t ret;

    BKF_LOG_DEBUG(BKF_LOG_HND, "before_codec, inValLen(%d)/keyStr(%s)\n", inValLen,
        BKF_CODEC_TLV_SLICE_KEY_2STR(codecer, val, buf, sizeof(buf)));

    if (inValLen != codecer->sliceKeyLen) {
        return BKF_ERR;
    }
    ret = codecer->sliceKeyCodec(val);
    if (ret == BKF_OK) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "after_codec, inValLen(%d)/keyStr(%s)\n", inValLen,
            BKF_CODEC_TLV_SLICE_KEY_2STR(codecer, val, buf, sizeof(buf)));
    }
    return ret;
}
STATIC uint32_t BkfCodecTlvTableType(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvTableType *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "tableTypeId(%u/%u), inValLen(%d)\n", temp->tableTypeId, VOS_HTONS(temp->tableTypeId),
        inValLen);

    temp->tableTypeId = VOS_HTONS(temp->tableTypeId);
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvTupleIdlData(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvTupleIdlData *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "seq(%" VOS_PRIu64 "/%" VOS_PRIu64 "), inValLen(%d)\n", temp->seq, VOS_HTONLL(temp->seq),
        inValLen);

    temp->seq = VOS_HTONLL(temp->seq);
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvCondIdlData(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    (void)codecer;
    (void)val;
    (void)inValLen;
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvResultIdlData(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    (void)codecer;
    (void)val;
    (void)inValLen;
    return BKF_OK;
}

STATIC uint32_t BkfCodecTlvRpc(BkfMsgCodecer *codecer, void *val, int32_t inValLen)
{
    BkfTlvRpc *temp = BKF_CODEC_VAL_2TLV(val);
    BKF_LOG_DEBUG(BKF_LOG_HND, "type(%u/%u), oper(%u/%u), seq(%" VOS_PRIu64 "/%" VOS_PRIu64 "), inValLen(%d)\n",
        temp->type, VOS_HTONS(temp->type), temp->operCode, VOS_HTONS(temp->operCode), temp->seq, VOS_HTONLL(temp->seq),
        inValLen);

    temp->type = VOS_HTONS(temp->type);
    temp->operCode = VOS_HTONS(temp->operCode);
    temp->seq = VOS_HTONLL(temp->seq);
    return BKF_OK;
}

#define BKF_TLV_PART1_LEN_FIXED ((uint32_t)1 << 0)
#define BKF_TLV_PART1_LEN_RANGE ((uint32_t)1 << 1)
typedef uint32_t (*F_BKF_TLV_CODEC)(BkfMsgCodecer *codecer, void *val, int32_t inValLen);
typedef struct tagBkfTlvCodecVTbl {
    uint32_t flag;
    F_BKF_TLV_CODEC codec;
    int32_t valRealLenFixed;
    int32_t valRealLenMin;
    int32_t valRealLenMax;
} BkfTlvCodecVTbl;

const BkfTlvCodecVTbl g_BkfTlvCodecVTbl[] = {
    {
        BKF_TLV_PART1_LEN_FIXED,
        VOS_NULL,
        0,
        0,
        0,
    },
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvTransNum, BKF_MBR_SIZE(BkfTlvTransNum, num), 0, 0},
    {BKF_TLV_PART1_LEN_RANGE, BkfCodecTlvProjectName, 0, 2, BKF_TLV_NAME_LEN_MAX + 1},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvProjectVersion, BKF_MBR_SIZE(BkfTlvProjectVersion, version), 0, 0},
    {BKF_TLV_PART1_LEN_RANGE, BkfCodecTlvServiceName, 0, 2, BKF_TLV_NAME_LEN_MAX + 1},
    {BKF_TLV_PART1_LEN_RANGE, BkfCodecTlvPuberName, 0, 2, BKF_TLV_NAME_LEN_MAX + 1},
    {BKF_TLV_PART1_LEN_RANGE, BkfCodecTlvSuberName, 0, 2, BKF_TLV_NAME_LEN_MAX + 1},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvIdlVersion,
        BKF_MBR_SIZE(BkfTlvIdlVersion, major) + BKF_MBR_SIZE(BkfTlvIdlVersion, minor), 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvReasonCode, BKF_MBR_SIZE(BkfTlvReasonCode, reasonCode), 0, 0},
    {BKF_TLV_PART1_LEN_RANGE, BkfCodecTlvSliceKey, 0, 4, BKF_DC_SLICE_KEY_LEN_MAX},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvTableType, BKF_MBR_SIZE(BkfTlvTableType, tableTypeId), 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvTupleIdlData, BKF_MBR_SIZE(BkfTlvTupleIdlData, seq), 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvRpc, BKF_OFFSET(BkfTlvRpc, idlData) - BKF_OFFSET(BkfTlvRpc, type), 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvCondIdlData, 0, 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvResultIdlData, 0, 0, 0},
    {BKF_TLV_PART1_LEN_FIXED, BkfCodecTlvSuberLsnUrl, sizeof(BkfUrl), 0, 0},
};

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义, 编码")
#ifdef BKF_LOG_HND
#undef BKF_LOG_HND
#define BKF_LOG_HND ((coder)->codecer.logOrNull)
#endif
#ifdef BKF_MOD_NAME
#undef BKF_MOD_NAME
#define BKF_MOD_NAME ((coder)->codecer.name)
#endif

void BkfMsgCodeResetSliceKeyPara(BkfMsgCoder *coder, uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec,
    F_BKF_GET_STR sliceKeyGetStrOrNull)
{
    if (coder == VOS_NULL) {
        return;
    }

    coder->codecer.sliceKeyLen = sliceKeyLen;
    coder->codecer.sliceKeyCodec = sliceKeyCodec;
    coder->codecer.sliceKeyGetStrOrNull = sliceKeyGetStrOrNull;
}

/* func */
uint32_t BkfMsgCodeInit(BkfMsgCoder *coder, const char *name, uint8_t *sendBuf, int32_t sendBufLen,
    uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec, F_BKF_GET_STR sliceKeyGetStrOrNull, BkfLog *logOrNull)
{
    if ((coder == VOS_NULL) || (name == VOS_NULL) || (sendBuf == VOS_NULL) || (sendBufLen <= 0)) {
        return BKF_ERR;
    }

    (void)memset_s(coder, sizeof(BkfMsgCoder), 0, sizeof(BkfMsgCoder));
    coder->codecer.name = name;
    coder->codecer.sliceKeyLen = sliceKeyLen;
    coder->codecer.sliceKeyCodec = sliceKeyCodec;
    coder->codecer.sliceKeyGetStrOrNull = sliceKeyGetStrOrNull;
    coder->codecer.logOrNull = logOrNull;
    coder->sendBuf = sendBuf;
    coder->sendBufLen = sendBufLen;
    coder->codedErr = VOS_FALSE;
    coder->validMsgLen = 0;
    coder->curMsgHead = VOS_NULL;
    coder->tempUsedLen = 0;
    coder->curTl = VOS_NULL;
    coder->curTlvHasAddRaw = VOS_FALSE;
    return BKF_OK;
}

STATIC uint32_t BkfMsgCodeMsgHeadChkParam(BkfMsgCoder *coder, uint16_t msgId)
{
    uint32_t msgBodyLen;

    if (coder == VOS_NULL) {
        return BKF_ERR;
    }
    msgBodyLen = (coder->curMsgHead != VOS_NULL) ? VOS_NTOHL(coder->curMsgHead->bodyLen) : 0;
    BKF_LOG_DEBUG(BKF_LOG_HND, "msgId(%u, %s), sendBufLen(%d)/validMsgLen(%d)/msgBodyLen(%u)/tempUsedLen(%d)\n", msgId,
        BkfMsgGetIdStr(msgId), coder->sendBufLen, coder->validMsgLen, msgBodyLen, coder->tempUsedLen);

    if (coder->codedErr) {
        BKF_LOG_INFO(BKF_LOG_HND, "codedErr(%u)\n", coder->codedErr);
        return BKF_ERR;
    }
    if (coder->tempUsedLen > 0) {
        /* 此处会将上个消息中没有更新到msgBody的数据扔掉。可能是错误。 */
        BKF_LOG_WARN(BKF_LOG_HND, "tempUsedLen(%d), drop some data\n", coder->tempUsedLen);
        /* 继续 */
    }
    if (!BKF_MSG_IS_VALID(msgId)) {
        BKF_LOG_INFO(BKF_LOG_HND, "msgId(%u), ng\n", msgId);
        return BKF_ERR;
    }

    return BKF_OK;
}
uint32_t BkfMsgCodeMsgHead(BkfMsgCoder *coder, uint16_t msgId, uint8_t flag)
{
    uint32_t ret;
    int32_t leftLen;
    int32_t needLen;
    BkfMsgHead *curMsgHead = VOS_NULL;

    ret = BkfMsgCodeMsgHeadChkParam(coder, msgId);
    if (ret != BKF_OK) {
        goto error;
    }
    coder->curMsgHead = VOS_NULL;
    coder->tempUsedLen = 0;
    coder->curTl = VOS_NULL;
    coder->curTlvHasAddRaw = VOS_FALSE;

    leftLen = BKF_MSG_CODE_GET_LEFT_LEN(coder);
    needLen = sizeof(BkfMsgHead);
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), send buf not enough & OK!\n", leftLen, needLen);
        goto error;
    }
    curMsgHead = (BkfMsgHead *)BKF_MSG_CODE_GET_LEFT_BEGIN(coder);
    curMsgHead->sign = VOS_HTONL(BKF_MSG_SIGN);
    curMsgHead->msgId = VOS_HTONS(msgId);
    curMsgHead->flag = flag;
    curMsgHead->reserved = BKF_MSG_RSV_VAL;
    curMsgHead->bodyLen = 0;

    coder->validMsgLen += needLen;
    coder->curMsgHead = curMsgHead;
    BKF_LOG_DEBUG(BKF_LOG_HND,
        "msgId(%u, %s)/flag(%#x), sendBufLen(%d)/validMsgLen(%d)/"
        "msgBodyLen(%u)/tempUsedLen(%d)\n",
        msgId, BkfMsgGetIdStr(msgId), flag, coder->sendBufLen, coder->validMsgLen, VOS_NTOHL(curMsgHead->bodyLen),
        coder->tempUsedLen);
    return BKF_OK;

error:

    if (coder != VOS_NULL) {
        coder->codedErr = VOS_TRUE;
    }
    return BKF_ERR;
}

STATIC uint32_t BkfMsgCodeTLVChkParam(BkfMsgCoder *coder, uint16_t typeId, uint8_t flag, void *val, BOOL updMsgBodyLen)
{
    BkfMsgHead *curMsgHead = VOS_NULL;

    if (coder == VOS_NULL) {
        return BKF_ERR;
    }
    curMsgHead = coder->curMsgHead;
    if (curMsgHead == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "curMsgHead(%#x), ng\n", BKF_MASK_ADDR(curMsgHead));
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "msgId(%u, %s)/typeId(%u, %s), flag(%#x)/val(%#x)/updMsgBodyLen(%u)\n",
        VOS_NTOHS(curMsgHead->msgId), BkfMsgGetIdStr(VOS_NTOHS(curMsgHead->msgId)), typeId, BkfTlvGetTypeStr(typeId),
        flag, BKF_MASK_ADDR(val), updMsgBodyLen);

    if (coder->codedErr) {
        BKF_LOG_INFO(BKF_LOG_HND, "codedErr(%u)\n", coder->codedErr);
        return BKF_ERR;
    }
    if (!BKF_TLV_IS_VALID(typeId)) {
        BKF_LOG_INFO(BKF_LOG_HND, "typeId(%u), ng\n", typeId);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void BkfMsgCodeTLVDoAfter(BkfMsgCoder *coder, BOOL updMsgBodyLen, int32_t needLen)
{
    BkfMsgHead *curMsgHead = coder->curMsgHead;
    BkfTL *curTl = coder->curTl;

    if (!updMsgBodyLen) {
        coder->tempUsedLen += needLen;
    } else {
        coder->validMsgLen += (coder->tempUsedLen + needLen);
        curMsgHead->bodyLen = VOS_NTOHL(curMsgHead->bodyLen) + (uint32_t)(coder->tempUsedLen + needLen);
        curMsgHead->bodyLen = VOS_HTONL(curMsgHead->bodyLen);
        coder->tempUsedLen = 0;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND,
        "msg(%u, %s)/type(%u, %s)/flag(%#x)/len(%u), "
        "sendBufLen(%d)/validMsgLen(%d)/msgBodyLen(%u)/tempUsedLen(%d)\n",
        VOS_NTOHS(curMsgHead->msgId), BkfMsgGetIdStr(VOS_NTOHS(curMsgHead->msgId)), VOS_NTOHS(curTl->typeId),
        BkfTlvGetTypeStr(VOS_NTOHS(curTl->typeId)), curTl->flag, VOS_NTOHL(curTl->valLen), coder->sendBufLen,
        coder->validMsgLen, VOS_NTOHL(curMsgHead->bodyLen), coder->tempUsedLen);
    return;
}
static inline void BkfMsgFillReservedForAlign(void *valBegin, int32_t valRealLen)
{
    int32_t valAlignLen = BKF_GET_ALIGN4_LEN(valRealLen);
    int32_t fillLen = valAlignLen - valRealLen;
    if (fillLen > 0) {
        (void)memset_s((uint8_t *)valBegin + valRealLen, fillLen, BKF_MSG_RSV_VAL, fillLen);
    }
}
uint32_t BkfMsgCodeTLV(BkfMsgCoder *coder, uint16_t typeId, uint8_t flag, void *val, BOOL updMsgBodyLen)
{
    uint32_t ret;
    BkfTlvCodecVTbl *vTbl = VOS_NULL;
    BkfTL *curTl = VOS_NULL;

    if (coder == NULL) {
        return BKF_ERR;
    }
    ret = BkfMsgCodeTLVChkParam(coder, typeId, flag, val, updMsgBodyLen);
    if (ret != BKF_OK) {
        BKF_LOG_INFO(BKF_LOG_HND, "type(%u, %s), check TLV param failed.\n", typeId, BkfTlvGetTypeStr(typeId));
        goto error;
    }
    coder->curTl = VOS_NULL;
    coder->curTlvHasAddRaw = VOS_FALSE;

    vTbl = (BkfTlvCodecVTbl *)&g_BkfTlvCodecVTbl[typeId];
    if (!BKF_BIT_TEST(vTbl->flag, BKF_TLV_PART1_LEN_FIXED)) {
        BKF_LOG_INFO(BKF_LOG_HND, "type(%u, %s), part1 not fixed len, ng\n", typeId, BkfTlvGetTypeStr(typeId));
        goto error;
    }
    int32_t leftLen = BKF_MSG_CODE_GET_LEFT_LEN(coder);
    int32_t needLen = (int32_t)sizeof(BkfTL) + BKF_GET_ALIGN4_LEN(vTbl->valRealLenFixed);
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), send buf not enough & OK!\n", leftLen, needLen);
        goto error;
    }
    curTl = (BkfTL *)BKF_MSG_CODE_GET_LEFT_BEGIN(coder);
    curTl->typeId = VOS_HTONS(typeId);
    curTl->flag = flag;
    curTl->reserved = BKF_MSG_RSV_VAL;
    curTl->valLen = VOS_HTONL((uint32_t)vTbl->valRealLenFixed);
    if (vTbl->valRealLenFixed != 0 && val != VOS_NULL) {
        if (memcpy_s(curTl + 1, vTbl->valRealLenFixed, val, vTbl->valRealLenFixed) != EOK) {
            goto error;
        }
    }
    ret = vTbl->codec(&coder->codecer, curTl + 1, vTbl->valRealLenFixed);
    if (ret != BKF_OK) {
        BKF_LOG_INFO(BKF_LOG_HND, "ret(%u), ng\n", ret);
        goto error;
    }
    BkfMsgFillReservedForAlign(curTl + 1, vTbl->valRealLenFixed);
    coder->curTl = curTl;

    BkfMsgCodeTLVDoAfter(coder, updMsgBodyLen, needLen);
    return BKF_OK;

error:
    coder->codedErr = VOS_TRUE;
    return BKF_ERR;
}

uint32_t BkfMsgCodeRawTLV(BkfMsgCoder *coder, uint16_t typeId, uint8_t flag, void *val, int32_t valRealLen,
    BOOL updMsgBodyLen)
{
    uint32_t ret;
    BkfTlvCodecVTbl *vTbl = VOS_NULL;
    int32_t leftLen;
    int32_t needLen;
    BkfTL *curTl = VOS_NULL;

    ret = BkfMsgCodeTLVChkParam(coder, typeId, flag, val, updMsgBodyLen);
    if (ret != BKF_OK) {
        goto error;
    }
    coder->curTl = VOS_NULL;
    coder->curTlvHasAddRaw = VOS_FALSE;

    vTbl = (BkfTlvCodecVTbl *)&g_BkfTlvCodecVTbl[typeId];
    if (!BKF_BIT_TEST(vTbl->flag, BKF_TLV_PART1_LEN_RANGE)) {
        BKF_LOG_INFO(BKF_LOG_HND, "type(%u, %s), not range len, ng\n", typeId, BkfTlvGetTypeStr(typeId));
        goto error;
    }
    if ((valRealLen < vTbl->valRealLenMin) || (valRealLen > vTbl->valRealLenMax)) {
        BKF_LOG_INFO(BKF_LOG_HND, "valRealLe(%d)/[%d, %d], ng\n", valRealLen, vTbl->valRealLenMin, vTbl->valRealLenMax);
        goto error;
    }
    leftLen = BKF_MSG_CODE_GET_LEFT_LEN(coder);
    needLen = (int32_t)(sizeof(BkfTL) + BKF_GET_ALIGN4_LEN(valRealLen));
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), send buf not enough & OK!\n", leftLen, needLen);
        goto error;
    }
    curTl = (BkfTL *)BKF_MSG_CODE_GET_LEFT_BEGIN(coder);
    curTl->typeId = VOS_HTONS(typeId);
    curTl->flag = flag;
    curTl->reserved = BKF_MSG_RSV_VAL;
    curTl->valLen = VOS_HTONL((uint32_t)valRealLen);
    (void)memcpy_s(curTl + 1, valRealLen, val, valRealLen);
    ret = vTbl->codec(&coder->codecer, curTl + 1, valRealLen);
    if (ret != BKF_OK) {
        BKF_LOG_INFO(BKF_LOG_HND, "ret(%u), ng\n", ret);
        goto error;
    }
    BkfMsgFillReservedForAlign(curTl + 1, valRealLen);
    coder->curTl = curTl;
    coder->curTlvHasAddRaw = VOS_TRUE;

    BkfMsgCodeTLVDoAfter(coder, updMsgBodyLen, needLen);
    return BKF_OK;

error:

    if (coder != VOS_NULL) {
        coder->codedErr = VOS_TRUE;
    }
    return BKF_ERR;
}

STATIC uint32_t BkfMsgCodeAppendValLenChkParam(BkfMsgCoder *coder, uint8_t flag, int32_t valRealLen, BOOL updMsgBodyLen)
{
    BkfMsgHead *curMsgHead = VOS_NULL;
    BkfTL *curTl = VOS_NULL;
    BkfTlvCodecVTbl *vTbl = VOS_NULL;
    uint16_t typeId;

    if (coder == VOS_NULL) {
        return BKF_ERR;
    }
    curMsgHead = coder->curMsgHead;
    if (curMsgHead == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "curMsgHead(%#x), ng\n", BKF_MASK_ADDR(curMsgHead));
        return BKF_ERR;
    }
    curTl = coder->curTl;
    if (curTl == VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "curTl(%#x), ng\n", BKF_MASK_ADDR(curTl));
        return BKF_ERR;
    }
    typeId = VOS_NTOHS(curTl->typeId);
    if (!BKF_TLV_IS_VALID(typeId)) {
        BKF_LOG_INFO(BKF_LOG_HND, "typeId(%u), ng\n", typeId);
        return BKF_ERR;
    }
    vTbl = (BkfTlvCodecVTbl *)&g_BkfTlvCodecVTbl[typeId];
    if (!BKF_BIT_TEST(vTbl->flag, BKF_TLV_PART1_LEN_FIXED)) {
        BKF_LOG_INFO(BKF_LOG_HND, "type(%u, %s), part1 not fixed len, ng\n", typeId, BkfTlvGetTypeStr(typeId));
        return BKF_ERR;
    }
    if (!BKF_LEN_IS_ALIGN4(vTbl->valRealLenFixed) && vTbl->valRealLenFixed != 0) {
        BKF_LOG_INFO(BKF_LOG_HND, "lenInTlv(%u), not align4, ng\n", BKF_LEN_IS_ALIGN4(vTbl->valRealLenFixed));
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "msgId(%u, %s)/typeId(%u, %s), flag(%#x)/valLen(%u)/updMsgBodyLen(%u)\n",
        VOS_NTOHS(curMsgHead->msgId), BkfMsgGetIdStr(VOS_NTOHS(curMsgHead->msgId)), typeId, BkfTlvGetTypeStr(typeId),
        flag, valRealLen, updMsgBodyLen);

    if (coder->codedErr) {
        BKF_LOG_INFO(BKF_LOG_HND, "codedErr(%u)\n", coder->codedErr);
        return BKF_ERR;
    }
    if (valRealLen <= 0) {
        BKF_LOG_INFO(BKF_LOG_HND, "valLen(%d), ng\n", valRealLen);
        return BKF_ERR;
    }

    return BKF_OK;
}
uint32_t BkfMsgCodeAppendValLen(BkfMsgCoder *coder, uint8_t flag, int32_t valRealLen, BOOL updMsgBodyLen)
{
    uint32_t ret;
    int32_t leftLen;
    int32_t needLen;
    BkfTL *curTl = VOS_NULL;
    int32_t valRealTotalLen;

    ret = BkfMsgCodeAppendValLenChkParam(coder, flag, valRealLen, updMsgBodyLen);
    if (ret != BKF_OK) {
        goto error;
    }
    if (coder->curTlvHasAddRaw) {
        BKF_LOG_INFO(BKF_LOG_HND, "curTlvHasAddRaw(%u), one tlv can add raw once, ng\n", coder->curTlvHasAddRaw);
        goto error;
    }

    leftLen = BKF_MSG_CODE_GET_LEFT_LEN(coder);
    needLen = BKF_GET_ALIGN4_LEN(valRealLen);
    if (leftLen < needLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "leftLen(%d)/needLen(%d), send buf not enough & OK!\n", leftLen, needLen);
        goto error;
    }
    curTl = coder->curTl;
    BKF_BIT_SET(curTl->flag, flag);
    valRealTotalLen = (int32_t)(VOS_NTOHL(curTl->valLen)) + valRealLen;
    curTl->valLen = VOS_HTONL((uint32_t)valRealTotalLen);
    BkfMsgFillReservedForAlign(curTl + 1, valRealTotalLen);
    coder->curTlvHasAddRaw = VOS_TRUE;

    BkfMsgCodeTLVDoAfter(coder, updMsgBodyLen, needLen);
    return BKF_OK;

error:

    if (coder != VOS_NULL) {
        coder->codedErr = VOS_TRUE;
    }
    return BKF_ERR;
}

uint8_t *BkfMsgCodeGetLeft(BkfMsgCoder *coder, int32_t *leftLen)
{
    if ((coder == VOS_NULL) || coder->codedErr || (leftLen == VOS_NULL)) {
        goto error;
    }

    *leftLen = BKF_MSG_CODE_GET_LEFT_LEN(coder);
    if (*leftLen < 0) {
        BKF_ASSERT(0);
        BKF_LOG_ERROR(BKF_LOG_HND, "sendBufLen(%d)/validMsgLen(%d)/tempUsedLen(%d)\n", coder->sendBufLen,
            coder->validMsgLen, coder->tempUsedLen);
        goto error;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "sendBufLen(%d)/validMsgLen(%d)/tempUsedLen(%d)\n", coder->sendBufLen,
        coder->validMsgLen, coder->tempUsedLen);
    return (*leftLen > 0) ? BKF_MSG_CODE_GET_LEFT_BEGIN(coder) : VOS_NULL;

error:

    if (leftLen != VOS_NULL) {
        *leftLen = 0;
    }
    return VOS_NULL;
}

int32_t BkfMsgCodeGetValidMsgLen(BkfMsgCoder *coder)
{
    return (coder != VOS_NULL) ? coder->validMsgLen : 0;
}

#endif

#if BKF_BLOCK("公有函数定义, 解码")
#ifdef BKF_LOG_HND
#undef BKF_LOG_HND
#define BKF_LOG_HND ((decoder)->codecer.logOrNull)
#endif
#ifdef BKF_MOD_NAME
#undef BKF_MOD_NAME
#define BKF_MOD_NAME ((decoder)->codecer.name)
#endif

uint32_t BkfMsgDecodeAddSlice(BkfMsgDecoder *decoder, uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec,
    F_BKF_GET_STR sliceKeyGetStrOrNull)
{
    if (decoder == VOS_NULL) {
        return BKF_ERR;
    }
    decoder->codecer.sliceKeyLen = sliceKeyLen;
    decoder->codecer.sliceKeyCodec = sliceKeyCodec;
    decoder->codecer.sliceKeyGetStrOrNull = sliceKeyGetStrOrNull;
    return BKF_OK;
}

void BkfMsgDeCodeResetSliceKeyPara(BkfMsgDecoder *decoder, uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec,
    F_BKF_GET_STR sliceKeyGetStrOrNull)
{
    if (decoder == VOS_NULL) {
        return;
    }

    decoder->codecer.sliceKeyLen = sliceKeyLen;
    decoder->codecer.sliceKeyCodec = sliceKeyCodec;
    decoder->codecer.sliceKeyGetStrOrNull = sliceKeyGetStrOrNull;
}

/* func */
uint32_t BkfMsgDecodeInit(BkfMsgDecoder *decoder, const char *name, uint8_t *rcvData, int32_t dataLen,
    uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec, F_BKF_GET_STR sliceKeyGetStrOrNull, BkfLog *logOrNull)
{
    if ((decoder == VOS_NULL) || (name == VOS_NULL) || (rcvData == VOS_NULL) || (dataLen <= 0)) {
        return BKF_ERR;
    }

    (void)memset_s(decoder, sizeof(BkfMsgDecoder), 0, sizeof(BkfMsgDecoder));
    decoder->codecer.name = name;
    decoder->codecer.sliceKeyLen = sliceKeyLen;
    decoder->codecer.sliceKeyCodec = sliceKeyCodec;
    decoder->codecer.sliceKeyGetStrOrNull = sliceKeyGetStrOrNull;
    decoder->codecer.logOrNull = logOrNull;

    decoder->rcvData = rcvData;
    decoder->rcvDataLen = dataLen;
    decoder->decodedErr = VOS_FALSE;
    decoder->decodedLen = 0;
    decoder->curMsgHead = VOS_NULL;
    decoder->curMsgDecodedLen = 0;
    decoder->curTl = VOS_NULL;
    return BKF_OK;
}

STATIC uint32_t BkfMsgDecodeMsgHeadChkParam(BkfMsgDecoder *decoder, uint32_t *errCode)
{
    BkfMsgHead *curMsgHead = VOS_NULL;

    if ((decoder == VOS_NULL) || (decoder->rcvData == VOS_NULL) || (errCode == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "rcvData(%#x), decodedErr(%u), decodedLen(%d)/rcvDataLen(%d)\n",
        BKF_MASK_ADDR(decoder->rcvData), decoder->decodedErr, decoder->decodedLen, decoder->rcvDataLen);

    if (decoder->decodedErr) {
        BKF_LOG_WARN(BKF_LOG_HND, "decodedErr(%u)\n", decoder->decodedErr);
        return BKF_ERR;
    }
    if (decoder->decodedLen > decoder->rcvDataLen) {
        BKF_LOG_WARN(BKF_LOG_HND, "decodedLen(%d)/rcvDataLen(%d)\n", decoder->decodedLen, decoder->rcvDataLen);
        return BKF_ERR;
    }

    curMsgHead = decoder->curMsgHead;
    if (curMsgHead != VOS_NULL) {
        if (decoder->curMsgDecodedLen > (int32_t)curMsgHead->bodyLen) {
            BKF_LOG_WARN(BKF_LOG_HND, "curMsgDecodedLen(%d)/bodyLen(%d), cur msg decode error, ng\n",
                decoder->curMsgDecodedLen, curMsgHead->bodyLen);
            return BKF_ERR;
        } else if (decoder->curMsgDecodedLen < (int32_t)curMsgHead->bodyLen) {
            BKF_LOG_INFO(BKF_LOG_HND, "curMsgDecodedLen(%d)/bodyLen(%d), not decode some data & maybe error\n",
                decoder->curMsgDecodedLen, curMsgHead->bodyLen);
            /* 跳过msg没有解析过的tlv&继续 */
        }
    }

    return BKF_OK;
}
STATIC BkfMsgHead *BkfMsgDecodeMsgHeadGetCurMsgHead(BkfMsgDecoder *decoder, uint32_t *ret)
{
    BkfMsgHead *curMsgHead = VOS_NULL;
    BkfMsgHead peekCurMsgHead;
    int32_t needLen;
    int32_t leftLen;

    needLen = sizeof(BkfMsgHead);
    leftLen = decoder->rcvDataLen - decoder->decodedLen;
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), rcv data not enough, need more for msgHead\n", leftLen,
            needLen);
        *ret = BKF_OK;
        return VOS_NULL;
    }

    curMsgHead = (BkfMsgHead *)(decoder->rcvData + decoder->decodedLen);
    peekCurMsgHead = *curMsgHead;
    peekCurMsgHead.sign = VOS_NTOHL(peekCurMsgHead.sign);
    if (peekCurMsgHead.sign != BKF_MSG_SIGN) {
        BKF_LOG_ERROR(BKF_LOG_HND, "sign(%#x/%#x)\n", peekCurMsgHead.sign, BKF_MSG_SIGN);
        *ret = BKF_ERR;
        return VOS_NULL;
    }
    peekCurMsgHead.msgId = VOS_NTOHS(peekCurMsgHead.msgId);
    peekCurMsgHead.bodyLen = VOS_NTOHL(peekCurMsgHead.bodyLen);
    needLen += (int32_t)peekCurMsgHead.bodyLen;
    leftLen = decoder->rcvDataLen - decoder->decodedLen;
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), rcv data not enough, need more for msgBody\n", leftLen,
            needLen);
        *ret = BKF_OK;
        return VOS_NULL;
    }

    return curMsgHead;
}
BkfMsgHead *BkfMsgDecodeMsgHead(BkfMsgDecoder *decoder, uint32_t *errCode)
{
    uint32_t ret;
    BkfMsgHead *curMsgHead = VOS_NULL;

    ret = BkfMsgDecodeMsgHeadChkParam(decoder, errCode);
    if (ret != BKF_OK) {
        goto error;
    }
    if (decoder->decodedLen == decoder->rcvDataLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "decodedLen(%d)/rcvDataLen(%d)\n", decoder->decodedLen, decoder->rcvDataLen);
        *errCode = BKF_OK;
        return VOS_NULL;
    }

    curMsgHead = BkfMsgDecodeMsgHeadGetCurMsgHead(decoder, &ret);
    if (curMsgHead == VOS_NULL) {
        if (ret != BKF_OK) {
            goto error;
        }

        *errCode = BKF_OK;
        return VOS_NULL;
    }
    decoder->curMsgHead = curMsgHead;
    decoder->decodedLen += (int32_t)sizeof(BkfMsgHead);
    decoder->curMsgDecodedLen = 0;
    decoder->curTl = VOS_NULL;
    curMsgHead->msgId = VOS_NTOHS(curMsgHead->msgId);
    curMsgHead->bodyLen = VOS_NTOHL(curMsgHead->bodyLen);
    decoder->decodedLen += (int32_t)curMsgHead->bodyLen;

    BKF_LOG_DEBUG(BKF_LOG_HND, "msg(%u, %s)/flag(%#x)/bodyLen(%u)\n", curMsgHead->msgId,
        BkfMsgGetIdStr(curMsgHead->msgId), curMsgHead->flag, curMsgHead->bodyLen);
    *errCode = BKF_OK;
    return curMsgHead;

error:

    if (decoder != VOS_NULL) {
        decoder->decodedErr = VOS_TRUE;
    }
    if (errCode != VOS_NULL) {
        *errCode = ret;
    }
    return VOS_NULL;
}

uint8_t *BkfMsgDecodeGetLeft(BkfMsgDecoder *decoder, int32_t *leftLen)
{
    if ((decoder == VOS_NULL) || (leftLen == VOS_NULL)) {
        goto error;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "decodedErr(%u), rcvData(%d)/decodedLen(%d)\n", decoder->decodedErr, decoder->rcvDataLen,
        decoder->decodedLen);

    if (decoder->decodedErr) {
        goto error;
    }
    if ((decoder->rcvDataLen < decoder->decodedLen) || (decoder->decodedLen < 0)) {
        BKF_ASSERT(0);
        BKF_LOG_ERROR(BKF_LOG_HND, "rcvDataLen(%d)/decodedLen(%d)\n", decoder->rcvDataLen, decoder->decodedLen);
        goto error;
    }

    *leftLen = decoder->rcvDataLen - decoder->decodedLen;
    return (*leftLen > 0) ? (decoder->rcvData + decoder->decodedLen) : VOS_NULL;

error:

    if (leftLen != VOS_NULL) {
        *leftLen = 0;
    }
    return VOS_NULL;
}

STATIC uint32_t BkfMsgDecodeTLVChkParam(BkfMsgDecoder *decoder, uint32_t *errCode)
{
    BkfMsgHead *curMsgHead = VOS_NULL;

    if ((decoder == VOS_NULL) || (decoder->curMsgHead == VOS_NULL) || (errCode == VOS_NULL)) {
        return BKF_ERR;
    }
    curMsgHead = decoder->curMsgHead;
    BKF_LOG_DEBUG(BKF_LOG_HND, "msg(%u, %s)/bodyLen(%u), curMsgDecodedLen(%d)\n", curMsgHead->msgId,
        BkfMsgGetIdStr(curMsgHead->msgId), curMsgHead->bodyLen, decoder->curMsgDecodedLen);

    if (decoder->decodedErr) {
        BKF_LOG_WARN(BKF_LOG_HND, "decodedErr(%u)\n", decoder->decodedErr);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC uint32_t BkfMsgDecodeTLVCurTl(BkfMsgDecoder *decoder, BkfTL *tl)
{
    BkfTlvCodecVTbl *vTbl = VOS_NULL;

    if (!BKF_TLV_IS_VALID(tl->typeId)) {
        BKF_LOG_INFO(BKF_LOG_HND, "typeId(%u), ng\n", tl->typeId);
        return BKF_ERR;
    }

    vTbl = (BkfTlvCodecVTbl *)&g_BkfTlvCodecVTbl[tl->typeId];
    /*
    1. 如果定长后没有append， 那接收长度等于valRealLenFixed。
    2. 否则，接收的长度大于valRealLenFixed。
    */
    if (BKF_BIT_TEST(vTbl->flag, BKF_TLV_PART1_LEN_FIXED) &&
        (BKF_GET_ALIGN4_LEN(tl->valLen) < BKF_GET_ALIGN4_LEN(vTbl->valRealLenFixed))) {
        BKF_LOG_INFO(BKF_LOG_HND, "valLen(%d)/(%d), ng\n", tl->valLen, vTbl->valRealLenFixed);
        return BKF_ERR;
    }
    if (BKF_BIT_TEST(vTbl->flag, BKF_TLV_PART1_LEN_RANGE) &&
        (((int32_t)tl->valLen < vTbl->valRealLenMin) || ((int32_t)tl->valLen > vTbl->valRealLenMax))) {
        BKF_LOG_INFO(BKF_LOG_HND, "valLen(%d)/[%d, %d], ng\n", tl->valLen, vTbl->valRealLenMin, vTbl->valRealLenMax);
        return BKF_ERR;
    }

    return vTbl->codec(&decoder->codecer, tl + 1, tl->valLen);
}
STATIC void BkfMsgDecodeTLVProcError(BkfMsgDecoder *decoder, uint32_t *errCode, uint32_t ret)
{
    if (decoder != VOS_NULL) {
        decoder->decodedErr = VOS_TRUE;
    }
    if (errCode != VOS_NULL) {
        *errCode = ret;
    }
}
BkfTL *BkfMsgDecodeTLV(BkfMsgDecoder *decoder, uint32_t *errCode)
{
    uint32_t ret;
    int32_t needLen;
    int32_t leftLen;
    BkfMsgHead *curMsgHead = VOS_NULL;
    BkfTL *curTl = VOS_NULL;

    ret = BkfMsgDecodeTLVChkParam(decoder, errCode);
    if (ret != BKF_OK) {
        goto error;
    }

    curMsgHead = decoder->curMsgHead;
    if (decoder->curMsgDecodedLen == (int32_t)curMsgHead->bodyLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "curMsgDecodedLen(%d)/bodyLen(%u)\n", decoder->curMsgDecodedLen,
            curMsgHead->bodyLen);
        *errCode = BKF_OK;
        return VOS_NULL;
    }

    needLen = sizeof(BkfTL);
    leftLen = (int32_t)(curMsgHead->bodyLen) - decoder->curMsgDecodedLen;
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), rcv buf not enough & ng\n", leftLen, needLen);
        ret = BKF_ERR;
        goto error;
    }
    curTl = (BkfTL *)((uint8_t *)(curMsgHead + 1) + decoder->curMsgDecodedLen);
    decoder->curTl = curTl;
    decoder->curMsgDecodedLen += needLen;

    curTl->typeId = VOS_NTOHS(curTl->typeId);
    curTl->valLen = VOS_NTOHL(curTl->valLen);
    needLen = (int32_t)BKF_GET_ALIGN4_LEN(curTl->valLen);
    BKF_LOG_DEBUG(BKF_LOG_HND, "tlvType %u, len %u\n", curTl->typeId, curTl->valLen);
    if (leftLen < needLen) {
        BKF_LOG_INFO(BKF_LOG_HND, "leftLen(%d)/needLen(%d), rcv buf not enough & ng\n", leftLen, needLen);
        ret = BKF_ERR;
        goto error;
    }
    decoder->curMsgDecodedLen += needLen;

    ret = BkfMsgDecodeTLVCurTl(decoder, curTl);
    if (ret != BKF_OK) {
        goto error;
    }

    *errCode = BKF_OK;
    return curTl;

error:

    BkfMsgDecodeTLVProcError(decoder, errCode, ret);
    return VOS_NULL;
}
#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
