/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_MSG_CODEC_H
#define BKF_MSG_CODEC_H

#include "bkf_msg.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_log.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
/* pack
1. 打包三种模式
   1) 定长
   2) 变长
   3) 定长(长度自然对齐) + 业务自己添加后更新长度
2. 每次编码，消息头默认是完整的，但消息下的tlv。
   即多个tlv，或者一个tlv的多个部分编码后，才完整，更新到msg中生效。通过updMsgBodyLen来控制。
3. 长度解释:
   1) sendBufLen: 发送缓冲的长度
   2) validMsgLen: 通过updMsgBodyLen确认过的消息的有效长度，即可以发送的长度
   3) tempUsedLen: 还没有计入msg的，临时使用的长度
   usedLen = validMsgLen + tempUsedLen
*/
/* common */
typedef struct tagBkfMsgCodecer {
    const char *name;
    uint16_t sliceKeyLen;
    uint8_t pad1[2];
    F_BKF_DO sliceKeyCodec;
    F_BKF_GET_STR sliceKeyGetStrOrNull;
    BkfLog *logOrNull;
} BkfMsgCodecer;

/* coder */
typedef struct tagBkfMsgCoder {
    BkfMsgCodecer codecer;

    uint8_t *sendBuf;
    int32_t sendBufLen;

    BOOL codedErr;
    int32_t validMsgLen;
    BkfMsgHead *curMsgHead;
    int32_t tempUsedLen;
    BkfTL *curTl;
    BOOL curTlvHasAddRaw;
} BkfMsgCoder;
#define BKF_MSG_CODE_GET_USED_LEN(coder) ((coder)->validMsgLen + (coder)->tempUsedLen)
#define BKF_MSG_CODE_GET_LEFT_BEGIN(coder) ((coder)->sendBuf + BKF_MSG_CODE_GET_USED_LEN(coder))
#define BKF_MSG_CODE_GET_LEFT_LEN(coder) ((coder)->sendBufLen - BKF_MSG_CODE_GET_USED_LEN(coder))

/* decoder */
typedef struct tagBkfMsgDecoder {
    BkfMsgCodecer codecer;

    uint8_t *rcvData;
    int32_t rcvDataLen;

    BOOL decodedErr;
    int32_t decodedLen;
    BkfMsgHead *curMsgHead;
    int32_t curMsgDecodedLen;
    BkfTL *curTl;
} BkfMsgDecoder;

/* func */
uint32_t BkfMsgCodeInit(BkfMsgCoder *coder, const char *name, uint8_t *sendBuf, int32_t sendBufLen,
    uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec, F_BKF_GET_STR sliceKeyGetStrOrNull, BkfLog *logOrNull);
void BkfMsgCodeResetSliceKeyPara(BkfMsgCoder *coder, uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec,
    F_BKF_GET_STR sliceKeyGetStrOrNull);
uint32_t BkfMsgCodeMsgHead(BkfMsgCoder *coder, uint16_t msgId, uint8_t flag);
uint32_t BkfMsgCodeTLV(BkfMsgCoder *coder, uint16_t typeId, uint8_t flag, void *val, BOOL updMsgBodyLen);
uint32_t BkfMsgCodeRawTLV(BkfMsgCoder *coder, uint16_t typeId, uint8_t flag, void *val, int32_t valRealLen,
    BOOL updMsgBodyLen);
uint32_t BkfMsgCodeAppendValLen(BkfMsgCoder *coder, uint8_t flag, int32_t valRealLen, BOOL updMsgBodyLen);
uint8_t *BkfMsgCodeGetLeft(BkfMsgCoder *coder, int32_t *leftLen);
int32_t BkfMsgCodeGetValidMsgLen(BkfMsgCoder *coder);

uint32_t BkfMsgDecodeInit(BkfMsgDecoder *decoder, const char *name, uint8_t *rcvData, int32_t dataLen,
    uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec, F_BKF_GET_STR sliceKeyGetStrOrNull, BkfLog *logOrNull);
void BkfMsgDeCodeResetSliceKeyPara(BkfMsgDecoder *decoder, uint16_t sliceKeyLen, F_BKF_DO sliceKeyCodec,
    F_BKF_GET_STR sliceKeyGetStrOrNull);
BkfMsgHead *BkfMsgDecodeMsgHead(BkfMsgDecoder *decoder, uint32_t *errCode);
uint8_t *BkfMsgDecodeGetLeft(BkfMsgDecoder *decoder, int32_t *leftLen);
BkfTL *BkfMsgDecodeTLV(BkfMsgDecoder *decoder, uint32_t *errCode);

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
