/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_SESS_UTILS_H
#define BKF_PUBER_SESS_UTILS_H

#include "bkf_puber_sess.h"
#include "bkf_puber_sess_data.h"

#ifdef __cplusplus
extern "C" {
#endif

void BkfPuberSessChgStateAndTrigSched(BkfPuberSess *sess, uint8_t newState);
void BkfPuberSessChgStateAndTrigSlowSched(BkfPuberSess *sess, uint8_t newState);
void BkfPuberSessNtfAppSub(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId);
void BkfPuberSessNtfAppUnsub(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId);
int32_t BkfPuberSessNtfAppCode(BkfPuberSess *sess, BkfDcTupleInfo *tupleInfo, void *buf, int32_t bufLen);
void BkfPuberSessNtfAppBatchTimeout(BkfPuberSessMng *sessMng, uint8_t *sliceKey, uint16_t tableTypeId);
BkfTlvTransNum *BkfPuberSessParseTransNum(BkfMsgDecoder *decoder);
uint32_t BkfPuberSessPackSubAck(BkfPuberSess *sess, BkfMsgCoder *coder);
uint32_t BkfPuberSessPackBatchBegin(BkfPuberSess *sess, BkfMsgCoder *coder);
uint32_t BkfPuberSessPackBatchEnd(BkfPuberSess *sess, BkfMsgCoder *coder);
uint32_t BkfPuberSessPackSubDelNtf(BkfPuberSess *sess, BkfMsgCoder *coder);
uint32_t BkfPuberSessPackDataHead(BkfPuberSess *sess, BkfMsgCoder *coder);
#ifdef __cplusplus
}
#endif

#endif
