/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_MSG_WRAP_H
#define BKF_MSG_WRAP_H

#include "bkf_msg.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

typedef struct tagBkfWrapMsgHello {
    BkfTlvProjectName *projectName;
    BkfTlvProjectVersion *projectVersion;
    BkfTlvServiceName *serviceName;
    BkfTlvSuberName *suberName;
    BkfTlvIdlVersion *idlVersion;
    BkfTlvSuberLsnUrl *suberLsnUrl;
} BkfWrapMsgHello;

typedef struct tagBkfWrapMsgHelloAck {
    BkfTlvReasonCode *reasonCode;
    BkfTlvProjectName *puberName;
    BkfTlvIdlVersion *puberIdlVersion;
} BkfWrapMsgHelloAck;

typedef struct tagBkfWrapMsgSub {
    BkfTlvSliceKey *sliceKey;
    BkfTlvTableType *tableTypeId;
    BkfTlvTransNum *transNum;
} BkfWrapMsgSub;

typedef struct tagBkfWrapMsgSubAck {
    BkfTlvSliceKey *sliceKey;
    BkfTlvTableType *tableTypeId;
    BkfTlvTransNum *transNum;
    BkfTlvReasonCode *reasonCode;
} BkfWrapMsgSubAck;

typedef BkfWrapMsgSub BkfWrapMsgUnsub;
typedef BkfWrapMsgSub BkfWrapMsgBatchBegin;
typedef BkfWrapMsgSub BkfWrapMsgBatchEnd;
typedef BkfWrapMsgSub BkfWrapMsgVerifyBegin;
typedef BkfWrapMsgSub BkfWrapMsgVerifyEnd;
typedef BkfWrapMsgSub BkfWrapMsgDelSubNtf;

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
