/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_TLV_H
#define RA_TLV_H

#include "hccp_tlv.h"

struct RaTlvHandle {
    struct RaTlvOps *tlvOps;
    struct TlvInitInfo initInfo;
    unsigned int bufferSize;
    pthread_mutex_t mutex;
};

struct RaTlvOps {
    int (*raTlvInit)(struct RaTlvHandle *tlvHandle);
    int (*raTlvDeinit)(struct RaTlvHandle *tlvHandle);
    int (*raTlvRequest)(struct RaTlvHandle *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg,
        struct TlvMsg *recvMsg);
};
#endif // RA_TLV_H
