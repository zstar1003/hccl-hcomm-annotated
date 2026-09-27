/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_CTX_DFX_H
#define HCCP_CTX_DFX_H

#include <stdint.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CONTEXT_MAX_LEN 512U
#define ASYNC_EVENT_MAX_NUM 4U

struct AsyncEvent {
    uint32_t resId;
    uint32_t eventType;
    uint8_t context[CONTEXT_MAX_LEN];
    unsigned int len;
};

enum HccpAuxInfoInType {
    AUX_INFO_IN_TYPE_CQE = 0,
    AUX_INFO_IN_TYPE_AE = 1,
    AUX_INFO_IN_TYPE_MAX,
};

struct HccpAuxInfoIn {
    enum HccpAuxInfoInType type;
    union {
        struct {
            uint32_t status;
            uint8_t sR;
        } cqe;
        struct {
            uint32_t eventType;
        } ae;
    };
    uint8_t resv[7U];
};

#define AUX_INFO_NUM_MAX 256U

struct HccpAuxInfoOut {
    uint32_t auxInfoType[AUX_INFO_NUM_MAX];
    uint32_t auxInfoValue[AUX_INFO_NUM_MAX];
    uint32_t auxInfoNum;
};

#define CR_ERR_INFO_MAX_NUM 96U

struct CrErrInfo {
    uint32_t status;
    uint32_t jettyId;
    struct timeval time;
    uint32_t resv[2U];
};

#ifdef __cplusplus
}
#endif

#endif // HCCP_CTX_DFX_H
