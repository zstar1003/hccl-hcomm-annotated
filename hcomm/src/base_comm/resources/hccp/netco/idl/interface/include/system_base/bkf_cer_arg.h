/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CER_ARG__H
#define BKF_CER_ARG__H

#include "bkf_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
enum {
    BKF_CERA_TYPE_INVALID,
    BKF_CERA_TYPE_V8,

    BKF_CERA_TYPE_CNT
};
#define BKF_CERA_TYPE_IS_VALID(type) (((type) > BKF_CERA_TYPE_INVALID) && ((type) < BKF_CERA_TYPE_CNT))

typedef struct tagBkfCeraV8Key {
    uint32_t policyId;
} BkfCeraV8Key;
typedef struct tagBkfCeraV8Val {
    uint32_t vrId;
    int32_t role;
    int32_t verifyMod;
} BkfCeraV8Val;
typedef struct tagBkfCeraV8 {
    BkfCeraV8Key key;
    BkfCeraV8Val val;
} BkfCeraV8;

typedef struct tagBkfCera {
    uint8_t sign[2];
    uint8_t type;
    uint8_t pad1[1];
    union {
        BkfCeraV8 v8;
    };
} BkfCera;

uint32_t BkfCeraInit(BkfCera *cera);
uint32_t BkfCeraV8Set(BkfCera *cera, uint32_t policyId, uint32_t vrId, int32_t role, int32_t verifyMod);
BOOL BkfCeraIsValid(BkfCera *cera);
int32_t BkfCeraCmp(BkfCera *cera1Input, BkfCera *cera2InDs);
char *BkfCeraGetStr(BkfCera *cera, uint8_t *buf, int32_t bufLen);
uint32_t BkfCeraH2N(BkfCera *ceraH, BkfCera *ceraN);
uint32_t BkfCeraN2H(BkfCera *ceraN, BkfCera *ceraH);

#define BKF_CERA_IS_EQUAL(cera1, cera2) (BkfCeraCmp((cera1), (cera2)) == 0)

#ifdef __cplusplus
}
#endif

#endif
