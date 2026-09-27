/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_CONN_H
#define BKF_SUBER_CONN_H

#include "bkf_suber_env.h"
#include "bkf_suber_data.h"
#include "bkf_url.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

#define BKF_SUBER_CONN_NEED_DELETE 100
typedef struct tagBkfSuberConnMng BkfSuberConnMng;
typedef struct tagBkfSuberConnMngInitArg {
    BkfSuberEnv *env;
} BkfSuberConnMngInitArg;
#pragma pack()
BkfSuberConnMng *BkfSuberConnMngInit(BkfSuberConnMngInitArg *initArg);
void BkfSuberConnMngUnInit(BkfSuberConnMng *connMng);
uint32_t BkfSuberConnMngSetSelfUrl(BkfSuberConnMng *connMng, BkfUrl *selfUrl);
uint32_t BkfSuberConnEnable(BkfSuberConnMng *connMng);
void BkfSuberConnProcInstAddUrl(BkfSuberConnMng *connMng, BkfSuberInst *inst);
void BkfSuberConnProcInstDelUrl(BkfSuberConnMng *connMng, BkfSuberInst *inst);
void BkfSuberConnProcSliceAddInst(BkfSuberConnMng *connMng, BkfSuberSlice *slice, uint64_t instId);
void BkfSuberConnProcSliceDelInst(BkfSuberConnMng *connMng, BkfSuberSlice *slice, uint64_t instId);
void BkfSuberConnProcAddSub(BkfSuberConnMng *connMng, BkfSuberAppSub *appSub);
void BkfSuberConnProcDelSub(BkfSuberConnMng *connMng, BkfSuberAppSub *appSub);
#ifdef __cplusplus
}
#endif

#endif
