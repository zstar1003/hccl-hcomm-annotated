/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_CONN_H
#define BKF_PUBER_CONN_H

#include "bkf_puber_adef.h"
#include "bkf_puber_table_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
typedef struct TagBkfPuberConnMng BkfPuberConnMng;
#pragma pack()

BkfPuberConnMng *BkfPuberConnInit(BkfPuberInitArg *arg, BkfPuberTableTypeMng *tableTypeMng);
void BkfPuberConnUninit(BkfPuberConnMng *connMng);
uint32_t BkfPuberConnEnable(BkfPuberConnMng *connMng);
uint32_t BkfPuberConnSetSelfUrl(BkfPuberConnMng *connMng, BkfUrl *url);
void BkfPuberConnUnsetSelfUrl(BkfPuberConnMng *connMng, BkfUrl *url);
void BkfPuberConnDispSummary(BkfPuberConnMng *connMng, BkfDisp *disp);

#ifdef __cplusplus
}
#endif

#endif
