/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_UB_JETTY_H
#define RS_UB_JETTY_H

#include "urma_types.h"
#include "udma_u_ctl.h"
#include "dl_hal_function.h"
#include "rs_ctx_inner.h"

#define WQEBB_NUM_PER_SQE 4ULL
#define ALIGN_DOWN(x, a) ((x) & (~((a) - 1)))

static inline bool RsIsCcuJetty(int jettyMode)
{
    return (jettyMode == JETTY_MODE_CCU) || (jettyMode == JETTY_MODE_CCU_TA_CACHE);
}

void RsUbCtxExtJettyCreate(struct RsCtxJettyCb *jettyCb, urma_jetty_cfg_t *jettyCfg);
void RsUbCtxExtJettyDelete(struct RsCtxJettyCb *jettyCb);
void RsUbVaMunmapBatch(struct RsCtxJettyCb **jettyCbArr, unsigned int num);
void RsUbFreeJettyIdBatch(struct RsCtxJettyCb **jettyCbArr, unsigned int num);
#endif // RS_UB_JETTY_H
