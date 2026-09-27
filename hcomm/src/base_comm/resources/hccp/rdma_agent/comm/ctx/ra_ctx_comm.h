/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_CTX_COMM_H
#define RA_CTX_COMM_H

#include "hccp_ctx.h"
#include "ra_ctx.h"
#include "ra_rs_ctx.h"

int RaCtxPrepareLmemRegister(struct MrRegInfoT *lmemInfo, struct MemRegAttrT *memAttr);

void RaCtxGetLmemInfo(struct MemRegInfoT *memInfo, struct MrRegInfoT *lmemInfo, struct RaLmemHandle *lmemHandle);

void RaCtxPrepareRmemImport(struct MrImportInfoT *rmemInfo, struct MemImportAttrT *memAttr);

void RaCtxPrepareCqCreate(struct CqInfoT *info, struct CtxCqAttr *cqAttr);

void RaCtxGetCqCreateInfo(struct CtxCqInfo *cqInfo, struct CqInfoT *info);

int RaCtxPrepareQpCreate(struct QpCreateAttr *qpAttr, struct CtxQpAttr *ctxQpAttr);

void RaCtxGetQpCreateInfo(struct RaCtxHandle *ctxHandle, struct QpCreateAttr *qpAttr, struct QpCreateInfo *qpInfo,
    struct RaCtxQpHandle *qpHandle);

void RaCtxPrepareQpImport(struct QpImportInfoT *qpInfo, struct RaRsJettyImportAttr *importAttr);

void RaCtxGetQpImportInfo(struct RaCtxHandle *ctxHandle, struct QpImportInfoT *qpInfo,
    struct RaRsJettyImportInfo *importInfo, struct RaCtxRemQpHandle *qpHandle);

#endif // RA_CTX_COMM_H
