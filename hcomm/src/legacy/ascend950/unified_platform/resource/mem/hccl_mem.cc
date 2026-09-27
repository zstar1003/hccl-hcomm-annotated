/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem.h"
#include "hccl_mem_v2.h"

HcclResult HcclMemReg(HcclNetDev netDev, const HcclMem* mem, HcclBuf* buf) { return HcclMemRegV2(netDev, mem, buf); }

HcclResult HcclMemDereg(const HcclBuf* buf) { return HcclMemDeregV2(buf); }

HcclResult HcclMemExport(HcclBuf* buf, char** outDesc, uint64_t* outDescLen)
{
    return HcclMemExportV2(buf, outDesc, outDescLen);
}

HcclResult
HcclMemImport(const char* description, uint32_t descLen, bool isRemote, HcclBuf* outBuf, HcclNetDevCtx netDevCtx)
{
    return HcclMemImportV2(description, descLen, isRemote, outBuf, netDevCtx);
}

HcclResult HcclMemClose(HcclBuf* buf) { return HcclMemCloseV2(buf); }
