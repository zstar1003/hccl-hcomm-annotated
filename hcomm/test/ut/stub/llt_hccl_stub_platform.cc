/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "hccl_tbe_task.h"
#include "graph_ctx_mgr_common.h"

HcclResult HcclTbeTaskInit(int32_t deviceLogicId) { return HCCL_SUCCESS; }

HcclResult HcclTbeTaskDeInit(int32_t deviceLogicId) { return HCCL_SUCCESS; }

HcclResult
HcclTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, aclrtStream stream, int32_t deviceLogicId)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetVectorBlockSize(uint32_t* blockSize, int32_t deviceLogicId) { return HCCL_SUCCESS; }

HcclResult HcclTbeReduce(
    const TbeReduceParam* param, aclrtStream stream, void* overflowAddrs[], uint32_t overflowCount,
    int32_t deviceLogicId)
{
    return HCCL_SUCCESS;
}

HcclResult HcclTbeReduceGenArgs(
    const TbeReduceParam* param, aclrtStream stream, void* overflowAddrs[], uint32_t overflowCount, TbeReduceArg* args,
    int32_t deviceLogicId)
{
    return HCCL_SUCCESS;
}

void* GraphMgrInit()
{
    void* graphMgr = new u64(1);
    return graphMgr;
}

void GraphMgrDeInit(void* graphMgr)
{
    if (graphMgr != nullptr) {
        delete static_cast<u64*>(graphMgr);
    }
}

void* GetGraphCtx(void* graphMgr, const char* key, uint32_t keyLen) {}

void* GetGraphCtxV2(void* graphMgr, const char* key, uint32_t keyLen) {}

HcclResult LaunchGraph(void* graphMgr, void* streamPtr, void* ctx, uint32_t timeout, uint32_t* ctxNum)
{
    return HCCL_SUCCESS;
}

HcclResult LaunchGraphAndGetGraphInfo(
    void* fftsPubInfo, void* streamPtr, void* ctx, uint32_t timeout, uint32_t* ctxNum, void** descBuf,
    size_t* descBufLen)
{
    return HCCL_SUCCESS;
}

void GraphDump(void* graphMgr, void* ctx) {}

HcclResult GraphAddRecordTask(void* graphMgr, void* ctx, uint32_t streamId, void* signal, bool inchip, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddWaitTask(void* graphMgr, void* ctx, uint32_t streamId, void* signal, bool inchip, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddMemcpyTask(
    void* graphMgr, void* ctx, uint32_t streamId, void* dstAddr, void* srcAddr, uint64_t size, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddReduceTask(
    void* graphMgr, void* ctx, uint32_t streamId, void* dstAddr, const void* srcAddr, uint64_t dataCount,
    const HcclDataType datatype, HcclReduceOp redOp, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddInlineReduceTask(
    void* graphMgr, void* ctx, uint32_t streamId, void* dstAddr, const void* srcAddr, uint64_t dataCount,
    const HcclDataType datatype, HcclReduceOp redOp, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddRdmaSendTask(
    void* graphMgr, void* ctx, uint32_t streamId, u32 dbindex, u64 dbinfo, bool isCapture, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddVectorReduceTask(
    void* graphMgr, void* ctx, uint32_t streamId, int count, void* addrListDevMemPtr, void* funcAddr,
    uint32_t numBlocks, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddTailVectorReduceTask(
    void* graphMgr, void* ctx, uint32_t streamId, void* dst, const void* src, u64 cnt, uint32_t* ctxIdx)
{
    return HCCL_SUCCESS;
}

HcclResult GraphAddVectorReduceArgs(void* graphMgr, void* argsHandle) { return HCCL_SUCCESS; }

HcclResult GraphAddRecordTaskById(void* graphMgr, void* ctx, uint32_t streamId, u32 notifyID) { return HCCL_SUCCESS; }

HcclResult GraphAddWaitTaskById(void* graphMgr, void* ctx, uint32_t streamId, u32 notifyID) { return HCCL_SUCCESS; }
