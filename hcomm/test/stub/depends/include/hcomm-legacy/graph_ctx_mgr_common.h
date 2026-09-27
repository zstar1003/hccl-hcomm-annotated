/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_GRAPH_CTX_MGR_COMMON_H
#define HCOMM_GRAPH_CTX_MGR_COMMON_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern void* GraphMgrInit();
extern void GraphMgrDeInit(void* fftsPubInfo);
extern void* GetGraphCtx(void* fftsPubInfo, const char* key, uint32_t keyLen);
extern void* GetGraphCtxV2(void* fftsPubInfo, const char* key, uint32_t keyLen);
extern HcclResult LaunchGraph(void* fftsPubInfo, void* streamPtr, void* ctx, uint32_t timeout, uint32_t* ctxNum);
extern void GraphDump(void* fftsPubInfo, void* ctx);
extern HcclResult
GraphAddRecordTask(void* fftsPubInfo, void* ctx, uint32_t streamId, void* signal, bool inchip, uint32_t* ctxIdx);
extern HcclResult
GraphAddWaitTask(void* fftsPubInfo, void* ctx, uint32_t streamId, void* signal, bool inchip, uint32_t* ctxIdx);
extern HcclResult GraphAddMemcpyTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, void* dstAddr, void* srcAddr, uint64_t size, uint32_t* ctxIdx);
extern HcclResult GraphAddReduceTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, void* dstAddr, const void* srcAddr, uint64_t dataCount,
    const HcclDataType datatype, HcclReduceOp redOp, uint32_t* ctxIdx);
extern HcclResult GraphAddInlineReduceTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, void* dstAddr, const void* srcAddr, uint64_t dataCount,
    const HcclDataType datatype, HcclReduceOp redOp, uint32_t* ctxIdx);
extern HcclResult GraphAddRdmaSendTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, u32 dbindex, u64 dbinfo, bool isCapture, uint32_t* ctxIdx);
extern HcclResult GraphAddVectorReduceTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, int count, void* addrListDevMemPtr, void* funcAddr,
    uint32_t numBlocks, uint32_t* ctxIdx);
extern HcclResult GraphAddTailVectorReduceTask(
    void* fftsPubInfo, void* ctx, uint32_t streamId, void* dst, const void* src, u64 cnt, uint32_t* ctxIdx);
extern HcclResult GraphAddVectorReduceArgs(void* fftsPubInfo, void* argsHandle);
extern HcclResult GraphAddRecordTaskById(void* fftsPubInfo, void* ctx, uint32_t streamId, u32 notifyID);
extern HcclResult GraphAddWaitTaskById(void* fftsPubInfo, void* ctx, uint32_t streamId, u32 notifyID);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_GRAPH_CTX_MGR_COMMON_H
