/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "types.h"
#include "op_type.h"
#include "data_type.h"
#include "reduce_op.h"
#include "dev_type.h"
#include "env_config_v2.h"
#include <vector>
#include "context/st_ctx.h"
#include "hccl_st_test_case.h"
#include "rts_stub/rts_stub.h"

void PrepareCtxForAllReduceFP32(ThreadContext* ctx)
{
    auto* sendBuf = (float*)ctx->sendBuf;
    auto* expectBuf = (float*)ctx->expectedResBuf;

    auto count = ctx->situation.GetCount();

    float value = 3;
    auto rankSize = ctx->situation.GetRankSize();

    for (auto idx = 0; idx < count; idx++) {
        *sendBuf = value;
        if (ctx->situation.GetReduceOp() == ReduceOp::SUM) {
            *expectBuf = value * (float)rankSize;
        } else if (ctx->situation.GetReduceOp() == ReduceOp::MIN || ctx->situation.GetReduceOp() == ReduceOp::MAX) {
            *expectBuf = value;
        } else { // ReduceOp::PROD 乘积
            *expectBuf = value;
            for (auto i = 1; i < rankSize; i++) {
                *expectBuf *= value;
            }
        }
        sendBuf++;
        expectBuf++;
    }
}

void PrepareCtxForAllReduce(ThreadContext* ctx)
{
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    void* expectedResBuf = nullptr;

    auto dataTypeSize = Hccl::DataTypeSizeGet(ctx->situation.GetDataType());
    auto count = ctx->situation.GetCount();

    uint64_t memSize = (u64)dataTypeSize * (u64)count;
    constexpr int policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    aclrtMallocWithCfg(&sendBuf, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    aclrtMallocWithCfg(&recvBuf, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    aclrtMallocWithCfg(&expectedResBuf, memSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    ctx->sendBuf = sendBuf;
    ctx->recvBuf = recvBuf;
    ctx->expectedResBuf = expectedResBuf;

    switch (ctx->situation.GetDataType()) {
        case DataType::FP32:
            return PrepareCtxForAllReduceFP32(ctx);
        default:
            std::cout << "ST only support Allreduce FP32 now" << std::endl;
            throw std::exception();
    }
}

bool VerifyAllreduceFp32(ThreadContext* ctx)
{
    auto* recvBuf = (float*)ctx->recvBuf;
    auto* expectBuf = (float*)ctx->expectedResBuf;

    auto count = ctx->situation.GetCount();

    for (auto idx = 0; idx < count; idx++) {
        float f1 = *recvBuf;
        float f2 = *expectBuf;
        if (abs(f1 - f2) > 0.000001) {
            return false;
        }
        recvBuf++;
        expectBuf++;
    }
    return true;
}

bool VerifyAllreduce(ThreadContext* ctx)
{
    switch (ctx->situation.GetDataType()) {
        case DataType::FP32:
            return VerifyAllreduceFp32(ctx);
        default:
            return false;
    }
}

void PrepareCtxForAllgatherInt8(ThreadContext* ctx)
{
    auto* sendBuf = (char*)ctx->sendBuf;
    auto* expectBuf = (char*)ctx->expectedResBuf;
    auto count = ctx->situation.GetCount();

    char value = 3;
    auto rankSize = ctx->situation.GetRankSize();

    for (auto idx = 0; idx < count; idx++) {
        *sendBuf = value;
        sendBuf++;
    }

    for (auto idx = 0; idx < count * rankSize; idx++) {
        *expectBuf = value;
        expectBuf++;
    }
}

void PrepareCtxForAllgather(ThreadContext* ctx)
{
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    void* expectedResBuf = nullptr;

    auto dataTypeSize = Hccl::DataTypeSizeGet(ctx->situation.GetDataType());
    auto count = ctx->situation.GetCount();

    uint64_t sendMemSize = (u64)dataTypeSize * (u64)count;
    uint64_t recvMemSize = (u64)dataTypeSize * (u64)count * (u64)ctx->situation.GetRankSize();

    constexpr int policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    aclrtMallocWithCfg(&sendBuf, sendMemSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    aclrtMallocWithCfg(&recvBuf, recvMemSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    aclrtMallocWithCfg(&expectedResBuf, recvMemSize, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    ctx->sendBuf = sendBuf;
    ctx->recvBuf = recvBuf;
    ctx->expectedResBuf = expectedResBuf;

    switch (ctx->situation.GetDataType()) {
        case DataType::INT8:
            return PrepareCtxForAllgatherInt8(ctx);
        default:
            std::cout << "ST only support Allgather int8 now" << std::endl;
            throw std::exception();
    }
}

bool VerifyAllgather(ThreadContext* ctx)
{
    auto dataTypeSize = Hccl::DataTypeSizeGet(ctx->situation.GetDataType());
    auto count = ctx->situation.GetCount();
    uint64_t recvMemSize = (u64)dataTypeSize * (u64)count * (u64)ctx->situation.GetRankSize();
    if (memcmp(ctx->recvBuf, ctx->expectedResBuf, recvMemSize) == 0) {
        return true;
    }
    return false;
}

bool VerifyCtx(ThreadContext* ctx)
{
    switch (ctx->situation.GetOpType()) {
        case OpType::ALLREDUCE:
            return VerifyAllreduce(ctx);
        case OpType::ALLGATHER:
            return VerifyAllgather(ctx);
        default:
            return false;
    }
}

void PrepareCtx(ThreadContext* ctx)
{
    OpType opType = ctx->situation.GetOpType();
    switch (opType) {
        case OpType::ALLREDUCE:
            return PrepareCtxForAllReduce(ctx);
        case OpType::ALLGATHER:
            return PrepareCtxForAllgather(ctx);
        default:
            std::cout << "we don't support the opType" << opType.Describe() << std::endl;
            throw std::exception();
    }
}
