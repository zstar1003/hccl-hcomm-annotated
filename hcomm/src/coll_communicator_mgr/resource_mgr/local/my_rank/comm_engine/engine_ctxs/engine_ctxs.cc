/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine_ctxs.h"
#include "log.h"
#include "hcomm_c_adpt.h"
#include "comm_engine_utils.h"

namespace hccl {
EngineCtxs::EngineCtxs() {}

EngineCtxs::~EngineCtxs()
{
    // 析构兜底：遍历 contextMap_ 释放尚未显式 Destroy 的 ctx 内存
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& tagPair : contextMap_) {
        for (auto& enginePair : tagPair.second) {
            CommEngine engine = enginePair.first;
            void* addr = enginePair.second.addr;
            HcclResult ret = static_cast<HcclResult>(HcommEngineCtxDestroy(engine, addr));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] destroy engine ctx failed in destructor, tag[%s], engine[%s], ret[%d]", __func__,
                    tagPair.first.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret);
            }
        }
    }
    contextMap_.clear();
}

HcclResult EngineCtxs::CreateCommEngineCtx(const std::string& tag, CommEngine engine, uint64_t size, void** ctx)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // 阻止重复创建
    if (contextMap_.find(tag) != contextMap_.end()) {
        auto engineCtxMap = contextMap_[tag];
        CHK_PRT_RET(
            engineCtxMap.find(engine) != engineCtxMap.end(),
            HCCL_ERROR(
                "[%s] already exist a context with same key, tag[%s], engine[%s]", __func__, tag.c_str(),
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str()),
            HCCL_E_PARA);
    }

    CHK_RET(static_cast<HcclResult>(HcommEngineCtxCreate(engine, size, ctx)));
    contextMap_[tag][engine] = {HCCL_MEM_TYPE_NUM, *ctx, size}; // type不需要使用
    HCCL_INFO(
        "[%s]create context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcclResult EngineCtxs::GetCommEngineCtx(const std::string& tag, CommEngine engine, void** ctx, uint64_t* size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Ctx未创建返回
    const auto& tagIter = contextMap_.find(tag);
    if (tagIter == contextMap_.end()) {
        HCCL_INFO("[%s] not exist a context with tag[%s]", __func__, tag.c_str());
        return HCCL_E_NOT_FOUND;
    }

    const auto& engineCtxMap = tagIter->second;
    const auto& engineIter = engineCtxMap.find(engine);
    if (engineIter == engineCtxMap.end()) {
        HCCL_INFO(
            "[%s] not exist a context with tag[%s], engine[%s]", __func__, tag.c_str(),
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_NOT_FOUND;
    }

    const auto& ctxRes = engineIter->second;
    *ctx = ctxRes.addr;
    *size = ctxRes.size;
    HCCL_INFO(
        "[%s] get context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcclResult EngineCtxs::CopyCommEngineCtx(
    const std::string& tag, CommEngine engine, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    void* dstCtx;
    uint64_t dstSize = 0;
    CHK_RET(GetCommEngineCtx(tag, engine, &dstCtx, &dstSize));
    CHK_PRT_RET(
        dstCtxOffset + size > dstSize,
        HCCL_ERROR(
            "[%s]Copy engine ctx failed: buffer overflow detected. tag[%s], engine[%s], "
            "dstSize[%llu], dstCtxOffset[%llu], copySize[%llu]",
            __func__, tag.c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), dstSize, dstCtxOffset,
            size),
        HCCL_E_PARA);
    CHK_RET(static_cast<HcclResult>(HcommEngineCtxCopy(
        engine, reinterpret_cast<uint8_t*>(dstCtx) + dstCtxOffset, srcCtx, size))); // 增加大小判断，增加强转
    HCCL_INFO(
        "[%s]copy engine ctx success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcclResult EngineCtxs::DestroyEngineCtx(const std::string& tag, CommEngine engine)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Ctx不存在返回错误
    if (contextMap_.find(tag) == contextMap_.end()) {
        HCCL_ERROR("[%s] not exist a context with tag[%s]", __func__, tag.c_str());
        return HCCL_E_PARA;
    }
    auto& engineCtxMap = contextMap_[tag];
    if (engineCtxMap.find(engine) == engineCtxMap.end()) {
        HCCL_ERROR(
            "[%s] not exist a context with tag[%s], engine[%s]", __func__, tag.c_str(),
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }
    // 获取内存信息
    HcclMem& memInfo = engineCtxMap[engine];
    HcclResult ret = static_cast<HcclResult>(HcommEngineCtxDestroy(engine, memInfo.addr));
    engineCtxMap.erase(engine);
    if (engineCtxMap.empty()) {
        contextMap_.erase(tag);
    }
    CHK_RET(ret);

    HCCL_INFO(
        "[%s]destroy context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}
} // namespace hccl
