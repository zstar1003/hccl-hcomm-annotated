/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "independent_op_context_manager.h"
#include "log.h"
#include "adapter_rts_common.h"
#include "comm_engine_utils.h"

namespace hccl {
ContextManager::ContextManager() {}

ContextManager::~ContextManager() {}

HcclResult ContextManager::CreateCommEngineCtx(const std::string& tag, CommEngine engine, uint64_t size, void** ctx)
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

    void* ctxData = nullptr;
    // 区分设备类型
    HcclMemType type;
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        type = HCCL_MEM_TYPE_HOST;
        ctxData = malloc(size);
        CHK_PTR_NULL(ctxData);
        s32 sRet = memset_s(ctxData, size, 0, size);
        if (sRet != EOK) {
            HCCL_ERROR("[%s] memset_s failed, ret[%d]", __func__, sRet);
            free(ctxData);
            ctxData = nullptr;
            return HCCL_E_INTERNAL;
        }
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        type = HCCL_MEM_TYPE_DEVICE;
        CHK_RET(hrtMalloc(&ctxData, size));
    } else {
        HCCL_ERROR(
            "[%s] not support engine type[%s]", __func__, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }

    contextMap_[tag][engine] = {type, ctxData, size};
    *ctx = contextMap_[tag][engine].addr;
    HCCL_INFO(
        "[%s]create context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());

    return HCCL_SUCCESS;
}

HcclResult ContextManager::GetCommEngineCtx(const std::string& tag, CommEngine engine, void** ctx, uint64_t* size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Ctx未创建返回
    if (contextMap_.find(tag) == contextMap_.end()) {
        HCCL_INFO("[%s] not exist a context with tag[%s]", __func__, tag.c_str());
        return HCCL_E_PARA;
    } else {
        auto engineCtxMap = contextMap_[tag];
        if (engineCtxMap.find(engine) == engineCtxMap.end()) {
            HCCL_INFO(
                "[%s] not exist a context with tag[%s], engine[%s]", __func__, tag.c_str(),
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
            return HCCL_E_PARA;
        }
    }

    *ctx = contextMap_[tag][engine].addr;
    *size = contextMap_[tag][engine].size;
    HCCL_INFO(
        "[%s]get context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcclResult ContextManager::CopyCommEngineCtx(
    const std::string& tag, CommEngine engine, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    void* dstCtx;
    uint64_t dstSize = 0;
    if (engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AIV) {
        CHK_RET(GetCommEngineCtx(tag, engine, &dstCtx, &dstSize));
        // 从Host内存拷贝到Device Context内存上
        CHK_RET(hrtMemSyncCopy(
            static_cast<uint8_t*>(dstCtx) + dstCtxOffset, size, srcCtx, size,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    } else if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS || engine == COMM_ENGINE_CCU) {
        CHK_RET(GetCommEngineCtx(tag, engine, &dstCtx, &dstSize));
        (void)memcpy_s(static_cast<uint8_t*>(dstCtx) + dstCtxOffset, size, srcCtx, size);
    } else {
        HCCL_ERROR(
            "[%s]copy engine ctx failed, Unsupported engine[%s], tag[%s]", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), tag.c_str());
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s]copy engine ctx success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcclResult ContextManager::DestroyCommEngineCtx(const std::string& tag, CommEngine engine)
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
    // 释放内存
    if (memInfo.type == HCCL_MEM_TYPE_HOST) {
        free(memInfo.addr);
    } else if (memInfo.type == HCCL_MEM_TYPE_DEVICE) {
        CHK_RET(hrtFree(memInfo.addr));
    } else {
        HCCL_ERROR("[%s] invalid memory type[%d]", __func__, memInfo.type);
        return HCCL_E_PARA;
    }
    // 从映射中移除
    engineCtxMap.erase(engine);
    if (engineCtxMap.empty()) {
        contextMap_.erase(tag);
    }

    HCCL_INFO(
        "[%s]destroy context success, tag[%s], engine[%s]", __func__, tag.c_str(),
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}
} // namespace hccl
