/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_dispatcher_ctx.h"
#include "dispatcher_ctx.h"
#include "dispatcher_aicpu_pub.h"
#include <unordered_map>
#include "adapter_rts_common.h"

// 多个通信域能并发跑通信算子，一个通信域只绑定一个dispatch_ctx线程变量
// 若不用通信域绑定线程变量，需要创建默认dispatch_ctx
static std::unordered_map<std::string, DispatcherCtxPtr> g_ctx;
std::mutex g_mtx; // 考虑已有的universal_map，或读写锁
thread_local DispatcherCtxPtr gDispatcherCtx = nullptr;

bool FindDispatcherByCommId(DispatcherCtxPtr* ctx, const char* commId)
{
    if (commId == nullptr) {
        HCCL_ERROR("[%s] find dispatcher fail, commId is nullptr", __func__);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    std::string commIdkey = std::string(commId);
    auto it = g_ctx.find(commIdkey);
    if (it != g_ctx.end()) {
        *ctx = it->second;
        HCCL_INFO("[%s] commIdkey[%s] has been bound with ctx[%p]", __func__, commIdkey.c_str(), *ctx);
        return true;
    }
    *ctx = nullptr;
    HCCL_WARNING("[%s] commIdkey[%s] not found, ctx return nullptr", __func__, commIdkey.c_str());
    return false;
}

bool DeleteDispatcherByCommId(const char* commId)
{
    if (commId == nullptr) {
        HCCL_ERROR("[%s] delete dispatcher fail, commId is nullptr", __func__);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mtx);
    std::string commIdkey = std::string(commId);
    auto it = g_ctx.find(commIdkey);
    if (it != g_ctx.end()) {
        HCCL_INFO("[%s] ctx[%p] has been bound by commId[%s]", __func__, it->second, commIdkey.c_str());
        g_ctx.erase(it);
        return true;
    }
    HCCL_WARNING("[%s] ctx has not been bound by commId[%s]", __func__, commIdkey.c_str());
    return false;
}

HcclResult BindDispatcherCtxWithComm(DispatcherCtxPtr ctx, const char* commId)
{
    CHK_PTR_NULL(commId);
    std::lock_guard<std::mutex> lock(g_mtx);
    std::string commIdkey = std::string(commId);
    auto it = g_ctx.find(commIdkey);
    if (it != g_ctx.end()) {
        HCCL_WARNING("[%s] commId[%s] has been bound", __func__, commIdkey.c_str());
        return HCCL_E_PARA;
    }
    g_ctx[commIdkey] = ctx;
    HCCL_INFO("[%s] ctx[%p] bind commId[%s] success", __func__, ctx, commIdkey.c_str());
    return HCCL_SUCCESS;
}

HcclResult CreateDispatcherCtx(DispatcherCtxPtr* ctx, u32 devPhyId, const char* commId)
{
    CHK_PTR_NULL(commId);
    CHK_PRT_RET(devPhyId == INVALID_UINT, HCCL_ERROR("[CreateCtx] devPhyId invalid"), HCCL_E_PARA);
    CHK_PTR_NULL(ctx);
    hccl::DispatcherCtx* Ctx_tmp = new (std::nothrow) hccl::DispatcherCtx(devPhyId);
    CHK_PTR_NULL(Ctx_tmp);
    // 创建ctx，内部创建dispatcher和notify pool实例  目前没有pool
    HcclResult ret = Ctx_tmp->Init();
    if (ret != HCCL_SUCCESS) {
        delete Ctx_tmp;
        HCCL_ERROR("[CreateCtx] CTX init fail");
        return ret;
    }

    ret = BindDispatcherCtxWithComm(Ctx_tmp, commId);
    // 如果存在，销毁创建的DispatcherCtx，返回存在的DispatcherCtx
    if (ret != HCCL_SUCCESS) {
        Ctx_tmp->Destroy();
        delete Ctx_tmp;
        // 查找已有ctx
        if (!FindDispatcherByCommId(ctx, commId)) {
            HCCL_ERROR("[CreateCtx] Bind fail AND no existing ctx for commId[%s]", commId);
            return HCCL_E_NOT_FOUND; // 明确返回错误，而非SUCCESS
        }
        gDispatcherCtx = *ctx;
        HCCL_WARNING("[CreateCtx] CTX bind fail, reuse existing ctx[%p] commId[%s]", *ctx, commId);
        return HCCL_SUCCESS;
    }
    *ctx = Ctx_tmp;
    gDispatcherCtx = Ctx_tmp;
    HCCL_INFO("[CreateCtx] CTX create success, ctx[%p] commId[%s]", *ctx, commId);
    return HCCL_SUCCESS;
}

bool DeleteCommIdByDispatcherCtx(DispatcherCtxPtr ctx)
{
    std::lock_guard<std::mutex> lock(g_mtx);
    for (const auto& pair : g_ctx) {
        if (pair.second == ctx) {
            HCCL_INFO("[%s] ctx[%p] bound with commId[%s], delete it", __func__, ctx, pair.first.c_str());
            g_ctx.erase(pair.first);
            return true;
        }
    }
    HCCL_WARNING("[%s] no commId bound with ctx[%p]", __func__, ctx);
    return false;
}

// 调用方有可能通过SetDispatcherCtx设置默认线程变量
// 传入commId是为了快速索引并释放g_ctx中的内容
// 不分成两个接口，防止重复释放
HcclResult DestroyDispatcherCtx(DispatcherCtxPtr ctx, const char* commId)
{
    static std::mutex deleteMutex_;
    const std::lock_guard<std::mutex> lock(deleteMutex_);
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(ctx);
    HCCL_INFO("[DestroyCtx] Destroy Ctx, ctx[%p] commId[%s]", ctx, commId);
    if (gDispatcherCtx == ctx) {
        gDispatcherCtx = nullptr;
    } else {
        HCCL_WARNING("[DestroyCtx] gDispatcherCtx[%p] and ctx[%p] do not match.", gDispatcherCtx, ctx);
    }

    DispatcherCtxPtr otherCtx = nullptr;
    // 若找到commId绑定的dispatch_ctx，解除绑定后销毁dispatch_ctx
    // 若找不到commId绑定的dispatch_ctx，查找map中
    if (LIKELY(FindDispatcherByCommId(&otherCtx, commId))) {
        DeleteDispatcherByCommId(commId);
    } else {
        bool hasFound = DeleteCommIdByDispatcherCtx(ctx);
        if (!hasFound) {
            HCCL_WARNING("[DestroyCtx] ctx[%p] not found by commId[%s], it may have been destroyed", ctx, commId);
            return HCCL_SUCCESS;
        }
        HCCL_WARNING("[DestroyCtx] ctx[%p] not found by commId[%s], just destroy", ctx, commId);
    }
    hccl::DispatcherCtx* Ctx_tmp = reinterpret_cast<hccl::DispatcherCtx*>(ctx);
    HcclResult ret = Ctx_tmp->Destroy();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[DestroyCtx] CTX Destroy fail");
    }
    delete Ctx_tmp;
    ctx = nullptr;
    return ret;
}

// 设置当前的线程变量dispatcherCtx，必须先调用CreateDispatcherCtx
// 同一个通信域不可以切换dispatcherCtx
HcclResult SetDispatcherCtx(const DispatcherCtxPtr ctx)
{
    HCCL_INFO("[%s], param: ctx[%p]", __func__, ctx);
    CHK_PTR_NULL(ctx);
    gDispatcherCtx = ctx;
    return HCCL_SUCCESS;
}

// 获取当前设置的线程变量dispatcherCtx，必须先调用CreateDispatcherCtx
DispatcherCtxPtr GetDispatcherCtx(const char* commId)
{
    if (UNLIKELY(commId == nullptr)) {
        HCCL_ERROR("[%s] get dispatcher fail, commId is nullptr", __func__);
        return nullptr;
    }
    HCCL_DEBUG("[%s], commId[%s]", __func__, commId);
    if (LIKELY((gDispatcherCtx != nullptr))) {
        HCCL_INFO("[%s], gDispatcherCtx[%p] exist, commId[%s]", __func__, gDispatcherCtx, commId);
        return gDispatcherCtx;
    }
    DispatcherCtxPtr ctx = nullptr;
    if (FindDispatcherByCommId(&ctx, commId)) {
        HCCL_INFO("[%s], ctx[%p] found in g_ctx, commId[%s]", __func__, ctx, commId);
        gDispatcherCtx = ctx;
        return ctx;
    }
    return nullptr;
}

HcclResult SetDispatcherCtxOpIdx(u32 opRingBufferIdx)
{
    HCCL_INFO("%s start, %u", __func__, opRingBufferIdx);
    hccl::DispatcherCtx* ctx_temp = reinterpret_cast<hccl::DispatcherCtx*>(GetDispatcherCtx());
    CHK_PTR_NULL(ctx_temp);
    hccl::DispatcherAiCpu* dispatcherPtr = reinterpret_cast<hccl::DispatcherAiCpu*>(ctx_temp->GetDispatcher());
    CHK_PTR_NULL(dispatcherPtr);
    dispatcherPtr->SetOpRingBufferIdx(opRingBufferIdx);
    return HCCL_SUCCESS;
}

HcclResult AcquireDispatcherCtx(DispatcherCtxPtr* ctx, const char* commId)
{
    CHK_PTR_NULL(commId);
    DispatcherCtxPtr ctxPtr = GetDispatcherCtx(commId);
    if (ctxPtr != nullptr) {
        *ctx = ctxPtr;
        HCCL_INFO("[AcquireCtx] CTX get success, ctx[%p] commId[%s]", *ctx, commId);
        return HCCL_SUCCESS;
    }
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    u32 devPhyId = INVALID_UINT;
    CHK_RET(hrtGetDevicePhyIdByIndex(deviceLogicId, devPhyId));
    CHK_PRT_RET(devPhyId == INVALID_UINT, HCCL_ERROR("[CreateCtx] devPhyId invalid"), HCCL_E_PARA);
    CHK_PTR_NULL(ctx);
    hccl::DispatcherCtx* Ctx_tmp = new (std::nothrow) hccl::DispatcherCtx(devPhyId);
    CHK_PTR_NULL(Ctx_tmp);
    HcclResult ret = Ctx_tmp->Init();
    if (ret != HCCL_SUCCESS) {
        delete Ctx_tmp;
        HCCL_ERROR("[AcquireCtx] CTX init fail");
        return ret;
    }

    // 如果存在，销毁创建的DispatcherCtx，返回存在的DispatcherCtx
    if (BindDispatcherCtxWithComm(Ctx_tmp, commId) != HCCL_SUCCESS) {
        Ctx_tmp->Destroy();
        delete Ctx_tmp;
        // 查找已有ctx
        if (!FindDispatcherByCommId(ctx, commId)) {
            HCCL_ERROR("[AcquireCtx] Bind fail AND no existing ctx for commId[%s]", commId);
            return HCCL_E_NOT_FOUND; // 明确返回错误，而非SUCCESS
        }
        gDispatcherCtx = *ctx;
        HCCL_WARNING("[AcquireCtx] CTX bind fail, reuse existing ctx[%p] commId[%s]", *ctx, commId);
        return HCCL_SUCCESS;
    }

    *ctx = Ctx_tmp;
    gDispatcherCtx = Ctx_tmp;
    HCCL_INFO("[AcquireCtx] CTX create success, ctx[%p] commId[%s]", *ctx, commId);
    return HCCL_SUCCESS;
}
