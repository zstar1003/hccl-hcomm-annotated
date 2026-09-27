/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include "hccl/hccl_res.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include <string>
#include "param_check_pub.h"
#include "comm_engine_utils.h"

using namespace hccl;

const char* COMM_RESERVE_CTX_TAG = "";

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(ctx);
    const char* ctxTagTmp = (ctxTag == nullptr) ? COMM_RESERVE_CTX_TAG : ctxTag;
    CHK_PRT_RET(
        strlen(ctxTagTmp) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR(
            "[%s] ctxTag length exceeds maximum length, ctxTag length[%zu], max length[%u]", __func__,
            strlen(ctxTagTmp), HCCL_RES_TAG_MAX_LEN),
        HCCL_E_PARA);
    CHK_PRT_RET(
        size == 0, HCCL_ERROR("[%s]Invalid CtxSize, CtxSize[%llu]", __func__, static_cast<unsigned long long>(size)),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
        std::string commId = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-HcclEngineCtxCreate:comm[%s]", commId.c_str());
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        EngineCtxs* engineCtxs = myRank->GetEngineCtxs();
        CHK_PTR_NULL(engineCtxs);
        HcclResult ret = HCCL_SUCCESS;
        ret = engineCtxs->CreateCommEngineCtx(ctxTagTmp, engine, size, ctx);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] Failed to create CommEngineCtx with ctxTag[%s], engine[%s], ctx size[%llu], ret[%d]", __func__,
                ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                static_cast<unsigned long long>(size), ret),
            ret);
        HCCL_RUN_INFO(
            "HcclEngineCtxCreate success, ctxTag[%s], engine[%s], size[%llu], ctx[%p], group[%s]", ctxTagTmp,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), static_cast<unsigned long long>(size), *ctx,
            hcclComm->GetIdentifier().c_str());
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto& contextMgr = hcclComm->GetIndependentOp().GetContextManager();
    HcclResult ret = contextMgr.CreateCommEngineCtx(ctxTagTmp, engine, size, ctx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to create CommEngineCtx with ctxTag[%s], engine[%s], ctx size[%llu], ret[%d]", __func__,
            ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
            static_cast<unsigned long long>(size), ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] success, ctxTag[%s], engine[%s], size[%llu], ctx[%p], group[%s]", __func__, ctxTagTmp,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), static_cast<unsigned long long>(size), *ctx,
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    // 性能关键路径，禁止打印算子粒度频次的日志
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(ctx);
    CHK_PTR_NULL(size);
    const char* ctxTagTmp = (ctxTag == nullptr) ? COMM_RESERVE_CTX_TAG : ctxTag;
    CHK_PRT_RET(
        strlen(ctxTagTmp) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR(
            "[%s] ctxTag length exceeds maximum length, ctxTag length[%zu], max length[%u]", __func__,
            strlen(ctxTagTmp), HCCL_RES_TAG_MAX_LEN),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
        const std::string& commId = hcclComm->GetIdentifier();
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        EngineCtxs* engineCtxs = myRank->GetEngineCtxs();
        CHK_PTR_NULL(engineCtxs);
        HcclResult ret = HCCL_SUCCESS;
        ret = engineCtxs->GetCommEngineCtx(ctxTagTmp, engine, ctx, size);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_WARNING(
                "[%s] Failed to get CommEngineCtx with ctxTag[%s], engine[%s], ret[%d]", __func__, ctxTagTmp,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto& contextMgr = hcclComm->GetIndependentOp().GetContextManager();
    HcclResult ret = contextMgr.GetCommEngineCtx(std::string(ctxTagTmp), engine, ctx, size);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] Failed to get CommEngineCtx with ctxTag[%s], engine[%s], ret[%d]", __func__, ctxTagTmp,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] success, ctxTag[%s], engine[%s], ctx[%p], size[%llu], group[%s]", __func__, ctxTagTmp,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), *ctx, static_cast<unsigned long long>(*size),
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCopy(
    HcclComm comm, CommEngine engine, const char* ctxTag, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(srcCtx);
    const char* ctxTagTmp = (ctxTag == nullptr) ? COMM_RESERVE_CTX_TAG : ctxTag;
    CHK_PRT_RET(
        strlen(ctxTagTmp) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR(
            "[%s] ctxTag length exceeds maximum length, ctxTag length[%zu], max length[%u]", __func__,
            strlen(ctxTagTmp), HCCL_RES_TAG_MAX_LEN),
        HCCL_E_PARA);
    CHK_PRT_RET(
        size == 0, HCCL_ERROR("[%s]Invalid size, size[%llu]", __func__, static_cast<unsigned long long>(size)),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
        std::string commId = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s]", __func__, commId.c_str());
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        EngineCtxs* engineCtxs = myRank->GetEngineCtxs();
        CHK_PTR_NULL(engineCtxs);
        HcclResult ret = HCCL_SUCCESS;
        ret = engineCtxs->CopyCommEngineCtx(ctxTagTmp, engine, srcCtx, size, dstCtxOffset);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_WARNING(
                "[%s] Failed to copy CommEngineCtx with ctxTag[%s], engine[%s], size[%llu], dstCtxOffset[%llu],"
                " ret[%d]",
                __func__, ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                static_cast<unsigned long long>(size), static_cast<unsigned long long>(dstCtxOffset), ret),
            ret);
        HCCL_RUN_INFO(
            "[%s] success, ctxTag[%s], engine[%s], srcCtx[%p], size[%llu], dstCtxOffset[%llu], group[%s]", __func__,
            ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), srcCtx,
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(dstCtxOffset),
            hcclComm->GetIdentifier().c_str());
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto& contextMgr = hcclComm->GetIndependentOp().GetContextManager();
    HcclResult ret = contextMgr.CopyCommEngineCtx(std::string(ctxTagTmp), engine, srcCtx, size, dstCtxOffset);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to copy CommEngineCtx with ctxTag[%s], engine[%s], size[%llu], dstCtxOffset[%llu],"
            " ret[%d]",
            __func__, ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(dstCtxOffset), ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] success, ctxTag[%s], engine[%s], srcCtx[%p], size[%llu], dstCtxOffset[%llu], group[%s]", __func__,
        ctxTagTmp, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), srcCtx,
        static_cast<unsigned long long>(size), static_cast<unsigned long long>(dstCtxOffset),
        hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)
{
    CHK_PTR_NULL(comm);
    const char* ctxTagTmp = (ctxTag == nullptr) ? COMM_RESERVE_CTX_TAG : ctxTag;
    CHK_PRT_RET(
        strlen(ctxTagTmp) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR(
            "[%s] ctxTag length exceeds maximum length, ctxTag length[%zu], max length[%u]", __func__,
            strlen(ctxTagTmp), HCCL_RES_TAG_MAX_LEN),
        HCCL_E_PARA);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
        std::string commId = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s]", __func__, commId.c_str());
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        auto myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        EngineCtxs* engineCtxs = myRank->GetEngineCtxs();
        HcclResult ret = HCCL_SUCCESS;
        ret = engineCtxs->DestroyEngineCtx(ctxTagTmp, engine);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] Failed to destroy CommEngineCtx, ctxTag[%s], engine[%s], ret[%d]", __func__, ctxTagTmp,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);
        HCCL_RUN_INFO(
            "[%s] success, ctxTag[%s], engine[%s], group[%s]", __func__, ctxTagTmp,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), hcclComm->GetIdentifier().c_str());
        return HCCL_SUCCESS;
    }());
#endif

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto& contextMgr = hcclComm->GetIndependentOp().GetContextManager();
    HcclResult ret = contextMgr.DestroyCommEngineCtx(ctxTagTmp, engine);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to destroy CommEngineCtx, ctxTag[%s], engine[%s], ret[%d]", __func__, ctxTagTmp,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret);
        return ret;
    }
    HCCL_RUN_INFO(
        "[%s] success, ctxTag[%s], engine[%s], group[%s]", __func__, ctxTagTmp,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), hcclComm->GetIdentifier().c_str());
    return HCCL_SUCCESS;
}
