/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include "hccl/hccl_res.h"
#include "hccl_mem.h"
#include "stream_pub.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "param_check_pub.h"
#include "op_base.h"
#include "hccl_res.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hccl_team_c_adpt.h"

using namespace hccl;

HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)

{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[HcclCommMemReg]comm is null"), HCCL_E_PTR);
    CHK_PRT_RET(memTag == nullptr, HCCL_ERROR("[HcclCommMemReg]memTag is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        strlen(memTag) == 0 || strlen(memTag) > HCCL_RES_TAG_MAX_LEN,
        HCCL_ERROR("[HcclCommMemReg]memTag length is %zu", strlen(memTag)), HCCL_E_PARA);
    std::string memTagStr(memTag);
    CHK_PRT_RET(
        memTagStr.compare(0, strlen(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX), HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) == 0,
        HCCL_ERROR(
            "[HcclCommMemReg]memTag[%s] uses reserved symmetric memory prefix[%s]", memTag,
            HCCL_SYMMETRIC_MEMORY_TAG_PREFIX),
        HCCL_E_PARA);
    CHK_PRT_RET(
        memTagStr.compare(0, strlen(HCCL_TEAM_SYNCMEM_TAG_PREFIX), HCCL_TEAM_SYNCMEM_TAG_PREFIX) == 0,
        HCCL_ERROR(
            "[HcclCommMemReg]memTag[%s] uses reserved team syncmem prefix[%s]", memTag, HCCL_TEAM_SYNCMEM_TAG_PREFIX),
        HCCL_E_PARA);
    CHK_PRT_RET(mem == nullptr, HCCL_ERROR("[HcclCommMemReg]mem is null"), HCCL_E_PTR);
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[HcclCommMemReg]memHandle is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        (mem->type != COMM_MEM_TYPE_DEVICE) && (mem->type != COMM_MEM_TYPE_HOST) && (mem->type != COMM_MEM_TYPE_CCU),
        HCCL_ERROR("[HcclCommMemReg]memoryType[%d] must be device or host or ccu", mem->type), HCCL_E_PARA);
    CHK_PRT_RET(mem->addr == nullptr, HCCL_ERROR("[HcclCommMemReg]addr is null"), HCCL_E_PTR);
    CHK_PRT_RET(
        mem->size == 0, HCCL_ERROR("[HcclCommMemReg]size[%llu] invalid", static_cast<unsigned long long>(mem->size)),
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
        CommMems* commMem = myRank->GetCommMems();
        HcclResult ret = HCCL_SUCCESS;
        ret = commMem->CommRegMem(memTagStr, *mem, memHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommMemReg]Bind failed. memTag[%s], ret[%d]", memTag, ret), ret);
        HCCL_INFO("[HcclCommMemReg] success: raw handle[%p]", *memHandle);
        return HCCL_SUCCESS;
    }());
#endif
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    HcclMem hcclMem;
    hcclMem.addr = mem->addr;
    hcclMem.size = mem->size;
    hcclMem.type = (mem->type == COMM_MEM_TYPE_HOST) ? HCCL_MEM_TYPE_HOST : HCCL_MEM_TYPE_DEVICE;
    HcclRegMemAttr attr;
    attr.value = 0;
    HcclResult ret = hcclComm->GetIndependentOp().GetCommMemMgr().CommRegMem(memTagStr, hcclMem, attr, memHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommMemReg]legacy Bind failed. memTag[%s], ret[%d]", memTag, ret), ret);
    HCCL_INFO("[HcclCommMemReg]legacy success: raw handle[%p]", *memHandle);

    return HCCL_SUCCESS;
}

HcclResult HcclCommDeregMem(HcclComm comm, const char* memTag, const void* memHandle)
{
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[HcclCommDeregMem]comm is null"), HCCL_E_PTR);
    CHK_PRT_RET(memHandle == nullptr, HCCL_ERROR("[HcclCommDeregMem]memHandle is null"), HCCL_E_PTR);
    CHK_PRT_RET(memTag == nullptr, HCCL_ERROR("[HcclCommDeregMem]memTag is null"), HCCL_E_PARA);
    CHK_PRT_RET(strlen(memTag) == 0, HCCL_ERROR("[HcclCommDeregMem]memTag length is 0"), HCCL_E_PARA);

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    HCCL_RUN_INFO("Entry-%s: comm[%s], handle[%p]", __func__, commId.c_str(), memHandle);

    // 解绑某算子下的该句柄
    HcclResult ret = HCCL_SUCCESS;
    if (hcclComm->IsCommunicatorV2()) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        CommMemMgr* commMemMgr = collComm->GetCommMemMgr();
        CHK_PTR_NULL(commMemMgr);
        ret = commMemMgr->CommUnregMem(std::string(memTag), memHandle);
    } else {
        auto& commMemMgr = hcclComm->GetIndependentOp().GetCommMemMgr();
        ret = commMemMgr.CommUnregMem(std::string(memTag), memHandle);
    }

    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND, HCCL_WARNING("[HcclCommDeregMem]handle not bound in this domain. raw[%p]", memHandle),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDeregMem] unBind failed. handle[%p], ret[%d]", memHandle, ret), ret);
    HCCL_INFO("[HcclCommDeregMem]success: raw handle[%p]", memHandle);
    return HCCL_SUCCESS;
}

HcclResult GetHcclBufferWithClearFlag(HcclComm comm, void** buffer, uint64_t* size, bool clearFlag)
{
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::string& commId = hcclComm->GetIdentifier();
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    if (collComm->GetRankSize() == 1) {
        HCCL_RUN_WARNING("[%s] comm[%s] is single rank, return nullptr", __func__, commId.c_str());
        *buffer = nullptr;
        *size = 0;
        return HCCL_SUCCESS;
    }
    auto myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);
    CommMems* commMem = myRank->GetCommMems();
    CHK_PTR_NULL(commMem);
    CHK_RET(commMem->GetHcclBuffer(*buffer, *size));
    CHK_RET(commMem->HcclBufferMemset(*buffer, *size, clearFlag));

    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    CHK_PRT_RET(buffer == nullptr, HCCL_ERROR("[%s] buffer is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(size == nullptr, HCCL_ERROR("[%s] size is null", __func__), HCCL_E_PTR);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        return GetHcclBufferWithClearFlag(comm, buffer, size, false);
    }());
#endif

    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::string commId = hcclComm->GetIdentifier();
    CollComm* collComm = hcclComm->GetCollComm();
    hccl::MyRank* myRank = nullptr;
    if (collComm != nullptr) {
        myRank = collComm->GetMyRank();
        if (collComm->GetRankSize() == 1) {
            HCCL_RUN_WARNING("[%s] comm[%s] is single rank, return nullptr", __func__, commId.c_str());
            *buffer = nullptr;
            *size = 0;
            return HCCL_SUCCESS;
        }
        if (hcclComm->GetConnectMode() != 0 && myRank != nullptr) {
            CommMems* commMem = myRank->GetCommMems();
            CHK_PTR_NULL(commMem);
            CHK_RET(commMem->GetHcclBuffer(*buffer, *size));
            return HCCL_SUCCESS;
        }
    }

    HCCL_RUN_INFO("Entry-%s:comm[%s]", __func__, commId.c_str());
    HcclResult ret = HCCL_SUCCESS;
    CommBuffer commBuffer;

    auto& commMemMgr = hcclComm->GetIndependentOp().GetCommMemMgr();
    ret = commMemMgr.GetHcclBuffer(&commBuffer);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to get local cclBuffer ret[%d]", __func__, ret);
        return ret;
    }
    *buffer = commBuffer.addr;
    *size = commBuffer.size;
    HCCL_RUN_INFO(
        "Entry-%s: success: comm[%s], buffer[%p] size[%llu]", __func__, commId.c_str(), *buffer,
        static_cast<unsigned long long>(*size));
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBufferCleared(HcclComm comm, void** buffer, uint64_t* size)
{
    CHK_PRT_RET(buffer == nullptr, HCCL_ERROR("[%s] buffer is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    CHK_PRT_RET(size == nullptr, HCCL_ERROR("[%s] size is null", __func__), HCCL_E_PTR);

#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        return GetHcclBufferWithClearFlag(comm, buffer, size, true);
    }());
#endif

    HCCL_ERROR("HcclGetHcclBufferCleared is not supported");
    return HCCL_E_NOT_SUPPORT;
}
