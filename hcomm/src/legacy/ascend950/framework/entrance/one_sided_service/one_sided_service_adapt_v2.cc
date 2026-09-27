/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "one_sided_service_adapt_v2.h"
#include "task_param.h"
#include "hccl_one_sided_data.h"
#include "hccl_one_sided_service_v2.h"
#include "hccl_communicator.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "param_check_v2.h"

using namespace std;
using namespace Hccl;

constexpr u64 ONE_SIDE_DEVICE_MEM_MAX_SIZE = 64llu * 1024 * 1024 * 1024; // device侧支持内存注册大小上限为64GB
constexpr u64 ONE_SIDE_HOST_MEM_MAX_SIZE = 1024llu * 1024 * 1024 * 1024; // host侧支持内存注册大小上限为1TB
constexpr u64 ONE_SIDE_HOST_MEM_ZERO = 0;
constexpr u64 MAX_DESC_NUM = 64; // 批量操作描述符个数上限
constexpr u64 MEM_TYPE_DEVICE = 0;
constexpr u64 MEM_TYPE_HOST = 0;
constexpr u64 MEM_TYPE_NUM = 0;

const std::map<int, HcclMemType> HCCL_MEM_TYPE_V2{
    {MEM_TYPE_DEVICE, HcclMemType::HCCL_MEM_TYPE_DEVICE},
    {MEM_TYPE_HOST, HcclMemType::HCCL_MEM_TYPE_HOST},
    {MEM_TYPE_NUM, HcclMemType::HCCL_MEM_TYPE_NUM}};

HcclResult HcclRegisterMemV2(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc)
{
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commIdentifier = hcclCommunicator->GetId();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s], remoteRank[%u], memType[%d], memAddr[%p], memSize[%llu], memDescPtr[%p]", __func__,
        commIdentifier.c_str(), remoteRank, type, addr, size, desc);

    auto it = HCCL_MEM_TYPE_V2.find(type);
    CHK_PRT_RET(
        it == HCCL_MEM_TYPE_V2.end(),
        HCCL_ERROR("[HcclRegisterMemV2] HcclMemType[%d] is invalid, please check memory type", type), HCCL_E_PARA);
    HcclMemType memType = it->second;
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclCommunicator->GetRankId(localRank));

    CHK_PRT_RET(
        remoteRank == localRank,
        HCCL_WARNING(
            "remoteRank[%u] is equal to localRank[%u], no need to "
            "register memory, return HcclRegisterMem success",
            remoteRank, localRank),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        memType != HcclMemType::HCCL_MEM_TYPE_DEVICE && memType != HcclMemType::HCCL_MEM_TYPE_HOST,
        HCCL_ERROR("[HcclRegisterMem]memoryType[%d] must be device or host, please check type", type), HCCL_E_PARA);
    CHK_PRT_RET(
        size <= ONE_SIDE_HOST_MEM_ZERO,
        HCCL_ERROR(
            "[HcclRegisterMem]memory size[%llu] is invalid, "
            "please check memory size",
            size),
        HCCL_E_PARA);
    CHK_PRT_RET(
        memType == HcclMemType::HCCL_MEM_TYPE_DEVICE && size > ONE_SIDE_DEVICE_MEM_MAX_SIZE,
        HCCL_ERROR("[HcclRegisterMem]memory size[%llu] is too large, please check memory size", size), HCCL_E_PARA);
    CHK_PRT_RET(
        memType == HcclMemType::HCCL_MEM_TYPE_HOST,
        HCCL_ERROR("[HcclRegisterMem] HCCL_MEM_TYPE_HOST is not support, please check memory type"),
        HCCL_E_NOT_SUPPORT);

    HCCL_INFO("HcclRegisterMemV2 GetLocalRankID Success: localRank[%u]", localRank);

    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclRegisterMemV2 RegMem Begin");

    // HcclResult HcclOneSidedService::RegMem(void *addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc
    // &localMemDesc)
    CHK_RET(service->RegMem(addr, size, memType, remoteRank, *desc));

    HCCL_INFO("HcclRegisterMemV2 RegMem End");

    HCCL_RUN_INFO(
        "%s success:commPtr[%p], remoteRank[%u], memType[%d], memAddr[%p], memSize[%llu], memDescPtr[%p]", __func__,
        comm, remoteRank, type, addr, size, desc);
    return HCCL_SUCCESS;
}

HcclResult HcclDeregisterMemV2(HcclComm comm, HcclMemDesc* desc)
{
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commIdentifier = hcclCommunicator->GetId();
    HCCL_RUN_INFO("Entry-%s:comm[%s], memDescPtr[%p]", __func__, commIdentifier.c_str(), desc);

    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclRegisterMemV2 DeregMem Begin");
    CHK_RET(service->DeregMem(*desc));
    HCCL_INFO("HcclRegisterMemV2 DeregMem End");

    HCCL_RUN_INFO("%s success:commPtr[%p], memDescPtr[%p]", __func__, comm, desc);
    return HCCL_SUCCESS;
}

HcclResult HcclExchangeMemDescV2(
    HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum)
{
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commIdentifier = hcclCommunicator->GetId();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s], remoteRank[%u], localMemDescPtr[%p], timeout[%d s], remoteMemDescPtr[%p], "
        "actualNum[%u]",
        __func__, commIdentifier.c_str(), remoteRank, local, timeout, remote, *actualNum);

    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclCommunicator->GetRankId(localRank));
    CHK_PRT_RET(
        remoteRank == localRank,
        HCCL_WARNING(
            "remoteRank[%u] is equal to localRank[%u], no need to "
            "register memory, return HcclRegisterMem success",
            remoteRank, localRank),
        HCCL_SUCCESS);

    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclRegisterMemV2 ExchangeMemDesc Begin");
    CHK_RET(service->ExchangeMemDesc(remoteRank, *local, *remote, *actualNum));
    HCCL_INFO("HcclRegisterMemV2 ExchangeMemDesc end");

    HCCL_RUN_INFO(
        "%s success:commPtr[%p], remoteRank[%u], localMemDescPtr[%p], timeout[%d s], remoteMemDescPtr[%p], "
        "actualNum[%u]",
        __func__, comm, remoteRank, local, timeout, remote, *actualNum);
    return HCCL_SUCCESS;
}

HcclResult HcclEnableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem)
{
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commIdentifier = hcclCommunicator->GetId();
    HCCL_RUN_INFO(
        "Entry-%s:comm[%s], remoteMemDescPtr[%p], remoteMemPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc,
        remoteMem);

    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclRegisterMemV2 EnableMemAccess Begin");
    CHK_RET(service->EnableMemAccess(*remoteMemDesc, *remoteMem));
    HCCL_INFO("HcclRegisterMemV2 EnableMemAccess End");

    HCCL_RUN_INFO(
        "%s success:commPtr[%p], remoteMemDescPtr[%p], remoteMemPtr[%p]", __func__, comm, remoteMemDesc, remoteMem);
    return HCCL_SUCCESS;
}

HcclResult HcclDisableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc)
{
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    std::string commIdentifier = hcclCommunicator->GetId();
    HCCL_RUN_INFO("Entry-%s:comm[%s], remoteMemDescPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc);

    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclRegisterMemV2 DisableMemAccess Begin");
    CHK_RET(service->DisableMemAccess(*remoteMemDesc));
    HCCL_INFO("HcclRegisterMemV2 DisableMemAccess End");

    HCCL_RUN_INFO("%s success:commPtr[%p], remoteMemDescPtr[%p]", __func__, comm, remoteMemDesc);
    return HCCL_SUCCESS;
}

inline static HcclResult HcclBatchParaCheckV2(HcclComm comm, HcclBatchData& paraData, std::string& getTag)
{
    HCCL_INFO("HcclBatchParaCheckV2 Begin");
    // 参数校验和适配
    CHK_PTR_NULL(paraData.comm);
    CHK_PTR_NULL(paraData.stream);
    CHK_PTR_NULL(paraData.desc);
    std::string batchString = (paraData.cmdType == HcclCMDType::HCCL_CMD_BATCH_GET) ? "BatchGet" : "BatchPut";
    CHK_PRT_RET(
        paraData.descNum > MAX_DESC_NUM,
        HCCL_WARNING("[%s] the count of HcclOneSideOpDesc exceed specification.", batchString.c_str()), HCCL_E_PARA);

    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    // 同算子复用tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclCommunicator->GetRankId(localRank));

    const std::string tag = batchString + "_" + std::to_string(localRank) + "_" + std::to_string(paraData.remoteRank)
                            + "_" + hcclCommunicator->GetId();
    getTag = tag;

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclCommunicator->GetRankSize(&rankSize), tag.c_str());
    CHK_RET(HcomCheckUserRankV2(rankSize, paraData.remoteRank));
    CHK_PRT_RET(
        paraData.remoteRank == localRank,
        HCCL_ERROR("[%s] the remoteRank can't be equal to localRank, please check.", batchString.c_str()), HCCL_E_PARA);

    s32 streamId = 0;

    HCCL_RUN_INFO(
        "Entry-%s::tag[%s], descNum[%u], streamId[%d], localRank[%u], remoteRank[%u]", __func__, tag.c_str(),
        paraData.descNum, streamId, localRank, paraData.remoteRank);

    HCCL_INFO("HcclBatchParaCheckV2 End");
    return HCCL_SUCCESS;
}

HcclResult HcclBatchPutV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream)
{
    HCCL_INFO("HcclBatchPutV2 Begin");
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(stream);
    std::string getTag;
    CHK_PRT_RET(descNum == 0, HCCL_WARNING("[%s] the count of HcclOneSideOpDesc is zero.", __func__), HCCL_SUCCESS);
    HcclBatchData paraData = {comm, HcclCMDType::HCCL_CMD_BATCH_PUT, remoteRank, desc, descNum, stream};
    CHK_RET(HcclBatchParaCheckV2(comm, paraData, getTag));
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclBatchPutV2 BatchPut Begin");
    CHK_RET(service->BatchPut(remoteRank, desc, descNum, stream));
    HCCL_INFO("HcclBatchPutV2 End");
    return HCCL_SUCCESS;
}

HcclResult HcclBatchGetV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream)
{
    HCCL_INFO("HcclBatchGetV2 Begin");
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(desc);
    CHK_PTR_NULL(stream);
    std::string getTag;
    CHK_PRT_RET(descNum == 0, HCCL_WARNING("[%s] the count of HcclOneSideOpDesc is zero.", __func__), HCCL_SUCCESS);
    HcclBatchData paraData = {comm, HcclCMDType::HCCL_CMD_BATCH_PUT, remoteRank, desc, descNum, stream};
    CHK_RET(HcclBatchParaCheckV2(comm, paraData, getTag));
    Hccl::HcclCommunicator* hcclCommunicator = static_cast<Hccl::HcclCommunicator*>(comm);
    Hccl::HcclOneSidedService* service = nullptr;
    CHK_RET(hcclCommunicator->GetOneSidedService(&service));
    CHK_PTR_NULL(service);

    HCCL_INFO("HcclBatchGetV2 BatchGet Begin");
    CHK_RET(service->BatchGet(remoteRank, desc, descNum, stream));
    HCCL_INFO("HcclBatchGetV2 End");
    return HCCL_SUCCESS;
}
