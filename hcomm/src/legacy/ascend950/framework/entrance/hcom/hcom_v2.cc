/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "orion_adapter_rts.h"
#include "hccl_communicator.h"
#include "comm_manager.h"
#include "hcom_v2.h"
#include "param_check_v2.h"
#include "hccl_common_v2.h"
#include "task_param.h"
#include "types.h"
#include "comm_topo_desc_v2.h"

using namespace std;
using namespace Hccl;

static HcclResult GetHcclGroupParams(const std::string& strGroup, HcclGroupParamsV2& hcclGroupParamsV2)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    std::lock_guard<std::mutex> groupParaLock(hcomCommInfoV2.groupParamsLock);
    auto iter = hcomCommInfoV2.hcclGroupMap.find(strGroup);
    if (iter != hcomCommInfoV2.hcclGroupMap.end()) {
        hcclGroupParamsV2 = iter->second;
        return HCCL_SUCCESS;
    }
    HCCL_WARNING("comm group not in hcclGroupMap, please check groupName[%s].", strGroup.c_str());
    return HCCL_E_NOT_FOUND;
}

static HcclResult GetHcclCommV2(const char* group, std::shared_ptr<Hccl::HcclCommunicator>& hcclComm)
{
    std::string strGroup = (group == nullptr || strlen(group) == 0) ? HCCL_WORLD_GROUP : group;
    HcclGroupParamsV2 hcclGroupParamsV2;
    HcclResult ret = GetHcclGroupParams(strGroup, hcclGroupParamsV2);
    CHK_PRT_RET(
        ret == HCCL_E_NOT_FOUND,
        HCCL_WARNING(
            "[GetHcclCommV2]errNo[0x%016llx] group[%s] group is not exist", HCOM_ERROR_CODE(HCCL_E_NOT_FOUND),
            strGroup.c_str()),
        HCCL_E_NOT_FOUND);
    hcclComm = hcclGroupParamsV2.pComm;
    CHK_PTR_NULL(hcclComm);
    HCCL_INFO("[%s] success.", __func__);
    return HCCL_SUCCESS;
}

inline Hccl::CollOpParams GetHcclOpParams(
    void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, Hccl::OpType opType,
    HcclReduceOp op = HCCL_REDUCE_RESERVED, bool isSuperKernel = false)
{
    Hccl::CollOpParams opParams;
    opParams.opType = opType;
    opParams.sendBuf = inputPtr;
    opParams.recvBuf = outputPtr;
    opParams.count = count;
    opParams.staticShape = true;
    if (dataType != HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        opParams.dataType = HcclDataTypeToDataType(dataType);
    } else {
        opParams.dataType = Hccl::DataType::INVALID;
    }
    if (op == HCCL_REDUCE_SUM) {
        opParams.reduceOp = Hccl::ReduceOp::SUM;
    } else if (op == HCCL_REDUCE_PROD) {
        opParams.reduceOp = Hccl::ReduceOp::PROD;
    } else if (op == HCCL_REDUCE_MAX) {
        opParams.reduceOp = Hccl::ReduceOp::MAX;
    } else if (op == HCCL_REDUCE_MIN) {
        opParams.reduceOp = Hccl::ReduceOp::MIN;
    }

    if (opType == Hccl::OpType::ALLTOALL && isSuperKernel) {
        opParams.all2AllDataDes.sendCount = count;
        opParams.all2AllDataDes.recvCount = count;
        opParams.all2AllDataDes.sendType = HcclDataTypeToDataType(dataType);
        opParams.all2AllDataDes.recvType = HcclDataTypeToDataType(dataType);
    }
    return opParams;
}

HcclResult HcomAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group,
    rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AllGather] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, inputCount, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, inputCount, dataType, Hccl::OpType::ALLGATHER);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom allgather success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "inputCount[%llu], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, inputCount, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomAllGatherVV2(
    const char* tag, void* sendBuf, u64 sendCount, void* recvBuf, void* recvCounts, void* rdispls,
    HcclDataType dataType, const char* group, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);
    HcclUs startut = TIME_NOW();
    /* 获取通信域 */
    std::string opTag = tag;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AllGatherV] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    /* 获取rank信息 */
    uint32_t rankId;
    CHK_RET(hcclComm->GetRankId(rankId));
    uint32_t rankSize;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    /* 参数合法性校验 */
    if (rankSize == 1) {
        /* rankSize为1时，退化为AllGather */
        // 检查异常回退AGV的情况
        if (sendCount == 0) {
            HCCL_WARNING("[AllGatherV] sendCount is 0 when single rank");
            return HCCL_SUCCESS;
        } else {
            CHK_PRT_RET(sendBuf == nullptr, HCCL_ERROR("[AllGatherV] sendBuf is null when single rank"), HCCL_E_PTR);
            CHK_PRT_RET(recvBuf == nullptr, HCCL_ERROR("[AllGatherV] recvBuf is null when single rank"), HCCL_E_PTR);
        }
        return HcomAllGatherV2(tag, sendBuf, recvBuf, sendCount, dataType, group, stream);
    }
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag, sendCount, dataType, stream), tag);
    CHK_RET_AND_PRINT_IDE(HcomCheckVOpParamV2(rankId, rankSize, sendCount, recvCounts), tag);
    u64* counts = static_cast<u64*>(recvCounts);
    u64 inputCount = 0;
    for (size_t index = 0; index < rankSize; index++) {
        inputCount += counts[index];
    }
    if (inputCount == 0) {
        HCCL_INFO("[%s] inputCount[%llu] is equal to zero", __func__, inputCount);
        return HCCL_SUCCESS;
    }
    /* opParams组装 */
    Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::ALLGATHERV;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.dstRank = rankId;
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = sendCount;
    opParams.vDataDes.counts = recvCounts;
    opParams.vDataDes.displs = rdispls;
    opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom allgatherv success,take time [%lld]us, tag[%s], sendBuf[%p], sendCount[%llu], "
        "recvBuf[%p], recvCounts[%p], rdispls[%p], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AllReduce] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::ALLREDUCE, op);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom allreduce success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%d], op[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, dataType, op);

    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    const char* group, rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR(
            "[ReduceScatter] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams
        = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCESCATTER, op);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reducescatter success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "inputCount[%llu], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomReduceScatterVV2(
    const char* tag, void* sendBuf, void* sendCounts, void* sdispls, void* recvBuf, u64 recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);
    HcclUs startut = TIME_NOW();
    /* 获取通信域 */
    std::string opTag = tag;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR(
            "[ReduceScatterV] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    /* 获取rank信息 */
    uint32_t rankId;
    CHK_RET(hcclComm->GetRankId(rankId));
    uint32_t rankSize;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    /* 入参校验 */
    if (rankSize == 1) {
        /* rankSize为1时，退化为ReduceScatter */
        // 检查异常回退RSV的情况
        if (recvCount == 0) {
            HCCL_WARNING("[ReduceScatterV] recvCount is 0 when single rank");
            return HCCL_SUCCESS;
        } else {
            CHK_PRT_RET(
                sendBuf == nullptr, HCCL_ERROR("[ReduceScatterV] sendBuf is null when single rank"), HCCL_E_PTR);
            CHK_PRT_RET(
                recvBuf == nullptr, HCCL_ERROR("[ReduceScatterV] recvBuf is null when single rank"), HCCL_E_PTR);
        }
        return HcomReduceScatterV2(tag, sendBuf, recvBuf, recvCount, dataType, op, group, stream);
    }
    CHK_RET_AND_PRINT_IDE(HcomCheckOpParamV2(tag, recvCount, dataType, stream), tag);
    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOpV2(op), opTag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataTypeV2(dataType, op), opTag.c_str());
    CHK_RET_AND_PRINT_IDE(HcomCheckVOpParamV2(rankId, rankSize, recvCount, sendCounts), tag);
    u64* counts = static_cast<u64*>(sendCounts);
    u64 inputCount = 0;
    for (size_t index = 0; index < rankSize; index++) {
        inputCount += counts[index];
    }
    if (inputCount == 0) {
        HCCL_INFO("[%s] inputCount[%llu] is equal to zero", __func__, inputCount);
        return HCCL_SUCCESS;
    }
    /* opParams组装 */
    Hccl::CollOpParams opParams;
    opParams.opType = Hccl::OpType::REDUCESCATTERV;
    opParams.dataType = HcclDataTypeToDataType(dataType);
    opParams.reduceOp = HcclReduceOpToReduceOp(op);
    opParams.dstRank = rankId;
    opParams.sendBuf = sendBuf;
    opParams.recvBuf = recvBuf;
    opParams.count = recvCount;
    opParams.vDataDes.counts = sendCounts;
    opParams.vDataDes.displs = sdispls;
    opParams.vDataDes.dataType = HcclDataTypeToDataType(dataType);
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reducescatterv success,take time [%lld]us, tag[%s], sendBuf[%p], sendCounts[%p], "
        "sdispls[%p], recvBuf[%p], recvCount[%llu], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, sendCounts, sdispls, recvBuf, recvCount, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group,
    rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[SendV2] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, nullptr, count, dataType, Hccl::OpType::SEND);
    opParams.dstRank = destRank;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom send success,time[%lld]us,tag[%s],inputPtr[%p],count[%llu],dataType[%s],destRank[%u],"
        "srTag[%u]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(), destRank,
        srTag);

    return HCCL_SUCCESS;
}

HcclResult HcomReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group,
    rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Recv] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(nullptr, outputPtr, count, dataType, Hccl::OpType::RECV);
    opParams.dstRank = srcRank;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom receive success,time[%lld]us,tag[%s],outputPtr[%p],count[%llu],dataType[%s],srcRank[%u],"
        "srTag[%u]",
        DURATION_US(TIME_NOW() - startut), tag, outputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(), srcRank,
        srTag);
    return HCCL_SUCCESS;
}

HcclResult HcomGetRankIdV2(const char* group, u32* rankId)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();

    // 校验通信域非空
    CHK_PRT_RET(
        hcomCommInfoV2.pComm == nullptr,
        HCCL_ERROR("[Get][RankId]hcomCommInfoV2.pComm is null, "
                   "please check if the initialize process is called."),
        HCCL_E_PTR);

    // 校验worldgroup
    std::string strGroup = (group == nullptr || strlen(group) == 0) ? HCCL_WORLD_GROUP : group;
    if (strGroup == HCCL_WORLD_GROUP) {
        *rankId = hcomCommInfoV2.commParams.myRank;
        HCCL_INFO("hcom get world rank id success, rankId[%u]", *rankId);
        return HCCL_SUCCESS;
    }

    // 获取group
    HcclGroupParamsV2 hcclGroupParamsV2;
    CHK_RET(GetHcclGroupParams(strGroup, hcclGroupParamsV2));

    // 获取rankId
    *rankId = hcclGroupParamsV2.groupRank;

    HCCL_INFO("hcom get rank id success, group[%s], rankId[%u]", strGroup.c_str(), *rankId);
    return HCCL_SUCCESS;
}

HcclResult HcomGetWorkspaceSubStreamNumV2(
    const char* group, u64& streamNum, u64 dataSize, HcclDataType dataType, HcclCMDType optype)
{
    HCCL_INFO("[%s] start.", __func__);

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    HcclOpType hcclOpType = static_cast<HcclOpType::Value>(optype);

    if (OP_TYPE_MAP.find(optype) == OP_TYPE_MAP.end()) {
        HCCL_ERROR("[HcomGetWorkspaceSubStreamNumV2], does not support opType[%s].", hcclOpType.Describe().c_str());
        return HCCL_E_PARA;
    }
    Hccl::CollOffloadOpResReq resReq{};
    Hccl::OpType opType = OP_TYPE_MAP.at(optype);
    CHK_RET(hcclComm->CalcCollOffloadOpRes(opType, dataSize, dataType, resReq));
    streamNum = resReq.requiredSubQueNum;
    HCCL_INFO("[HcomGetWorkspaceSubStreamNumV2] GetWorkspaceSubStreamNum success, streamNum[%llu]", streamNum);
    return HCCL_SUCCESS;
}

HcclResult
HcomGetWorkspaceMemSizeV2(const std::string& opType, u64 count, HcclDataType dataType, const char* group, u64& memSize)
{
    HCCL_INFO("[%s] start.", __func__);
    if ((dataType < HCCL_DATA_TYPE_INT8) || (dataType > HCCL_DATA_TYPE_FP8E8M0)) {
        HCCL_ERROR("[%s] does not support data type[%s].", __func__, GetDataTypeEnumStrV2(dataType).c_str());
        return HCCL_E_PARA;
    }

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    if (OP_TYPE_STR.find(opType) == OP_TYPE_STR.end()) {
        HCCL_ERROR("[%s] does not support opType[%s].", __func__, opType.c_str());
        return HCCL_E_PARA;
    }
    Hccl::CollOffloadOpResReq resReq{};
    Hccl::OpType optype = OP_TYPE_STR.at(opType);
    u64 dataSize = SIZE_TABLE[dataType] * count;
    CHK_RET(hcclComm->CalcCollOffloadOpRes(optype, dataSize, dataType, resReq));
    memSize = resReq.requiredScratchMemSize;
    HCCL_INFO("[%s] GetWorkspaceMemSize success, memSize[%llu]", __func__, memSize);
    return HCCL_SUCCESS;
}

HcclResult HcomSetWorkspaceResourceV2(
    const std::string& tag, const char* group, std::vector<rtStream_t> stream, void* memPtr, u64 maxSize)
{
    HCCL_INFO("[%s] start.", __func__);

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    /* 设定 workspace 内存资源 */
    CHK_RET(hcclComm->SetCollOffloadSlaveStreams(tag, stream));
    CHK_RET(hcclComm->SetCollOffloadScratchBuf(tag, memPtr, maxSize));

    HCCL_INFO("[%s] success, maxSize[%llu]", __func__, maxSize);
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVV2(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream,
    const char* tag)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AlltoAllV]comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, 0, sendType, group, stream));
    CHK_RET(HcomCheckDataTypeV2(recvType));

    /* 根据ranksize校验相关入参 */
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    CHK_RET(HcomCheckAlltoAllVExternalMemV2(sendBuf, sendCounts, recvBuf, recvCounts, rankSize));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(
        const_cast<void*>(sendBuf), const_cast<void*>(recvBuf), 0, HcclDataType::HCCL_DATA_TYPE_RESERVED,
        Hccl::OpType::ALLTOALLV);
    opParams.all2AllVDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllVDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.all2AllVDataDes.sendCounts = const_cast<void*>(sendCounts);
    opParams.all2AllVDataDes.recvCounts = const_cast<void*>(recvCounts);
    opParams.all2AllVDataDes.sdispls = const_cast<void*>(sdispls);
    opParams.all2AllVDataDes.rdispls = const_cast<void*>(rdispls);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcomAlltoAllV success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCounts[%p],"
        "recvCounts[%p], sendType[%s], recvType[%s], group[%s].",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, recvBuf, sendCounts, recvCounts,
        GetDataTypeEnumStrV2(sendType).c_str(), GetDataTypeEnumStrV2(recvType).c_str(), group);
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllVCV2(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, rtStream_t stream, const char* tag)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 获取通信域句柄并入参校验 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AlltoAllVC]comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, 0, sendType, group, stream));
    CHK_RET(HcomCheckDataTypeV2(recvType));

    /* 根据ranksize校验相关入参 */
    u32 rankSize = 0;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    u32 myRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetRankId(myRank));
    bool isEmpty = false;
    CHK_RET(HcomCheckAlltoAllVCEmptyV2(sendBuf, sendCountMatrix, recvBuf, rankSize, isEmpty));
    if (isEmpty) {
        HCCL_INFO("[HcclAlltoAllVCV2] sendCountMatrix is Empty");
        return HCCL_SUCCESS;
    }
    CHK_RET(HcomCheckAlltoAllVCExternalMemV2(sendBuf, sendCountMatrix, recvBuf, rankSize, myRank));

    std::string strGroup = (group == nullptr) ? HCCL_WORLD_GROUP : group;
    s32 streamId = HrtGetStreamId(stream);
    s32 deviceLogicId = HrtGetDevice();
    u64 sendCountMatrixHash;
    HcomGetHashFromSendCountMatrixV2(sendCountMatrixHash, sendCountMatrix, rankSize, tag);
    /* 接口交互信息日志 */
    HCCL_RUN_INFO(
        "Entry-HcomAlltoAllVC:tag[%s], sendBuf[%p], sendCountMatrixHash[%llu], sendType[%s], "
        "recvBuf[%p], recvType[%s], group[%s], streamId[%d], deviceLogicId[%d]",
        tag, sendBuf, sendCountMatrixHash, GetDataTypeEnumStrV2(sendType).c_str(), recvBuf,
        GetDataTypeEnumStrV2(recvType).c_str(), strGroup.c_str(), streamId, deviceLogicId);

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(
        const_cast<void*>(sendBuf), const_cast<void*>(recvBuf), 0, HcclDataType::HCCL_DATA_TYPE_RESERVED,
        Hccl::OpType::ALLTOALLVC);
    opParams.all2AllVCDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllVCDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.all2AllVCDataDes.sendCountMatrix = const_cast<void*>(sendCountMatrix);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "HcomAlltoAllVC success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], "
        "sendType[%s], recvType[%s], group[%s].",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, recvBuf, GetDataTypeEnumStrV2(sendType).c_str(),
        GetDataTypeEnumStrV2(recvType).c_str(), group);
    return HCCL_SUCCESS;
}

HcclResult HcomAlltoAllV2(
    const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
    HcclDataType recvType, const char* group, rtStream_t stream, const char* tag)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[AlltoAll]comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, sendCount, sendType, stream));
    CHK_RET(HcomCheckOpParamV2(tag, recvCount, recvType, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(
        const_cast<void*>(sendBuf), const_cast<void*>(recvBuf), 0, HcclDataType::HCCL_DATA_TYPE_RESERVED,
        Hccl::OpType::ALLTOALL);
    opParams.all2AllDataDes.recvCount = recvCount;
    opParams.all2AllDataDes.sendCount = sendCount;
    opParams.all2AllDataDes.sendType = HcclDataTypeToDataType(sendType);
    opParams.all2AllDataDes.recvType = HcclDataTypeToDataType(recvType);
    opParams.dataType = HcclDataTypeToDataType(sendType);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    HCCL_RUN_INFO(
        "HcomAlltoAll success,take time [%lld]us, tag[%s], sendBuf[%p], recvBuf[%p], sendCount[%llu], "
        "recvCounts[%llu], sendType[%s], recvType[%s], group[%s].",
        DURATION_US(TIME_NOW() - startut), tag, sendBuf, recvBuf, sendCount, recvCount,
        GetDataTypeEnumStrV2(sendType).c_str(), GetDataTypeEnumStrV2(recvType).c_str(), group);

    return HCCL_SUCCESS;
}

HcclResult HcomGetAlltoAllStagedWorkSpaceMemSizeV2(
    const char* group, [[maybe_unused]] u64* sendCounts, [[maybe_unused]] u64* sdispls, HcclDataType sendType,
    [[maybe_unused]] u64* recvCounts, [[maybe_unused]] u64* rdispls, HcclDataType recvType, u64& memSize)
{
    HCCL_INFO("[%s] start.", __func__);

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckDataTypeV2(sendType));
    CHK_RET(HcomCheckDataTypeV2(recvType));

    Hccl::CollOffloadOpResReq resReq{};
    Hccl::OpType optype = Hccl::OpType::ALLTOALL;
    u64 dataSize = 0; // ??
    CHK_RET(hcclComm->CalcCollOffloadOpRes(optype, dataSize, sendType, resReq));
    memSize = resReq.requiredScratchMemSize;

    // memSize = 200 * 1024 * 1024; //  需要200M
    HCCL_INFO("[%s] success, memSize[%llu]", __func__, memSize);
    return HCCL_SUCCESS;
}

HcclResult HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2(const char* group, u64& memSize)
{
    HCCL_INFO("[%s] start.", __func__);

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    Hccl::CollOffloadOpResReq resReq;
    Hccl::OpType optype = Hccl::OpType::ALLTOALLVC;

    u64 dataSize = 0; // 不涉及ScratchMenSize
    // 为保证流程执行填写默认dataType
    CHK_RET(hcclComm->CalcCollOffloadOpRes(optype, dataSize, HCCL_DATA_TYPE_INT8, resReq));
    memSize = resReq.requiredScratchMemSize;

    // memSize = 200 * 1024 * 1024; //  需要200M
    HCCL_INFO("[%s] success, memSize[%llu]", __func__, memSize);
    return HCCL_SUCCESS;
}

HcclResult HcomBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Broadcast] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    CHK_RET(HcomCheckUserRankV2(rankSize, root));
    Hccl::CollOpParams opParams = GetHcclOpParams(ptr, ptr, count, dataType, Hccl::OpType::BROADCAST);
    opParams.root = root;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom broadcast success,take time [%lld]us,tag[%s], input_ptr[%p], count[%llu], data_type[%s], "
        "root[%u]",
        DURATION_US(TIME_NOW() - startut), tag, ptr, count, GetDataTypeEnumStrV2(dataType).c_str(), root);
    return HCCL_SUCCESS;
}

HcclResult HcomReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    const char* group, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("[Reduce] comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, group, stream));

    /* 入参的正确性由HCCL确保 */
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    CHK_RET(HcomCheckUserRankV2(rankSize, root));
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCE, op);
    opParams.root = root;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reduce success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], "
        "data_type[%s], op[%s], root[%u]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(),
        GetReduceOpEnumStrV2(op).c_str(), root);

    return HCCL_SUCCESS;
}

HcclResult HcomGetLocalRankSizeV2(const char* group, u32* localRankSize)
{
    CHK_RET(HcomCheckGroupNameV2(group));
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    u32 layer0NetInstanceNum = 0;
    u32* instSizeList = nullptr;
    CHK_RET(hcclComm->GetInstSizeListByNetLayer(0, &instSizeList, &layer0NetInstanceNum));
    if (layer0NetInstanceNum == 0) {
        HCCL_ERROR("[HcomGetLocalRankSizeV2] The layer0NetInstanceNum is zero, commId[%s]", hcclComm->GetId().c_str());
        return HCCL_E_INTERNAL;
    }
    *localRankSize = rankSize / layer0NetInstanceNum;
    HCCL_INFO(
        "[HcomGetLocalRankSizeV2] end, layer0NetInstanceNum[%u], localRankSize[%u], rankSize[%u], commId[%s]",
        layer0NetInstanceNum, *localRankSize, rankSize, hcclComm->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetLocalRankIdV2(const char* group, u32* localRankId)
{
    CHK_RET(HcomCheckGroupNameV2(group));
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    u32 rankId = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetRankId(rankId));
    u32 localRankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcomGetLocalRankSizeV2(group, &localRankSize));
    if (localRankSize == 0) {
        HCCL_ERROR("[HcomGetLocalRankIdV2] The localRankSize is zero, commId[%s]", hcclComm->GetId().c_str());
        return HCCL_E_INTERNAL;
    }
    *localRankId = rankId % localRankSize;
    HCCL_INFO(
        "[HcomGetLocalRankIdV2] end, rankId[%u], localRankSize[%u], localRankId[%u], commId[%s]", rankId, localRankSize,
        *localRankId, hcclComm->GetId().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommHandleByGroupV2(const char* group, HcclComm* commHandle)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    *commHandle = static_cast<HcclComm>(hcclComm.get());
    return HCCL_SUCCESS;
}

HcclResult HcomCalcTaskNumV2(HcomOpParam* hcomOpParam, u32& taskNum)
{
    /* 通信域 */
    HCCL_INFO("HcomCalcTaskNumV2 start.");
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(hcomOpParam->group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR(
            "comm with group name [%s] is not found",
            hcomOpParam->group == nullptr ? HCCL_WORLD_GROUP : hcomOpParam->group),
        HCCL_E_NOT_FOUND);

    Hccl::DataType hcclDataType = Hccl::DataType::INVALID;
    if (hcomOpParam->dataType != HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        hcclDataType = HcclDataTypeToDataType(hcomOpParam->dataType);
    }

    if (OP_TYPE_STR.find(hcomOpParam->opType) == OP_TYPE_STR.end()) {
        HCCL_ERROR("[HcomCalcTaskNumV2], does not support opType[%s].", hcomOpParam->opType);
        return HCCL_E_PARA;
    }
    Hccl::OpType hcclOpType = OP_TYPE_STR.at(hcomOpParam->opType);
    CHK_RET(hcclComm->CalcTaskNum(hcclOpType, hcclDataType, hcomOpParam->count, taskNum));
    return HCCL_SUCCESS;
}

HcclResult HcomGetTopoDescV2(const char* group, HcclTopoDescs* topoDescs, uint32_t topoSize)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    CHK_RET(hcclComm->GetTopoDesc(topoDescs, topoSize));

    return HCCL_SUCCESS;
}

HcclResult HcomCreateCommCclBufV2(const char* group)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->CreateCommCclBuf());
    return HCCL_SUCCESS;
}

HcclResult HcomGetInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetInCclBuf(commInputPtr, commInputSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGetOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetOutCclBuf(commOutputPtr, commOutputSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGetIndirectInCclBufV2(const char* group, void*& commInputPtr, u64& commInputSize)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetIndirectInputCclBuf(commInputPtr, commInputSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGetIndirectOutCclBufV2(const char* group, void*& commOutputPtr, u64& commOutputSize)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetIndirectOutputCclBuf(commOutputPtr, commOutputSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGraphCreateCommCclBufV2(const int64_t& hcomComm)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(hcomComm);
    CHK_RET(hcclComm->CreateCommCclBuf());
    return HCCL_SUCCESS;
}

HcclResult HcomGraphGetInCclBufV2(const int64_t& hcomComm, void*& commInputPtr, u64& commInputSize)
{
    HCCL_INFO("[%s] start.", __func__);
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(hcomComm);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetInCclBuf(commInputPtr, commInputSize));
    return HCCL_SUCCESS;
}

HcclResult HcomGraphGetOutCclBufV2(const int64_t& hcomComm, void*& commOutputPtr, u64& commOutputSize)
{
    HCCL_INFO("[%s] start.", __func__);
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(hcomComm);
    CHK_RET(hcclComm->CreateCommCclBuf());
    CHK_RET(hcclComm->GetOutCclBuf(commOutputPtr, commOutputSize));
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetRankIdV2(s64 opBaseHcom, u32* rankId)
{
    HCCL_INFO("[%s] start.", __func__);
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_RET(hcclComm->GetRankId(*rankId));
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphGetRankSizeV2(s64 opBaseHcom, u32* rankSize)
{
    HCCL_INFO("[%s] start.", __func__);
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_RET(hcclComm->GetRankSize(rankSize));
    return HCCL_SUCCESS;
}

HcclResult HcclCommGraphAllGatherV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, s64 opBaseHcom,
    rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参校验 */
    CHK_RET(HcomCheckOpParamV2(tag, inputCount, dataType, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, inputCount, dataType, Hccl::OpType::ALLGATHER);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom graph allgather success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "inputCount[%llu], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, inputCount, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomGraphAllReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::ALLREDUCE, op);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom allreduce success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "count[%llu], data_type[%d], op[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, dataType, op);

    return HCCL_SUCCESS;
}

HcclResult HcomGraphReduceScatterV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, s64 opBaseHcom,
    rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams
        = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCESCATTER, op);
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reducescatter success,take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], "
        "inputCount[%llu], data_type[%d]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, dataType);

    return HCCL_SUCCESS;
}

HcclResult HcomGraphReduceV2(
    const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root,
    s64 opBaseHcom, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 入参校验 */
    CHK_RET(HcomCheckReductionOpV2(op));
    CHK_RET(HcomCheckReduceDataTypeV2(dataType, op));
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参的正确性由HCCL确保 */
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    CHK_RET(HcomCheckUserRankV2(rankSize, root));
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, Hccl::OpType::REDUCE, op);
    opParams.root = root;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom reduce success, take time [%lld]us, tag[%s], input_ptr[%p], output_ptr[%p], count[%llu], "
        "data_type[%s], op[%s], root[%u]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, outputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(),
        GetReduceOpEnumStrV2(op).c_str(), root);

    return HCCL_SUCCESS;
}

HcclResult HcomGraphSendV2(
    const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参校验 */
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, nullptr, count, dataType, Hccl::OpType::SEND);
    opParams.dstRank = destRank;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom send success,time[%lld]us,tag[%s],inputPtr[%p],count[%llu],dataType[%s],destRank[%u],"
        "srTag[%u]",
        DURATION_US(TIME_NOW() - startut), tag, inputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(), destRank,
        srTag);

    return HCCL_SUCCESS;
}

HcclResult HcomGraphReceiveV2(
    const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, s64 opBaseHcom,
    rtStream_t& stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参校验 */
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 入参的正确性由HCCL确保 */
    Hccl::CollOpParams opParams = GetHcclOpParams(nullptr, outputPtr, count, dataType, Hccl::OpType::RECV);
    opParams.dstRank = srcRank;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));
    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom receive success,time[%lld]us,tag[%s],outputPtr[%p],count[%llu],dataType[%s],srcRank[%u],"
        "srTag[%u]",
        DURATION_US(TIME_NOW() - startut), tag, outputPtr, count, GetDataTypeEnumStrV2(dataType).c_str(), srcRank,
        srTag);
    return HCCL_SUCCESS;
}

HcclResult HcomGraphBroadcastV2(
    const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, s64 opBaseHcom, rtStream_t stream)
{
    HCCL_INFO("[%s] start.", __func__);

    HcclUs startut = TIME_NOW();

    /* 通信域 */
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(opBaseHcom);
    CHK_PTR_NULL(hcclComm);

    /* 入参校验 */
    CHK_RET(HcomCheckOpParamV2(tag, count, dataType, stream));

    /* 入参的正确性由HCCL确保 */
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(hcclComm->GetRankSize(&rankSize));
    CHK_RET(HcomCheckUserRankV2(rankSize, root));
    Hccl::CollOpParams opParams = GetHcclOpParams(ptr, ptr, count, dataType, Hccl::OpType::BROADCAST);
    opParams.root = root;
    std::string opTag = tag;
    CHK_RET(hcclComm->LoadOffloadCollOp(opTag, opParams, stream));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "hcom broadcast success,take time [%lld]us,tag[%s], input_ptr[%p], count[%llu], data_type[%s], "
        "root[%u]",
        DURATION_US(TIME_NOW() - startut), tag, ptr, count, GetDataTypeEnumStrV2(dataType).c_str(), root);
    return HCCL_SUCCESS;
}

HcclResult HcomGetDevTypeV2(Hccl::DevType& devType)
{
    HCCL_INFO("[%s] start.", __func__);
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    devType = hcomCommInfoV2.commParams.devType;
    HCCL_INFO("HcomGetDevTypeV2, devType[%s]", devType.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult HcomGetDevIdV2(const char* group, s32* devId)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    auto retDevId = static_cast<s32>(hcclComm->GetDeviceLogicId());
    *devId = retDevId;
    HCCL_INFO("HcomGetDeviceIdV2, devId[%d]", *devId);
    return HCCL_SUCCESS;
}

HcclResult HcomSetGlobalWorkSpaceV2(const char* group, const std::vector<void*>& globalWorkSpaceAddr)
{
    HCCL_INFO("[%s] start.", __func__);
    (void)globalWorkSpaceAddr;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->SetGlobalWorkSpace());
    return HCCL_SUCCESS;
}

HcclResult HcomGetInitStatusV2(bool& initiated)
{
    HCCL_INFO("[%s] start.", __func__);
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    initiated = !(hcomCommInfoV2.pComm == nullptr);
    HCCL_INFO("[%s] initiated[%d].", __func__, initiated);
    return HCCL_SUCCESS;
}

/* 实现和1.0一致，获取到通信域指针后就返回成功 */
HcclResult HcomCheckCommValidityV2(const char* group)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult HcomSupportDeterministicOptimV2(const char* group, bool& isDeterministicOptim)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    isDeterministicOptim = true;
    HCCL_WARNING("HcomSupportDeterministicOptimV2 does not support at A5! set isDeterministicOptim to true.");
    return HCCL_SUCCESS;
}

HcclResult HcomSetAivCoreLimitV2(const char* group, u32 aivCoreLimit)
{
    HCCL_INFO("[%s] start.", __func__);
    CHK_PRT_RET(
        aivCoreLimit == 0, HCCL_ERROR("[HcomSetAivCoreLimitV2] aivCoreLimit[%u] invalid", aivCoreLimit), HCCL_E_PARA);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(hcclComm->SetAivCoreLimit(aivCoreLimit));
    HCCL_RUN_INFO("HcomSetAivCoreLimitV2 group[%s] aivCoreLimit[%u]", group ? group : HCCL_WORLD_GROUP, aivCoreLimit);
    return HCCL_SUCCESS;
}

HcclResult HcomSetQosCfgV2(const char* group, const u32 qosCfg)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    (void)qosCfg;
    HCCL_WARNING("HcomSetQosCfgV2 does not support at A5!");
    return HCCL_SUCCESS;
}

HcclResult HcomGraphSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName)
{
    HCCL_INFO("[%s] start.", __func__);
    (void)comm;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcomCheckOpParamV2(count, dataType, group));

    HcclOpType hcclOpType = static_cast<HcclOpType::Value>(opType);
    if (OP_TYPE_MAP.find(opType) == OP_TYPE_MAP.end()) {
        HCCL_ERROR("[HcomGraphSelectAlgV2], does not support opType[%s].", hcclOpType.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::OpType optype = OP_TYPE_MAP.at(opType);
    Hccl::CollOpParams opParams = GetHcclOpParams(nullptr, nullptr, count, dataType, optype, op, true);
    CHK_RET(hcclComm->ExecAlgSelect(opParams, aivCoreLimit, ifAiv, algName));
    return HCCL_SUCCESS;
}

HcclResult HcomSelectAlgV2(
    s64 comm, const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, HcclReduceOp op,
    int32_t aivCoreLimit, bool& ifAiv, std::string& algName)
{
    /* 通信域 */
    HCCL_INFO("[%s] start.", __func__);
    (void)group;
    Hccl::HcclCommunicator* hcclComm = reinterpret_cast<Hccl::HcclCommunicator*>(comm);

    CHK_RET(HcomCheckOpParamV2(count, dataType));

    HcclOpType hcclOpType = static_cast<HcclOpType::Value>(opType);
    if (OP_TYPE_MAP.find(opType) == OP_TYPE_MAP.end()) {
        HCCL_ERROR("[HcomSelectAlgV2], does not support opType[%s].", hcclOpType.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::OpType optype = OP_TYPE_MAP.at(opType);
    Hccl::CollOpParams opParams = GetHcclOpParams(nullptr, nullptr, count, dataType, optype, op, true);
    CHK_RET(hcclComm->ExecAlgSelect(opParams, aivCoreLimit, ifAiv, algName));
    return HCCL_SUCCESS;
}

HcclResult HcomUnloadTaskV2(const std::string group, const char* tag)
{
    HCCL_INFO("[%s] start.", __func__);
    HcclUs startut = TIME_NOW();
    CHK_RET(HcomCheckGroupNameV2(group.c_str()));
    CHK_RET(HcomCheckTagV2(tag));
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    HcclResult ret = GetHcclCommV2(group.c_str(), hcclComm);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING(
            "[HcomUnloadTaskV2]errNo[0x%016llx] group[%s] group is not exist", HCOM_ERROR_CODE(HCCL_E_NOT_FOUND),
            group.c_str()),
        HCCL_SUCCESS);
    std::string opTag = tag;
    CHK_RET(hcclComm->ClearOpResource(opTag));
    HCCL_RUN_INFO("hcom unload task success,take time [%lld]us,tag[%s]", DURATION_US(TIME_NOW() - startut), tag);
    return HCCL_SUCCESS;
}

HcclResult HcclCommResetQosCfgV2()
{
    HCCL_WARNING("HcclCommResetQosCfgV2 does not support!");
    return HCCL_SUCCESS;
}

HcclResult HcomResetQosCfgV2()
{
    HCCL_WARNING("HcomGetCommCCLBufferSizeV2 does not support!");
    return HCCL_SUCCESS;
}

HcclResult HcclCommSetQosCfgV2()
{
    HCCL_WARNING("HcclCommSetQosCfgV2 does not support!");
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommCCLBufferSizeV2()
{
    HCCL_WARNING("HcomGetCommCCLBufferSizeV2 does not support!");
    return HCCL_SUCCESS;
}

HcclResult HcomSetAivClearEnableV2(const char* group, bool aivClearEnable)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);
    CHK_RET(HcomCheckGroupNameV2(group));

    CHK_RET(hcclComm->SetAivClearEnable(aivClearEnable));
    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcomCalcNumBlocksV2(
    const char* group, HcclCMDType opType, u64 count, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
    u32& numBlocks)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcomCheckOpParamV2(count, dataType, group));

    HcclOpType hcclOpType = static_cast<HcclOpType::Value>(opType);
    if (OP_TYPE_MAP.find(opType) == OP_TYPE_MAP.end()) {
        HCCL_ERROR("[HcomGraphSelectAlgV2], does not support opType[%s].", hcclOpType.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::OpType optype = OP_TYPE_MAP.at(opType);
    Hccl::CollOpParams opParams
        = GetHcclOpParams(nullptr, nullptr, count, dataType, optype, HCCL_REDUCE_RESERVED, true);
    CHK_RET(hcclComm->CalcNumBlocks(opParams, aivCoreLimit, algName, numBlocks));
    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclGetAlgExecParamV2(
    const std::string& tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType,
    bool clearEnable, HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit)
{
    HCCL_INFO("[%s] start.", __func__);
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_PRT_RET(
        GetHcclCommV2(group, hcclComm) == HCCL_E_NOT_FOUND,
        HCCL_ERROR("comm with group name [%s] is not found", group == nullptr ? HCCL_WORLD_GROUP : group),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcomCheckOpParamV2(count, dataType, group));

    HcclOpType hcclOpType = static_cast<HcclOpType::Value>(opType);
    if (OP_TYPE_MAP.find(opType) == OP_TYPE_MAP.end()) {
        HCCL_ERROR("[HcomGraphSelectAlgV2], does not support opType[%s].", hcclOpType.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    Hccl::OpType optype = OP_TYPE_MAP.at(opType);
    Hccl::CollOpParams opParams = GetHcclOpParams(inputPtr, outputPtr, count, dataType, optype, op, true);
    opParams.opTag = tag;
    CHK_RET(hcclComm->GetAlgExecParam(opParams, clearEnable, commContext, len, aivCoreLimit));
    HCCL_INFO("[%s] end.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcomGetL0TopoTypeExV2(const char* group, CommTopo* topoType, uint32_t flag)
{
    CHK_PTR_NULL(topoType);
    CHK_PTR_NULL(group);

    bool isSetDevice = static_cast<bool>(flag & (~(0xfffffffe)));
    if (isSetDevice) {
        HCCL_ERROR("current only support no setdevice, flag[%u]", flag);
        return HCCL_E_PARA;
    }

    std::string identifier(group);
    return CommTopoDesc::GetInstance().GetL0TopoType(identifier, topoType);
}

HcclResult HcomGetRankSizeExV2(const char* group, uint32_t* rankSize, uint32_t flag)
{
    CHK_PTR_NULL(rankSize);
    CHK_PTR_NULL(group);

    bool isSetDevice = static_cast<bool>(flag & (~(0xfffffffe)));
    if (isSetDevice) {
        HCCL_ERROR("current only support no setdevice, flag[%u]", flag);
        return HCCL_E_PARA;
    }

    std::string identifier(group);
    return CommTopoDesc::GetInstance().GetRankSize(identifier, rankSize);
}

HcclResult HcomMc2AiCpuStreamAllocAndGetV2(const char* group, u32 streamMode, rtStream_t* aiCpuStream)
{
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(aiCpuStream);
    (void)streamMode;

    std::shared_ptr<Hccl::HcclCommunicator> hcclComm;
    CHK_RET(GetHcclCommV2(group, hcclComm));
    CHK_RET(hcclComm->Mc2AiCpuStreamAllocAndGetV2(aiCpuStream));
    CHK_PTR_NULL(*aiCpuStream);
    return HCCL_SUCCESS;
}

HcclResult HcomSetAttachedStreamV2()
{
    HCCL_WARNING("HcomSetAttachedStreamV2 does not support!");
    return HCCL_SUCCESS;
}

HcclResult HcomReleaseSubCommsV2()
{
    HCCL_WARNING("HcomReleaseSubCommsV2 does not support!");
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
