/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ios>
#include <iostream>

#include "log.h"

#include "ccu_instruction_all_reduce_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_all_reduce_mesh1d.h"
#include "ccu_temp_all_reduce_mesh_1D.h"

namespace Hccl {
static CcuInstRegister<CcuContextAllReduceMesh1D> g_registrarAllReduce(CcuInstType::CCU_ALL_REDUCE_MESH_1D_DIRECT);

CcuTempAllReduceMesh1D::CcuTempAllReduceMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempAllReduceMesh1D::~CcuTempAllReduceMesh1D() {}

void CcuTempAllReduceMesh1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult
CcuTempAllReduceMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    CHK_RET(CalcSliceInfoAllReduce(allignInfo, tempRankSize_, dataSize, sliceInfoVec));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

/* CCU数据类型校验规则
 * Reduce算子：
 *      高精度模式，当dataType==outputDataType时，可选类型为FP32、FP16、BF16、UINT8、INT16、INT32；
 *      低精度模式，当dataType!=outputDataType时，dataType可选范围HIF8、E4M3、E5M2、INT8；outputDataType可选范围FP32、FP16、BF16；
 * 非Reduce算子：任意数据类型，dataType==outputDataType即可。
 */
HcclResult CcuTempAllReduceMesh1D::CheckCcuDataType() const
{
    if (op_.dataType == op_.outputDataType) {
        // allreduce算子高精度模式
        HCCL_INFO("HIGH PRECISION");
        set<DataType> highPrecisionSupportedInputDataType
            = {DataType::FP32, DataType::FP16, DataType::BFP16, DataType::UINT8, DataType::INT16, DataType::INT32};
        if (highPrecisionSupportedInputDataType.count(op_.dataType) == 0) {
            HCCL_ERROR(
                "Unsupported DataType [%s] For OpType [%s].", op_.dataType.Describe().c_str(),
                op_.opType.Describe().c_str());
            return HcclResult::HCCL_E_PARA;
        }
    } else if (op_.outputDataType != DataType::INVALID) {
        // allreduce算子低精度模式
        HCCL_INFO("LOW PRECISION");
        HCCL_ERROR(
            "Unsupported LOW PRECISION, Output DataType [%s] For OpType [%s].", op_.outputDataType.Describe().c_str(),
            op_.opType.Describe().c_str());
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO("CheckCcuDataType Success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(tempInsQues.empty(), HCCL_ERROR("[CcuTempAllReduceMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CcuInstructionAllReduceMesh1D ccuInsAllReduceMesh1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);
    uint64_t inputAddr;
    uint64_t outputAddr;

    if (op_.outputDataType == DataType::INVALID) {
        op_.outputDataType = op_.dataType;
    }
    CHK_RET(CheckCcuDataType());
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[0].GetType())
                        + tempFuncs.usrData.usrInSlices[0].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        }
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                         + tempFuncs.usrData.usrOutSlices[0].GetOffset();
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff;
        }
    } else {
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff
                    + tempFuncs.usrData.usrInSlices[0].GetOffset();
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + tempFuncs.usrData.usrOutSlices[0].GetOffset();
    }
    HCCL_INFO("inputAddr[%llu], outputAddr[%llu]", inputAddr, outputAddr);

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t offset = sliceInfoVec[myRank_][0].offset;  // 自己需要 reduce 的数据基于 inputAddr 的偏移
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    ccuInsAllReduceMesh1D.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, offset, token, op_, tempVTopo_);
    HCCL_INFO(
        "[CcuTempAllReduceMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu], outputAddr[%llu],"
        "sliceSize[%llu], offset[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, offset);

    std::vector<LinkData> links;
    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempAllReduceMesh1D] links.size[%zu]", links.size());
    ccuInsAllReduceMesh1D.SetLinks(links);

    RankGroup rankGroup;

    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 4;
    ccuInsAllReduceMesh1D.SetCntCkeNum(cntCkeNum);
    ccuInsAllReduceMesh1D.SetRankGroup(rankGroup);
    HCCL_INFO("CCUInsAllReducemesh1D is [%s]", ccuInsAllReduceMesh1D.Describe().c_str());
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionAllReduceMesh1D>(ccuInsAllReduceMesh1D)));

    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
