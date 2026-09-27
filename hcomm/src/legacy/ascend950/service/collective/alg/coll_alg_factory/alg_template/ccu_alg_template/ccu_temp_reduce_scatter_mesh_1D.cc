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

#include "ccu_instruction_reduce_scatter_mesh1d.h"
#include "ccu_rank_group.h"
#include "ccu_ctx_creator_registry.h"
#include "ccu_context_reduce_scatter_mesh1d.h"
#include "ccu_temp_reduce_scatter_mesh_1D.h"

namespace Hccl {

static CcuInstRegister<CcuContextReduceScatterMesh1D>
    g_registrarReduceScatter(CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_DIRECT);

CcuTempReduceScatterMesh1D::CcuTempReduceScatterMesh1D(
    const RankId virtualRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const std::map<RankId, u32>& tempVirtRankMap)
    : CcuAlgTemplateBase(virtualRank, tempRankSize, tempVTopo, tempVirtRankMap)
{}

CcuTempReduceScatterMesh1D::~CcuTempReduceScatterMesh1D() {}

void CcuTempReduceScatterMesh1D::InitReduceInfo(const ReduceOp& reduceOp, const DataType& dataType)
{
    reduceOp_ = reduceOp;
    dataType_ = dataType;
}

HcclResult
CcuTempReduceScatterMesh1D::CalcSliceInfo(const AllignInfo& allignInfo, const u64 dataSize, RankSliceInfo& sliceInfoVec)
{
    std::vector<SliceInfo> tmp(tempVTopo_.size());
    sliceInfoVec.resize(tempRankSize_, tmp);
    CHK_RET(CalcRsAgSliceInfoMesh(myRank_, tempRankSize_, allignInfo, dataSize, sliceInfoVec));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1D::CalcRes(AlgTempResReq& tempResReq)
{
    tempResReq.queNum = 1;
    tempResReq.streamNum = tempResReq.queNum;
    HCCL_INFO("[CalcRes] tempResReq.queNum[%u]", tempResReq.queNum);
    CHK_RET(CalcResLinksMesh(myRank_, tempRankSize_, tempVTopo_, linkNumBtwPeers_, tempResReq));
    return HcclResult::HCCL_SUCCESS;
}

uint64_t CcuTempReduceScatterMesh1D::GetMaxSliceSize() const { return UB_MAX_DATA_SIZE; }

/* CCU数据类型校验规则
 * Reduce算子：
 *      高精度模式，当dataType==outputDataType时，可选类型为FP32、FP16、BF16、UINT8、INT16、INT32;
 *      低精度模式，当dataType!=outputDataType时，dataType可选范围HIF8、E4M3、E5M2、INT8;outputDataType可选范围FP32、FP16、BF16;
 * 非Reduce算子：任意数据类型，dataType==outputDataType即可。
 */
void CcuTempReduceScatterMesh1D::CheckCcuDataType() const
{
    if (op_.opType == OpType::REDUCESCATTER && op_.reduceOp == ReduceOp::SUM) {
        if (op_.dataType == op_.outputDataType) {
            // reduce算子高精度模式
            HCCL_INFO("HIGH PRECISION");
            set<DataType> highPrecisionSupportedInputDataType
                = {DataType::FP32,  DataType::FP16,  DataType::BFP16, DataType::UINT8,
                   DataType::UINT8, DataType::INT16, DataType::INT32};
            if (highPrecisionSupportedInputDataType.count(op_.dataType) == 0) {
                THROW<CcuApiException>(StringFormat(
                    "Unsupported DataType [%s] For OpType [%s].", op_.dataType.Describe().c_str(),
                    op_.opType.Describe().c_str()));
            }
        } else {
            // reduce算子的低精度模式
            HCCL_INFO("LOW PRECISION");
            set<DataType> lowPrecisionSupportedInputDataType
                = {DataType::HIF8, DataType::FP8E4M3, DataType::FP8E5M2, DataType::INT8};
            set<DataType> lowPrecisionSupportedOutputDataType = {DataType::FP32, DataType::FP16, DataType::BFP16};
            if (lowPrecisionSupportedInputDataType.count(op_.dataType) == 0) {
                THROW<CcuApiException>(StringFormat(
                    "Unsupported Input DataType [%s] For OpType [%s].", op_.dataType.Describe().c_str(),
                    op_.opType.Describe().c_str()));
            }
            if (lowPrecisionSupportedOutputDataType.count(op_.outputDataType) == 0) {
                THROW<CcuApiException>(StringFormat(
                    "Unsupported Output DataType [%s] For OpType [%s].", op_.outputDataType.Describe().c_str(),
                    op_.opType.Describe().c_str()));
            }
        }
    } else {
        if (op_.dataType != op_.outputDataType) {
            THROW<CcuApiException>(StringFormat(
                "Inconsistent DataType[%s]--OutputDataType[%s] for OpType[%s].", op_.dataType.Describe().c_str(),
                op_.outputDataType.Describe().c_str(), op_.opType.Describe().c_str()));
        }
    }
    HCCL_INFO("CheckCcuDataType Success!");
}

HcclResult CcuTempReduceScatterMesh1D::Run(
    const TempFuncs& tempFuncs, const RankSliceInfo& sliceInfoVec, const BuffInfo& buffInfo, const ResLinks& tempLinks,
    std::vector<InsQuePtr>& tempInsQues)
{
    CHK_PRT_RET(
        tempInsQues.empty(), HCCL_ERROR("[CcuTempReduceScatterMesh1D] empty queue"), HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempInsQues[0]);
    opMode_ = tempFuncs.opMode;
    buffInfo_ = buffInfo;
    CcuInstructionReduceScatterMesh1D ccuInsReduceScatterMesh1D;
    std::vector<uint64_t> dimSize;
    dimSize.push_back(tempRankSize_);

    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t offset;
    if (op_.outputDataType == DataType::INVALID) {
        op_.outputDataType = op_.dataType;
    }

    uint64_t expandingtimes
        = DataTypeSizeGet(op_.outputDataType) / DataTypeSizeGet(op_.dataType); // 膨胀的倍数是输出类型/输入类型
    HCCL_INFO(
        "[CcuTempReduceScatterMesh1D] dataType outputDatatype %s %s", op_.dataType.Describe().c_str(),
        op_.outputDataType.Describe().c_str());
    CheckCcuDataType();
    if (opMode_ == OpMode::OPBASE) {
        if (tempFuncs.isForepart) {
            inputAddr = BufferTypeToAddr(tempFuncs.usrData.usrInSlices[myRank_].GetType());
            // 需要加上UserIn的偏移，包含了loop偏移和rank偏移
            offset = tempFuncs.usrData.usrInSlices[myRank_].GetOffset();
        } else {
            inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
            // 从inBuff获取数据，只需要加rank偏移
            offset = sliceInfoVec[myRank_][0].offset;
        }
        if (tempFuncs.isBottom) {
            outputAddr = BufferTypeToAddr(tempFuncs.usrData.usrOutSlices[0].GetType())
                         + (tempFuncs.usrData.usrOutSlices[0].GetOffset()) * expandingtimes;
        } else {
            outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff * expandingtimes;
        }
    } else {
        // 图模式没有tempFuncs.usrData，直接通过buffInfo_来获取输入输出地址
        inputAddr = BufferTypeToAddr(buffInfo_.inBuffType) + buffInfo_.inBuffBaseOff;
        outputAddr = BufferTypeToAddr(buffInfo_.outBuffType) + buffInfo_.outBuffBaseOff
                     + (tempFuncs.usrData.usrOutSlices[0].GetOffset());
        offset = tempFuncs.usrData.usrInSlices[myRank_].GetOffset();
    }
    uint64_t sliceSize = sliceInfoVec[myRank_][0].size; // 获取本rank需要处理的数据量
    uint64_t token;
    CHK_RET(GetToken(op_, token));
    ccuInsReduceScatterMesh1D.Init(
        static_cast<uint32_t>(myRank_), inputAddr, outputAddr, sliceSize, offset, token, op_, tempVTopo_);
    HCCL_INFO(
        "[CcuTempReduceScatterMesh1D] Run Init: myRank_[%d], dimSize[%llu], inputAddr[%llu],"
        "outputAddr[%llu], sliceSize[%llu], offset[%llu]",
        myRank_, dimSize[0], inputAddr, outputAddr, sliceSize, offset);

    std::vector<LinkData> links;

    for (auto& pair : tempLinks) {
        if (pair.second.empty()) {
            continue;
        }
        links.push_back(pair.second[0]);
    }
    HCCL_INFO("[CcuTempReduceScatterMesh1D] links.size[%zu]", links.size());
    ccuInsReduceScatterMesh1D.SetLinks(links);
    RankGroup rankGroup;

    for (auto& peer : tempVTopo_[0]) {
        rankGroup.AddRank(peer);
    }
    u32 cntCkeNum = 3;
    ccuInsReduceScatterMesh1D.SetCntCkeNum(cntCkeNum);
    ccuInsReduceScatterMesh1D.SetRankGroup(rankGroup);
    ccuInsReduceScatterMesh1D.Describe();
    tempInsQues[0]->Append(std::move(std::make_unique<CcuInstructionReduceScatterMesh1D>(ccuInsReduceScatterMesh1D)));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1D::GenExtIns(
    const RankGraph* rankGraph, const TemplateInfo& tmpInfo, const std::vector<InsQuePtr>& tempInsQues) const
{
    (void)rankGraph;
    (void)tmpInfo;
    (void)tempInsQues;
    // 框架解析aicpuIns，算法的algCompnnetLite在device侧直接调用Run()
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
