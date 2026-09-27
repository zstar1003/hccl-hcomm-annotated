/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mesh_opbase_small_count_deterministic_executor.h"

const u32 RANK_SIZE_FOUR = 4;
namespace hccl {
// 准入条件: 确定性&小数据量
CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    CCLMemSlice_ = false;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum;
    if (IsPowerOfTwo(topoAttr_.deviceNumPerAggregation)) {
        // level0 为HD staged
        totalStreamNum = 2U;
    } else {
        totalStreamNum = 1U;
    }

    streamNum = totalStreamNum - 1U;
    HCCL_INFO(
        "[CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcCommInfo(
    std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

bool CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::IsPowerOfTwo(u32 num)
{
    return (num & (num - 1)) == 0;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommType commType;
    if (topoAttr_.deviceNumPerAggregation >= RANK_SIZE_FOUR && IsPowerOfTwo(topoAttr_.deviceNumPerAggregation)) {
        // HD stage
        commType = CommType::COMM_TAG_HALVING_DOUBLING;
    } else {
        // NHR
        commType = CommType::COMM_TAG_WHOLE_NHR;
    }
    CommParaInfo commParaInfo(COMM_LEVEL0, commType);
    commParaInfo.meshSinglePlane = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] start", __func__, tag_.c_str());
    CommParaInfo commParaLevel1(COMM_LEVEL1, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        commParaLevel1.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc RingCommInfo", __func__, tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc NHRCommInfo", __func__, tag_.c_str());
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        commParaLevel1.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc NBCommInfo", __func__, tag_.c_str());
    } else {
        commParaLevel1.commType = CommType::COMM_TAG_HALVING_DOUBLING;
        HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc HDCommInfo", __func__, tag_.c_str());
    }
    commParaLevel1.forceRdma = false;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel1, opTransport[commParaLevel1.commPlane], inputType, outputType));
    HCCL_INFO("[%s][CalcLevel1CommInfo]tag[%s] Calc CommInfo Finish", __func__, tag_.c_str());

    return HCCL_SUCCESS;
}

u64 CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

bool CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::IsHugeData(
    const u64 curSize, [[maybe_unused]] OpParam* param)
{
    bool hugeData = (curSize * topoAttr_.userRankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE);
    return hugeData;
}

bool CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::IsSmallData(
    [[maybe_unused]] const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    // 小数据量才选到该执行器，默认为true
    return true;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor][Run]"
                  "CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor begins.");
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 curSize = execMem.count * unitSize; // 单位：字节
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_LEVEL1, level0CommInfo.localRank + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, level0CommInfo.localRank);

    execMem.inputMem = execMem.inputMem.range(0, curSize * topoAttr_.userRankSize);
    execMem.outputMem = execMem.outputMem.range(0, curSize * topoAttr_.userRankSize);
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);
    CHK_RET(RunAlgLevel1(param, reduceAttr, execMem, level1CommInfo));
    CHK_RET(RunAlgLevel0(param, reduceAttr, execMem, level0CommInfo, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::CopyFromUserInToCclIn(
    const OpParam& param, ExecMem& execMem)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    const bool preloadCopyOpt = IsPreloadCopyOptimizeCondition(param, execMem);
    DeviceMem dstMem;
    DeviceMem srcMem;
    if (preloadCopyOpt) {
        // 中转内存大小足够时，一次性搬完
        const u64 copySize = execMem.count * unitSize * topoAttr_.userRankSize;
        dstMem = execMem.inputMem.range(0, copySize);
        srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), copySize);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    } else {
        u64 copySizeOnce = execMem.count * unitSize;
        for (u32 i = 0; i < topoAttr_.userRankSize; i++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            dstMem = execMem.inputMem.range(copySizeOnce * i, copySizeOnce);
            srcMem = DeviceMem::create(
                static_cast<u8*>(execMem.inputPtr) + param.DataDes.count * unitSize * i, copySizeOnce);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::RunAlgLevel1(
    const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level1CommInfo)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    CHK_RET(CopyFromUserInToCclIn(param, execMem));

    // 第一步：节点间
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HCCL_INFO("ReduceScatter smallcount deterministic: using ring algo inter-server.");
        u64 ringSize = execMem.inputMem.size() / level1CommInfo.localRankSize;
        u64 ringCount = ringSize / unitSize;
        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        HCCL_INFO("ReduceScatter smallcount deterministic: using nhr algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
        u64 ringSize = execMem.inputMem.size() / level1CommInfo.localRankSize;
        u64 ringCount = ringSize / unitSize;
        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
        level1TempAlg->CloseBarrier();
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
        HCCL_INFO("ReduceScatter smallcount deterministic: using nonuniform-bruck algo inter-server.");
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        u64 ringSize = execMem.inputMem.size() / level1CommInfo.localRankSize;
        u64 ringCount = ringSize / unitSize;
        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, ringCount, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_REDUCESCATTER_RECURSIVE_HD, dispatcher_);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        HCCL_INFO("ReduceScatter smallcount deterministic: using halving-doubling algo inter-server.");
        u64 inputDataCount = execMem.inputMem.size() / unitSize; // count是output的数据个数
        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, inputDataCount, param.DataDes.dataType,
            param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0)));
    }
    CHK_RET(level1TempAlg->RegisterProfiler(
        (level1CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_0,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::RunAlgLevel0(
    const OpParam& param, u64 reduceAttr, ExecMem& execMem, SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    // 第二步：节点内
    // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareReduceScatterSliceData(execMem.count, unitSize, level0CommInfo.localRankSize, dataSegsSlice));

    // 每个server分配的slice大小
    u64 serverSliceSize = execMem.inputMem.size() / level1CommInfo.localRankSize;
    // 每个服务器对应的偏移
    u64 serverSliceOffset = serverSliceSize * level1CommInfo.localRank;

    HCCL_DEBUG(
        "inputMem.size=%llu, level0CommInfo.localRankSize=%u, serverSliceSize=%llu, serverSliceOffset=%llu "
        "level0CommInfo.localRank=%u level1CommInfo.localRank=%u",
        execMem.inputMem.size(), level0CommInfo.localRankSize, serverSliceSize, serverSliceOffset,
        level0CommInfo.localRank, level1CommInfo.localRank);

    DeviceMem reduceScatterMeshInput = execMem.inputMem.range(serverSliceOffset, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshInput);
    DeviceMem reduceScatterMeshOutput = execMem.outputMem.range(0, serverSliceSize);
    CHK_SMART_PTR_NULL(reduceScatterMeshOutput);

    HcomCollOpInfo opInfo
        = {"",
           reduceScatterMeshInput.ptr(),
           execMem.outputPtr,
           param.DataDes.count,
           param.DataDes.dataType,
           param.root,
           param.reduceType,
           0};
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    // HD stage 模板中ranksize必须大于等于4
    if (level0CommInfo.localRankSize >= RANK_SIZE_FOUR && IsPowerOfTwo(level0CommInfo.localRankSize)) {
        level0TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, dispatcher_);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(
            reduceScatterMeshInput, reduceScatterMeshOutput, reduceScatterMeshOutput, execMem.count,
            param.DataDes.dataType, param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), 0,
            reduceAttr, algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
            topoAttr_.userRank, &opInfo));
        HCCL_INFO("ReduceScatter smallcount deterministic: using hd stage algo inter-server.");
    } else {
        level0TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        CHK_SMART_PTR_NULL(level0TempAlg);
        CHK_RET(level0TempAlg->Prepare(reduceAttr, false));
        u64 ringSize = reduceScatterMeshInput.size() / level0CommInfo.localRankSize;
        u64 ringCount = ringSize / unitSize;
        CHK_RET(level0TempAlg->Prepare(
            reduceScatterMeshInput, reduceScatterMeshInput, reduceScatterMeshOutput, ringCount, param.DataDes.dataType,
            param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), serverSliceOffset));
        level0TempAlg->CloseBarrier();
        HCCL_INFO("ReduceScatter smallcount deterministic: using nhr algo inter-server.");
    }
    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    if (level0CommInfo.localRankSize < RANK_SIZE_FOUR || !IsPowerOfTwo(level0CommInfo.localRankSize)) {
        DeviceMem srcMem = execMem.inputMem.range(
            serverSliceOffset + dataSegsSlice[level0CommInfo.localRank].offset, execMem.count * unitSize);
        DeviceMem dstMem = DeviceMem::create(execMem.outputPtr, execMem.count * unitSize);
        CHK_SMART_PTR_NULL(srcMem);
        CHK_SMART_PTR_NULL(dstMem);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    }
    return HCCL_SUCCESS;
}

bool CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor::IsPreloadCopyOptimizeCondition(
    const OpParam& param, ExecMem& execMem)
{
    // 通信buffer足够大时，将user in到ccl的拷贝任务合并成一个
    return param.DataDes.count == execMem.count;
}

REGISTER_EXEC(
    "ReduceScatterMeshOpbaseSmallCountDeterministicExecutor", ReduceScatterMeshOpbaseSmallCountDeterministic,
    CollReduceScatterMeshOpbaseSmallCountDeterministicExecutor);

} // namespace hccl
