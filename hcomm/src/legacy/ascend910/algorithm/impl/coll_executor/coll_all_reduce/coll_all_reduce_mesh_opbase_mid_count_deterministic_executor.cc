/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mesh_opbase_mid_count_deterministic_executor.h"

namespace hccl {
CollAllReduceMeshOpbaseMidCountDeterministicExecutor::CollAllReduceMeshOpbaseMidCountDeterministicExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::CalcStreamNum(u32& streamNum)
{
    const u32 level0AlltoallStreamNum = topoAttr_.deviceNumPerAggregation - 1;
    const u32 level0LocalReduceStreamNum = 1 << static_cast<int>(std::floor(log2(topoAttr_.deviceNumPerAggregation)));
    streamNum = level0AlltoallStreamNum + level0LocalReduceStreamNum;

    HCCL_INFO(
        "[%s]tag[%s] level0AlltoallStreamNum[%u], level0LocalReduceStreamNum[%u], streamNum[%u]", __func__,
        tag_.c_str(), level0AlltoallStreamNum, level0LocalReduceStreamNum, streamNum);
    return HCCL_SUCCESS;
}

HcclResult
CollAllReduceMeshOpbaseMidCountDeterministicExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO(
        "[CollAllReduceMeshOpbaseMidCountDeterministicExecutor][CalcTransportMemType]"
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

bool CollAllReduceMeshOpbaseMidCountDeterministicExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    HCCL_DEBUG(
        "[%s]isHugeData[%d], curSize[%llu], topoAttr_.deviceNumPerAggregation[%u]", __func__, hugeData, curSize,
        topoAttr_.deviceNumPerAggregation);
    return hugeData;
}

bool CollAllReduceMeshOpbaseMidCountDeterministicExecutor::IsSmallData(
    [[maybe_unused]] const u64 totalSize, const u64 curSize)
{
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::PrepareSlicesInfo(
    const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice, GroupSlicesInfo& groupSlicesInfo,
    const u32 sliceSize) const
{
    const u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    MemBlockInfo memInfo;
    u64 sizePerBlock = (execMem.count + sliceSize - 1) / sliceSize * perDataSize;
    sizePerBlock = AlgTemplateBase::RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN);
    const u64 totalSize = execMem.count * perDataSize;
    u64 sizeRemain = totalSize;
    for (u32 dataId = 0; dataId < sliceSize; dataId++) {
        u64 size = (sizeRemain > sizePerBlock) ? sizePerBlock : sizeRemain;
        u64 offset = totalSize - sizeRemain;
        memInfo.size.push_back(size);
        memInfo.userInputOffsets.push_back(offset);
        memInfo.inputOffsets.push_back(offset);
        memInfo.outputOffsets.push_back(offset);
        Slice slice{offset, size};
        dataSegsSlice.emplace_back(std::move(slice));
        sizeRemain -= size;
    }
    groupSlicesInfo.push_back(memInfo);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::RunReduceScatterLevel0(
    const OpParam& param, ExecMem& execMem, GroupSlicesInfo& groupSlicesInfo)
{
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    const u32 all2allOffset = 0;
    level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);

    CHK_RET(level0TempAlg->Prepare(
        execMem.inputPtr, execMem.inputMem, execMem.outputMem, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, groupSlicesInfo, param.reduceType, all2allOffset,
        param.DataDes.dataType, true, true));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::RunAllReduceLevel1(
    const OpParam& param, ExecMem& execMem, const std::vector<Slice>& dataSegsSlice)
{
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    const u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];
    const u32 commIndex = level0CommInfo.localRank;

    CHK_PRT_RET(
        commIndex >= dataSegsSlice.size(),
        HCCL_ERROR("[%s]commIndex[%u] >= dataSegsSlice size[%zu]", __func__, commIndex, dataSegsSlice.size()),
        HCCL_E_INTERNAL);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    const u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, param.DataDes.dataType, param.reduceType);

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_INFO("[%s]: using ring algo inter-server.", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
        HCCL_INFO("[%s]: using nhr algo inter-server.", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
        level1TempAlg->CloseBarrier();
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR_V1, dispatcher_);
        HCCL_INFO("[%s]: using nhr_v1 algo inter-server.", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NB, dispatcher_);
        HCCL_INFO("[%s]: using nb algo inter-server.", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING, dispatcher_);
        HCCL_INFO("[%s]: using Recursive halving-doubling algo inter-server.", __func__);
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(reduceAttr));
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    const u32 rankSize = level1CommInfo.localRankSize;

    const u64 hdCount = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(level1TempAlg->Prepare(
        allreduceInput, allreduceOutput, allreduceOutput, hdCount, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    CHK_RET(level1TempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::RunAllGatherLevel0(
    const OpParam& param, ExecMem& execMem, const std::vector<Slice>& dataSegsSlice)
{
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    level0TempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);

    CHK_SMART_PTR_NULL(level0TempAlg);
    CHK_RET(level0TempAlg->Prepare(
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
        level0CommInfo.localRank, level0CommInfo.localRankSize));

    u32 rankSize = level0CommInfo.localRankSize;
    CHK_RET(level0TempAlg->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

    CHK_RET(level0TempAlg->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s]userRank[%u] starts.", __func__, topoAttr_.userRank);

    CHK_RET(CheckCommSize(COMM_LEVEL0, 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    const u32 sliceNum = level0CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;
    GroupSlicesInfo groupSlicesInfoLevel0;
    CHK_RET(PrepareSlicesInfo(param, execMem, dataSegsSlice, groupSlicesInfoLevel0, sliceNum));
    CHK_RET(ActiveSlaveStreams(param.stream));

    /* STAGE 0: level 0 reduce scatter - plant local reduce */
    CHK_RET(RunReduceScatterLevel0(param, execMem, groupSlicesInfoLevel0));
    HCCL_INFO("[%s]AllReduce stage0 run success.", __func__);

    /* STAGE 1: level1 all_reduce - auto selected */
    CHK_RET(RunAllReduceLevel1(param, execMem, dataSegsSlice));
    HCCL_INFO("[%s]AllReduce stage1 run success.", __func__);

    /* STAGE 2: level0 all_gather - mesh atomic */
    CHK_RET(RunAllGatherLevel0(param, execMem, dataSegsSlice));
    HCCL_INFO("[%s]AllReduce stage2 run success", __func__);

    const u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    DeviceMem outCommMem = execMem.outputMem.range(0, curSize);
    DeviceMem outMem(execMem.outputPtr, curSize);
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outMem, outCommMem, const_cast<Stream&>(param.stream)));

    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMeshOpbaseMidCountDeterministicExecutor::RunLoopInner(
    OpParam& param, const ReduceType& reduceType, ExecMem& execMem)
{
    const u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    const u64 curSize = execMem.count * unitSize;
    HCCL_DEBUG(
        "[%s]inputMem[%p][%llu], outputMem[%p][%llu], "
        "inputPtr[%p], outputPtr[%p], curCount[%llu], curSize[%llu]",
        __func__, execMem.inputMem.ptr(), execMem.inputMem.size(), execMem.outputMem.ptr(), execMem.outputMem.size(),
        execMem.inputPtr, execMem.outputPtr, execMem.count, curSize);
    CHK_PRT_RET((execMem.count == 0), HCCL_ERROR("[%s]In OP_BASE curCount is zero.", __func__), HCCL_E_PARA);

    /* init task */
    const auto autoSelectedAlgTypeLevel1 = static_cast<u32>(algType_.algoLevel1);
    const bool hugeData = IsHugeData(curSize);
    const bool smallData = IsSmallData(param.DataDes.count * unitSize, curSize);
    u64 sliceNum = 0;
    CHK_RET(GetSliceNum(execMem.count * unitSize, smallData, sliceNum, unitSize));
    const bool dataSplit = true;
    const u8 deterministic = topoMatcher_->GetExternalInputHcclDeterministic();
    const CopyPattern copy = CopyPattern::ZCOPY;
    const auto opMeta = HcclOpMetaInfo::GetOneForAllReduce(
        autoSelectedAlgTypeLevel1, param.DataDes.dataType, reduceType, smallData, 1, hugeData, copy, sliceNum, false,
        true, dataSplit, deterministic);
    CHK_RET(InitTask(dispatcher_, param.stream, opMeta.isEnableCache, opMeta.GetCacheKey()));

    /* kernel run */
    CHK_RET(KernelRun(param, execMem));
    CHK_RET(LaunchTaskExtend(
        dispatcher_, const_cast<Stream&>(param.stream), const_cast<std::vector<Stream>&>(algResResp_->slaveStreams)));

    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "AllReduceMeshOpbaseMidCountDeterministicExecutor", AllReduceMeshOpbaseMidCountDeterministic,
    CollAllReduceMeshOpbaseMidCountDeterministicExecutor);

} // namespace hccl
