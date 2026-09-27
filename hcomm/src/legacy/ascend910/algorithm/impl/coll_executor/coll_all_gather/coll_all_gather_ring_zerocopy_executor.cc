/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_ring_zerocopy_executor.h"

namespace hccl {
CollAllGatherRingZerocopyExecutor::CollAllGatherRingZerocopyExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true; // 设为true，以禁用RunLoop中的本地拷贝
    desc_.isZeroCopy = true;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING,
           AlgTypeLevel1::ALG_LEVEL1_AHC, AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

HcclResult CollAllGatherRingZerocopyExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? (LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE + 1) :
                                                                             LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[%s] tag[%s] streamNum_[%u]", __func__, tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

void CollAllGatherRingZerocopyExecutor::ParseParam(const OpParam& param) { tag_ = param.tag; }

HcclResult CollAllGatherRingZerocopyExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_LEVEL0];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        commTransportLevel0[subCommIndex].isZeroCopy = true;
    }
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingZerocopyExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.serverNum / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

HcclResult CollAllGatherRingZerocopyExecutor::SemiRingAllGather(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream, s32 profStage,
    const u64 baseOffset, const HcomCollOpInfo* opInfo, const std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    (void)multRingsSliceZero;
    (void)tag;
    (void)baseOffset;
    (void)opInfo;
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    // 执行
    std::unique_ptr<AlgTemplateBase> executor
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_UNIFIED_MARCH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_UNIFIED_MARCH in COMM_LEVEL0", __func__);
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(
        stream, level0CommInfo, algResResp_->paramInputMem, algResResp_->paramOutputMem, inputMem, outputMem,
        count * SIZE_TABLE[dataType], algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        multRingsUserMemSlice));
    HcclResult ret = executor->RegisterProfiler(
        ((COMM_INDEX_0 + 1) << PROF_RINGINDEX_OFFSET_OF_PLANEID)
            + (level0CommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank,
        profStage, HCCL_EXEC_STEP_NOT_SET, stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollAllGatherRingZerocopyExecutor][SemiRingAllGather]Double ring "
            "AllGather failed, return[%d]",
            ret),
        ret);
    CHK_RET(executor->RunAsync());
    return ret;
}

HcclResult CollAllGatherRingZerocopyExecutor::KernelRunIntraServerPost(const OpParam& param, ExecMem& execMem)
{
    bool isAHCAlgo = algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
                     || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_, isAHCAlgo));

    // 计算slice信息
    std::vector<Slice> dataSegsSlice;
    CHK_RET(CalcLevel0DataSlices(param, execMem, dataSegsSlice));
    // 执行AllGather
    u64 level0Count = (dataSegsSlice.size() > level0RankSize_) ? // 如果是非连续数据通信
                          (execMem.count) :
                          (execMem.count * level1RankSize_ * level2RankSize_);
    std::vector<std::vector<Slice>> multRingsUserMemSlice = {dataSegsSlice};
    if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        HCCL_INFO("[%s] single ring AllGather", __func__);
        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, level0Count, param.DataDes.dataType, multRingsUserMemSlice,
            param.stream, PROF_STAGE_0, 0, nullptr, multRingsUserMemSlice));
    } else {
        CHK_PRT_RET(
            topoType_ != TopoType::TOPO_TYPE_NP_DOUBLE_RING,
            HCCL_ERROR("[%s] unknown topoType: %u", __func__, topoType_), HCCL_E_NOT_SUPPORT);
        HCCL_INFO("[%s] semi ring AllGather", __func__);
        CHK_RET(SemiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, level0Count, param.DataDes.dataType, multRingsUserMemSlice,
            param.stream, PROF_STAGE_0, 0, nullptr, multRingsUserMemSlice));
    }

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyExecutor::KernelRunInterServerPreProcess(const OpParam& param, const ExecMem& execMem)
{
    // 将数据从User Input拷到CCL Output
    u32 dataIndex = level1Rank_ * level2RankSize_ + level2Rank_;
    u64 curSize = execMem.inputMem.size();
    DeviceMem dstMem = execMem.outputMem.range(curSize * dataIndex, curSize);
    DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), curSize);
    Stream stream = param.stream;
    return HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream);
}

HcclResult CollAllGatherRingZerocopyExecutor::KernelRunInterServer(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllGatherRingZerocopyExecutor][KernelRunInterServer] The AllGatherDoubleRingExecutor starts");
    bool isAHCAlgo = algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
                     || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    CHK_RET(GetCommRankInfoNormal(
        level0Rank_, level0RankSize_, level1Rank_, level1RankSize_, level2Rank_, level2RankSize_, isAHCAlgo));

    // 前处理
    CHK_RET(KernelRunInterServerPreProcess(param, execMem));

    // 计算slice
    std::vector<Slice> level1DataSegsSlice;
    CalcLevel1DataSlices(execMem.inputMem.size(), level1RankSize_, level2RankSize_, level1DataSegsSlice);

    // 超节点间通信
    if (level2RankSize_ > 1 && !isAHCAlgo) {
        // 获取对应算法的Template
        std::unique_ptr<AlgTemplateBase> level2AGTemplage;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGTemplage
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGTemplage
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_RING) {
            level2AGTemplage
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_HD) {
            level2AGTemplage = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
            HCCL_CONFIG_INFO(
                HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING in COMM_LEVEL2", __func__);
        } else {
            HCCL_ERROR("AllGather ring: unsupported level2 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level2AGTemplage);
        // 执行算法编排
        DeviceMem level2OutputMem
            = execMem.outputMem.range(level1DataSegsSlice[level1Rank_].offset, level1DataSegsSlice[level1Rank_].size);
        CHK_RET(level2AGTemplage->Prepare(
            level2OutputMem, level2OutputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, std::vector<Slice>(0),
            level1DataSegsSlice[level1Rank_].offset));
        CHK_RET(level2AGTemplage->RegisterProfiler(
            (level2RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2Rank_, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        CHK_RET(RunTemplate(level2AGTemplage, level2CommInfo));
        HCCL_INFO("AllGather double ring [superpod] level2 AllGather run success");
    }

    // 超节点内、节点间通信
    if (level1RankSize_ > 1) {
        // 获取对应算法的Template
        std::unique_ptr<AlgTemplateBase> level1AGTemplate;
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NB in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1AGTemplate
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        } else if (isAHCAlgo) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(COMM_LEVEL1_AHC, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1AGTemplate = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC in COMM_LEVEL1", __func__);
            } else {
                level1AGTemplate = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            CHK_SMART_PTR_NULL(level1AGTemplate);
            CHK_RET(level1AGTemplate->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
        } else {
            HCCL_ERROR("AllGather ring: unsupported level1 algtype [%s]", AlgTypeToStr(algType_).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_SMART_PTR_NULL(level1AGTemplate);
        // 执行算法编排
        CHK_RET(level1AGTemplate->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, INVALID_U64, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice));
        CHK_RET(level1AGTemplate->RegisterProfiler(
            (level1RankSize_ << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1Rank_, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CommPlane commPlaneLevel1 = isAHCAlgo ? COMM_LEVEL1_AHC : COMM_LEVEL1;
        CHK_RET(CheckCommSize(commPlaneLevel1, level0Rank_ + 1));
        SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0Rank_);
        CHK_RET(RunTemplate(level1AGTemplate, level1CommInfo));
        HCCL_INFO("AllGather double ring [superpod] level1 AllGather run success");
    }

    // 后处理
    CHK_RET(KernelRunInterServerPostProcess(param, execMem));

    return HCCL_SUCCESS;
}

HcclResult
CollAllGatherRingZerocopyExecutor::KernelRunInterServerPostProcess(const OpParam& param, const ExecMem& execMem)
{
    u32 unitSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, unitSize));

    DeviceMem dstMem;
    DeviceMem srcMem;
    u64 curSize = execMem.inputMem.size();
    Stream stream = param.stream;
    for (u32 i = 0; i < level1RankSize_; i++) {
        for (u32 j = 0; j < level2RankSize_; j++) {
            // 拷贝input上每个slice的数据到中转内存，源端每个slice的size固定为output的size
            u32 dstIndex = i * level2RankSize_ + j;
            u32 srcIndex = j * level1RankSize_ + i;
            srcMem = execMem.outputMem.range(dstIndex * curSize, curSize);
            dstMem = DeviceMem::create(
                static_cast<u8*>(execMem.outputPtr) + param.DataDes.count * unitSize * level0RankSize_ * srcIndex
                    + param.DataDes.count * unitSize * level0Rank_,
                curSize);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, stream));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingZerocopyExecutor::CalcLevel0DataSlices(
    const OpParam& param, const ExecMem& execMem, std::vector<Slice>& dataSegsSlice)
{
    return CalcIntraServerDataSlicesDiscontinuous(
        param, execMem, level0RankSize_, level1RankSize_, level2RankSize_, dataSegsSlice);
}

REGISTER_EXEC("AllGatherRingZerocopyExecutor", AllGatherRingZerocopy, CollAllGatherRingZerocopyExecutor);

} // namespace hccl
