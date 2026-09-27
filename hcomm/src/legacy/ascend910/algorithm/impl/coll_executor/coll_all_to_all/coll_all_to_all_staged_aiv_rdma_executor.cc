/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_staged_aiv_rdma_executor.h"

namespace hccl {
constexpr u32 A_X_SIZE = 16;

CollRunAlltoAllStagedAivRdmaExecutor::CollRunAlltoAllStagedAivRdmaExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    desc_.isAivMode = true;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    // alltoall aiv暂不支持图模式
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    execMem.scratchMem = algRes.aivInputMem;
    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollRunAlltoAllStagedAivRdmaExecutor][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "[CollRunAlltoAllStagedAivRdmaExecutor]tag[%s], orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));

    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 0; // AIV通信不需要申请从流

    HCCL_INFO("[CollRunAlltoAllStagedAivRdmaExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalcScratchMemSize(u64& scratchMemSize)
{
    scratchMemSize = 0U; // AIV模式不需要scratch内存，直接在cclbuffer上进行内存重排
    HCCL_INFO(
        "[CollRunAlltoAllStagedAivRdmaExecutor][CalcScratchMemSize]tag[%s] scratchMemSize_ is [%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_MESH_L0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_MESH_L0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalcLevel1CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaInfo(COMM_MESH_L1, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_MESH_L1], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    // aiv阶段使用cclin做数据搬移，使用aivout做标记
    CHK_RET(CalcLevel0CommInfo(TransportMemType::CCL_INPUT, TransportMemType::AIV_INPUT, opTransport));
    CHK_RET(CalcLevel1CommInfo(TransportMemType::CCL_INPUT, TransportMemType::CCL_OUTPUT, opTransport));

    HCCL_DEBUG("[CollRunAlltoAllStagedAivRdmaExecutor][CalcCommInfo] ends");
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::CalNumBlocks(
    u32& numBlocks, u32 rankSize, [[maybe_unused]] u64 dataSize, [[maybe_unused]] HcclCMDType cmdType)
{
    numBlocks = rankSize; // 默认情况使用rankSize个AIV
    u32 bestNumBlocks = numBlocks;

    CHK_PRT_RET(
        numBlocks_ < numBlocks,
        HCCL_WARNING(
            "[CollRunAlltoAllStagedAivRdmaExecutor][CalNumBlocks]aivCore[%u] is invalid, at least need [%u].",
            numBlocks_, numBlocks),
        HCCL_E_PARA);

    HCCL_INFO(
        "[CollRunAlltoAllStagedAivRdmaExecutor][CalNumBlocks] numBlocks is set to [%u], limit[%u], recommended[%u]",
        numBlocks, numBlocks_, bestNumBlocks);
    return HCCL_SUCCESS;
}

// run aiv kernel
HcclResult CollRunAlltoAllStagedAivRdmaExecutor::RunAlltoAllStaged1InAIV(const OpParam& param, ExecMem& execMem)
{
    void* dataBuffers[MAX_RANK_SIZE];
    void* flagBuffers[MAX_RANK_SIZE];

    u32 serverNum = innerCommInfo_.localRankSize;
    u64 sendCount = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix));

    CHK_RET(PrepareAivBuffers(execMem.inputMem, execMem.scratchMem, dataBuffers, flagBuffers));

    AivOpArgs opArgs{
        HcclCMDType::HCCL_CMD_ALLTOALL,
        execMem.inputPtr,
        execMem.outputPtr,
        sendCount,
        param.All2AllDataDes.sendType,
        HCCL_REDUCE_RESERVED,
        0,
        true};
    AivTopoArgs topoArgs{
        outerCommInfo_.localRank, outerCommInfo_.localRankSize,
        topoAttr_.isDiffDeviceModule ? topoAttr_.devicePhyId : A_X_SIZE, 0, serverNum};
    topoArgs.identify = algoAttr_.identifier;
    u32 numBlocks;
    CHK_PRT_RET(
        CalNumBlocks(numBlocks, outerCommInfo_.localRankSize) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
    numBlocks_ = numBlocks;
    AivResourceArgs resourceArgs{param.tag,  param.stream.ptr(), dataBuffers, flagBuffers, execMem.inputMem.size(),
                                 numBlocks_, param.aivTag};
    AivAlgArgs algArgs{0};
    algArgs.execTimeOut = topoMatcher_->GetExecTimeOutConfig();
    algArgs.execTimeOutSet = true;
    struct AivProfilingInfo aivProfilingInfo;
    aivProfilingInfo.counter = opCounter_;
    HCCL_DEBUG("[CollRunAlltoAllStagedAivRdmaExecutor]RunAlltoAllStaged1InAIV for numBlocks is %u", numBlocks_);
    CHK_RET(ExecuteKernelLaunch(opArgs, topoArgs, resourceArgs, algArgs, aivProfilingInfo));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::RunAlltoAllStaged2(const OpParam& param, ExecMem& execMem)
{
    std::map<u32, std::list<OneSendRecvAddrInfo>> sendAddrInfosInter;
    std::map<u32, std::list<OneSendRecvAddrInfo>> recvAddrInfosInter;

    CalcInterMeshAggregationAlltoAllMemInfo(param, sendAddrInfosInter, recvAddrInfosInter);

    HcclOpMetaInfoDef opMeta
        = HcclOpMetaInfo::GetOneForAllToAll(CopyPattern::ZCOPY, algResResp_->paramInputMem.size(), false, true);
    CHK_RET(InitTask(dispatcher_, const_cast<Stream&>(param.stream), opMeta.isEnableCache, opMeta.GetCacheKey()));

    std::unique_ptr<AlgTemplateBase> alltoallInner = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE, dispatcher_);
    CHK_SMART_PTR_NULL(alltoallInner);

    CHK_RET(alltoallInner->Prepare(
        execMem.inputMem, execMem.outputMem, sendAddrInfosInter, recvAddrInfosInter, true,
        const_cast<Stream&>(param.stream)));

    CHK_RET(RunAlltoAllVTemplateStaged(alltoallInner, innerCommInfo_));
    return HCCL_SUCCESS;
}

void CollRunAlltoAllStagedAivRdmaExecutor::CalcInterMeshAggregationAlltoAllMemInfo(
    const OpParam& param, std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
    std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter)
{
    u64 sendCount = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix));
    // serverLength表示每个rank给每个server需要发送的数据总量
    u64 serverSendLength = outerCommInfo_.localRankSize * sendCount * sendDataSize_;
    u64 serverRecvLength = outerCommInfo_.localRankSize * sendCount * recvDataSize_;
    // 数据中转时每个rank按server顺序存储对应的中转数据，使用userRankOffset表示根据当前rank所在server计算对端的偏移
    for (u32 i = 0; i < innerCommInfo_.localRankSize; i++) {
        OneSendRecvAddrInfo sendAddrInfo;
        sendAddrInfo.localOffset = i * serverSendLength;
        sendAddrInfo.remoteOffset = innerCommInfo_.localRank * serverSendLength;
        sendAddrInfo.localLength = serverSendLength;
        sendAddrInfo.remoteLength = serverSendLength;
        sendAddrInfosInter[i].push_back(sendAddrInfo);

        OneSendRecvAddrInfo recvAddrInfo;
        recvAddrInfo.localOffset = i * serverRecvLength;
        recvAddrInfo.remoteOffset = innerCommInfo_.localRank * serverRecvLength;
        recvAddrInfo.localLength = serverRecvLength;
        recvAddrInfo.remoteLength = serverRecvLength;
        recvAddrInfosInter[i].push_back(recvAddrInfo);
    }
    return;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollRunAlltoAllStagedAivRdmaExecutor][KernelRun] AllToAll staged starts");
    CHK_PRT_RET(
        topoAttr_.userRankSize % topoAttr_.meshAggregationRankSize != 0,
        HCCL_ERROR(
            "userRankSize[%u] is not an Integer multiple of MeshAggregation Dev Num[%u]", topoAttr_.userRankSize,
            topoAttr_.meshAggregationRankSize),
        HCCL_E_PARA);

    CHK_RET(SalGetDataTypeSize(param.All2AllDataDes.sendType, sendDataSize_));
    CHK_RET(SalGetDataTypeSize(param.All2AllDataDes.recvType, recvDataSize_));

    CHK_RET(CheckCommSize(COMM_MESH_L0, COMM_INDEX_0 + 1));
    outerCommInfo_ = GetSubCommInfo(COMM_MESH_L0, COMM_INDEX_0);

    CHK_RET(CheckCommSize(COMM_MESH_L1, COMM_INDEX_0 + 1));
    innerCommInfo_ = GetSubCommInfo(COMM_MESH_L1, COMM_INDEX_0);

    // step1：每个server内通过aiv进行数据交换，将中转数据存储在ipc_buffer中
    CHK_RET(RunAlltoAllStaged1InAIV(param, execMem));
    HCCL_INFO("[CollRunAlltoAllStagedAivRdmaExecutor][kernelRun] stage0 run aiv in level0 success!");

    // step2：server间通过rdma进行数据交换，将中转数据分发到各个rank的ccl_out中
    CHK_RET(RunAlltoAllStaged2(param, execMem));
    HCCL_INFO("[CollRunAlltoAllStagedAivRdmaExecutor][kernelRun] stage1 run rdma in level1 success!");

    // 每个rank上将数据从ccl_out中搬运到用户输出buffer中
    DeviceMem srcMem = (execMem.outputMem).range(0, algResResp_->paramOutputMem.size());
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, algResResp_->paramOutputMem, srcMem, const_cast<Stream&>(param.stream)));

    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(param.stream)));
    HCCL_INFO("[CollRunAlltoAllStagedAivRdmaExecutor][kernelRun] AllToAll staged ends");
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllStagedAivRdmaExecutor::PrepareAivBuffers(
    DeviceMem& inputMem, DeviceMem& outputMem, void** dataBuffers, void** flagBuffers)
{
    void* tmpCCLBufferData = nullptr;
    void* tmpCCLBufferFlag = nullptr;
    for (u32 i = 0; i < outerCommInfo_.localRankSize; i++) {
        if (i != outerCommInfo_.localRank) {
            if (outerCommInfo_.links[i] != nullptr) {
                CHK_RET(outerCommInfo_.links[i]->GetRemoteMem(UserMemType::INPUT_MEM, &(tmpCCLBufferData)));
                CHK_RET(outerCommInfo_.links[i]->GetRemoteMem(UserMemType::OUTPUT_MEM, &(tmpCCLBufferFlag)));
                // cclbuffer后32K数据作为数据标志位
                dataBuffers[i] = static_cast<u8*>(tmpCCLBufferData);
                flagBuffers[i] = static_cast<u8*>(tmpCCLBufferFlag) + HCCL_MID_COUNT_32_MB;
            }
        } else {
            dataBuffers[i] = static_cast<u8*>(inputMem.ptr());
            flagBuffers[i] = static_cast<u8*>(outputMem.ptr()) + HCCL_MID_COUNT_32_MB;
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AlltoAllStagedAIVRdmaExecutor", AlltoAllStagedAIVRdma, CollRunAlltoAllStagedAivRdmaExecutor);
} // namespace hccl
