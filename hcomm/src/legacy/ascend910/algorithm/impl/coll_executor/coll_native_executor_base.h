/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_NATIVE_EXECUTOR_BASE_H
#define COLL_NATIVE_EXECUTOR_BASE_H

#include "coll_executor_base.h"
#include "device_capacity.h"
#include "dispatcher.h"
#include "stream_active_manager.h"
#include "comm_factory_pub.h"
#include "rank_consistentcy_checker.h"
#include "hccl_aiv.h"
#include "config_log.h"

namespace hccl {
constexpr u64 HCCL_INPLACE_MEMCOPY_SIZE = 131072;   // 128K数据量 = 131072B数据量
constexpr u64 HCCL_POST_SYNC_MEMCOPY_SIZE = 131072; // 128K数据量 = 131072B数据量
struct ExecMem {
    u64 count = 0;
    DeviceMem inputMem;  /* 单算子模式时是InCCLMem, 图模式时是InUserMem */
    DeviceMem outputMem; /* 单算子模式时是OutCCLMem, 图模式时是OutUserMem */
    DeviceMem scratchMem;
    void* inputPtr = nullptr;  /* InUserMem的地址，图模式时与inputMem的地址相同 */
    void* outputPtr = nullptr; /* OutUserMem的地址，图模式时与outputMem的地址相同 */
};

class CollNativeExecutorBase : public CollExecutorBase {
public:
    CollNativeExecutorBase(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollNativeExecutorBase() override = default;

    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;

protected:
    /* *************** 资源计算 *************** */
    virtual void ParseParam(const OpParam& param);
    virtual HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport); // 默认情况下可根据algType_支持NHR、NHRV1、NB、HD、Ring算法。
    virtual HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport);
    virtual HcclResult CalcStreamNum(u32& streamNum);
    virtual HcclResult CalcScratchMemSize(u64& scratchMemSize);
    virtual HcclResult CalcNotifyNum(u32 streamNum, u32& notifyNum);
    virtual HcclResult CalcAivBufferRequest(u64& aivBufferRequest);

    // 考虑新建一个资源计算类ResourceCalculator，将资源推导、资源解析的都放进去。
    // 推导通信域信息的公用函数，不同Executor的在计算Level0、Level1、Level2时使用。
    HcclResult CalcCommPlaneInfo(
        const std::string& tag, const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport,
        TransportMemType inPutMemType, TransportMemType outPutMemType);
    HcclResult BuildResourceRequest(
        u64 scratchMemSize, u32 streamNum, u32 notifyNum, u64 aivBufferRequest,
        std::vector<LevelNSubCommTransport>& opTransport, AlgResourceRequest& resourceRequest) const;
    HcclResult PrintTransportRequest(AlgResourceRequest& resourceRequest);
    virtual HcclResult CalcOptimalIntraRing(const OpParam& param);
    HcclResult SetCommInfoForARS(u32 ringSize);
    HcclResult SetCommInfoForIntraARS(u32 intraRingsize, std::vector<u32> commPlaneVector);
    HcclResult SetCommInfoForInterARS(u32 intraRingsize, std::vector<u32> commPlaneVector);
    /* *************** 算法编排 *************** */
    // 非零拷贝场景走KernelRun
    virtual HcclResult KernelRun(const OpParam& param, ExecMem& execMem);
    // 零拷贝场景走KernelRunIntraServerPre、KernelRunInterServer、KernelRunIntraServerPost
    virtual HcclResult KernelRunInterServer([[maybe_unused]] const OpParam& param, [[maybe_unused]] ExecMem& execMem)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult KernelRunInterServer(
        [[maybe_unused]] const OpParam& param, [[maybe_unused]] ExecMem& execMem, [[maybe_unused]] u64 baseOffset)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult KernelRunIntraServerPre([[maybe_unused]] const OpParam& param, [[maybe_unused]] ExecMem& execMem)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult
    KernelRunIntraServerPost([[maybe_unused]] const OpParam& param, [[maybe_unused]] ExecMem& execMem)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo);
    virtual HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize);
    virtual HcclResult GetDevNumInlocalPod(u32& devNumInlocalPod);

    // 图模式下激活从流
    HcclResult ActiveSlaveStreams(const Stream& stream);
    // 将从流添加至Profiling
    HcclResult AddSubStreamToProfiling();
    // 检查通信域大小
    HcclResult CheckCommSize(const CommPlane levelIndex, const u32 subLevelIndex);

    // 获取不同类型通信域中的 transport 信息
    // 为了避免循环调用时反复校验Range引发性能问题，此处不做Range校验，建议调用该接口前先调用CheckCommSize避免OutOfRange问题
    SubCommInfo GetSubCommInfo(const CommPlane levelIndex, const u32 subLevelIndex);

    HcclResult GetRankByUserRank(CommPlane levelIndex, u32 subLevelIndex, u32 userRank, u32& rank);
    HcclResult GetUserRankByRank(CommPlane levelIndex, u32 subLevelIndex, u32 rank, u32& userRank);
    HcclResult GenerateStreams(PrepareData& prepareData, std::vector<Stream>& streams);
    HcclResult NotifySubStreamStart(
        Stream& stream, std::vector<Stream>& substreams, std::vector<std::shared_ptr<LocalNotify>>& signalsSubToMain,
        u32 substreamNum);
    HcclResult WaitSubStreamFinish(
        Stream& stream, std::vector<Stream>& substreams, std::vector<std::shared_ptr<LocalNotify>>& signalsMainToSub,
        u32 substreamNum);
    HcclResult GenerateRecordWaitStreams(
        std::vector<Stream>& streams, u32 recordStreamNum, u32 waitStreamNum, std::vector<Stream>& recordStreams,
        std::vector<Stream>& waitStreams);
    HcclResult
    HoldAllRanksOnCurrentOp(const OpParam& param, ExecMem& execMem, PrepareData& prepareData, std::vector<LINK> links);
    HcclResult HoldAllRanksOnCurrentOpWithSingleStream(OpParam& param, ExecMem& execMem, std::vector<LINK> links);
    HcclResult SendRecvSignalOnLinks(OpParam& param, ExecMem& execMem, std::vector<LINK> links);
    bool OpSyncCheckCommSize(const CommPlane levelIndex, const u32 expectedSize);
    HcclResult PostSyncWithSubstream(OpParam& param, ExecMem& execMem, PrepareData& prepareData);
    HcclResult PostSyncWithoutSubstream(OpParam& param, ExecMem& execMem);
    HcclResult InplaceOpSync(OpParam& param, ExecMem& execMem);

    virtual HcclResult
    CopyAivCommInfoToDevice(const CommPlane levelIndex, const u32 subLevelIndex, AlgResourceResponse& algResource);

    HcclResult SetOpCache(
        const AivOpArgs& opArgs, const AivTopoArgs& topoArgs, const AivResourceArgs& resourceArgs,
        const AivAlgArgs& algArgs, ExtraArgs& extraArgs, AivProfilingInfo& aivProfilingInfo, bool isA3CrossNode);

    void SetWorkflowMode(HcclWorkflowMode workflowMode) override;
    HcclWorkflowMode GetExecutorWorkflowMode() const override;

    /* ---------------以下为 protected 成员变量定义领域-------------------------- */
    std::string tag_;
    u32 root_ = INVALID_VALUE_RANKID;
    AlgResourceResponse* algResResp_ = nullptr;
    HcclCMDType opType_ = HcclCMDType::HCCL_CMD_INVALID;

    // Infos got from topoMatcher_
    const HcclTopoInfo topoAttr_;
    const HcclAlgoInfo algoAttr_;
    TopoType topoType_;
    bool is310P3Common_ = false;
    bool aicpuUnfoldMode_ = false;
    HcclWorkflowMode workflowMode_;
};
std::vector<std::vector<u32>> GetARSRingsOrder(u32 ranksSize, TopoType topoType, std::vector<u32>& RingList);
} // namespace hccl
#endif
