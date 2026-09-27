/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALG_H
#define HCCL_ALG_H

#include "hccl_common.h"
#include "common.h"
#include "mem_device_pub.h"
#include "dispatcher.h"
#include "parallel_task_loader.h"
#include "comm_factory_pub.h"
#include "ccl_buffer_manager.h"
#include "workspace_resource.h"
#include "hccl_impl_pub.h"
#include "hccl_trace_info.h"
#include "queue_notify_manager.h"
#include "topo_matcher.h"
#include "coll_alg_operator.h"
#include "topo_info_extractor.h"
#include "alg_configurator.h"

namespace hccl {
class hcclImpl;
class HcclAlg {
public:
    explicit HcclAlg(
        CCLBufferManager& cclBufferManager, const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher);
    virtual ~HcclAlg();
    HcclResult Init(
        std::unique_ptr<WorkspaceResource>& workSpaceRes, const std::unique_ptr<NotifyPool>& notifyPool,
        std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,
        const std::unique_ptr<QueueNotifyManager>& queueNotifyManager, HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr,
        bool isHeterogComm = false);
    HcclResult Init(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, bool isHeterogComm = false);
    HcclResult ReleaseCommInfos();

    HcclResult GetTinyMem(DeviceMem& tinySendRecvMem);

    // legacy code
    HcclResult
    GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize);
    HcclResult GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize);
    HcclResult ClearOpResource(const std::string& tag);
    HcclResult CreateMutiStreamRes(
        const std::string& tag, Stream& stream, level1StreamInfo_t& streamInfo, AlgType algType,
        bool isAicpuModeEn = false);
    HcclResult CreateComm(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType,
        std::unique_ptr<CommInfo>& commInfo, u32 root = INVALID_VALUE_RANKID, bool isP2p = false,
        bool isAicpuModeEn = false);
    HcclResult CreateComm(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType,
        u32 root = INVALID_VALUE_RANKID, bool isP2p = false);
    void CancelCommRes(const std::string& tag);
    void Break();
    HcclResult SetAlgType(AlgType algType, HcclCMDType opType);
    HcclResult GetAlgType(AlgType& algType, HcclCMDType opType);
    HcclResult SupportDeterministicOptim(bool& isDeterministicOptim);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
        std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort);

    u8 GetDeterministicConfig() const;                         // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic); // 设置确定性计算配置
    HcclResult SetAivModeConfig(const bool aivMode);           // 设置aiv模式配置
    HcclResult SetOnlyAivModeConfig(const bool isOnlyAiv);
    HcclResult SetAicpuUnfoldConfig(const bool aicpuUnfold); // 设置aicpu配置
    HcclResult SetExecTimeOutConfig(const s32 execTimeOut);  // 设置HCCL执行超时时间
    HcclResult SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap); // 设置HCCL_Algo
    bool GetAicpuUnfoldConfig() const;
    bool GetAivModeConfig() const;
    HcclResult GetIsBridgeVector(std::vector<bool>& isBridgeVector);
    HcclResult GetRankVecInfo(std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank);
    HcclResult GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>>& commPlaneRanks);
    void GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>>& commPlaneVector);
    HcclResult GetIsUsedRdmaMap(std::unordered_map<u32, bool>& isUsedRdmaMap);
    HcclResult
    GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>>& commPlaneSubGroupVector);
    HcclResult GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption);

    __attribute__((weak)) std::unique_ptr<CollAlgOperator> GetAlgOperator(
        const HcclCMDType& opType,
        HcclWorkflowMode workflowMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED); // 只在host侧使用的方法

    HcclResult GetTopoType(TopoType& topoType);

private:
#ifndef CCL_KERNEL_AICPU
#ifndef OPEN_HCCL_TEST
    // 只有流流程和异构场景在使用
    std::unique_ptr<hcclImpl> pimpl_;
#endif
#endif
    HcclResult InitTopoInfo(HcclTopoInfo& topoInfo, HcclTopoAttr& topoAttr);
    HcclResult InitAlgoInfo(HcclAlgoInfo& algoInfo, HcclAlgoAttr& algoAttr);
    HcclResult InitExternalEnable(HcclExternalEnable& externalEnable);

    // 缓存初始传入传入的属性值
    HcclAlgoAttr algoAttr_;
    HcclTopoAttr topoAttr_;

    std::shared_ptr<AlgConfigurator> algConfigurator_;
    std::shared_ptr<TopoInfoExtractor> topoInfoEx_;

    std::unique_ptr<TopoMatcher> topoMatcher_;

    CCLBufferManager& cclBufferManager_;
    const HcclDispatcher dispatcher_;

    // 历史继承特性使用的环境变量
    const HcclDispatcher vDispatcher_;
    std::unique_ptr<ParallelTaskLoader> parallelTaskLoader_; // 并行下发taskloader管理
    DeviceMem tinySendRecvMem_; // 在sendCount/recvCount全0时, 使用tinySendRecvMem_, 避免使用空deviceMem
};
} // namespace hccl

#endif // HCCL_ALG_H
