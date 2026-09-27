/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>
#include "stream_active_manager.h"
#include "profiling_manager_pub.h"
#include "coll_alg_utils.h"
#include "sal_pub.h"
#include "hccl_impl.h"

using namespace std;

namespace hccl {

std::array<DeviceMem, MAX_MODULE_DEVICE_NUM> hcclImpl::inOutPutTempMem_;
std::array<std::mutex, MAX_MODULE_DEVICE_NUM> hcclImpl::inOutPutTempMemMutex_;
std::array<Referenced, MAX_MODULE_DEVICE_NUM> hcclImpl::instanceRef_;
RegisterToHeartBeatCallBack g_RegisterToHeartBeatCallBack = nullptr;
UnRegisterToHeartBeatCallBack g_UnRegisterToHeartBeatCallBack = nullptr;
SetRankPortInfoCallBack g_SetRankPortInfoCallBack = nullptr;

hcclImpl::hcclImpl(
    const HcclDispatcher dispatcher, const std::unique_ptr<NotifyPool>& notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, const std::unique_ptr<QueueNotifyManager>& queueNotifyManager,
    std::unique_ptr<WorkspaceResource>& workSpaceRes, CCLBufferManager& cclBufferManager, HcclAlgoAttr& algoAttr,
    HcclTopoAttr& topoAttr, std::shared_ptr<AlgConfigurator> algConfigurator,
    std::shared_ptr<TopoInfoExtractor> topoInfoEx)
    : dispatcher_(dispatcher),
      notifyPool_(notifyPool),
      netDevCtxMap_(netDevCtxMap),
      queueNotifyManager_(queueNotifyManager),
      workSpaceRes_(workSpaceRes),
      cclBufferManager_(cclBufferManager),
      algConfigurator_(algConfigurator),
      topoInfoEx_(topoInfoEx),
      topoAttr_(topoAttr),
      algoAttr_(algoAttr)
{
    SetAlgoAttr(algoAttr);
    SetTopoAttr(topoAttr);

    s32 deviceLogicId = 0;
    if (hrtGetDevice(&deviceLogicId) != HCCL_SUCCESS) {
        HCCL_INFO("start hccl resources build:no get deviceLogicId[%d]", deviceLogicId);
        return;
    }
    if ((static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM) || (deviceLogicId < 0)) {
        HCCL_WARNING("start hccl resources build:get fail deviceLogicId[%d]", deviceLogicId);
        return;
    }

    HCCL_INFO("start hccl resources build:get deviceLogicId[%d]", deviceLogicId_);
    instanceRef_[deviceLogicId].Ref();
    if (SalGetBareTgid(&pid_) != HCCL_SUCCESS) {
        HCCL_INFO("get pid is unsuccessful");
        return;
    }
}

hcclImpl::~hcclImpl()
{
    HCCL_INFO("start hccl resources destruction:deviceLogicId[%d]", deviceLogicId_);

    WaitCommThread(commThreadPtrLevel0_);
    WaitCommThread(commThreadPtrLevel1_);
    WaitCommThread(commThreadPtrLevel2_);

    /* 销毁通信域关联资源 */
    for (auto& iter : tagCommInfo_) {
        DestroyLevel0Comm(iter.first);
        DestroyLevel1Comm(iter.first);
        DestroyIntraServerComm(iter.first);
        // Workspace资源需要根据tag销毁（临时方案）
        workSpaceRes_->DestroyWorkspaceResource(iter.first);
    }

    cclBufferManager_.ReleaseAlltoAllvParaBuffer();

    for (auto& level1_stream_info : tagStreamInfo_) {
        if (ReleaseSignal(level1_stream_info.second) != HCCL_SUCCESS) {
            HCCL_WARNING("tag[%s],signal is not released successfully", level1_stream_info.first.c_str());
        }
        (void)StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(level1_stream_info.second.ringStreams);
    }

    tagCommInfo_.clear();
    tagStreamInfo_.clear();
    commMeshPtr_.reset();
    commMeshLevel2_.reset();
    commMeshMap_.clear();

    commFactory_ = nullptr;

    if ((static_cast<u32>(deviceLogicId_) >= MAX_MODULE_DEVICE_NUM) || (deviceLogicId_ < 0)) {
        HCCL_WARNING("start hccl resources destruction:get fail deviceLogicId[%d]", deviceLogicId_);
        return;
    }

    if (instanceRef_[deviceLogicId_].Unref() == 0) {
        std::unique_lock<std::mutex> lock(inOutPutTempMemMutex_[deviceLogicId_]);
        inOutPutTempMem_[deviceLogicId_].free();
    }
}

void hcclImpl::SetAlgoAttr(HcclAlgoAttr& algoAttr)
{
    isHaveCpuRank_ = algoAttr.isHaveCpuRank;
    inlineReduceSwitchOn_ = algoAttr.inlineReduceSwitchOn;
    isUsedRdmaLevel0_ = algoAttr.isUsedRdmaLevel0;
    isUsedInterHccsMode_ = algoAttr.isUsedInterHccsMode;

    identifier_ = algoAttr.identifier;
    collectiveId_ = algoAttr.collectiveId;

    nicDeployment_ = algoAttr.nicDeployment;
    commWorkMode_ = algoAttr.commWorkMode;
    return;
}

void hcclImpl::SetTopoAttr(HcclTopoAttr& topoAttr)
{
    serverNum_ = topoAttr.serverNum;
    superPodNum_ = topoAttr.superPodNum;
    moduleNum_ = topoAttr.moduleNum;
    deviceNumPerServer_ = topoAttr.deviceNumPerServer;
    deviceNumPerAggregation_ = topoAttr.deviceNumPerAggregation;
    multiModuleDiffDeviceNumMode_ = topoAttr.multiModuleDiffDeviceNumMode;
    multiSuperPodDiffServerNumMode_ = topoAttr.multiSuperPodDiffServerNumMode;
    multiSuperPodDiffDeviceNumMode_ = topoAttr.multiSuperPodDiffDeviceNumMode;

    meshAggregationRankSize_ = topoAttr.meshAggregationRankSize;
    isDiffDeviceModule_ = topoAttr.isDiffDeviceModule;
    isSingleMeshAggregation_ = topoAttr.isSingleMeshAggregation;
    isAllRankSamePlane_ = topoAttr.isAllRankSamePlane;

    userRank_ = topoAttr.userRank;
    realUserRank_ = topoAttr.realUserRank;
    userRankSize_ = topoAttr.userRankSize;
    rankInfoList_ = topoAttr.rankInfoList;

    devicePhyId_ = topoAttr.devicePhyId;
    deviceLogicId_ = topoAttr.deviceLogicId;
    useSuperPodMode_ = topoAttr.useSuperPodMode;
    deviceType_ = topoAttr.deviceType;
    isStandardCard_ = topoAttr.isStandardCard;
    is310PDuoCard_ = topoAttr.is310PDuoCard;

    nicList_ = topoAttr.nicList;
    pairLinkCounter_ = topoAttr.pairLinkCounter;
    pairLinkInfo_ = topoAttr.pairLinkInfo;
    isSupportRdmaLite_ = topoAttr.isSupportRdmaLite;
    isSupportHccsAndSio_ = topoAttr_.isSupportHccsAndSio;
    localNicPort_ = topoAttr.localNicPort;
    isNeedInitNic_ = topoAttr.isNeedInitNic;
    return;
}

HcclResult hcclImpl::Init(bool isHeterogComm)
{
    algConfigurator_->GetTopoType(topoType_);

    commFactory_.reset(new (std::nothrow) CommFactory(
        identifier_, userRank_, userRankSize_, dispatcher_, notifyPool_, netDevCtxMap_, topoInfoEx_, isUsedRdmaLevel0_,
        topoType_, deviceType_, rankInfoList_, nicDeployment_, isHeterogComm, meshAggregationRankSize_, isHaveCpuRank_,
        isUsedInterHccsMode_, useSuperPodMode_));
    CHK_SMART_PTR_NULL(commFactory_);
    CHK_RET(commFactory_->Init());

    HCCL_INFO("hcclImpl init success.");
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ReleaseCommInfos()
{
    auto iter = tagCommInfo_.begin();
    while (iter != tagCommInfo_.end()) {
        for (auto& comm : iter->second.commLevel1) {
            if (comm != nullptr) {
                CHK_RET(comm->DeInit());
            }
        }
        iter++;
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateP2pComm(const std::string& tag, CommInfo& commInfo, DeviceMem& inOutMem, u32 peerUserRank)
{
    CommParaInfo commP2P(COMM_COMBINE, CommType::COMM_TAG_P2P);
    commP2P.peerUserRank = peerUserRank;
    CHK_RET(commFactory_->CreateCommPlane(tag, inOutMem, inOutMem, commP2P, commInfo.commP2P));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::WaitCommThread(std::unique_ptr<std::thread>& ThreadPtr) const
{
    // 若线程指针为空，为此线程从未被拉起使能，不返回异常日志
    if (ThreadPtr != nullptr && ThreadPtr->joinable()) {
        ThreadPtr->join();                       // 等待线程执行完毕
        CHK_RET(hrtResetDevice(deviceLogicId_)); // 防止线程里面异常退出，在进程中reset
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::InitMultiStreamResource(
    const std::string& tag, level1StreamInfo_t& streamInfo, AlgType algType, bool isAicpuModeEn, bool isBatchSendRecv,
    u32 ringNum)
{
    if (!isBatchSendRecv) {
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING) {
            if (deviceType_ == DevType::DEV_TYPE_910_93) {
                if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                    streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
                } else {
                    streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
                }
            }
        } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE * STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
            } else {
                streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
            }
        } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_8P_RING) {
            streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_8PRING;
        } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH) {
            if ((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                && (deviceType_ == DevType::DEV_TYPE_910B) && isSingleMeshAggregation_) {
                streamInfo.ringNum = deviceNumPerAggregation_;
            } else if ((deviceType_ == DevType::DEV_TYPE_910_93) && (isAicpuModeEn == true)) {
                streamInfo.ringNum = deviceNumPerAggregation_;
            } else if (
                (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
                && (deviceType_ == DevType::DEV_TYPE_910B)
                && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
                streamInfo.ringNum = deviceNumPerAggregation_ + 1; /* pipeline ring场景下性能优化 */
            } else {
                streamInfo.ringNum = deviceNumPerAggregation_ - 1;
            }
        } else if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH) {
            streamInfo.ringNum = LEVEL0_PLANE_NUM_IN_4PMESH;
        }
    } else {
        // 批量send/recv需要2条流
        streamInfo.ringNum = 2;
    }

    if (pipelineSliceNum_ > 0) {
        streamInfo.ringNum++; // 流水并行算法, Server间需要额外一条从流
    }
    streamInfo.ringNum = std::max(streamInfo.ringNum, ringNum);
    HCCL_INFO("algType:[%u] InitMultiStreamResource streamInfo.ringNum %u", algType.algoLevel0, streamInfo.ringNum);
    if (streamInfo.ringNum > 1) {
        u32 resNum = streamInfo.ringNum - 1;
        streamInfo.ringStreams.resize(resNum);   // 只有主环以外会用,减去主环1
        streamInfo.ringSignal.resize(resNum);    // 只有主环以外会用,减去主环1
        streamInfo.ringSignalAux.resize(resNum); // 只有主环以外会用,减去主环1
        streamInfo.ringThreadsManage.resize(resNum);
        streamInfo.tidInfo.resize(resNum);

        for (auto& signal : streamInfo.ringSignal) {
            signal = nullptr;
        }
        for (auto& signal : streamInfo.ringSignalAux) {
            signal = nullptr;
        }

        u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys));
        for (u32 i = 0; i < resNum; i++) {
            streamInfo.ringSignal[i] = notifys[2 * i];
            streamInfo.ringSignalAux[i] = notifys[2 * i + 1];
        }
        for (u32 ringIndex = 0; ringIndex < resNum; ringIndex++) {
            if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                streamInfo.ringThreadsManage[ringIndex].reset(new (std::nothrow)
                                                                  ThreadManage(deviceLogicId_, userRank_, dispatcher_));
                CHK_SMART_PTR_NULL(streamInfo.ringThreadsManage[ringIndex]);
                HcclResult ret = streamInfo.ringThreadsManage[ringIndex]->Init();
                CHK_PRT_RET(
                    ret != HCCL_SUCCESS,
                    HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]", ringIndex, ret),
                    ret);
                streamInfo.tidInfo[ringIndex] = streamInfo.ringThreadsManage[ringIndex]->GetTid();
                HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
            }
        }
    }
    if (isAicpuModeEn == true) {
        HCCL_INFO("aicpu resource num[%u]", streamInfo.ringNum);
        streamInfo.ringDeviceStreams.resize(streamInfo.ringNum);

        if (streamInfo.ringNum > 1) {
            u32 resNum = streamInfo.ringNum - 1;
            streamInfo.ringDeviceSignal.resize(resNum);
            streamInfo.ringDeviceSignalAux.resize(resNum);

            for (auto& signal : streamInfo.ringDeviceSignal) {
                signal = nullptr;
            }

            for (auto& signal : streamInfo.ringDeviceSignalAux) {
                signal = nullptr;
            }

            u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
            std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
            CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
            for (u32 i = 0; i < resNum; i++) {
                streamInfo.ringDeviceSignal[i] = notifys[2 * i];
                streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ReplaceCommInfoByTag(const std::string& tag, std::unique_ptr<CommInfo>& commInfo)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    tagCommInfo_.erase(tag);
    tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType,
    std::unique_ptr<CommInfo>& commInfo, u32 root, bool isP2p, bool isAicpuModeEn, bool isBatchSendRecv,
    bool meshSinglePlane, bool aivMode, std::set<u32> batchSendRecvtargetRanks)
{
    (void)batchSendRecvtargetRanks;
    (void)isBatchSendRecv;
    // Comm资源的唯一性，由上层调用保证
    // tag 多线程并行调度时唯一标识，不能为空
    CHK_PRT_RET(
        tag.empty(), HCCL_ERROR("[Create][Comm]errNo[0x%016llx] tag is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    commInfo.reset(new (std::nothrow) CommInfo);
    CHK_SMART_PTR_NULL(commInfo);

    DeviceMem inputMemComm(inputMem);
    DeviceMem outputMemComm(outputMem);
    DeviceMem expMemComm = cclBufferManager_.GetCommCCLBuffer();
    if (!isHaveCpuRank_) {
        inputMemComm = cclBufferManager_.GetCommRegMem(inputMem, MemAttr::IN_CCL_BUFFER, aivMode);
        outputMemComm = cclBufferManager_.GetCommRegMem(outputMem, MemAttr::OUT_CCL_BUFFER, aivMode);
    }

    if (isP2p) {
        CHK_RET(CreateP2pComm(tag, *commInfo, inputMemComm, root));
    } else if (isAicpuModeEn && deviceType_ == DevType::DEV_TYPE_910_93) {
        // level0 mesh通信域
        std::vector<std::unique_ptr<CommBase>> commMeshL0;
        CommParaInfo commCombinePara(COMM_MESH_L0, CommType::COMM_TAG_MESH);
        commCombinePara.isAicpuModeEn = isAicpuModeEn;
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMemComm, outputMemComm, commCombinePara, commInfo->commLevel0));
    } else {
        bool isA2MC2MultiServer = false;
        const std::string& suffix = HCCL_MC2_MULTISERVER_SUFFIX;
        if (tag.size() > suffix.size() && tag.compare(tag.size() - suffix.size(), suffix.size(), suffix) == 0) {
            isA2MC2MultiServer = true;
        }
        CHK_RET(CreateCommByAlg(
            tag, algType, *commInfo, inputMemComm, outputMemComm, expMemComm, root, isAicpuModeEn, meshSinglePlane,
            isA2MC2MultiServer));
    }

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType, u32 root, bool isP2p,
    bool isBatchSendRecv, bool meshSinglePlane, bool aivMode, std::set<u32> batchSendRecvtargetRanks)
{
    // tag 多线程并行调度时唯一标识，不能为空
    CHK_PRT_RET(
        tag.empty(), HCCL_ERROR("[Create][Comm]errNo[0x%016llx] tag is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    // 作下重复的判断，在Gather等逻辑梳理清楚后，再清理
    CHK_PRT_RET(
        IsExistCommRes(tag), HCCL_DEBUG("[HcclImpl][CreateComm] tag[%s] comm has existed, do nothing", tag.c_str()),
        HCCL_SUCCESS);

    std::unique_ptr<CommInfo> commInfo = nullptr;
    HcclResult ret = CreateComm(
        tag, inputMem, outputMem, algType, commInfo, root, isP2p, false, isBatchSendRecv, meshSinglePlane, aivMode,
        batchSendRecvtargetRanks);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[hcclImpl][CreateComm]create comminfo by tag[%s] failed. return[%d]", tag.c_str(), ret), ret);

    // 根据上下层逻辑，这里其实只是Save/Insert。
    CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetCommTypeInLevel0(const AlgType algType, const TopoType topoType, CommType& commType)
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_HD) {
            commType = CommType::COMM_TAG_HALVING_DOUBLING;
        } else {
            commType = CommType::COMM_TAG_RING_INNER;
        }
        HCCL_DEBUG(
            "[Get][CommTypeForLevel0]The algType is %s, topoType is %d, while commType is %d",
            AlgTypeToStr(algType).c_str(), topoType, commType);
        return HCCL_SUCCESS;
    }

    bool isMesh
        = ((topoType_ == TopoType::TOPO_TYPE_4P_MESH) || (topoType_ == TopoType::TOPO_TYPE_2P_MESH)
           || (topoType_ == TopoType::TOPO_TYPE_1P_MESH) || (topoType_ == TopoType::TOPO_TYPE_NP_MESH));

    // 根据算法类型创建内层拓扑
    if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_STAR) {
        commType = CommType::COMM_TAG_STAR;
    } else if (isMesh) {
        commType = CommType::COMM_TAG_MESH;
    } else {
        commType = CommType::COMM_TAG_RING_INNER;
    }
    HCCL_DEBUG(
        "[Get][CommTypeForLevel0]The algType is %s, topoType is %d, while commType is %d",
        AlgTypeToStr(algType).c_str(), topoType, commType);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::GetCommTypeInLevel1(const AlgType algType, CommType& commType)
{
    // 根据算法类型创建内层拓扑
    if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING) {
        commType = CommType::COMM_TAG_RING_COMBINED;
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_HD) {
        commType = CommType::COMM_TAG_HALVING_DOUBLING;
        /* pipeline ring场景下性能优化 */
    } else if (
        algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
        || algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        commType = CommType::COMM_TAG_RING_INNER;
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_STAR) {
        commType = CommType::COMM_TAG_STAR;
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
            commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        } else {
            commType = CommType::COMM_TAG_WHOLE_NHR;
        }
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
            commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1;
        } else {
            commType = CommType::COMM_TAG_WHOLE_NHR_V1;
        }
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
        if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
            commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE;
        } else {
            commType = CommType::COMM_TAG_WHOLE_AHC;
        }
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
            commType = CommType::COMM_TAG_ASYMMETRIC_HIERARCHICAL_CONCATENATE_BROKE;
        } else {
            commType = CommType::COMM_TAG_WHOLE_AHC_BROKE;
        }
    } else if (algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED) {
            commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        } else {
            commType = CommType::COMM_TAG_WHOLE_NB;
        }
    } else {
        HCCL_ERROR("[Get][CommTypeInLevel1]algType[%s] is not support", AlgTypeToStr(algType).c_str());
        return HCCL_E_PARA;
    }
    HCCL_DEBUG(
        "[Get][CommTypeInLevel1]The algType is %s, while commType is %d", AlgTypeToStr(algType).c_str(), commType);
    return HCCL_SUCCESS;
}

CommPlane hcclImpl::GetCommPlaneInLevel1(CommType& commType)
{
    CommPlane commPlane;
    switch (commType) {
        case CommType::COMM_TAG_RING_COMBINED: {
            commPlane = COMM_COMBINE;
            break;
        }

        case CommType::COMM_TAG_WHOLE_NB:
        case CommType::COMM_TAG_WHOLE_NHR:
        case CommType::COMM_TAG_WHOLE_NHR_V1:
        case CommType::COMM_TAG_MESH_COMBINED: {
            commPlane = COMM_COMBINE_ORDER;
            break;
        }

        default: {
            commPlane = COMM_LEVEL1;
            break;
        }
    }
    HCCL_DEBUG("[Get][CommPlaneInLevel1]The commType is %d, commPlane is %d", commType, commPlane);
    return commPlane;
}

HcclResult hcclImpl::CreateCommByAlg(
    const std::string& tag, const AlgType algType, CommInfo& commInfo, DeviceMem& inputMem, DeviceMem& outputMem,
    DeviceMem& expMem, u32 root, bool isAicpuModeEn, bool meshSinglePlane, bool isA2MC2MultiServer)
{
    CHK_RET(algConfigurator_->CheckAlgType(algType));
    CHK_RET(commFactory_->SetHDCModeInfo(
        rankDevicePhyIdNicInfoMap_, ranksPort_, vnicRanksPort_, isSetHDCModeInfo_, isUseRankPort_));

    HcclResult commThreadWaitResultLevel0 = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel0Rdma = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel1 = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel1Rdma = HCCL_SUCCESS;
    HcclResult commThreadWaitResultLevel2 = HCCL_SUCCESS;

    workflowMode_ = GetWorkflowMode(); // 后续会起新线程，因此更新workflowMode
    /* Level0通信域 */
    CommType commTypeInLevel0;
    HcclResult commThreadResultLevel0 = HCCL_SUCCESS;
    HcclResult commThreadResultLevel0Rdma = HCCL_SUCCESS;
    CHK_RET(GetCommTypeInLevel0(algType, topoType_, commTypeInLevel0));
    bool isUsedRdma = false;
    if (isA2MC2MultiServer) {
        HCCL_INFO("commInfo create commLevel0Rdma/commLevel1Rdma for EnableRdmaSdma start");
        isUsedRdma = true;
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        if (isAicpuModeEn) {
            commTypeInLevel0 = CommType::COMM_TAG_MESH;
        }
        // level0 通信域
        CommParaInfo commParaLevel0(COMM_LEVEL0, commTypeInLevel0);
        commParaLevel0.isAicpuModeEn = isAicpuModeEn;
        std::vector<std::unique_ptr<CommBase>> commVec;
        CHK_RET(commFactory_->CreateCommPlane(tag, inputMem, outputMem, commParaLevel0, commVec));

        CHK_PRT_RET(
            commVec.empty() || !commVec[0],
            HCCL_ERROR(
                "[Create][CommIntraServer]errNo[0x%016llx] tag[%s], created commIntraServer fail.",
                HCCL_ERROR_CODE(HCCL_E_NOT_FOUND), tag.c_str()),
            HCCL_E_NOT_FOUND);
        commInfo.commIntraServer = std::move(commVec[0]);
        return HCCL_SUCCESS;
    }
    CommParaInfo commInfoLevel0(
        COMM_LEVEL0, commTypeInLevel0, root, INVALID_VALUE_RANKID, isAicpuModeEn, meshSinglePlane);
    // default、whole_nhr和whole_nb算法不创建外层拓扑
    if (algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING
        && algType.algoLevel0 != AlgTypeLevel0::ALG_LEVEL0_RESERVED && !isA2MC2MultiServer) {
        commThreadPtrLevel0_.reset(new (std::nothrow) std::thread(
            &hcclImpl::CreateCommThread, this, hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem),
            std::ref(outputMem), std::ref(expMem), std::ref(commInfoLevel0), std::ref(commInfo.commLevel0),
            std::ref(commThreadResultLevel0)));
        CHK_PRT_RET(
            !commThreadPtrLevel0_,
            HCCL_ERROR("[Create][CommByAlg]commTypeInLevel0[%d] threads reset failed.", commInfoLevel0.commType),
            HCCL_E_INTERNAL);
        commThreadWaitResultLevel0 = WaitCommThread(commThreadPtrLevel0_);
        if (isUsedRdma) {
            commInfoLevel0.forceRdma = isUsedRdma;
            commThreadPtrLevel0Rdma_.reset(new (std::nothrow) std::thread(
                &hcclImpl::CreateCommThread, this, hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem),
                std::ref(outputMem), std::ref(expMem), std::ref(commInfoLevel0), std::ref(commInfo.commLevel0Rdma),
                std::ref(commThreadResultLevel0Rdma)));
            CHK_PRT_RET(
                !commThreadPtrLevel0Rdma_,
                HCCL_ERROR(
                    "[Create][CommByAlg]commTypeInLevel0[%d]"
                    " commLevel0Rdma threads reset failed.",
                    commInfoLevel0.commType),
                HCCL_E_INTERNAL);
            commThreadWaitResultLevel0Rdma = WaitCommThread(commThreadPtrLevel0Rdma_);
        }
    }

    /* Level1通信域 */
    HcclResult commThreadResultLevel1 = HCCL_SUCCESS;
    HcclResult commThreadResultLevel1Rdma = HCCL_SUCCESS;
    CommType commTypeInLevel1;
    CHK_RET(GetCommTypeInLevel1(algType, commTypeInLevel1));
    if (isA2MC2MultiServer) {
        commTypeInLevel1 = CommType::COMM_TAG_MESH_COMBINED;
    }

    CommPlane commPlaneInLevel1 = GetCommPlaneInLevel1(commTypeInLevel1);
    CommParaInfo commInfoLevel1(commPlaneInLevel1, commTypeInLevel1, root, INVALID_VALUE_RANKID, isAicpuModeEn);
    if (commTypeInLevel1 != CommType::COMM_TAG_STAR) {
        if (!isA2MC2MultiServer) {
            commThreadPtrLevel1_.reset(new (std::nothrow) std::thread(
                &hcclImpl::CreateCommThread, this, hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem),
                std::ref(outputMem), std::ref(expMem), std::ref(commInfoLevel1), std::ref(commInfo.commLevel1),
                std::ref(commThreadResultLevel1)));
            CHK_PRT_RET(
                !commThreadPtrLevel1_,
                HCCL_ERROR("[Create][CommByAlg]commTypeInLevel1[%d] threads reset failed.", commInfoLevel1.commType),
                HCCL_E_INTERNAL);
            commThreadWaitResultLevel1 = WaitCommThread(commThreadPtrLevel1_);
        }

        if (isUsedRdma) {
            commInfoLevel1.forceRdma = isUsedRdma;
            commThreadPtrLevel1Rdma_.reset(new (std::nothrow) std::thread(
                &hcclImpl::CreateCommThread, this, hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem),
                std::ref(outputMem), std::ref(expMem), std::ref(commInfoLevel1), std::ref(commInfo.commLevel1Rdma),
                std::ref(commThreadResultLevel1Rdma)));
            CHK_PRT_RET(
                !commThreadPtrLevel1Rdma_,
                HCCL_ERROR(
                    "[Create][CommByAlg]commTypeInLevel1[%d]"
                    " commLevel1Rdma threads reset failed.",
                    commInfoLevel1.commType),
                HCCL_E_INTERNAL);
            commThreadWaitResultLevel1Rdma = WaitCommThread(commThreadPtrLevel1Rdma_);
        }
    }

    /* Level2通信域 */
    HcclResult commThreadResultLevel2 = HCCL_SUCCESS;
    CommParaInfo commInfoLevel2(COMM_LEVEL2, CommType::COMM_TAG_RING_INNER);
    commThreadPtrLevel2_.reset(new (std::nothrow) std::thread(
        &hcclImpl::CreateCommThread, this, hrtErrMGetErrorContextPub(), std::ref(tag), std::ref(inputMem),
        std::ref(outputMem), std::ref(expMem), std::ref(commInfoLevel2), std::ref(commInfo.commLevel2),
        std::ref(commThreadResultLevel2)));
    CHK_PRT_RET(
        !commThreadPtrLevel2_,
        HCCL_ERROR("[Create][CommByAlg]commTypeInLevel2[%d] threads reset failed.", commInfoLevel2.commType),
        HCCL_E_INTERNAL);
    commThreadWaitResultLevel2 = WaitCommThread(commThreadPtrLevel2_);

    CHK_PRT_RET(
        static_cast<bool>(commThreadWaitResultLevel0) || static_cast<bool>(commThreadWaitResultLevel1)
            || static_cast<bool>(commThreadWaitResultLevel2) || static_cast<bool>(commThreadWaitResultLevel0Rdma)
            || static_cast<bool>(commThreadWaitResultLevel1Rdma),
        HCCL_ERROR(
            "[Create][CommByAlg]wait thread failed.algoLevel0[%d] Level1[%d] Level2[%d] Level0rdma[%d]"
            " Level1rdma[%d]",
            commThreadWaitResultLevel0, commThreadWaitResultLevel1, commThreadWaitResultLevel2,
            commThreadWaitResultLevel0Rdma, commThreadWaitResultLevel1Rdma),
        HCCL_E_INTERNAL);

    CHK_PRT_RET(
        static_cast<bool>(commThreadResultLevel0) || static_cast<bool>(commThreadResultLevel1)
            || static_cast<bool>(commThreadResultLevel2) || static_cast<bool>(commThreadResultLevel0Rdma)
            || static_cast<bool>(commThreadResultLevel1Rdma),
        HCCL_ERROR(
            "[Create][CommByAlg]CreateComm failed. result: Level0[%d] Level1[%d] Level2[%d]"
            " Level0rdma[%d] Level1rdma[%d].",
            commThreadResultLevel0, commThreadResultLevel1, commThreadResultLevel2, commThreadResultLevel0Rdma,
            commThreadResultLevel1Rdma),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateCommThread(
    const ErrContextPub& error_context, const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem,
    DeviceMem& expMem, const CommParaInfo& commParaInfo, std::vector<std::unique_ptr<CommBase>>& commVec,
    HcclResult& retOut)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_CreateComm");

    hrtErrMSetErrorContextPub(error_context);
    retOut = hrtSetDevice(deviceLogicId_);
    CHK_PRT_RET(
        retOut != HCCL_SUCCESS, HCCL_ERROR("[Create][CommThread]set device[%d] failed", deviceLogicId_), retOut);
    SetWorkflowMode(workflowMode_);

    retOut = commFactory_->CreateCommPlane(tag, inputMem, outputMem, commParaInfo, commVec, expMem);
    CHK_PRT_RET(
        retOut != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Create][CommThread]tag[%s], create comm level[%d] commType[%d] fail", tag.c_str(), commParaInfo.commPlane,
            commParaInfo.commType),
        retOut);

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateMutiStreamRes(
    const std::string& tag, Stream& stream, level1StreamInfo_t& streamInfo, AlgType algType, bool isAicpuModeEn,
    bool isBatchSendRecv, u32 ringNum)
{
    /* 多环资源初始化 */
    HcclResult ret = InitMultiStreamResource(tag, streamInfo, algType, isAicpuModeEn, isBatchSendRecv, ringNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Create][MutiStreamRes]tag[%s] init multi ring resource failed, return[%d]", tag.c_str(), ret),
        ret);

    CHK_RET(hccl::ProfilingManagerPub::CallMsprofReportMultiThreadInfo(streamInfo.tidInfo));

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        // GE OffloadStreamManager中set的流都是从流
        CHK_RET(workSpaceRes_->RegisterMaster(tag, stream));
        streamInfo.ringStreams = workSpaceRes_->AllocSlaveStreams(tag, streamInfo.ringNum - 1);
    } else if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        CHK_RET(opBaseStreamManager_.RegisterMaster(stream));
        streamInfo.ringStreams
            = opBaseStreamManager_.AllocSlaves(StreamType::STREAM_TYPE_ONLINE, streamInfo.ringNum - 1);

        if (isAicpuModeEn == true) {
            if (auxRingStreamsDev_.empty()) {
                auxRingStreamsDev_.reserve(MAX_SUBSTREAM_NUM + 1);
                HCCL_DEBUG("CreateMutiStreamRes: reserve auxRingStreamsDev_[%u]", MAX_SUBSTREAM_NUM);
            }
            if (auxRingStreamsDev_.size() < streamInfo.ringNum) {
                HCCL_DEBUG(
                    "CreateMutiStreamRes:tag[%s], auxRingStreamsDev_.size[%u], less than [%u], need create new streams",
                    tag.c_str(), auxRingStreamsDev_.size(), streamInfo.ringNum);
                CHK_PRT_RET(
                    streamInfo.ringNum > MAX_SUBSTREAM_NUM + 1,
                    HCCL_ERROR(
                        "[Create][MutiStreamRes]tag[%s] streamInfo.ringNum[%u] is larger than MAX_SUBSTREAM_NUM+1[%u].",
                        tag.c_str(), streamInfo.ringNum, MAX_SUBSTREAM_NUM + 1),
                    HCCL_E_INTERNAL);
                u32 ringNum = auxRingStreamsDev_.size();
                for (u32 ringIndex = ringNum; ringIndex < streamInfo.ringNum; ringIndex++) {
                    auxRingStreamsDev_.emplace_back(Stream(StreamType::STREAM_TYPE_DEVICE));
                    // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
                }
            }
            for (u32 ringIndex = 0; ringIndex < streamInfo.ringNum; ringIndex++) {
                streamInfo.ringDeviceStreams[ringIndex] = auxRingStreamsDev_[ringIndex];
                CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
            }
        }
    } else {
        HCCL_ERROR("[Create][MutiStreamRes]WorkflowMode[%d] invalid", GetWorkflowMode());
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(
        (streamInfo.ringStreams.size() != streamInfo.ringNum - 1),
        HCCL_ERROR(
            "[Create][MutiStreamRes]tag[%s] get slave stream failed, "
            "expect to get size [%u], but only alloc [%u].",
            tag.c_str(), streamInfo.ringNum - 1, streamInfo.ringStreams.size()),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::CreateMutiStreamRes(
    const std::string& tag, Stream& stream, AlgType algType, bool isBatchSendRecv, u32 ringNum)
{
    std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
    CHK_PRT_RET(
        tagStreamInfo_.find(tag) != tagStreamInfo_.end(),
        HCCL_DEBUG("[Create][MutiStreamRes]tag[%s] already exists, do nothing", tag.c_str()), HCCL_SUCCESS);

    level1StreamInfo_t streamInfo;
    CHK_RET(CreateMutiStreamRes(tag, stream, streamInfo, algType, false, isBatchSendRecv, ringNum));

    // 构建线程和内部流维护关系
    tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(streamInfo)));
    mutiStreamLock.unlock();
    HCCL_INFO("[Create][MutiStreamRes]tag[%s], ringNum[%u]", tag.c_str(), streamInfo.ringNum);
    return HCCL_SUCCESS;
}

void hcclImpl::DestroyLevel1Comm(const std::string& tag)
{
    // vector成员是智能指针, 自动destroy
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commLevel1.clear();
    }
}

void hcclImpl::DestroyLevel0Comm(const std::string& tag)
{
    // vector成员是智能指针, 自动destroy
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commLevel0.clear();
    }
}

void hcclImpl::DestroyIntraServerComm(const std::string& tag)
{
    tagCommInfo_t::iterator itr = tagCommInfo_.find(tag);
    if (itr != tagCommInfo_.end()) {
        itr->second.commIntraServer.reset();
    }
}

HcclResult hcclImpl::ReleaseSignal(level1StreamInfo_t& level1Stream)
{
    for (auto& signal : level1Stream.ringSignal) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto& signal : level1Stream.ringSignalAux) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto& signal : level1Stream.ringDeviceSignal) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    for (auto& signal : level1Stream.ringDeviceSignalAux) {
        if (signal != nullptr) {
            signal = nullptr;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult hcclImpl::ClearOpResource(const std::string& tag)
{
    // 链接资源释放
    commMeshMap_.erase(tag);
    tagCommInfo_.erase(tag);
    // stream解绑定
    auto iterStream = tagStreamInfo_.find(tag);
    if (iterStream != tagStreamInfo_.end()) {
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
    }
    tagStreamInfo_.erase(tag);
    // scratchMemMap_清理
    scratchMemMap_.erase(tag);
    return HCCL_SUCCESS;
}

HcclResult hcclImpl::SetRankPortInfo(s32 deviceLogicID, bool isUseRankPort, std::vector<u32>& ranksPort)
{
    if (g_SetRankPortInfoCallBack != nullptr) {
        return g_SetRankPortInfoCallBack(deviceLogicID, isUseRankPort, ranksPort);
    } else {
        HCCL_RUN_WARNING("[SetRankPortInfo] g_SetRankPortInfoCallBack is nullptr");
    }
    return HCCL_SUCCESS;
}

void hcclImpl::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
    std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    rankDevicePhyIdNicInfoMap_ = rankDevicePhyIdNicInfoMap;
    ranksPort_ = ranksPort;
    vnicRanksPort_ = vnicRanksPort;
    isSetHDCModeInfo_ = isSetHDCModeInfo;
    isUseRankPort_ = isUseRankPort;
}
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void RegisterHeartBeatCallBack(
    RegisterToHeartBeatCallBack p1, UnRegisterToHeartBeatCallBack p2, SetRankPortInfoCallBack p3)
{
    g_RegisterToHeartBeatCallBack = p1;
    g_UnRegisterToHeartBeatCallBack = p2;
    g_SetRankPortInfoCallBack = p3;
}
#ifdef __cplusplus
}
#endif // __cplusplus
} // namespace hccl
// namespace hccl
