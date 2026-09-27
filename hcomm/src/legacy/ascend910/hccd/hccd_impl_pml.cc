/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <numeric>
#include <sys/time.h>
#include <dlog_pub.h>
#include "sal.h"

#include "adapter_prof.h"
#include "dlprof_function.h"
#include "dlrt_function.h"
#include "externalinput_pub.h"
#include "transport_heterog_event_roce.h"
#include "transport_heterog_roce.h"
#include "device_capacity.h"
#include "rank_consistentcy_checker.h"
#include "hccd_impl_pml.h"

using namespace std;

namespace hccl {

HccdImplPml::HccdImplPml()
    : initializedFlag_(ATOMIC_FLAG_INIT),
      userRank_(INVALID_VALUE_RANKID),
      realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE),
      devicePhyId_(INVALID_UINT),
      deviceLogicId_(-1),
      hcomGroupNicInit_(false),
      heterogRaInit_(false),
      hostRdmaInitFlag_(false),
      commHandle_(nullptr),
      mrManager_(nullptr),
      pMsgInfosMem_(nullptr),
      pReqInfosMem_(nullptr),
      memBlocksManager_(nullptr),
      pRecvWrInfosMem_(nullptr),
      transportResourceInfo_(mrManager_, pMsgInfosMem_, pReqInfosMem_, memBlocksManager_, pRecvWrInfosMem_),
      profilingInitiated_(false),
      mrManagerInit_(false),
      srqInit_(false)
{}

HccdImplPml::~HccdImplPml()
{
    // 销毁异构通信资源
    DestroyHeterogTransport();
    DestroySrq();
    DeInitTransportMem();
    MrManagerDeInit();

    /* 网络资源销毁 */
    DeinitHeterogRaResource();
}

HcclResult HccdImplPml::Init(HcclCommParams& params, const RankTable_t& rankTable)
{
    CHK_RET(InitCommParams(params));

    CHK_RET(InitTcpMode(rankTable));

    // 获取serverId
    CHK_RET(GetServerId(rankTable));

    // 根据server整理rank信息
    CHK_RET(TransformRankInfoByServerId(rankTable.rankList, servRankInfo_));

    // 生成nicList
    for (auto iter : servRankInfo_[serverId_]) {
        if (((!iter.hostIp.IsInvalid()) || (!iter.deviceInfo.deviceIp[0].IsInvalid()))
            && (iter.deviceInfo.devicePhyId != HOST_DEVICE_ID)) {
            nicList_.push_back(iter.deviceInfo.devicePhyId);
        }
    }
    std::sort(nicList_.begin(), nicList_.end());

    // 解析ranktable信息(生成rankInfoList_)，供给commfactory使用
    CHK_RET(GetRankInfoList(rankTable));

    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        HCCL_DEBUG(
            " host ip: %s host port: %u dev phy id: %d", rankInfoList_[i].hostIp.GetReadableAddress(),
            rankInfoList_[i].hostPort, rankInfoList_[i].devicePhyId);
        if (rankInfoList_[i].userRank == userRank_) {
            devIpAddr_ = rankInfoList_[i].nicIp;
            devicePhyId_ = rankInfoList_[i].devicePhyId;
            break;
        }
    }

    ranksPort_.resize(userRankSize_, 0);
    for (auto rankInfo : rankTable.rankList) {
        ranksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT || rankInfo.deviceInfo.port == 0 ?
                                          HETEROG_CCL_PORT :
                                          rankInfo.deviceInfo.port;
    }

    // 在确定 servRankInfo_ 和 serverId_ 信息后，就完成初始判断

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    HCCL_INFO("init heterog comm, rank id[%u] device id[%u]", userRank_, devicePhyId_);

    HcclResult ret = InitPara(rankTable.collectiveId);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclImplBase][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
            HCCL_ERROR_CODE(ret), rankTable.collectiveId.c_str()),
        ret);

    CHK_PRT_RET(
        devIpAddr_.empty(),
        HCCL_ERROR(
            "[HcclImplBase][Init]devIpAddr_ size[%llu] "
            "should be greater than 0.",
            devIpAddr_.size()),
        HCCL_E_UNAVAIL);

    if (params.attr.mode != WorkMode::HCCL_MODE_AI_CPU && params.attr.mode != WorkMode::HCCL_MODE_PS) {
        CHK_RET(InitHeterogRaResource(rankTable));
        CHK_RET(InitHeterogRecvExecutor());
        CHK_RET(MrManagerInit());
        CHK_RET(InitRecvMsgAndRequestBuffer());
        CHK_RET(InitMemBlocksAndRecvWrMem());
        CHK_RET(CreateSrq());
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::InitCommParams(HcclCommParams& params)
{
    commHandle_ = params.commHandle;
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    profilingOption_ = params.profilingOption;
    profilingInitiated_ = params.profilingInitiated;
    hcomGroupNicInit_ = params.hcomGroupNicInit;
    identifier_ = params.identifier;
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::GetServerId(const RankTable_t& rankTable)
{
    for (u32 i = 0; i < rankTable.rankList.size(); i++) {
        if (rankTable.rankList[i].rankId == userRank_) {
            serverId_ = rankTable.rankList[i].serverId;
            break;
        }
    }
    if (serverId_.empty()) {
        HCCL_ERROR("[Get][ServerId]GetServerId fail");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult
HccdImplPml::TransformRankInfoByServerId(const std::vector<RankInfo_t>& rankList, ServRankInfo_t& servRankInfo) const
{
    // 按server重新组织rank信息，便于后续校验及信息填写
    for (size_t index = 0; index < rankList.size(); ++index) {
        const RankInfo_t& rankInfo = rankList[index];
        std::string serverId = SalTrim(rankInfo.serverId);
        // 以serverID为索引，将server下的ranks放入vector
        ServRankInfo_t::iterator itr = servRankInfo.find(serverId);
        if (itr != servRankInfo.end()) {
            itr->second.push_back(rankInfo);
        } else {
            std::vector<RankInfo_t> rankInfoList;
            rankInfoList.push_back(rankInfo);
            std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair(serverId, rankInfoList);
            servRankInfo.insert(rankInfoPair);
        }
    }
    // 每个server下的rank列表按设备Id从小到大的顺序排序
    for (auto& iter : servRankInfo) {
        std::sort(iter.second.begin(), iter.second.end(), CompareWithDevicePhyId);
    }
    return HCCL_SUCCESS;
}

bool HccdImplPml::CompareWithDevicePhyId(const RankInfo_t& left, const RankInfo_t& right)
{
    return left.deviceInfo.devicePhyId < right.deviceInfo.devicePhyId;
}

HcclResult HccdImplPml::InitTcpMode(const RankTable_t& rankTable) const
{
    bool isTcpMode = false;
    HCCL_INFO("[TcpMode][%u] [1:TCP, 2:RDMA, 3:RESERVED]", GetExternalInputProtocolType());
    if (GetExternalInputProtocolType() == ProtocolType::TCP) {
        isTcpMode = true;
    } else if (GetExternalInputProtocolType() == ProtocolType::RDMA) {
        // 通信协议选择RDMA
    } else {
        isTcpMode = (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST);
        HCCL_INFO("[Init][TcpMode]isTcpMode[%d] nicDeploy[%d]", isTcpMode, rankTable.nicDeploy);
    }
    SetTcpMode(isTcpMode);
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::GetRankInfoList(const RankTable_t& rankTable)
{
    // 遍历rank table获取rank信息
    rankInfoList_.clear();
    for (auto iter = servRankInfo_.begin(); iter != servRankInfo_.end(); ++iter) {
        for (u32 index = 0; index < iter->second.size(); ++index) {
            const RankInfo_t& orgRankInfo = iter->second[index];
            // 构建comm 使用的rank 信息
            RankInfo rankInfo;
            rankInfo.userRank = orgRankInfo.rankId;
            rankInfo.worldRank = orgRankInfo.rankId;
            rankInfo.devicePhyId = orgRankInfo.deviceInfo.devicePhyId;

            rankInfo.serverId = orgRankInfo.serverId;
            rankInfo.serverIdx = orgRankInfo.serverIdx;
            rankInfo.hostIp = orgRankInfo.hostIp;
            rankInfo.hostPort = orgRankInfo.hostPort;
            rankInfo.localRank = orgRankInfo.localRank;
            rankInfo.superPodId = orgRankInfo.superPodId;
            CHK_RET(GetNicInfo(rankTable.nicDeploy, index, iter->second, rankInfo));
            rankInfo.nicIdx.assign(nicList_.begin(), nicList_.end());
            rankInfoList_.push_back(rankInfo);
        }
    }
    // 将rank id从小到大的顺序返回
    CHK_RET(SortRankInfoList());

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::GetNicInfo(
    const NICDeployment& nicDeploy, const u32 curRankIndex, const std::vector<RankInfo_t>& servRankList,
    RankInfo& rankInfo) const
{
    CHK_PRT_RET(
        servRankList.empty(),
        HCCL_ERROR("[Get][NicInfo]errNo[0x%016llx] server rank list is empty", HCCL_ERROR_CODE(HCCL_E_PARA)),
        HCCL_E_PARA);

    rankInfo.nicDeploy = nicDeploy;
    if (nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST) {
        // 检查网卡个数
        // 网卡挂载位置在host时，按rank index从网卡列表中获取
        const RankInfo_t& curRankInfo = servRankList[curRankIndex];
        rankInfo.nicIp.push_back(curRankInfo.hostIp);
    } else {
        CHK_PRT_RET(
            curRankIndex >= servRankList.size(),
            HCCL_ERROR(
                "[Get][NicInfo]rankindex[%u] invalid,rank list "
                "size is[%zu]",
                curRankIndex, servRankList.size()),
            HCCL_E_PARA);

        const RankInfo_t& curRankInfo = servRankList[curRankIndex];
        CHK_PRT_RET(
            curRankInfo.deviceInfo.deviceIp.size() == 0,
            HCCL_ERROR("[Get][NicInfo]rankindex[%u] invalid,deviceIp is zero", curRankIndex), HCCL_E_PARA);
        rankInfo.nicIp.push_back(curRankInfo.deviceInfo.deviceIp[0]);
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::SortRankInfoList()
{
    // 按rank id从小到大的顺序返回
    std::sort(rankInfoList_.begin(), rankInfoList_.end(), CompareWithUserRank);

    for (u32 index = 0; index < rankInfoList_.size(); ++index) {
        CHK_PRT_RET(
            (index != rankInfoList_[index].userRank),
            HCCL_ERROR(
                "[HcclImplBase][SortRankInfoList]errNo[0x%016llx] index[%u] != rankInfoList.userRank[%u]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, rankInfoList_[index].userRank),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

bool HccdImplPml::CompareWithUserRank(const RankInfo& left, const RankInfo& right)
{
    return left.userRank < right.userRank;
}

HcclResult HccdImplPml::InitPara(const std::string& colectiveId)
{
    // 检查当前user_rank 对应的devid和rt查到的一致
    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        if ((userRank_ == rankInfoList_[i].userRank)
            && (static_cast<s32>(devicePhyId_) != rankInfoList_[i].devicePhyId)) {
            HCCL_ERROR(
                "[Init][Para]errNo[0x%016llx] parameter check failed,userrank[%u] == rankInfoList.userrank[%u],"
                "phyid[%d] != rankInfoList.devid[%d]",
                HCCL_ERROR_CODE(HCCL_E_PARA), userRank_, rankInfoList_[i].userRank, static_cast<s32>(devicePhyId_),
                rankInfoList_[i].devicePhyId);
            return HCCL_E_PARA;
        }
    }
    collectiveId_ = colectiveId;

    workSpaceRes_.reset(new (std::nothrow) WorkspaceResource(devicePhyId_, deviceLogicId_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::InitHeterogRaResource(const RankTable_t& rankTable)
{
    CHK_PRT_RET(
        rankTable.rankList.size() != userRankSize_,
        HCCL_ERROR(
            "[Init][HeterogRaResourc] rank list size[%u]"
            " is different from user rank size[%u]",
            rankTable.rankList.size(), userRankSize_),
        HCCL_E_PARA);
    ranksPort_.resize(userRankSize_, 0);
    for (auto rankInfo : rankTable.rankList) {
        ranksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT || rankInfo.deviceInfo.port == 0 ?
                                          HETEROG_CCL_PORT :
                                          rankInfo.deviceInfo.port;
    }

    heterogRaInit_ = true;
    CHK_RET(
        NetworkManager::GetInstance(deviceLogicId_).HeterogInit(devicePhyId_, devIpAddr_[0], ranksPort_[userRank_]));
    if (!GetExternalInputHcclIsTcpMode()) {
        hostRdmaInitFlag_ = true;
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::InitRecvMsgAndRequestBuffer()
{
    // 拉远、下沉、推理场景(ps、worker)支持使用msg/request内存池
    if (pMsgInfosMem_ == nullptr) {
        pMsgInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclMessageInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pMsgInfosMem_);
        CHK_RET(pMsgInfosMem_->Init());
        HCCL_INFO("InitRecvMsgBuffer Success!");
    }

    if (pReqInfosMem_ == nullptr) {
        pReqInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<HcclRequestInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pReqInfosMem_);
        CHK_RET(pReqInfosMem_->Init());
        HCCL_INFO("InitRequestBuffer Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::InitMemBlocksAndRecvWrMem()
{
    u32 memBlockNum = MEM_BLOCK_NUM;
    CHK_PRT(GetMemBlockNum(devicePhyId_, memBlockNum));

    if (!GetExternalInputHcclIsTcpMode()) {
        // 初始化信封内存
        memBlocksManager_.reset(new (std::nothrow) HeterogMemBlocksManager());
        CHK_SMART_PTR_NULL(memBlocksManager_);
        CHK_RET(memBlocksManager_->Init(memBlockNum));

        // 信封内存注册
        CHK_RET(mrManager_->GetKey(
            memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize(), transportResourceInfo_.lkey));

        // 初始化wr内存
        pRecvWrInfosMem_.reset(new (std::nothrow) LocklessRingMemoryAllocate<RecvWrInfo>(MEMORY_CAPACITY));
        CHK_SMART_PTR_NULL(pRecvWrInfosMem_);
        CHK_RET(pRecvWrInfosMem_->Init());
        HCCL_INFO("InitMemBlocksAndRecvWrMem Success!");
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::CreateSrq()
{
    u32 info = 0;
    CHK_RET(hrtDrvGetPlatformInfo(&info));
    if (info == 0) {
        std::string chipName;
        HcclResult ret = hrtHalGetChipInfo(devicePhyId_, chipName);
        if (ret == HCCL_SUCCESS) {
            if (chipName.find(SOC_NAME_910B) != std::string::npos) {
                HCCL_INFO("not support chip[%s] create srq", chipName.c_str());
                return HCCL_SUCCESS;
            }
        }
    }

    if (!srqInit_ && !GetExternalInputHcclIsTcpMode()) {
        RaResourceInfo raResourceInfo;
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
        void* nicRdmaHandle = raResourceInfo.nicSocketMap[devIpAddr_[0]].nicRdmaHandle;

        // 创建srq
        transportResourceInfo_.tagSrqInfo.srqEvent = HCCL_EVENT_RECV_REQUEST_MSG;
        transportResourceInfo_.dataSrqInfo.srqEvent = HCCL_EVENT_SEND_COMPLETION_MSG;

        transportResourceInfo_.tagSrqInfo.srqDepth = MAX_SRQ_DEPTH;
        transportResourceInfo_.dataSrqInfo.srqDepth = MAX_SRQ_DEPTH;
        CHK_RET(hrtRaCreateSrq(nicRdmaHandle, transportResourceInfo_.tagSrqInfo));
        CHK_RET(hrtRaCreateSrq(nicRdmaHandle, transportResourceInfo_.dataSrqInfo));
        HCCL_INFO("CreateSrq Success!");

        std::unique_ptr<TransportHeterogEventRoce> transportPtr;
        transportPtr.reset(new (std::nothrow) TransportHeterogEventRoce(transportResourceInfo_));
        CHK_SMART_PTR_NULL(transportPtr);
        CHK_RET(transportPtr->InitSrqRecvWqe());

        srqInit_ = true;
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::AtomicInitSet()
{
    CHK_PRT_RET(
        initializedFlag_.test_and_set(),
        HCCL_ERROR(
            "[HcclImplBase][AtomicInitSet]errNo[0x%016llx] instance "
            "has already been initialized",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void HccdImplPml::AtomicInitClear() { initializedFlag_.clear(); }

HcclResult HccdImplPml::RegisterMemory(void* buffer, uint64_t size)
{
    // 拉远、推理场景PS侧支持注册全局内存
    if (hostRdmaInitFlag_) {
        CHK_RET(mrManager_->RegGlobalMr(buffer, size));
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::UnregisterMemory(void* buffer)
{
    if (hostRdmaInitFlag_) {
        CHK_RET(mrManager_->DeRegGlobalMr(buffer));
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::CheckCount(const u64 count) const
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR(
            "[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::CheckDataType(const HcclDataType dataType, bool needReduce)
{
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT64) || (dataType == HCCL_DATA_TYPE_UINT8)
            || (dataType == HCCL_DATA_TYPE_UINT16) || (dataType == HCCL_DATA_TYPE_UINT32)
            || (dataType == HCCL_DATA_TYPE_FP64) || (dataType == HCCL_DATA_TYPE_RESERVED)) {
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),
                GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported", HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT),
                GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::Isend(
    void* buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest& requestHandle, u32 userRequire)
{
    if ((buffer == nullptr) && (count != 0)) {
        HCCL_ERROR(
            "[Check][Buffer]errNo[0x%016llx] buffer[%p] or count[%d] is invalid", HCCL_ERROR_CODE(HCCL_E_PARA), buffer,
            count);
        return HCCL_E_PARA;
    }
    if (peerRank >= userRankSize_) {
        HCCL_ERROR(
            "[Check][UserRank]errNo[0x%016llx] peerRank:[%u] is out of range[0 ~ %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            peerRank, userRankSize_);
        return HCCL_E_PARA;
    }

    TransportHandle transportHandle = nullptr;
    CHK_RET(BuildHeterogeneousTransport(0, peerRank, tag, transportHandle));

    TransportHeterog* transportPtr = reinterpret_cast<TransportHeterog*>(transportHandle);
    HcclRequestInfo* request = nullptr;
    TransData sendData(
        reinterpret_cast<u64>(buffer), reinterpret_cast<u64>(nullptr), count, dataType, false, userRequire);
    TransportEndPointInfo srcEp(0, userRank_, tag);
    TransportEndPointInfo dstEp(0, peerRank, tag);
    TransportEndPointParam epParam(srcEp, dstEp);
    CHK_RET(transportPtr->Isend(sendData, epParam, request));
    request->commHandle = commHandle_;
    requestHandle = request;
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::BuildHeterogeneousTransport(u32 commId, u32 peerRank, s32 tag, TransportHandle& transportHandle)
{
    TransportEndPointInfo commRankTagKey(commId, peerRank, tag);
    std::unique_lock<SpinMutex> transportMapLock(transportMapSpinMutex_);
    std::unique_ptr<TransportHeterog>& transportInfo = transportStorage_[commRankTagKey];
    transportMapLock.unlock();

    if (transportInfo == nullptr) {
        std::string transTag;
        if (userRank_ > peerRank) {
            transTag = collectiveId_ + "_" + std::to_string(peerRank) + "_" + std::to_string(userRank_) + "_";
        } else {
            transTag = collectiveId_ + "_" + std::to_string(userRank_) + "_" + std::to_string(peerRank) + "_";
        }
        transTag += std::to_string(tag);
        std::unique_ptr<TransportHeterog> transportPtr;
        // 当前代码只保留TransportHeterogEventRoce
        transportPtr.reset(new (std::nothrow) TransportHeterogEventRoce(
            transTag, rankInfoList_[userRank_].nicIp[0], rankInfoList_[peerRank].nicIp[0], ranksPort_[peerRank],
            ranksPort_[userRank_], transportResourceInfo_));
        CHK_SMART_PTR_NULL(transportPtr);
        CHK_RET(transportPtr->SetDeviceIndex(deviceLogicId_));
        CHK_RET(transportPtr->Init());
        transportInfo = std::move(transportPtr);
    }
    transportHandle = transportInfo.get();
    CHK_PTR_NULL(transportHandle);
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::Improbe(u32 peerRank, s32 tag, s32& flag, HcclMessage& msgHandle, HcclStatus& status)
{
    if (peerRank >= userRankSize_) {
        HCCL_ERROR(
            "[Check][UserRank]errNo[0x%016llx] peerRank:[%u] is out of range[0 ~ %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            peerRank, userRankSize_);
        return HCCL_E_PARA;
    }

    void* transportHandle = nullptr;
    CHK_RET(BuildHeterogeneousTransport(0, peerRank, tag, transportHandle));

    TransportHeterog* transportPtr = reinterpret_cast<TransportHeterog*>(transportHandle);
    TransportEndPointInfo srcEp(0, peerRank, tag);
    TransportEndPointInfo dstEp(0, userRank_, tag);
    TransportEndPointParam epParam(srcEp, dstEp);
    HcclMessageInfo* msg = nullptr;
    CHK_RET(transportPtr->Improbe(epParam, flag, msg, status));
    msgHandle = msg;
    return HCCL_SUCCESS;
}

HcclResult
HccdImplPml::Imrecv(void* buffer, s32 count, HcclDataType dataType, HcclMessage msgHandle, HcclRequest& requestHandle)
{
    HcclMessageInfo* msg = static_cast<HcclMessageInfo*>(msgHandle);
    CHK_PTR_NULL(msg);
    TransportHeterog* transportPtr = reinterpret_cast<TransportHeterog*>(msg->transportHandle);
    CHK_PTR_NULL(transportPtr);

    HcclRequestInfo* request = nullptr;
    TransData recvData(reinterpret_cast<u64>(nullptr), reinterpret_cast<u64>(buffer), count, dataType);
    CHK_RET(transportPtr->Imrecv(recvData, *msg, request));
    requestHandle = request;
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::HcclTest(HcclRequest requestHandle, s32& flag, HcclStatus& compState)
{
    HcclRequestInfo* request = reinterpret_cast<HcclRequestInfo*>(requestHandle);
    CHK_PTR_NULL(request->transportHandle);

    TransportHeterog* transportPtr = reinterpret_cast<TransportHeterog*>(request->transportHandle);
    return transportPtr->Test(*request, flag, compState);
}

u32 HccdImplPml::GetUserRank() { return realUserRank_; }

u32 HccdImplPml::GetRankSize() { return userRankSize_; }

void HccdImplPml::DestroyHeterogTransport()
{
    std::unique_lock<SpinMutex> transportMapLock(transportMapSpinMutex_);
    transportStorage_.clear();
    return;
}

HcclResult HccdImplPml::DestroySrq()
{
    if (srqInit_) {
        RaResourceInfo raResourceInfo;
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
        void* nicRdmaHandle = raResourceInfo.nicSocketMap[devIpAddr_[0]].nicRdmaHandle;

        // 销毁srq
        CHK_RET(hrtRaDestroySrq(nicRdmaHandle, transportResourceInfo_.tagSrqInfo));
        CHK_RET(hrtRaDestroySrq(nicRdmaHandle, transportResourceInfo_.dataSrqInfo));
        transportResourceInfo_.tagSrqInfo = SrqInfo();
        transportResourceInfo_.dataSrqInfo = SrqInfo();
        HCCL_INFO("DestroySrq Success!");
        srqInit_ = false;
    }

    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::DeInitTransportMem()
{
    if (memBlocksManager_ != nullptr) {
        // 解注册内存
        CHK_RET(mrManager_->ReleaseKey(memBlocksManager_->GetMemAddr(), memBlocksManager_->GetMemSize()));
        memBlocksManager_ = nullptr;
    }

    if (pMsgInfosMem_ != nullptr) {
        pMsgInfosMem_ = nullptr;
    }

    if (pReqInfosMem_ != nullptr) {
        pReqInfosMem_ = nullptr;
    }

    if (pRecvWrInfosMem_ != nullptr) {
        pRecvWrInfosMem_ = nullptr;
    }

    HCCL_INFO("DeInitTransportMem Success!");
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::MrManagerInit()
{
    // 拉远、下沉、推理场景(ps、worker)支持使用mrManager
    if (!GetExternalInputHcclIsTcpMode()) {
        mrManager_.reset(new (std::nothrow) MrManager());
        CHK_SMART_PTR_NULL(mrManager_);

        RaResourceInfo raResourceInfo;
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
        void* nicRdmaHandle = raResourceInfo.nicSocketMap[devIpAddr_[0]].nicRdmaHandle;

        CHK_RET(mrManager_->Init(nicRdmaHandle));
        mrManagerInit_ = true;
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::MrManagerDeInit()
{
    if (mrManagerInit_) {
        RaResourceInfo raResourceInfo;
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
        void* nicRdmaHandle = raResourceInfo.nicSocketMap[devIpAddr_[0]].nicRdmaHandle;

        CHK_SMART_PTR_NULL(mrManager_);
        CHK_RET(mrManager_->DeInit(nicRdmaHandle));
        mrManager_ = nullptr;
        mrManagerInit_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::DeinitHeterogRaResource()
{
    if (heterogRaInit_) {
        HCCL_INFO("deinit heterog ra resource!");
        CHK_RET(NetworkManager::GetInstance(deviceLogicId_)
                    .HeterogDeinit(devicePhyId_, devIpAddr_[0], ranksPort_[userRank_]));
        heterogRaInit_ = false;
    }
    return HCCL_SUCCESS;
}

HcclResult HccdImplPml::InitHeterogRecvExecutor() const
{
    std::vector<SocketWlistInfoT> whiteList(userRankSize_);
    for (u32 i = 0; i < rankInfoList_.size(); i++) {
        whiteList[i].remoteIp.addr = rankInfoList_[i].nicIp[0].GetBinaryAddress().addr;
        whiteList[i].remoteIp.addr6 = rankInfoList_[i].nicIp[0].GetBinaryAddress().addr6;
        whiteList[i].connLimit = CONN_LIMIT;
    }

    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).GetRaResourceInfo(raResourceInfo));
    SocketHandle nicSocketHandle = raResourceInfo.nicSocketMap[devIpAddr_[0]].nicSocketHandle;

    HCCL_DEBUG("ip[%s] device[%d]", devIpAddr_[0].GetReadableAddress(), deviceLogicId_);

    CHK_RET(hrtRaSocketWhiteListAdd(nicSocketHandle, whiteList.data(), userRankSize_));

    return HCCL_SUCCESS;
}

std::string HccdImplPml::GetUniqueId(void)
{
    static std::atomic<u32> idCounter(0);

    std::string uniqueId("");
    uniqueId += std::to_string(SalGetPid());
    uniqueId += '-';
    uniqueId += std::to_string(idCounter.fetch_add(1));
    uniqueId += '-';
    uniqueId += std::to_string(SalGetSysTime());

    return uniqueId;
}

} // namespace hccl
