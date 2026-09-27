/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCD_IMPL_PML_H
#define HCCD_IMPL_PML_H

#include <atomic>
#include <memory>
#include <unordered_set>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "hccl_comm_pub.h"
#include "topoinfo_parse.h"
#include "network_manager_pub.h"
#include "workspace_resource.h"
#include "transport_heterog_def.h"
#include "transport_heterog_pub.h"
#include "dlhal_function.h"
#include "mr_manager.h"
#include "spin_mutex.h"

namespace hccl {
constexpr u32 MEMORY_CAPACITY = 256 * 1024;
constexpr u32 WAIT_PREPARE_SLEEP_TIME = 5000;
constexpr u32 SINGLE_SERVER_NUM = 1;
constexpr u32 CONN_LIMIT = 4096;

using TransportStorageMap
    = std::unordered_map<TransportEndPointInfo, std::unique_ptr<TransportHeterog>, TransportEndPointInfoHash>;
using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t>>;
class HccdImplPml {
public:
    explicit HccdImplPml();
    ~HccdImplPml();
    HcclResult Init(HcclCommParams& params, const RankTable_t& rankTable);
    HcclResult GetServerId(const RankTable_t& rankTable);
    HcclResult InitCommParams(HcclCommParams& params);
    HcclResult TransformRankInfoByServerId(const std::vector<RankInfo_t>& rankList, ServRankInfo_t& servRankInfo) const;
    HcclResult InitTcpMode(const RankTable_t& rankTable) const;
    HcclResult GetServerNum(const std::vector<RankInfo_t>& ranks);
    HcclResult GetInnerServerAverageDevice(const RankTable_t& rankTable);
    HcclResult GetModuleInfo(const std::vector<RankInfo_t>& rankList);
    bool IsDiffDeviceModule(const std::vector<RankInfo_t>& rankList) const;
    HcclResult CheckSingleServerComm(const std::vector<RankInfo_t>& rankList) const;
    HcclResult GetRankInfoList(const RankTable_t& rankTable);
    HcclResult SortRankInfoList();
    static bool CompareWithUserRank(const RankInfo& left, const RankInfo& right);
    HcclResult InitPara(const std::string& colectiveId);
    HcclResult CalAndSetMeshAggRankSize();
    u32 CalMeshAggRankSize(int halfDevNum) const;
    HcclResult SetMeshAggregationRankSize(u32 size);
    HcclResult InitHeterogRaResource(const RankTable_t& rankTable);
    HcclResult MrManagerInit();
    HcclResult InitRecvMsgAndRequestBuffer();
    HcclResult InitMemBlocksAndRecvWrMem();
    HcclResult CreateSrq();
    HcclResult InitPreResource(const RankTable_t& rankTable);
    bool IsEnableRoce();
    HcclResult InitNic();
    HcclResult InitHeterogHostNic(void);
    HcclResult InitProfiling();
    HcclResult RegistTaskExceptionHandler() const;
    void NotifyPrepareComm();
    HcclResult UnRegistTaskExceptionHandler() const;
    void DestroyHeterogTransport();
    HcclResult DestroySrq();
    HcclResult DeInitTransportMem();
    HcclResult MrManagerDeInit();
    HcclResult DestroyNetworkResources();
    HcclResult DeinitNic();
    HcclResult DeinitHeterogHostNic();
    HcclResult DeinitHeterogRaResource();
    HcclResult DisablePreResource();
    HcclResult DeinitProfiling();
    HcclResult InitHeterogRecvExecutor() const;
    static bool CompareWithDevicePhyId(const RankInfo_t& left, const RankInfo_t& right);
    HcclResult AtomicInitSet();
    void AtomicInitClear();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);
    HcclResult DestroyCDomainResource(s32 tag);
    HcclResult RegisterMemory(void* buffer, uint64_t size);
    HcclResult UnregisterMemory(void* buffer);
    HcclResult CheckCount(const u64 count) const;
    HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);
    HcclResult Isend(
        void* buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest& requestHandle,
        u32 userRequire);
    HcclResult
    Imrecv(void* buffer, s32 count, HcclDataType dataType, HcclMessage msgHandle, HcclRequest& requestHandle);
    HcclResult Improbe(u32 peerRank, s32 tag, s32& flag, HcclMessage& msgHandle, HcclStatus& status);
    HcclResult HcclTest(HcclRequest requestHandle, s32& flag, HcclStatus& compState);
    HcclResult GetNicInfo(
        const NICDeployment& nicDeploy, const u32 curRankIndex, const std::vector<RankInfo_t>& servRankList,
        RankInfo& rankInfo) const;
    HcclResult BuildHeterogeneousTransport(u32 commId, u32 peerRank, s32 tag, TransportHandle& transportHandle);
    static std::string GetUniqueId(void);

    u32 GetUserRank();
    u32 GetRankSize();
    std::mutex tagOpExecutorMapMutex_;
    std::mutex transferMemsMapMutex_;
    std::mutex transportHeterogMapMutex_;
    std::string identifier_;
    std::unique_ptr<WorkspaceResource> workSpaceRes_;
    HcomProfilingMode profilingMode_;
    ServRankInfo_t servRankInfo_;
    std::string profilingOption_;
    std::string serverId_;
    std::vector<RankInfo> rankInfoList_; // world group内rank的信息, 按照rank id递增依次排列
    std::vector<HcclIpAddress> devIpAddr_;
    std::vector<u32> ranksPort_;
    std::vector<u32> nicList_;
    std::string collectiveId_;
    bool csCommInitFlag_;
    std::atomic_flag initializedFlag_;
    u32 userRank_;     // 本group中的userrank
    u32 realUserRank_; // world group中的userrank
    u32 userRankSize_;
    u32 devicePhyId_;
    s32 deviceLogicId_;
    bool hcomGroupNicInit_;
    bool heterogRaInit_;
    bool hostRdmaInitFlag_;
    HcclComm commHandle_;
    std::unique_ptr<MrManager> mrManager_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>> pMsgInfosMem_;
    std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>> pReqInfosMem_;
    std::unique_ptr<HeterogMemBlocksManager> memBlocksManager_;
    std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>> pRecvWrInfosMem_;
    TransportResourceInfo transportResourceInfo_;
    bool profilingInitiated_;
    bool mrManagerInit_;
    bool srqInit_;

private:
    SpinMutex transportMapSpinMutex_;
    TransportStorageMap transportStorage_;
};
} // namespace hccl
#endif
