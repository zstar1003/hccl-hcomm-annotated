/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SERVICE_COLLECTIVE_COLL_SERVICE_AI_CPU_IMPL
#define HCCLV2_SERVICE_COLLECTIVE_COLL_SERVICE_AI_CPU_IMPL

#include <unordered_map>
#include "stream.h"
#include "kernel_param_lite.h"
#include "coll_operator.h"
#include "communicator_impl.h"
#include "coll_service_base.h"
#include "hccl_params_pub.h"
#include "virtual_topo.h"
#include "connections_builder.h"
#include "task_param.h"
#include "ins_queue.h"
#include "coll_alg_params.h"

namespace Hccl {

constexpr u32 MAX_ALLTOALLV_MEM_NUM = 64;
constexpr u32 ALLTOALL_INFO_SIZE = 4;
constexpr u32 KERNEL_PARAM_BUF_SIZE = 32 * 1024;

class CollServiceAiCpuImpl : public CollServiceBase {
public:
    explicit CollServiceAiCpuImpl(CommunicatorImpl* comm);

    void Init() override;
    void LoadWithOpBasedMode(CollOperator& op, unique_ptr<Stream> stream) override;
    void LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream) override;
    void RecoverTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair) override;
    HcclResult GetSnapShotDynamicBuf(CollOperator& op, BinaryStream& buf) override;

    void Resume() override;

    HcclResult AllocCollOpResource(CollOperator& op, const std::string& opAlgTag, void** addr) override;

    void ReLoadWithOpBasedMode(CollOperator& op) override;
    void ReLoadWithOffloadMode(CollOperator& op) override;

    HcclResult ClearOpLoadedInfo(const std::string& opTag);

private:
    // 创建rmaConnections
    unordered_map<std::string, unique_ptr<ConnectionsBuilder>> connectionsBuilders;

    DevBuffer* OpBasedCollProcess(CollOperator& op, const std::string& algName);
    void SetOpbaseBufferParam(HcclKernelLaunchParam& param, CommunicatorImpl* comm, CollOperator& op) const;
    void SetOffloadBufferParam(HcclKernelLaunchParam& param, CommunicatorImpl* comm, CollOperator& op) const;
    void SetHcclKernelLaunchParam(HcclKernelLaunchParam& param, CommunicatorImpl* comm, bool isLaunch = true);
    void SetDeviceEnvConfigParam(HcclKernelLaunchParam& param) const;
    void AicpuKernelLaunch(HcclKernelLaunchParam& param, Stream& stream, OpMode opMode);
    void AicpuKernelEntranceLaunch(Stream& stream, const CollOperator& op, const string& algName, const DevBuffer* mem);
    void AicpuUpdateCommLaunch(Stream& stream, const DevBuffer* mem);
    HcclResult AicpuMc2CommResourcePrepare(
        const CollOperator& op, const string& algName, const DevBuffer* mem, const std::string& opAlgTag, void** addr);

    void AllocQueueNotify(std::vector<std::tuple<QId, QId, u32>>& queueNotifyReq) const;
    void AllocBcastPostCntNotify(std::vector<std::pair<QId, u32>>& bcastPostCntNotifyReq) const;
    void AllocWaitGroupCntNotify(std::vector<std::pair<QId, u32>>& waitGroupCntNotifyReq) const;
    void AddSyncPointsToUserStream(const Stream& stream);
    void AddPostToUserStream(const Stream& stream);
    void AddWaitToUserStream(const Stream& stream);
    void AllocWorkStream(u32 primQueueNum) const;
    void AllocNotifies(const vector<LinkData>& links);
    void AllocOpMem(const CollOperator& op);
    u32 GetRemoteRankIdsHashValue(const CollOperator& op) const;
    u64 CalcOpDynamicDataSize(const CollOperator& op, const OpType& opType, const u32& rankSize) const;
    HcclResult FillBatchSendRecvData(const CollOperator& op);
    HcclResult FillAllToAllvData(const CollOperator& op);
    HcclResult FillAllToAllvcData(const CollOperator& op);

    std::set<LinkData> availableLinks;
    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> collOpLoadedMap; // 集合通信算子资源加载到device侧的内存
    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> aicpuMc2CommResourceMap_;

    u32 index{0};
    std::string curTagKey{};
    std::shared_ptr<HostBuffer> kernelParamBuf_;
    u64 dynamicDataSize{0};

    std::vector<char> PackOpData(const std::string& opTag, const CollAlgOpReq& req) const;
    std::vector<char> PackAllTransportData() const;

    shared_ptr<DevBuffer> devBatchSendRecvItemBufs;
    void SaveDfxTaskInfo(const TaskParam& taskParam, const RankId remoteRankId, const bool isMaster) const;

    void LoadWithOpBasedModeNoRegister(CollOperator& op);
    void LoadWithOffloadModeNoRegister(CollOperator& op);
    HcclResult AllocCollOpResourceNoRegister(CollOperator& op, const std::string& opAlgTag, void** addr);

    void AllocQueueNotify(const InsQueue& insQueue) override;
    void AllocQNotifyForSingleQ(const InsQueue& insQueue) const override;
};
void InitAicpuLocBufLite(HcclAicpuLocBufLite& lite, u64 addr, u64 size, const string& desc);
} // namespace Hccl

#endif
