/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMUNICATOR_H
#define HCCL_COMMUNICATOR_H

#include <atomic>
#include <memory>
#include <vector>
#include <hccl/hccl_types.h>
#include "acl/acl_rt.h"
#include "hccl_communicator_attrs.h"
#include "hccl/base.h"
#include "hccl_impl_pub.h"
#include "opexecounter_pub.h"
#include "op_base_stream_manager_pub.h"
#include "offload_stream_manager_pub.h"
#include "prof_common.h"
#include "profiler_manager.h"

#include "topoinfo_parse.h"
#include "hccl_alg.h"
#include "hccl_aiv.h"
#include "ccl_buffer_manager.h"
#include "hccl_trace_info.h"
#include "hccl_callback_task.h"
#include "aicpu_operator_pub.h"
#include "h2d_dto/transport_h2d.h"
#include "transport_pub.h"
#include "mr_manager.h"
#include "transport_heterog_def.h"
#include "resource_manager/queue_notify_manager.h"
#include "hccl_network_pub.h"
#include "comm.h"
#include "device_capacity.h"
#include "transport_manager.h"
#include "zero_copy/zero_copy_memory_agent.h"
#include "coll_alg_operator.h"
#include "alltoall_operator.h"
#include "peterson_lock.h"
#include "coll_alg_utils.h"
#include "heartbeat.h"
#include "../nslbdp/hccl_nslbdp_pub.h"
#include "i_hccl_one_sided_service.h"
#include "opretry_manager.h"
#include "aclgraph/zero_copy_acl_graph.h"
#include "../nslbdp/hccl_nslbdp.h"
#include "hccl/hccl_res.h"
#include "independent_op.h"
#include "comm_config_pub.h"
#include "new/hccl_dispatcher_ctx.h"
#include "rank_graph.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hccl_dpu_manager.h"
#include "../../../unified_platform/resource/buffer/dev_buffer.h"

namespace hccl {
using ServRankInfo_t = std::map<std::string, std::vector<RankInfo_t>>;

constexpr u32 COMM_MAX_WORK_SPACE_SIZE = 16 * 1024 * 1024; // 默认16MB
constexpr u32 INPUT = 0;
constexpr u32 OUTPUT = 1;
const std::string COMM_LOOPBACK_IP = "127.0.0.1";
constexpr u8 INPLACE_PRESYNC_STATUS_SEVEN = 7;
constexpr u32 NSLBDP_HCCP_VERSION = 1;
constexpr u32 NSLBDP_HCCP_NICPOSION = 1;
constexpr u32 AICPU_LOCAL_NOTIFY_SIZE = 8; // aicpu场景本地控制时序的notify数量，对应枚举：enum AicpuLocalNotifyIdx
constexpr u32 AICPU_LOCAL_EVENT_SIZE = 2; // aicpu场景本地控制时序的event数量，对应枚举：enum AicpuLocalEventIdx
constexpr u32 CACHEMAP_MAXSIZE = 65536;
constexpr float CACHEMAP_CLEARPERCENT = 0.1;
constexpr u32 RDMA_NOTIFY_MIN_NUM = 3;
constexpr u32 RDMA_NOTIFY_MAX_NUM = 8192;
constexpr u32 COMM_LAYER_NUM_MAX = 2;

struct RemoteRes {
    u64 inbufferSize;
    u64 outbufferSize;
    u64 inbuffer;
    u32 inbufferKey;
    u64 outbuffer;
    u32 outbufferKey;
};
constexpr u32 HCCL_AICPU_HOST_BASE_TIME_MS
    = 50 * 1000; // 50秒, 停流的超时时间可能为46s，停止npu的超时时间需要比停流时间长
struct AicpuOpTiling {
    std::string newTag;
    std::string algName;
    AlgType algType;
    bool isUsedMainStream = false;
    u8 floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    u8 dumpDebug = false;
};

struct InitTask {
    u64 context;
    bool isCustom;
};

using rankTagSignalInfo_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<HcclSignalInfo>>>;
using rankTagKey_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<u32>>>;
using rankTagAddr_t = std::unordered_map<u32, std::unordered_map<std::string, std::vector<u64>>>;
using rankTagChipId_t = std::unordered_map<u32, std::unordered_map<std::string, s64>>;

class HcclCommunicator {
public:
    explicit HcclCommunicator();
    explicit HcclCommunicator(const CommConfig& commConfig);

    virtual ~HcclCommunicator();

    virtual HcclResult Stop();
    virtual HcclResult Resume();
    HcclResult Suspend();
    HcclResult TraverseAlgResourceResponse(bool isStop);
    HcclResult TraverseOpCommTransport(OpCommTransport& opCommTransport, bool isStop);
    HcclResult TraverseLevelNSubCommTransport(LevelNSubCommTransport& levelNSubCommTransport, bool isStop);
    HcclResult TraverseSingleSubCommTransport(SingleSubCommTransport& commTransport, bool isStop);

    // 对外接口
    virtual HcclResult StopExec();
    virtual HcclResult Clean();
    virtual HcclResult Init(HcclCommParams& params, const RankTable_t& rankTable);
    virtual HcclResult
    Init(HcclCommParams& params, const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData);

    virtual HcclResult GetAlgType(AlgType& algType, HcclCMDType opType);
    virtual HcclResult InitHccpChannel();
    virtual std::vector<RankInfo> GetRankLists();

    virtual HcclResult GetDeviceNumPerAggregation(u32& deviceNumPerAggregation);

    virtual HcclResult GetBandWidthPerNPU(u32 level, float& bandWidth);

    u32 GetRankTableCrc() const;

    u32 GetServerNum() const;

    u32 GetModuleNum() const;

    u32 GetRealUserRank() const;

    HcclResult InitMyRankConnectMode(const HcclCommParams& params, const RankTable_t& rankTable);
    uint32_t GetConnectMode() const;
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);

    HcclResult GetCommParams(HcclCommParams& params); // 逆向解析获取HcclCommParams参数

    HcclResult GetCommRankTable(RankTable_t& rankTable); // 逆向解析获取RankTable_t参数

    virtual HcclResult AllGather(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType,
        HcclRtStream stream, HcomCollOpInfo* opInfo = nullptr);

    virtual HcclResult AllGatherV(
        const std::string& tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts,
        const void* rdispls, HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllGatherOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType,
        HcclRtStream stream);

    virtual HcclResult AllGatherVOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, const void* outputCounts,
        const void* outputDispls, HcclDataType dataType, HcclRtStream stream);

    virtual HcclResult AllReduce(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream, SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE,
        const HcomCollOpInfo* opInfo = nullptr);

    virtual HcclResult AllReduceOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream, SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);

    virtual HcclResult AlltoAllV(
        const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
        const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag);

    virtual HcclResult AlltoAllVOutPlace(
        const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
        const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag);

    virtual HcclResult AlltoAllVC(
        const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);

    virtual HcclResult AlltoAllVCOutPlace(
        const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);

    virtual HcclResult AlltoAll(
        const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);

    virtual HcclResult
    Broadcast(const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult BroadcastOutPlace(
        const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, HcclRtStream stream);

    virtual HcclResult Scatter(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult ScatterOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
        HcclRtStream stream);

    virtual HcclResult Reduce(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        u32 root, HcclRtStream stream);

    virtual HcclResult ReduceOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        u32 root, HcclRtStream stream);

    virtual HcclResult ReduceScatter(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream, HcomCollOpInfo* opInfo = nullptr);

    virtual HcclResult ReduceScatterOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream);

    virtual HcclResult ReduceScatterV(
        const std::string& tag, void* inputPtr, const void* inputCounts, const void* inputDispls, void* outputPtr,
        u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream, HcomCollOpInfo* opInfo = nullptr);

    virtual HcclResult ReduceScatterVOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, const void* inputCounts, const void* inputDispls,
        u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);

    virtual HcclResult
    BatchSendRecv(const std::string& tag, HcclSendRecvItem* sendRecvItemsPtr, u32 itemNum, rtStream_t stream);

    virtual HcclResult Send(
        const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream,
        u32 srTag = 0, u32 localGroupRank = 0);

    virtual HcclResult SendOutPlace(
        const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream);

    virtual HcclResult Receive(
        const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream,
        u32 srTag = 0, u32 localGroupRank = 0);

    virtual HcclResult ReceiveOutPlace(
        const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream);

    virtual HcclResult SupportDeterministicOptim(bool& isDeterministicOptim);

    virtual HcclResult GetCqeError(HcclResult& result);

    virtual HcclResult GetOpInconsistentError(HcclResult& result);

    //  对内接口
    virtual HcclResult CheckDataType(const HcclDataType dataType, bool needReduce);

    virtual HcclResult CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op);

    virtual HcclResult ReleaseCommInfos();

    virtual HcclResult GetAlltoAllStagedWorkSpaceMemSize(
        u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls, HcclDataType recvType,
        u64& memSize);

    virtual HcclResult
    GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize);

    virtual HcclResult GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize) const;

    virtual bool IsStandardCard();

    virtual bool Is310PDuoCard();

    HcclResult HcclSelectAlg(
        HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, int32_t aivCoreLimit,
        bool& ifAiv, std::string& algName);

    HcclResult HcclCalcNumBlocks(
        HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
        u32& numBlocks);

    HcclResult HcclGetAlgExecParam(
        const std::string& tag, HcclCMDType opType, u64 count, void* inputPtr, void* outputPtr, bool clearEnable,
        HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit);

    HcclResult GetAivTag(s32 tagNum, bool isCapture, s32& aivTag);

    HcclResult CheckDeviceType(const DevType deviceType) const;

    HcclResult CheckReductionOp(const HcclReduceOp op) const;

    HcclResult CheckUserRank(const u32 userRank) const;

    HcclResult CheckCount(const u64 count) const;

    HcclResult GetGroupCommonData(WorldGroupInfo& groupCommonData) const;

    HcclResult GetHccsLinkNum(u32& numHccsLink);

    HcclResult GetGroupRanksInfo(const std::vector<u32>& groupRanks, std::vector<RankInfo>& ranksInfo);

    static bool CompareWithUserRank(const RankInfo& left, const RankInfo& right);

    static bool CompareWithServerId(const ServerInfo_t& left, const ServerInfo_t& right);

    static bool CompareWithNicName(const NetworkInfo_t& left, const NetworkInfo_t& right);

    HcclResult GetOneSidedService(IHcclOneSidedService** service);
    HcclResult InitOneSidedServiceNetDevCtx(u32 remoteRankId);
    HcclResult OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx);
    HcclResult GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port);
    HcclResult DeInitOneSidedServiceNetDevCtx();
    HcclResult DeinitOneSidedService();

    u32 GetUserRank() const;
    u32 GetGroupRank() const;
    u32 GetRankSize() const;
    u32 GetRankInParentComm() const; // 获取父通信域中的rank ID (MC2场景使用)
    /* * 以下两函数用于防止重复初始化 */
    HcclResult AtomicInitSet();
    HcclResult HostMC2EnvResume();
    HcclResult ClearWinBuffer();
    HcclResult AivResume();
    void AtomicInitClear();
    bool GetNicInitialized();
    void DestroyAlgResource(AlgResourceResponse& res, bool aclGraphDestroyCbk = false);
    void DestroyOpTransportResponse(OpCommTransport& opTransportResponse, bool aclGraphDestroyCbk = false);
    HcclResult ReleasePreemptSocket();
    HcclResult DestroyNetworkResources();
    HcclResult DisablePreResource();
    HcclResult GetWorkspaceSubStreamNum(
        u64 count, HcclDataType dataType, HcclReduceOp op, const std::string& algName, u64& streamNum, u64 dataSize,
        bool ifAiv, HcclCMDType opType);
    HcclResult GetWorkspaceMemSize(
        const std::string& opType, u64 count, HcclDataType dataType, u32& rankSize, u64& memSize,
        DevType& deviceType) const;
    HcclResult
    SetWorkspaceResource(const std::string& tag, void* memPtr, u64& maxSize, std::vector<rtStream_t>& stream);
    HcclResult CreateOpBasedResources(const HcclCMDType& opType, const std::string& tag, const HcomCollOpInfo& opInfo);
    HcclResult CreateRemoteOpBasedResources(u64 memSize, const std::string& tag);
    HcclResult DestroyRemoteOpBasedMem(const std::string& tag);
    void DestroyWorkspaceResource(const std::string& tag);
    HcclResult GetInCCLbuffer(void*& buffer, u64& size);
    HcclResult GetOutCCLbuffer(void*& buffer, u64& size);
    void ReleaseCommCCLbuffer();
    HcclResult CreateCommCCLbuffer();
    HcclResult InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize);

    // 目前支持按tag对资源释放、解绑定
    HcclResult ClearResMap(const std::string& tag, bool& findTag, bool aclGraphDestroyCbk = false);
    virtual HcclResult ClearOpResource(const std::string& tag, bool aclGraphDestroyCbk = false);
    HcclResult SetClearAivSyncBuf(bool aivClearEnable);

    HcclResult SetGlobalWorkSpace(std::vector<void*>& globalWorkSpaceAddr);
    HcclResult SetAttachedStream(u32 graphId, const std::vector<rtStream_t>& streams);
    // 获得rdma with reduce算子溢出的task信息后清除
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo);

    HcclResult GetDeviceId(s32& deviceId) const;
    virtual void Break();
    HcclResult SetDevicePid(s32 devicePid);
    HcclResult DestroyCDomainResource(s32 tag);

    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> GetPhyIdNicInfo();
    std::vector<u32> GetRanksPort();
    std::vector<RankInfo> GetRanksList();
    HcclResult SetWorldGroupInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
        std::vector<RankInfo> worldRankInfoList, std::vector<u32>& nicRanksPort, std::vector<u32>& vnicRanksPort);
    virtual HcclResult SaveTraceInfo(std::string& logInfo);
    virtual bool GetCommResource(const std::string& tag, void** commContext);
    virtual bool GetCommResource(void*& commContext);

    virtual HcclResult GetAicpuOpStreamNotify(HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify);

    HcclResult GetAlgInfo(
        const std::string& algConfig, const std::string& tag, HcclCMDType commType, std::string& algName,
        std::string& newTag);
    HcclResult GetAlgInfo(const std::string& algConfig, const std::string& tag, std::string& algName);
    HcclResult
    FillOpParam(const HcclCMDType commType, OpParam& opParam, const uint64_t count, void* pCount, void* pDispls) const;
    HcclResult AllocComResource(
        const std::string& newTag, const std::string& algName, const HcclCMDType commType, const OpParam& opParam,
        rtStream_t stream, bool isNeedHostSlaveStream = true);
    HcclResult AllocComResourceByTiling(const std::string& algConfig, void* param);

    virtual HcclResult CreateCommResource(
        const std::string& tag, rtStream_t aiCpuStream, bool isOpbaseMode, void** commContext,
        const std::string& algConfig = "");
    virtual HcclResult AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string& kernelName);
    virtual HcclResult AicpuUnfoldKernelLaunch(
        void* inputPtr, void* outputPtr, const rtStream_t stm, u64 addr, void* tilingDataPtr, u32 tilingDataSize,
        const std::string& kernelName, HcclWorkflowMode mode, const std::string& tag);
    virtual HcclResult AicpuUnfoldKernelLaunchV2(
        void* inputPtr, void* outputPtr, const rtStream_t stm, u64 addr, void* tilingDataPtr, u32 tilingDataSize,
        const std::string& kernelName, HcclWorkflowMode mode, const std::string& tag, bool isCustom);
    HcclResult KernelLaunchChooseAicpuOrCustom(
        void* inputPtr, void* outputPtr, const rtStream_t stm, u64 addr, void* tilingDataPtr, u32 tilingDataSize,
        const std::string& kernelName, HcclWorkflowMode mode, const std::string& tag, bool isCustom);
    HcclResult InitAndCheckAicpuOrderNotify(u8& orderLaunchMode);
    // aclgraph 销毁时批量清理 aicpu 端 tags；走 RunAicpuKfcClearOpRes kernel，超 MAX_BATCH 分批
    HcclResult AicpuKfcClearOpResLaunch(const std::unordered_set<std::string>& tags);
    HcclResult ClearAclgraphHostLinks(const std::unordered_set<std::string>& tags);
    virtual HcclResult Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t& aiCpuStream);
    HcclResult Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t& aiCpuStream);
    HcclResult GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize) const;
    static std::string GetUniqueId(void);

    u8 GetDeterministicConfig() const;                         // 获取确定性计算配置
    HcclResult SetDeterministicConfig(const u8 deterministic); // 设置确定性计算配置
    HcclResult SetAivModeConfig(const bool aivMode);           // 设置aiv模式配置
    HcclResult SetOnlyAivModeConfig(const bool isOnlyAiv);     // 设置aiv only模式配置
    HcclResult SetAicpuUnfoldConfig(const bool aicpuUnfold);   // 设置aicpu配置
    HcclResult SetExecTimeOutConfig(const s32 execTimeOut);    // 设置HCCL执行超时时间
    HcclResult SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap); // 设置HCCL_ALGO
    bool GetAivModeConfig();                               // 获取通信域粒度aiv模式配置
    bool GetConfigIsOnlyAivMode();                         // 获取通信域粒度aiv only模式配置
    bool GetAicpuUnfoldConfig();                           // 获取通信域粒度aicpu配置
    void SetQpQosAttr(u32 trafficClass, u32 serviceLevel); // 设置TC/SL配置
    HcclResult SetMC2EnvFlag();
    bool GetMC2EnvFlag() const;
    bool GetAicpuCommEngine() const;
    HcclResult SetAicpuCommEngine(bool isAicpuCommEngine);
    HcclResult SetStopFlag(bool value);
    HcclResult SetState(HcclCommState state);
    HcclCommState GetState();
    HcclResult ResetNotify();
    HcclResult ResetNotifyForDestRank(s64 destRank);
    HcclResult InitZeroCopyMemoryAgent();
    HcclResult DeinitZeroCopyMemoryAgent(bool inDestructor = false);
    u8 GetConfigAclGraphZeroCopyEnable(); // 从commConfig_里通过函数获取用户配置的aclGraphZeroCopyEnable值
    HcclResult SetMemoryRange(void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);
    HcclResult UnsetMemoryRange(void* baseVirPtr);
    HcclResult ActivateCommMemory(void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags);
    HcclResult DeactivateCommMemory(void* virPtr);
    HcclResult GetNumBlocks(u32& numBlocks)
    {
        numBlocks = numBlocks_;
        return HCCL_SUCCESS;
    }
    HcclResult SetAivCoreLimit(u32 aivCoreLimit);
    HcclResult SwitchNic(uint32_t nRanks, uint32_t* ranks, bool* useBackup);
    HcclResult GetSwitchRanks(
        u32* distSwitchRankList, bool* distSwitchUseBackup, u32& distSwitchRankNum, u8* distRemoteRankNicStatus,
        u32& distNicStatusNum, bool& needCheckDefaultNic, bool& needCheckBackupNic);
    HcclResult SetTransportStatus(
        const HcclOpIdentifier& opId, bool statusStop, const std::map<u32, bool>& remoteRankPortMap,
        const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag);
    static HcclResult GetTransportCqeErrors(const HcclNetDevCtx netDevCtx, std::vector<ErrCqeInfo>& infos, u32& num);
    ErrorMessageReport GetAicpuTaskException();

    // 独立算子专用
    HcclResult IndOpTransportAlloc(
        const std::string& tag, OpCommTransport& opCommTransport, TransportIOMem& transMem, bool isAicpuModeEn);
    aclrtBinHandle GetBinHandle();
    HcclResult
    GetHDCommunicate(HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams);
    HcclResult SetGetAicpuCommState(std::function<bool()> getAicpuCommState);
    CCLBufferManager& GetCCLbufferManager();

    HcclResult RegisterCommUserMem(void* addr, u64 size, void** handle);
    HcclResult DeregisterCommUserMem(void* handle);
    HcclResult ExchangeCommUserMem(void* handle, std::vector<u32>& peerRanks);
    HcclResult GetCommUserMemSize(uint64_t& size);
    HcclResult GetCacheMap(
        std::unique_ptr<CollAlgOperator>& algOperator, OpParam& opParam, AlgType& algType, bool selectAivAlg,
        std::string& newTag);
    HcclResult ExecOpCache(HcclCMDType opType, OpParam& opParam, HcclCacheInfo& cacheInfo);
    void SplitBsrData(
        OpParam& opParam, std::vector<u8>& isDirectRemoteRank, std::vector<HcclSendRecvItem>& hostSendRecvInfo,
        std::vector<HcclSendRecvItem>& aicpuSendRecvInfo);
    HcclResult SetInvalidComm(bool isInvalid);
    HcclResult SnapshotCheckPreProcess();
    HcclResult SnapshotCheckPostProcess();

    // decouple for MC2
    HcclResult GetLocalCCLBuf(void** addr, uint64_t* size);
    HcclResult GetRemoteCCLBuf(uint32_t remoteRank, void** addr, uint64_t* size);
    HcclResult GetKFCWorkSpace(void** addr, uint64_t* size);

    HcclResult CommGetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult CommGetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) const;
    HcclResult CommGetInstTopoTypeByNetLayer(uint32_t netLayer, u32* topoType) const;
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType);
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum);
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);

    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
    HcclResult GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);

    HcclResult GetRankGraph(GraphType type, void** graph, uint32_t* len);
    void* GetRankGraphV1(); // 获取V1 RankGraph指针，用于CollComm初始化

    HcclResult GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
    HcclResult GetHeterogMode(HcclHeterogMode* mode);
    HcclTopoAttr GetTopoAttr();
    void ForceProf(bool isForce);
    // for Group
    HcclResult SetGroupMode(bool isGroup);
    bool GetGroupMode() const;

    void SetReleaseChannel(std::function<HcclResult()> releaseChannel);

    void SetHcclQos(u32 hcclQos);
    u32 GetHcclQos() const;
    HcclResult RegisterWindow(void* ptr, size_t size, HcclCommSymWindow* winHandle);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcclResult InitSymmetricMemory();
    HcclResult GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset);
    void EraseCaptureModelId(u64 modelId);

private:
    bool IsEnableRoce();
    bool IsEnableBackupLink();
    HcclResult CheckOneSidedBackupAndSetDevId(
        u32& backupDevPhyId, u32& backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
        bool& isOneSidedTaskAndBackupInitA3);
    HcclResult OneSidedBackupInitNetResource(
        HcclNetDevCtx& nicPortBackUpCtx, u32& backupDevPhyId, u32& backupDevLogicId,
        std::vector<HcclIpAddress>& localIpList);
    HcclResult OneSidedBackupServerInit(HcclNetDevCtx& nicPortBackUpCtx);
    void SetAttrs();
    u32 HcclGetCmdTimeout() const;
    HcclResult InitCommParams(HcclCommParams& params);
    HcclResult InitRankInfo(const RankTable_t& rankTable);
    HcclResult InitRankInfoSubGroup(WorldGroupInfo& groupCommonData);
    HcclResult CheckSingleServerComm(const std::vector<RankInfo_t>& rankList) const;
    HcclResult SetRanksPort(const std::vector<RankInfo_t>& rankList);
    HcclResult InitNetResource(const RankTable_t& rankTable);
    HcclResult InitDebug();
    HcclResult InitDebugSubGroup();
    HcclResult InitATraceInfo();
    HcclResult InitNotifyManager();
    HcclResult InitDispatcher();
    HcclResult InitStreamManager();
    HcclResult InitSocketManager();
    HcclResult InitTransportManager();
    HcclResult InitHcclAlg();
    HcclResult InitAlgResource();
    HcclResult InitProfiling();
    HcclResult DeinitProfiling();
    HcclResult InitProfiler();
    HcclResult InitOneSidedService(const RankTable_t& rankTable);

    HcclResult RegistTaskExceptionHandler() const;
    HcclResult UnRegistTaskExceptionHandler() const;
    HcclResult UnRegisterBackGroundThread();
    HcclResult
    UnRegisterBackGroundThread(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult DestroyAicpuComm();
    HcclResult DestroyAicpuComm(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult InitPreResource(const RankTable_t& rankTable);
    HcclResult InitTcpMode(const RankTable_t& rankTable) const;
    HcclResult InitRaResource();
    HcclResult InitRaNetResource();
    HcclResult InitRaNic();
    HcclResult InitDevicePrimaryNic(bool isMC2ReInit, bool isOneSidedTaskAndBackupInitA3);
    HcclResult InitDeviceBackupNic(
        u32 backupDevPhyId, u32 backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
        bool isOneSidedTaskAndBackupInitA3);
    HcclResult InitNicDeviceDeploy(
        bool isMC2ReInit, u32 backupDevPhyId, u32 backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
        bool isOneSidedTaskAndBackupInitA3);
    HcclResult InitNicHostDeploy();
    bool IsNeedNicInit();
    HcclResult InitNic(bool isMC2ReInit = false);
    HcclResult DeinitNic();
    HcclResult DeinitNicHostDeploy();
    HcclResult AddOpInfoToHeartBeat(const OpInfoDesc& opInfo, const std::string& tag);
    void DeleteOpInfoToHeartBeat();
    HcclResult RegisterToHeartBeat();
    HcclResult RegisterToHeartBeat(u32 peerRankId, std::string& tag);
    void UnRegisterToHeartBeat();
    void UnRegisterToCommConfiger();
    HcclResult PrintOpbaseKeyTraceInfo(void);
    HcclResult InitPara();
    HcclResult GetComm(const std::string& tag, CommBase** comm);
    HcclResult Mc2CreateAndLaunchContext(
        rtStream_t aiCpuStream, bool isOpbaseMode, void** commContext, const std::string& tag = "");
    HcclResult SetCommResource(
        u64 commBufferSize, void* commInPtr, void* commOutPtr, void* commExpPtr, CommBase* comm,
        level1StreamInfo_t& streamInfo, Stream& stream);
    HcclResult GetAicpuOpStreamAndNotify(HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify);
    HcclResult SetAicpuNotifyInvalid();
    HcclResult AicpuKfcTilingDataLaunch(
        const OpParam& opParam, const HcclCMDType& opType, const DeviceMem& deviceContext,
        const std::string& kernelName, const AicpuOpTiling opTilingInfo);
    HcclResult AicpuKfcTilingDataLaunchExt(
        const OpParam& opParam, const HcclCMDType& opType, const DeviceMem& deviceContext,
        const std::string& kernelName, const AicpuOpTiling opTilingInfo, bool isCustom = false);
    u64 CalcOpTilingDynamicDataSize(
        const OpParam& opParam, const HcclCMDType& opType, const u32& rankSize, const std::string& algName = "");
    u64 CalcOpTilingVDataDesVDataLen(const u32 rankSize) const;
    HcclResult AicpuInitOpTilingDataFromOpParam(
        const OpParam& opParam, const HcclCMDType& opType, struct OpTilingData* opTilingData);
    HcclResult AicpuInitOpTilingDataAicpuCache(
        const OpParam& opParam, const HcclCMDType& opType, struct OpTilingData* opTilingData);
    HcclResult AicpuInitOpTilingDataBuf(
        const OpParam& opParam, const HcclCMDType& opType, const std::string& kernelName,
        const AicpuOpTiling opTilingInfo, u64 dynamicDataSize);
    HcclResult AicpuKfcTilingDataLaunchIn(
        const OpParam& opParam, const DeviceMem& deviceContext, const std::string& kernelName,
        const AicpuOpTiling opTilingInfo, u64 opTilingDataSize, bool isCustom = false);
    HcclResult AllReduceAicpuUnfold(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream);
    HcclResult CreateMutiStreamResFor310P(const std::string& tag, level1StreamInfo_t& streamInfo);
    HcclResult SetDynamicTilingDataAlltoall(const OpParam& opParam, HostMem& dynamicDataMem);
    HcclResult UnRegisterDfxInfo(const OpParam& param, const std::vector<Stream>& slaveStreams);
    HcclResult RegisterDfxInfo(
        const OpParam& param, AlgType algType, const std::vector<Stream>& slaveStreams, bool isAiv = false,
        const std::string& tag = "");
    HcclResult AddGroupTagInfo(const std::string& tag, bool isAiv);
    HcclResult
    SetDynamicTilingDataAlltoallv(const OpParam& opParam, HostMem& dynamicDataMem, const std::string& algName = "");
    HcclResult SetDynamicTilingDataAlltoallvc(const OpParam& opParam, HostMem& dynamicDataMem);
    HcclResult SetDynamicTilingDataV(const OpParam& opParam, HostMem& dynamicDataMem);
    HcclResult GetReportHcclMC2Info(const Stream& kfcStream, const std::vector<Stream>& aicpuStreams);
    u8 GetOrderLaunchMode(bool isCapture);

    HcclResult ReAllocTransports(const std::string& tag, const std::string& newTag);
    HcclResult SetTransportStatusImpl(
        OpCommTransport& opCommTransport, bool statusStop, const HcclOpIdentifier& opId, u32 remoteRank,
        const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault);
    HcclResult SetBsrTransportStatusImpl(
        OpCommTransport& opCommTransport, bool statusStop, const HcclOpIdentifier& opId, u32 remoteRank);
    HcclResult SetTransportStatusImplForChange(
        OpCommTransport& opCommTransport, const HcclOpIdentifier& opId, u32 remoteRank,
        const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault, const std::map<u32, bool>& isChangeLinkMap,
        bool isCurTag);
    HcclResult SetTransportResumeStatus(
        const std::map<u32, bool>& remoteRankPortMap, const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag,
        bool statusStop);
    HcclResult ResumeTransportsImplForChange(
        OpCommTransport& opCommTransport, const std::map<u32, bool>& remoteRankPortMap,
        const std::map<u32, bool>& isChangeLinkMap, bool isUseDefault);
    HcclResult ResumeTransportsImpl(
        OpCommTransport& opCommTransport, const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault,
        bool statusStop);
    HcclResult SetBsrTransportStatusImplforchange(
        OpCommTransport& opCommTransport, const HcclOpIdentifier& opId, u32 remoteRank,
        const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault, const std::map<u32, bool>& isChangeLinkMap,
        bool isCurTag);
    void ClearOpTransportResponseLinks(OpCommTransport& opTransportResponse);
    HcclResult SetSignalTransport(SingleSubCommTransport& singleSubCommTransport, u32 linkIdx, bool statusStop);
    void InsertNewTagToTagMap(std::string& newTag, std::string& tag);
    HcclResult GetTagFromNewTag(const std::string& newTag, std::string& tag);
    HcclResult
    ParseSwitchRanks(uint32_t nRanks, uint32_t* ranks, bool* useBackup, std::unordered_map<u32, bool>& switchRanks);
    HcclResult PrepareLinkForSwitchNic(std::unordered_map<u32, bool>& switchRanks, ChangeLinkInfo& changeLinkInfo);
    HcclResult SetRemoteRankLinkInfo(std::unordered_map<u32, bool>& switchRanks, ChangeLinkInfo& changeLinkInfo);
    HcclResult
    SetSingleLinkInfo(std::unordered_map<u32, bool>& switchRanks, u32 remoteRankId, ChangeLinkInfo& changeLinkInfo);
    HcclResult
    ActiveStoppedLink(std::map<u32, bool>& remoteRankPortMap, OpCommTransport& opTransportResponse, bool isBackup);
    HcclResult setVnicIpToRankInfoList();
    HcclResult GetRemoteUserMemResource();

    HcclResult Suspend(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult StopExec(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult Clean(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult SwitchNic(
        uint32_t nRanks, uint32_t* ranks, bool* useBackup, std::shared_ptr<HDCommunicate>& controlH2D,
        std::shared_ptr<HDCommunicate>& statusD2H);
    HcclResult SaveRankInfoHasLinked(const AlgResourceRequest& resRequest);
    HcclResult RecordOpPara(HcclCMDType opType, const OpParam& opParam);
    HcclResult SaveTopoDesc(std::string& identifier);

    HcclResult SetAicpuUnfoldFlag();
    bool GetAicpuUnfoldFlag() const;

    HcclResult ReAllocScratchMemForAlltoall(
        HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest,
        AlgResourceResponse& algResResponse);

    HcclResult HandleExistAlgResource(
        const std::string& newTag, const std::string& algName, HcclCMDType opType, const OpParam& opParam,
        std::unique_ptr<CollAlgOperator>& algOperator, bool selectAivAlg, bool aicpuUnfoldModeFor910B,
        bool needRecreateAlltoallComm);
    u32 deviceNumPerServer_{0};
    HcclDispatcher dispatcher_; // dispatcher放到最后析构
    DispatcherCtxPtr dispatcherCtx_{nullptr};
    HcclDispatcher vDispatcher_; // virtualDispatcher放到最后析构
    std::unique_ptr<NotifyPool> notifyPool_;
    std::unique_ptr<HcclCallbackTask> callbackTask_;
    std::atomic_flag initializedFlag_;
    u32 userRank_;     // 本group中的userrank
    u32 realUserRank_; // world group中的userrank
    u32 userRankSize_;
    std::vector<RankInfo> rankInfoList_;            // world group内rank的信息, 按照rank id递增依次排列
    std::vector<RankInfo> rankInfoListIntraServer_; // 节点内rank信息，用于零拷贝
    bool drvInit_;                                  // ra是否初始化
    ServRankInfo_t servRankInfo_;
    std::string serverId_;
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo_; // server内所有device间的链路类型
    bool inlineReduceSwitchOn_;
    NICDeployment nicDeployment_;
    u32 devicePhyId_;
    u32 deviceBackUpPhyId_{0};
    s32 deviceLogicId_;
    u32 deviceBackUpLogicId_{0};
    std::unordered_set<s32> aicpuStreamIds_;
    std::vector<HcclIpAddress> devIpAddr_;
    std::vector<HcclIpAddress> devBackupIpAddr_;
    u32 devBackupPort_{HCCL_INVALID_PORT};
    HcclIpAddress hostIp_;
    HcclIpAddress deviceVnicIp_;
    u32 hostPort_{HCCL_INVALID_PORT};
    u32 localRank_;
    SocketHandle hostSocketHandle_;
    SocketHandle loopbackHeterogSocketHandle_{nullptr};
    bool isUsedRdmaLevel0_; // 节点内是否使用rdma, 包括a+x和标卡
    std::atomic<s32> nicInitialized_;
    bool hcomGroupNicInit_;
    // profiling 相关资源
    HcomProfilingMode profilingMode_;
    std::string profilingOption_;
    ProfilingDeviceCommResInfo hcclMc2Info_{};
    bool raResourceInit_;
    bool interServer_;
    std::unique_ptr<WorkspaceResource> workSpaceRes_;
    std::vector<u32> enableP2PDevices_;
    std::unordered_set<u32> enableP2PRankIds_;
    bool isSingleMeshAggregation_;
    CCLBufferManager cclBufferManager_;
    bool isExecuteProfilingInit_;
    DevType deviceType_;
    std::string collectiveId_;
    HcclComm commHandle_;
    std::vector<u32> nicRanksPort_;
    std::vector<u32> groupNicRanksPort_;
    std::vector<u32> vnicRanksPort_;
    std::vector<u32> groupVnicRanksPort_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::unordered_map<u32, HcclRtContext> rtCtxMap_; // {devPhyId, rtCtx}
    WorkMode commWorkMode_;
    u32 meshAggregationRankSize_;
    std::map<HcomOperationType, std::string> opTypeTagMap_;
    bool isHaveCpuRank_;
    bool isUseRankPort_{true};
    bool isSetHDCModeInfo_{false};
    std::map<std::string, HostMem> tagWorkSpaceMem_;
    std::string identifier_;
    u32 ranktableCrc_;
    s32 devicePid_{0};
    bool multiModuleDiffDeviceNumMode_;
    bool multiSuperPodDiffServerNumMode_;
    bool multiSuperPodDiffDeviceNumMode_{false};
    DeviceMem commContext_;
    std::shared_ptr<ProfilerManager> profilerManager_;
    bool isStandardCard_ = false;
    bool is310PDuoCard_ = false;
    bool isCommon310P3DUO_ = false;
    s32 hccsPortNum_ = -1;
    std::string superPodId_;
    u32 superDeviceId_ = INVALID_UINT;
    bool useSuperPodMode_ = false;
    bool isUsedInterHccsMode_ = false;
    bool isNeedInitNic_ = false;
    std::vector<RankInfo> worldRankInfoList_;
    std::unique_ptr<HcclTraceInfo> opBaseAtraceInfo_;
    bool aivClearEnable_ = false;
    u32 numBlocks_ = 0;
    std::map<OpParam, HcclCacheInfo> hcclCacheMap_; // 存储aiv cache信息
    std::string cclBuffName_;
    bool isShareComm_ = false; // 是否共享cclbuffer
private:
    bool IsAtomicInit();
    HcclResult MigrateLinkToStopOrResume(LINK& link, bool isStop);
    HcclResult MigrateLinkVectorToStopOrResume(const std::vector<LINK>& links, bool isStop);
    HcclResult TraverseLinkVector(std::vector<std::unique_ptr<CommBase>>& commBaseVector, bool isStop);
    HcclResult CheckSuspendingStatus();
    HcclResult InitCombinOpara();
    HcclResult RegisterRanksToDca();
    HcclResult InitWorkSpace();
    void ReleaseWorkSpacebuffer();
    HcclResult CreateWorkSpace(u64 size, DeviceMem& buffer) const;
    HcclResult GetWorkSpace(u64* workSpaceSize, u64* workSpace) const;
    void ReleaseCommContextbuffer();
    HcclResult CreateDeviceCommContext(u64 size, DeviceMem& buffer) const;
    HcclResult CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo);
    HcclResult GetAiCpuNotifyData(const std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo) const;
    HcclResult ReplaceCommInfoByTag(const std::string& tag, std::unique_ptr<CommInfo>& commInfo);
    HcclResult CreateCommAndStreamRes(const std::string& tag, Stream& stream);
    HcclResult SetInfoToDevice(const std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, Stream& stream);
    HcclResult GetInfoFromDevice(
        const std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, const HcclWorkflowMode& mode,
        HostMem& hostCollectBuffer);
    HcclResult RegressCalPreOp(
        AlltoAllOperator*& alltoAllOperator, const OpParam& opParam, std::unique_ptr<PreProcessMetaInfo>& preMetaInfo);
    HcclResult RegressCalPreOp(
        AlltoAllOperator*& alltoAllOperator, const OpParam& opParam, std::unique_ptr<PreProcessMetaInfo>& preMetaInfo,
        Stream& preProcessStream);
    DevType NslbGetDeviceType() const;
    u32 NslbGetServerNum() const;
    HcclResult NslbDp_CollectOperTable(HcclCMDType opType, OpParam& opParam, AlgType nslbAlgType, std::string& algName);
    HcclResult
    NslbDp_CollectSendAdjTable(HcclCMDType opType, const OpParam& opParam, AlgType nslbAlgType, AdjInfo& nslbAdjInfo);
    HcclResult ExecOp(HcclCMDType opType, OpParam& opParam, bool isCustom = false);
    // alltoall专用
    HcclResult ExecOpAlltoAll(HcclCMDType opType, OpParam& opParam, bool isCustom = false);
    HcclResult FreeScratchMemOnOpBaseMode(DeviceMem& scratchMem, const OpParam& opParam, const HcclCMDType& opType);
    HcclResult
    CalcTinySendRecvMem(const OpParam& opParam, AlgResourceResponse& algResResponse, DeviceMem& tinySendRecvMem) const;
    bool IsForceAicpuOpBaseMode(const OpParam& opParam, const HcclCMDType& opType);
    HcclResult AllocOpBaseModeScratchMem(
        HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest,
        AlgResourceResponse& algResResponse);
    HcclResult AllocAlgResource(
        const std::string& newTag, HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest,
        AlgResourceResponse& algResResponse, bool selectAivAlg = false);
    HcclResult IncreAllocLink(
        const std::string& newTag, const OpParam& opParam, AlgResourceRequest& resRequest,
        AlgResourceResponse& algResResponse);
    bool HasRoceTransportLinks(OpCommTransport& opTransportReq) const;
    HcclResult CleanTransportLinks(OpCommTransport& opTransportReq, OpCommTransport& opTransportResponse);
    DeviceMem GetWorkspaceScracthMem(const std::string& tag, u64 allocMemSize);
    std::vector<Stream> GetWorkspaceSubStreams(const std::string& tag, u32 num);
    // HcclImplBase中Comm资源是否存在
    inline bool IsExistCommRes(const std::string& tag)
    {
        std::unique_lock<std::mutex> commLock(commLock_);
        return (tagCommInfo_.find(tag) != tagCommInfo_.end());
    }
    // HcclImplBase中MutiStream资源是否存在
    inline bool IsExistMutiStreamRes(const std::string& tag)
    {
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        return (tagStreamInfo_.find(tag) != tagStreamInfo_.end());
    }
    void GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode);
    void RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode);
    HcclResult AicpuUnfold(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        HcclRtStream stream, HcclCMDType cmdType);
    u32 GetHostPort(s32 devicePhyId);
    u32 GetLocalNicPort(NicType nicType);
    std::string GetSupportDataType(bool needReduce);
    HcclResult InitHDCommunicate();
    bool GetSupportHDCommunicate();
    HcclResult InitOpRetry();
    HcclResult InitOpResPara();
    bool IsSupportSymmetricMemory(HcclCMDType opType, OpParam& opParam);
    bool IsSupportZeroCopy(const OpParam& opParam) const;
    HcclResult PrepareZeroCopy(const std::string& algName, const AlgDesc& algDesc, OpParam& opParam);
    HcclResult UpdateZeroCopy(const OpParam& opParam, const AlgResourceResponse& algResource);
    HcclResult BuildZeroCopyParam();
    HcclResult AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem>& bufferPtr) const;
    HcclResult AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem>& bufferPtr) const;
    HcclResult updateList() const;
    HcclResult BuildOpLocalResParam(const AlgResourceResponse& algResource, const std::string& newTag);
    HcclResult BuildOpLocalScratchMemResParam(
        const AlgResourceResponse& algResource, const std::string& newTag, LocalResInfoV2* localResHostPtr);
    HcclResult BuildOpTopoResTlvParam(
        const std::string& algName, const std::vector<std::vector<std::vector<u32>>>& inputVectorInfo,
        DeviceMem& dstTlvDeviceMem, u64& tlvLen);
    HcclResult BuildOpTopoResVectorTlvParam(
        const std::string& algName, const std::vector<std::vector<std::vector<std::vector<u32>>>>& inputVectorInfo,
        DeviceMem& dstTlvDeviceMem, u64& tlvLen);
    HcclResult BuildPairLinkCounter(const std::string& algName);
    HcclResult BuildIsUsedRdmaRank(const std::string& algName);
    HcclResult BuildNicList(const std::string& algName);
    HcclResult BuildBridgeRank(const std::string& algName);
    HcclResult BuildCommPlanRank(const std::string& algName);
    HcclResult BuildServerAndsuperPodRank(const std::string& algName);
    HcclResult BuildCommPlaneSubGroupRank(const std::string& algName);
    HcclResult BuildHierarchicalAlgOption(u32* ahcConfInfo);
    HcclResult BuildOpTopoResParam(const std::string& algName, const AlgResourceResponse& algResource);
    HcclResult BuildAicpuCustomParam();
    HcclResult BuildAicpuOrderLaunchNotify();
    HcclResult BuildOpRemoteResParam(
        const AlgResourceResponse& algResource, const std::string& newTag, const HcclCMDType opType,
        bool isRetry = false);
    HcclResult BuildOpResParam(
        const std::string& algName, const AlgResourceResponse& algResource, const std::string& newTag,
        const HcclCMDType opType, const rtStream_t aicpuStream);
    HcclResult BuildCustomOpResParam();
    HcclResult BuildOpRetryParam(const AlgResourceResponse& algResource, const std::string& newTag);
    HcclResult CopyHostListResToDeviceParam(const std::string& newTag, const ListCommon* headHostList, const u64 size);
    HcclResult
    CopyHostAirmaInfoToDeviceParam(const std::string& newTag, const HcclCMDType opType, const rtStream_t aiCpuStream);
    HcclResult CopyHostOpRemoteResToDeviceParam(const std::string& newTag);
    HcclResult CopyHostOpResToDeviceParam(const std::string& newTag);
    HcclResult AicpuResourceInit(
        const std::string& algName, const AlgResourceResponse& algResource, const std::string& newTag,
        const rtStream_t& aicpuStream, const HcclCMDType opType, bool isCustom = false);
    HcclResult
    AicpuResourceRefresh(const AlgResourceResponse& algResource, const std::string& newTag, const HcclCMDType opType);
    HcclResult OrchestrateAicpu(
        const HcclCMDType& opType, const std::string& algName, const OpParam& param,
        const AlgResourceResponse& algResource, const std::string& newTag, AlgType algType, bool isCustom = false,
        bool needIncreLink = false, bool needRecreateAlltoallComm = false);
    template <typename T>
    HcclResult CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec);
    template <typename T>
    HcclResult CreateListNode(T** resHostPtr, T** resDevicePtr);
    HcclResult ParseRemoteDataToMem(
        const OpCommTransport& opTransportResponse, const std::string& newTag, const HcclCMDType opType,
        bool isBackup = false, bool isRetry = false);
    HcclResult BuildRelationResByRemoteRankId(
        const TransportRequest& transportRequest, const LINK& link, HcclRankRelationResV2*& rankRelationResHostPtr,
        HcclRankRelationResV2*& rankRelationResDevicePtr);
    HcclResult BuildRemoteResByTag(
        const std::string& newTag, const u32& usrRankId, HcclRankRelationResV2*& rankRelationResHostPtr,
        HcclRankRelationResV2*& rankRelationResDevicePtr, bool isBackup, bool isRetry);
    HcclResult BuildOpRemoteLinkP2pResParam(
        const LINK& link, HccltagRemoteResV3& tagRemoteRes, TransportLinkType linkType = TransportLinkType::RESERVED);
    HcclResult BuildOpRemoteLinkRoceResParam(
        const LINK& link, HccltagRemoteResV3& tagRemoteRes, bool isBackup, bool isRetry, bool isSecondBuild);
    HcclResult BuildAiRmaInfoParam(const std::string& newTag, const std::string& algName, const HcclCMDType opType);
    HcclResult CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes);
    HcclResult AllocAlgNotifys(
        const std::string& tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
        std::vector<std::shared_ptr<LocalNotify>>& notifiesMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifiesAux);
    HcclResult CreateAndGetAiCpuNotifyWithNotifyRes(HcclSignalInfo& notifyInfo);
    void SaveLinkRes(const OpCommTransport& opTransportResponse);
    HcclResult
    SetDevIbverbsData(CommBase* comm, bool isSupportNormalQP, u64 commBufferSize, void* commInPtr, void* commOutPtr);

    // 获取 Transport 本端内存信息
    HcclResult
    GetTransportLocalMem(const std::shared_ptr<Transport>& transport, UserMemType memType, MemDetails& detail) const;
    // 获取 Transport 远端内存信息
    HcclResult
    GetTransportRemoteMem(const std::shared_ptr<Transport>& transport, UserMemType memType, MemDetails& detail) const;

    // 收集全部 Transport 内存/QP信息
    HcclResult GenAiRMAInfo(CommBase* comm);
    HcclResult GenAiRMAInfoV2(const std::string& tag);
    // 同步全部信息到Device
    HcclResult H2DAiRMAInfo(const std::string& tag, rtStream_t aiCpuStream);
    HcclResult H2DAiRMAInfoV2(const std::string& tag, rtStream_t aiCpuStream);
    HcclResult GetAIVNormalQPInfo(CommBase* comm, const std::string& tag);
    HcclResult GetAIVNormalQPInfoV2(std::vector<LINK>& links, const std::string& tag);
    template <typename T>
    HcclResult
    GenIbvAiRMAInfo(u32 rankid, const std::shared_ptr<Transport>& transport, const std::string& tag, T* aiRMAInfoPtr);
    HcclResult GetAivQPInfoV2(std::vector<LINK>& links, const std::string& tag);
    HcclResult CaptureSlaveStreams(rtStream_t mainStream, std::vector<Stream>& slaveStreams);
    HcclResult HandleAclGraphFirstOpAivBuff(rtStream_t mainStream);
    bool StreamIsCapture(rtStream_t mainStream) const;
    HcclResult AllocAndGetStreamContextBuff(u32 streamId, u64& addr, u64& size);
    u32 UpdateOpIndex(const OpParam& opParam); // 更新opIndex
    HcclResult LoadCustomFile(
        const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle);
    void UnloadBinary(aclrtBinHandle& binHandle);
    bool IsEnableCustom() const;
    void UnloadCustomKernel(void);
    HcclResult LoadCustomKernel(void);
    HcclResult LoadAICPUKernel(void);
    void UnloadAICPUKernel(void);
    u32 LargestPowerOfTwoLessThan(const u32 localRankSize);
    u32 CalcStreamNumForReduceOrderPreservation();

    HcclResult CheckSetRetryStateToWaitResume();
    HcclResult CheckExitWaitResumeState(bool& isChangedLink);

    HcclResult RegisterToSnapshot();
    HcclResult UnRegisterFromSnapshot();

    bool EnableAicpuUnfold(bool isCapture = false);

    // reduce类算子的prod操作或者int64数据类型不支持重执行
    bool IsReduceWithInt64OrProd(HcclCMDType opType, const OpParam& opParam) const;
    // 控制当前通信域首次检测到reduce类算子的prod操作或者int64数据类型时打印不能重执行的约束
    bool needWarnAboutReduceProdInt64_{true};

    bool isOnlyAiv_{false};
    HcclIpAddress loopBackIp_;
    bool profilingInitiated_;
    u64 callbackThreadId_;
    u32 role_;
    std::map<u64, std::vector<rtStream_t>> callbackStreamMap_;
    bool isHostUseDevNic_;
    std::mutex socketListenMutex_;

    std::unique_ptr<HcclAlg> implAlg_ = nullptr;
    HcclCommunicatorAttrs attrCollector_;

    u32 deviceNumPerAggregation_{0};
    std::vector<u32> nicList_;
    std::unordered_map<u32, u32> pairLinkCounter_; // server内所有device间的链路类型计数
    bool isAllRankSamePlane_;
    std::unique_ptr<TopoInfoParse> topoInfoParse_; // 对rank table device选取的校验模块
    u32 serverNum_;
    u32 moduleNum_;
    u32 superPodNum_ = 0;
    bool isAlgoLevel1Default_ = false;
    std::shared_ptr<HostMem> combinOparaMem_ = nullptr;
    Stream opStream_;
    Stream aicpuInitStream_;
    std::vector<Stream> attachedStreams_;
    std::vector<std::shared_ptr<LocalNotify>> localAiCpuNotifyRes_;
    std::shared_ptr<LocalNotify> localAiCpuOpNotify_[AICPU_LOCAL_NOTIFY_SIZE] = {nullptr};
    HcclRtEvent localAicpuOpEvent_[AICPU_LOCAL_EVENT_SIZE] = {nullptr}; // 用于控制Aclgraph模式按序下发控制流入图的event
    u32 workSpaceSize_{0};
    DeviceMem workSpace_;
    DeviceMem mc2DeviceMem_;
    std::vector<DeviceMem> extraMem_;
#ifndef CCL_KERNEL_AICPU
    // aclgraph 销毁时投递 HcclKfcClearOpResTilingData 的 HBM buffer，lazy alloc，RAII 析构
    DeviceMem aicpuCleanupBuf_;
    // host 侧 payload buffer（heap 持有），避免 ~2.5MB 结构体在栈上 value-init 爆栈
    std::unique_ptr<HcclKfcClearOpResTilingData> aicpuCleanupHostBuf_;
#endif
    std::vector<HcclRtEvent> aiCpuNoIpcEvnet_;
    bool isDiffDeviceModule_{false};
    bool isDiffDeviceType_{false};
    bool isARSDoubleRing_{false};
    u32 gcdDeviceNumPerAggregation_{0};
    tagCommInfo_t tagCommInfo_; // 以tag为粒度分配comm实例和资源
    std::mutex commLock_;
    tagStreamInfo_t tagStreamInfo_;
    std::mutex tagStreamInfoLock_;

    std::vector<Stream> auxRingCommStreamsDev_;
    bool isServerInter_{false};
    bool isSupportRdmaLite_{false};   // 是否支持RDMA Lite
    bool isSupportHccsAndSio_{false}; // 是否支持hccs sio并发

    HcclIpAddress localVnicIp_;
    u32 localVnicListenPort_{0};
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;

    std::unique_ptr<OpBaseStreamManager> opStreamManager_ = {nullptr};
    std::unique_ptr<QueueNotifyManager> queueNotifyManager_ = {nullptr};
    std::unique_ptr<QueueNotifyManager> queueNotifyManagerRefac_ = {nullptr};
    std::unique_ptr<HcclSocketManager> socketManager_;
    std::unique_ptr<TransportManager> transportManager_ = {nullptr};
    std::unique_ptr<TransportManager> indptOpTransportManager_ = {nullptr};

    std::unique_ptr<ZeroCopyMemoryAgent> zeroCopyMemoryAgent_ = {nullptr};

    std::unordered_map<std::string, AlgResourceResponse> resMap_; // tag : AlgResourceResponse
    std::unordered_set<std::string> hostResMap_;
    std::unordered_set<std::string> hbSendRecvTags_;
    std::vector<DeviceMem> deviceResOrigMem_;
    bool isSuspending = false;
    bool retryEnable_ = false;
    bool rtsSupportChangeLink_
        = true; // RTS是否支持借轨（部分ASCEND_RT_VISIBLE_DEVICES自定义场景不支持访问同chip内的另一个die）
    bool inplaceSupportRetry_ = false; // inplace是否支持重执行
    u8 isInplaceStatus_ = 0;           // 算子是不是inplace的状态
    // 算子在inplace时，是否支持重执行的状态
    InplaceSupportRetryStatus inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    bool isInplacePreSync_ = false;
    bool isPostSync_ = false;
    HcclWorkflowMode retryOrigWorkflowMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    HcclOpResParam opResPara_{};
    DeviceMem opResDevicePara_;
    HcclOpResParam* opResDeviceParaPtr_{nullptr};
    Stream opMainStream_;
    Stream aicpuOrderStream_;
    bool isContextLaunched_{false};
    std::vector<std::shared_ptr<DeviceMem>> deviceMemVec_;
    std::vector<std::shared_ptr<HostMem>> hostMemVec_;
    DeviceMem nicListDevice_;
    DeviceMem complanRankDevice_;
    DeviceMem pairLinkCounterDevice_;
    DeviceMem isUsedRdmaRankPairDevice_;
    std::unordered_set<std::string> newTagResAlloced_;
    DeviceMem bridgeRankDevice_;
    DeviceMem serverAndsuperPodToRankDevice_;
    DeviceMem commplaneSubGroupRankDevice_;
    DeviceMem hierarchicalAlgOptionDevice_;

    // aicpu-custom共享内存区
    std::unordered_map<u32, DeviceMem> streamIdToStreamContext_;
    DeviceMem aicpuCustomDev_; // aicpu-custom共享内存区，对应AicpuCustomParam结构体

    std::unordered_map<s32, u32>
        opIndexMap_; // 记录aicpu/custom的算子计数, key值用来区分, bsr: -1, sendrecv: 对端rank, 其他算子: 本端rank

    std::unique_ptr<IHcclOneSidedService> oneSideService_ = {nullptr};
    HcclIpAddress onesidedServiceNicIpAddr_;
    HcclRankLinkInfo hcclRankLinkInfo_{};
    std::atomic<bool> isOneSidedServiceNetDevCtxInited{false};
    std::atomic<bool> isOneSidedServiceNicInited{false};
    std::atomic<bool> isOneSidedServiceNicStartListen_{false};

    std::unique_ptr<OpRetryManager> opRetryManager_ = {nullptr};
    std::shared_ptr<HcclOpStreamRes> opRetryStreamPtr_;
    std::unordered_set<u64> captureModelIds_;

    std::unordered_map<u32, std::unordered_map<std::string, HccltagRemoteResV3>>
        rankTagRemoteRes_; // 以rankid&tag粒度保存HccltagRemoteResV3
    // aicpu进程使用的host-device共享内存
    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_;
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_;
    // custom进程使用的host-device共享内存
    std::shared_ptr<HDCommunicate> customControlTransferH2D_;
    std::shared_ptr<HDCommunicate> customStatusTransferD2H_;

    HcclCommConnections commConnections_;
    HcclSocketPortConfig commPortConfig_;
    std::shared_ptr<PetersonLock> hostDeviceLock_;
    bool isNsRecovery_{false};
    bool isAicpuCommEngine_{false};
    bool isAicpuUnfold_{false};
    HostMem opTilingDataBuf_;
    HostMem apiTilingDataMem_;
    DeviceMem tilingDataMemDevice_;
    // 单机场景下多卡间能互相访问的共享buffer，除了自己rank是申请的，其余均是Ipc打开的
    DeviceMem zeroCopyLocalBuffer_;
    void* zeroCopyIpcPtrs_[AICPU_ZERO_COPY_MAX_DEVICE_NUM_A3]{};
    std::atomic<HcclCommState> state_{HcclCommState::IDLE};
    std::unordered_map<std::string, std::string> newTagToTagMap_;
    // zerocopy hex prefix tag + cnt 副本 _CaptureN tag，跨 iter 累积新 entry
    std::unordered_set<std::string> tagsRequiringHostCleanup_;
    // 保护 rankTagRemoteRes_ / hostMemVec_ / deviceMemVec_ / tagsRequiringHostCleanup_ /
    // ibverbsLocalNotify_ / ibverbsRemoteNotify_ / resMap_ 的并发访问
    std::mutex commResMutex_;
    static std::mutex linkResMapMutex_;
    static std::unordered_map<Transport*, LinkInfo> linkResMap_;
    std::shared_ptr<HostMem> transDevIbverbsDataMem_ = nullptr;
    bool isA2MC2MultiServer_{false};
    bool isA2MC2IntraHie_{false};
    DeviceMem ibverbsDataBuffer_;
    std::unordered_map<std::string, std::list<DeviceMem>> ibverbsLocalNotify_;
    std::unordered_map<std::string, std::list<DeviceMem>> ibverbsRemoteNotify_;

    // 按序下发notify的工作区
    DeviceMem aicpuOrderNotifyAddr_;
    u32 graphId_{0};

    // alltoallv
    HostMem hostCollectBuffer_;

    // batchsendrecv
    std::set<u32> ranksLinked_{};

    // AIV通信同步标识
    s32 aivOpbaseTag_ = 1;                         // 动态图或者单算子非Capture模式的tag
    s32 aivOffloadTag_ = 1;                        // 静态图或者Capture模式的tag
    std::vector<DeviceMem> aivOffloadCommInfoMem_; // 图模式每个算子单独一块内存维护通信域信息

    // Host侧收集的数据
    std::shared_ptr<HostMem> aiRMAInfoMem_ = nullptr;
    std::shared_ptr<HostMem> rmaInfoMem_ = nullptr; // for aiv
    std::shared_ptr<HostMem> aiSqMem_ = nullptr;
    std::shared_ptr<HostMem> aiScqMem_ = nullptr;
    std::shared_ptr<HostMem> aiRqMem_ = nullptr;
    std::shared_ptr<HostMem> aiRcqMem_ = nullptr;
    std::shared_ptr<HostMem> aiMemMem_ = nullptr;
    std::shared_ptr<HostMem> aiMemDetailsMem_ = nullptr;

    // Host侧同步到Device的内存空间
    DeviceMem aiRMAInfoDev_;
    DeviceMem aiSqDev_;
    DeviceMem aiScqDev_;
    DeviceMem aiRqDev_;
    DeviceMem aiRcqDev_;
    DeviceMem aiMemDev_;
    DeviceMem aiMemDetailsDev_;

    // 通信能力支持信息，提供给融合算子获取
    std::shared_ptr<HostMem> combinedCapabilityMem_ = nullptr;
    DeviceMem combinedCapabilityBuffer_;

    aclrtBinHandle binHandle_ = nullptr;
    aclrtBinHandle binCustomHandle_ = nullptr;

    std::unique_ptr<ZeroCopyAclGraph> zeroCopyAclGraph_;

    u32 switchRanksNum_{0};
    u32 switchRankList_[AICPU_MAX_RANK_NUM]{};
    bool switchUseBackup_[AICPU_MAX_RANK_NUM]{};
    u8 remoteRankNicStatus_[AICPU_MAX_RANK_NUM]{};
    bool needCheckDefaultNic_{false};
    bool needCheckBackupNic_{false};
    bool switchNicWaitingResult_{false};
    u32 captureCnt_ = 0;
    bool isUserMemRegisted_{false};                                    // 是否已注册user Mem，与ccl buffer互斥
    std::unordered_map<void*, std::shared_ptr<DeviceMem>> userMemMap_; //  key: window handle, value: window ptr
    OpCommTransport userMemTransport_;
    std::vector<LINK> channelLinks_{};

    void* p2pCclBuf_[AICPU_MAX_RANK_NUM]{};
    void* cclBuf_[AICPU_MAX_RANK_NUM]{};
    std::map<u32, TransportType> remoteTransportMap_;
    uint32_t netLayer_[COMM_LAYER_NUM_MAX]{};
#ifndef CCL_KERNEL_AICPU
    RankGraphV1 rankGraph_;
    std::unique_ptr<DpuManager> dpuManager_;
#endif
    uint32_t myRankConnectMode_{0}; // 0: normal mode 1: host nic对接device nic异构模式

    // for group
    bool isGroupMode_{false};

    // 独立算子
    std::vector<std::shared_ptr<DeviceMem>> channelRemoteParamMem_;
    CommConfig commConfig_;
    std::function<bool()> getAicpuCommState_; // 获取自定义算子aicpu通信域是否初始化
    bool isInvalidComm_{false};
    std::function<HcclResult()> releaseChannel_ = nullptr;

    u32 hcclQos_ = EnvConfig::HCCL_QOS_DEFAULT;
    u32 rankInParentComm_{0}; // 父通信域中的rank ID (MC2场景使用)
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_;
    std::unique_ptr<SymmetricMemory> symmetricMemory_;
};
} // end namespace hccl
#endif // HCCL_IMPL_BASE_H
