/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMUNICATOR_IMPL_H
#define HCCL_COMMUNICATOR_IMPL_H

#include <set>
#include <vector>
#include <unordered_map>
#include <string>
#include <atomic>
#include "types.h"
#include "coll_service_default_impl.h"
#include "coll_service_ai_cpu_impl.h"
#include "conn_local_notify_manager.h"
#include "conn_local_cnt_notify_manager.h"
#include "data_buf_manager.h"
#include "local_rma_buf_manager.h"
#include "queue_notify_manager_v2.h"
#include "remote_rma_buf_manager.h"
#include "rma_conn_manager.h"
#include "stream_manager.h"
#include "queue_wait_group_cnt_notify_manager.h"
#include "host_device_sync_notify_manager.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#include "mem_transport_manager.h"
#include "rank_gph.h"
#include "rank_table_info.h"
#include "topo_info.h"
#include "snap_shot_parse.h"
#include "comm_type.h"
#include "mirror_task_manager.h"
#include "aicpu_stream_manager.h"
#include "profiling_reporter.h"
#include "hdc.h"
#include "hccl_one_sided_service_v2.h"
#include "coll_alg_component.h"
#include "ub_memory_transport_mgr.h"
#include "ccu_super_fast_load.h"
#include "ccu_stream_sync_notify_manager.h"
#include "ccu_driver_handle.h"
#include "hccl_common_v2.h"
#include "hccl_rank_graph.h"
#include "hccl_aiv_utils.h"
#include "error_message_v2.h"
#include "hccp.h"
#include "aicpu/launch_device_v2.h"
#include "communicator_dpu_types.h"

namespace Hccl {

using HcclUs = std::chrono::steady_clock::time_point;
class CommunicatorImpl {
public:
    HcclResult Init(const CommParams& commParams, const std::string& rankTablePath);
    HcclResult Init(const CommParams& commParams, const std::string& ranktableM, const HcclCommConfig& config);
    HcclResult Init(const CommParams& commParams, const RankTableInfo& ranktable, const HcclCommConfig& config);

    HcclResult
    CreateSubComm(const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl);
    HcclResult CreateSubComm(
        const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
        HcclCommConfig& subConfig);

    HcclResult LoadOpbasedCollOp(const CollOpParams& opParams, void* stream);

    HcclResult AllocCollOpResource(const CollOpParams& opParams, void** addr);

    HcclResult AllocCommResource(void* mc2Tiling, void** commContext);

    HcclResult GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup) const;
    HcclResult GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize) const;

    HcclResult
    CalcCollOffloadOpRes(const OpType opType, u64 dataSize, HcclDataType dataType, CollOffloadOpResReq& resReq);
    HcclResult SetCollOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams);
    HcclResult SetCollOffloadScratchBuf(const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize);
    HcclResult LoadOffloadCollOp(std::string& opTag, const CollOpParams& opParams, void* stream);

    HcclResult SaveTopoDesc(std::string& identifier);
    HcclResult GetConfigInCCLbufferSize(uint64_t* cclBufSize);
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, uint32_t* topoType);
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
    HcclResult GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
    HcclResult GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);

    HcclResult GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);

    u32 GetCcuMc2ServerNum();

    const string& GetId() const;

    u32 GetIdIndex() const;

    RankId GetMyRank() const;

    u32 GetRankSize() const;

    u32 GetDeviceLogicId() const;

    u32 GetDevicePhyId() const;

    u64 GetBufferSize() const;

    const DevType& GetDevType() const;

    shared_ptr<RankGraph> GetRankGraph() const;

    bool GetOpAiCpuTSFeatureFlag() const;

    bool GetOpAivFeatureFlag() const;

    bool GetOpCcuFeatureFlag() const;

    bool GetCommAiCpuTSFeatureFlag() const;

    bool GetCommCcuFeatureFlag() const;

    virtual DataBufManager& GetDataBufferManager() const; // NOTE:添加 virtual用于UT打桩

    virtual LocalRmaBufManager& GetLocalRmaBufManager() const;

    virtual RemoteRmaBufManager& GetRemoteRmaBufManager() const;

    virtual QueueNotifyManager& GetAicpuQueueNotifyManager() const;

    virtual QueueNotifyManager& GetCcuQueueNotifyManager() const;

    virtual ConnLocalNotifyManager& GetConnLocalNotifyManager() const;

    virtual ConnLocalCntNotifyManager& GetConnLocalCntNotifyManager() const;

    virtual QueueWaitGroupCntNotifyManager& GetQueueWaitGroupCntNotifyManager() const;

    virtual QueueBcastPostCntNotifyManager& GetBcastPostCntNotifyManager() const;

    virtual StreamManager& GetStreamManager() const;

    virtual AicpuStreamManager& GetAicpuStreamManager() const;

    virtual CollServiceBase* GetCollService() const;

    virtual CollServiceBase* GetCcuCollService() const;

    virtual SocketManager& GetSocketManager() const;

    virtual RmaConnManager& GetRmaConnManager() const;

    virtual const string& GetEstablishLinkSocketTag() const;

    virtual CollOperator* GetCurrentCollOperator() const;

    virtual MemTransportManager* GetMemTransportManager() const;

    virtual HostDeviceSyncNotifyManager& GetHostDeviceSyncNotifyManager() const;

    virtual Trace& GetTrace() const;

    virtual u32 GetOpBaseOpIndex() const;

    virtual u32 GetOpIndex() const;

    u32 GetSubmittedOpCnt() const;

    HDCommunicate& GetKfcControlTransferH2D() const;

    HDCommunicate& GetKfcStatusTransferD2H() const;

    HcclResult Suspend();

    HcclResult Clean();

    HcclResult Resume();

    void SetAicpuKernelLaunched(bool flag) { isAicpuKernelLaunched = flag; }

    const NotifyTimeoutCfg& GetNotifyTimeoutCfg() const;

    const shared_ptr<DevBuffer>& GetCclBuffer() const
    {
        // 接口设计不合理，其他数据结构无权管理hccl buffer生命周期
        // 建议改为传递裸指针，但整改范围较大，本次性能优化暂不处理
        return cclBuffer;
    }

    const shared_ptr<DevBuffer> GetAivTagBuffer() const { return aivTagBuffer; }

    const shared_ptr<DevBuffer> GetAivOffloadTagBuffer() const { return aivOffloadTagBuffer; }

    const shared_ptr<DevBuffer> GetInCclBuffer() const { return inCclBuffer; }

    const shared_ptr<DevBuffer> GetOutCclBuffer() const { return outCclBuffer; }

    const shared_ptr<DevBuffer> GetKFCWorkSpace(const char* memTag) const
    {
        std::string tag = memTag != nullptr ? std::string(memTag) : "";
        auto it = tagWorkspaceMap_.find(tag);
        return it != tagWorkspaceMap_.end() ? it->second : nullptr;
    }

    HcclResult CreateCommCclBuf();
    HcclResult GetInCclBuf(void*& commInputPtr, u64& commInputSize);
    HcclResult GetOutCclBuf(void*& commOutputPtr, u64& commOutputSize);
    HcclResult GetIndirectInCclBuf(void*& commIndirectInputPtr, u64& commIndirectInputSize);
    HcclResult GetIndirectOutCclBuf(void*& commIndirectOutputPtr, u64& commIndirectOutputSize);

    HcclResult GetLocalCclBuffer(void** addr, uint64_t* size);
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);
    HcclResult CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated);
    HcclResult AllocAndRegKFCWorkSpace(uint64_t size, const std::string& memTag);
    HcclResult CleanupKFCWorkSpaceOnFailure(DpuShmem& shmem, HcclResult ret);
    HcclResult GetKFCWorkSpaceVA(const std::string& memTag, const uint64_t* size, void** addr, bool* newCreated);
    HcclResult DestroyKFCWorkSpaceVA();

    bool IsWorldGroup() const;

    // 静态信息序列化,供获取保存快照size时使用
    BinaryStream& GetStaticBinaryInfo() { return staticBinaryInfo; }
    // 获取rankTable的字节流，供一致性校验crc时带localId使用
    u32 GetRanktableCrc(bool isContainLoaId) const;
    HcclResult RecoverRankGraphData(SnapShotComm& snapShotComm, const char* changeInfo);
    HcclResult RecoverTransportData(
        u32 savedSubmittedOpCnt, const std::vector<std::pair<u32, RankId>>& levelRankPairs, u32 savedStep,
        vector<std::pair<LinkGroup, u32>> linkGroupPair);
    HcclResult
    RecoverExeCfgData(const OpExecuteConfig& inOpExeCfg, const OpExecuteConfig& inCommExeCfg, bool inIsLoadOp);
    virtual HcclResult GetSnapShotDynamicBuf(BinaryStream& buf) const;

    HcclResult RecoverComm(SnapShotComm& snapShotComm, u32 stepParam, const char* changeInfo);
    HcclResult
    RecoverComm(const SnapShotSubComm& snapShotSubComm, std::unique_ptr<RankGraph>& inputRankGraph, u32 inputStep);
    HcclResult RecoverSubComm(const SnapShotSubComm& snapShotSubComm, CommunicatorImpl* subCommImpl, u32 step);
    HcclResult RecoverOpMode(u32 opMode);
    std::set<RankId> GetNeighboorRanks() const;
    virtual u32 GetCollOpIndex() const;

    virtual u32 GetStep() const;
    bool IsCommReady();
    void CovertToCurrentCollOperator(
        std::string& opTag, const CollOpParams& opParams, OpMode opMode, bool isLaunch = true,
        bool isHcomSelectAlg = false);

    virtual MirrorTaskManager& GetMirrorTaskManager() const;
    virtual ProfilingReporter& GetProfilingReporter() const;

    ~CommunicatorImpl();
    HcclResult NotifyAicpuDestroyComm();

    virtual HcclResult GetOneSidedService(HcclOneSidedService** service) const;
    u32 GetUsedChannelCount(u32 dieId);
    void PrintChannelInfoCallback() const;
    void RegisterPrintChannelInfoCallback(std::function<void()> callback);
    void SetCommStatus(CommStatus commStatus);
    CommStatus GetCommStatus() const;

    /* mc2数据上报 */
    void ReportHcclMC2Info(const Stream& kfcStream, Stream& stream, const std::vector<Stream*>& aicpuStreams);

    const OpExecuteConfig& GetOpExecuteConfig() const // 获取算子粒度 加速模式
    {
        return opExecuteConfig;
    }
    const OpExecuteConfig& GetCommExecuteConfig() const // 获取通讯域粒度 加速模式
    {
        return commExecuteConfig;
    }
    void SetOpExecuteConfig(const OpExecuteConfig& inConfig);
    void SetCommExecuteConfig(const OpExecuteConfig& inConfig);
    const std::string& GetCurAlgName() const { return curAlgName; }
    CollAlgComponent* GetCollAlgComponent() { return collAlgComponent.get(); }

    std::map<AivOpCacheArgs, std::shared_ptr<InsQueue>> hcclCacheMap_; // 存储aiv cache信息
    HcclResult GetCacheMap(AivOpCacheArgs& opCacheParam, std::shared_ptr<InsQueue>& tempInsQue);
    HcclResult SetAccelerator(HcclAccelerator hcclAccelerator, bool isCcuMsAvailable);
    HcclResult GetAccelerator(int32_t* accelerator) const;
    void ExecAlgSelect(const CollOpParams& opParams, const OpMode& opMode);

    bool IsOpUsingCcuMs() const;      // 算子粒度
    bool IsOpUsingCcuSched() const;   // 算子粒度
    bool IsOpUsingAiv() const;        // 算子粒度
    bool IsOpUsingAivOnly() const;    // 算子粒度
    bool IsCommUsingCcuMs() const;    // 通信域粒度
    bool IsCommUsingCcuSched() const; // 通信域粒度
    void RegisterAcceStateCallBack(
        std::function<HcclResult(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)> inCallback);
    HcclResult AcceleratorFallback(); // 加速模式回退

    virtual UbMemoryTransportMgr* GetUbMemoryTransportMgr() const;
    u32 GetAivTag() const { return aivTag; }

    u32 GetAivOffloadTag() const { return aivOffloadTag; }
    u8 GetAlgorithmType() const { return algorithmType_; }

    void SetAivTag(u32 tag) { aivTag = tag; }

    void SetAivClearEnable(bool enable) { aivClearEnable = enable; }

    bool GetAivClearEnable() const { return aivClearEnable; }

    void SetAivCoreLimit(u32 newAivCoreLimit) { aivCoreLimit = newAivCoreLimit; }
    HcclResult CalcTaskNum(OpType opType, DataType dataType, u64 count, u32& taskNum) const;
    void CollAlgComponentInit();

    virtual CcuStreamSyncNotifyManager& GetCcuStreamSyncNotifyManager() const;

    void saveCCUParams(
        std::vector<std::vector<CcuTaskParam>>&& ccuParams,
        std::vector<std::vector<CcuProfilingInfo>>&& ccuProfilingInfo, u64 execId, CcuInstType insType,
        bool isSlave = false)
    {
        auto& ccuParamsMapping = colCcuParamMapping[currentCollOperator->opType];
        auto& ccuParamsNotCacheKey = colParamsNotCacheKey[currentCollOperator->opType];
        if (ccuParamsMapping.find(ccuParamsMappingKey) == ccuParamsMapping.end()
            && ccuParamsNotCacheKey.find(ccuParamsMappingKey) == ccuParamsNotCacheKey.end()) {
            ccuParamsMapping.emplace(
                std::piecewise_construct, std::forward_as_tuple(ccuParamsMappingKey),
                std::forward_as_tuple(
                    std::move(ccuParams), std::move(ccuProfilingInfo), execId, insType, isSlave,
                    static_cast<void*>(this)));
        } else {
            ccuParamsMapping.erase(ccuParamsMappingKey);
            if (ccuParamsMapping.empty()) {
                colCcuParamMapping.erase(currentCollOperator->opType);
            }
            ccuParamsNotCacheKey.emplace(ccuParamsMappingKey);
        }
    }

    inline bool isEnableSuperFasterLoad() const { return superFasterLoad; }

    HcclResult CreateBarrierMemory(void*& sendBuf, void*& recvBuf, uint64_t count);

    HcclResult HcomSelectAlg(const CollOpParams& opParams, int32_t aivCoreLimit, bool& ifAiv, std::string& algName);
    HcclResult
    CalcNumBlocks(const CollOpParams& opParams, int32_t aivCoreLimit, std::string& algName, u32& numBlocks) const;
    HcclResult
    GetAlgExecParam(const CollOpParams& opParams, bool clearEnable, void*& commContext, u64& len, u32 aivCoreLimit);

    HcclResult ClearOpResource(const std::string& opTag); // 清空opTag所属资源
    HcclResult GetAicpuOpStreamNotify(rtStream_t* opStream, u8 aicpuNotifyNum, void** aicpuNotify) const;
    static std::string GetTopoFilePath();
    std::vector<LinkData> GetFullMeshLinks() const;
    ErrorMessageReport GetAicpuTaskException();
    u32 GetRankInParentComm() const;
    u8 GetCommDepth() const;
    aclrtFuncHandle GetAicpuKernelFuncHandle(const char* kernelName) const;
    bool IsCommWithPCIEProtocol() const; // 判断通信域内是否有rank之间存在PCIE链路
    HcclResult Mc2AiCpuStreamAllocAndGetV2(rtStream_t* aiCpuStream);
    HcclResult SaveDpuStreamId();
    uint32_t GetDpuStreamId() const { return dpuStreamId; }

    HcclResult GetRankIpPortMap(RankIpPortMapPtr& rankIpPortMap);
    HcclResult SetRankIpPortMap(const RankIpPortMapPtr& rankIpPortMap);

private:
    std::string id;
    static std::atomic<u32> globalIndex; // 全局通信域唯一一个index, 对应锁保护
    u32 idIndex{0};                      // 每个通信域唯一一个index
    std::string establishLinkSocketTag;
    RankId myRank;
    u32 rankSize;
    RankId rankInParentComm;
    u8 commDepth{0};
    DevType devType;
    DevId devPhyId;
    DevId devLogicId;
    HcclCommConfig config;
    std::shared_ptr<RankGraph> rankGraph;
    uint32_t dpuStreamId{0};
    unique_ptr<DataBufManager> dataBufferManager;
    unique_ptr<LocalRmaBufManager> localRmaBufManager;
    unique_ptr<RemoteRmaBufManager> remoteRmaBufManager;
    unique_ptr<QueueNotifyManager> aicpuQueueNotifyManager_;
    unique_ptr<QueueNotifyManager> ccuQueueNotifyManager_;
    unique_ptr<QueueWaitGroupCntNotifyManager> queueWaitGroupCntNotifyManager;
    unique_ptr<QueueBcastPostCntNotifyManager> queueBcastPostCntNotifyManager;
    unique_ptr<ConnLocalNotifyManager> connLocalNotifyManager;
    unique_ptr<ConnLocalCntNotifyManager> connLocalCntNotifyManager;
    unique_ptr<StreamManager> streamManager;
    unique_ptr<AicpuStreamManager> aicpuStreamManager;
    unique_ptr<SocketManager> socketManager;
    unique_ptr<RmaConnManager> rmaConnectionManager;
    CollServiceBase* collService{nullptr};
    unique_ptr<CollOperator> currentCollOperator;
    void* notifyFixedValue{nullptr}; // 2MB内存，用于临时解决性能问题，实际该内存未使用
    unique_ptr<HostDeviceSyncNotifyManager> hostDeviceSyncNotifyManager;
    unique_ptr<Trace> trace;
    unique_ptr<MemTransportManager> memTransportManager{};
    unique_ptr<MirrorTaskManager> mirrorTaskManager;
    unique_ptr<UbMemoryTransportMgr> ubMemoryTransportMgr{};
    unique_ptr<ProfilingReporter> profilingReporter;
    unique_ptr<HDCommunicate> kfcControlTransferH2D;
    unique_ptr<HDCommunicate> kfcStatusTransferD2H;
    unique_ptr<HcclOneSidedService> oneSidedService;
    std::function<void()> printChannelInfoCallback{nullptr};
    unique_ptr<CcuStreamSyncNotifyManager> ccuStreamSyncNotifyManager;
    std::shared_ptr<CcuDriverHandle> ccuDrvHandle{nullptr};

    std::vector<u32> netLayersVec;
    std::vector<uint32_t> instSizeVec;
    std::vector<uint32_t> rankListVec;
    std::vector<CommLink> linkListVec;
    std::vector<uint32_t> ranksVec;
    std::vector<uint32_t> topoInstsVec;
    std::vector<u32> enableP2PDevices_;

    NotifyTimeoutCfg notifyTimeoutCfg;

    u32 step = 0;          // 全局device信息的step
    u32 opBaseOpIndex = 0; // 单算子次数
    u32 collOpIndex = 0;   // 集合通信算子次数
    u32 opIndex = 0;       // 下发算子总计数(单算子/图模式/CCU快速下发)
    u32 sendRecvIndex = 0; // send/recv 算子次数
    u32 submittedOpCnt = 0;
    u32 aivCoreLimit = MAX_NUM_BLOCKS;

    void RegisterOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams) const;
    void RegisterOffloadScratchBuffer(const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize);
    bool initFlag = false;
    bool devModeFlag = false;
    bool isSuspended = false;
    bool isCleaned = false;
    bool isAicpuKernelLaunched = false;
    bool isDpuKernelLaunched = false;
    bool isWorldGroup = false;
    bool aivClearEnable = false;

    std::shared_ptr<DevBuffer> cclBuffer;
    u64 cclBufferSize = 0;
    std::shared_ptr<DevBuffer> aivTagBuffer;
    std::shared_ptr<DevBuffer> indirectInCclBuffer;
    std::shared_ptr<DevBuffer> indirectOutCclBuffer;
    std::shared_ptr<DevBuffer> aivOffloadTagBuffer;
    std::shared_ptr<DevBuffer> inCclBuffer;
    std::shared_ptr<DevBuffer> outCclBuffer;
    // 为barrier算子新增的buffer与判断;
    std::shared_ptr<DevBuffer> barrierInMemory;
    std::shared_ptr<DevBuffer> barrierOutMemory;
    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> tagWorkspaceMap_;
    bool isFirstBarrier = true;
    DpuKernelLaunchParam hostArgs;
    std::unordered_map<std::string, DpuShmem> tagDpuShmemArgsMap_
        = {{"DPUTAG", DpuShmem{}}, {"DPUTASKEXCEPTION", DpuShmem{}}};
    void* hostShareBuf{nullptr}; // 在dpu线程中，从DPUTAG共享内存拷出算法信息的存放地址，作为入参给算法注册的回调
    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> tagWorkspaceVAMap_;
    aclrtStream dpuStream;
    aclrtContext dpuContext;
    aclrtContext npuContext;

    std::unordered_map<std::string, std::shared_ptr<Buffer>> offloadScrachBufferMap;
    BinaryStream staticBinaryInfo; // 静态信息序列化流

    std::atomic<CommStatus> status_{CommStatus::COMM_IDLE}; // 通信域状态
    std::vector<u32> rankIdsVec;                            // 子通信域使用：序列化解析
    std::unique_ptr<RankTableInfo> ranktableInfo;           // 主通信域使用：序列化解析
    std::shared_ptr<TopoInfo> topoInfo;                     // 主通信域使用：序列化解析

    std::map<AcceleratorState, std::shared_ptr<CollServiceBase>> collServices; // 初始化3种collService，供算法选择
    std::shared_ptr<CollAlgComponent> collAlgComponent;                        // 初始化算法组件
    OpExecuteConfig opExecuteConfig;                                           // 算子粒度 加速模式
    OpExecuteConfig commExecuteConfig;                                         // 通信域粒度 加速模式
    std::string curAlgName;                                                    // 当前算法名称
    bool isLoadOp{false}; // 是否已加载过算子,只要算子下发过就不让改加速模式 loadop offload AllocCommResource
    u32 aivTag{1};        // aiv kernel内部用于标志位计数
    u32 aivOffloadTag{0}; // aiv kernel内部用于标志位计数
    u8 algorithmType_{0};
    std::atomic<u32> tagResourceIndex_{0};

    std::function<HcclResult(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)> callback;
    CollOpParams curOpParams; // 当前算子参数
    std::map<std::pair<OpType, string>, std::pair<AcceleratorState, string>>
        opAcceStateCache{}; // opType + algName --> acceleratorState + newAlgName
    AicpuBinaryHolder aicpuKernelHolder_;

    void InitCommonData(const CommParams& commParams);
    void InitCommonDataNotInitDevType(const CommParams& commParams, const HcclCommConfig& commConfig);
    void InitCommonData(const CommParams& commParams, const HcclCommConfig& commConfig);
    void InitRankGraph(const string& ranktableM);
    void InitRankGraph(std::unique_ptr<RankGraph>& inputRankGraph);
    void InitRankGraph(
        std::shared_ptr<RankGraph> inputRankGraph, const RankTableInfo& inputRankTableInfo,
        const TopoInfo& inputTopoInfo);
    void InitRankGraph(const RankTableInfo& ranktable);
    void CheckRankGraph() const;
    void CheckRankGraphAddrs() const;
    HcclResult CheckCommStatus() const;
    void InitDataBufferManager();
    void InitNotifyManager();
    void InitStreamManager();
    void InitCollService();
    void InitHccpHdc() const;
    void InitCcuSuperFastLoad();
    void InitPreResource();
    void DeInitPreResource();
    void InitSocketManager();
    void InitRmaConnManager();
    void InitNotifyFixedValue();
    void InitMemTransportManager();
    void InitHostDeviceSyncNotifyManager();
    HcclResult InitTraceManager();
    void InitHDCommunicate();
    void InitOneSidedService();
    void InitUbMemoryTransportMgr();
    void RefreshSubmittedOpcnt();
    void SingleRankProc(const CollOpParams& opParams, void* stream) const;
    void ConvertCollOperatorA2A(const CollOpParams& opParams, bool isLaunch = true, bool isHcomSelectAlg = false);
    void DefaultConvertCollOperatorA2A(const CollOpParams& opParams);
    void LaunchConvertCollOperatorA2A(const CollOpParams& opParams, bool isHcomSelectAlg = false);
    void ConvertCollOperatorMem(const CollOpParams& opParams, u64 size);
    void
    CalcA2ASendRecvMem(const CollOpParams& opParams, u64& sendSize, u64& recvSize, bool isHcomSelectAlg = false) const;
    void ConvertCollOperatorMemV(const CollOpParams& opParams, bool isHcomSelectAlg = false);
    void RegisterAicpuKernel();

    // dpu相关
    void InitHccpPeer() const; // 拉起peer模式HCCP进程
    bool IsNeedDpu();          // 判断是否需要Host网卡参与集合通信
    void InitDpuKernel();
    std::unordered_set<IpAddress> GetHostIpFromRankGraph();
    HcclResult LaunchDpuKernel(aclrtFuncHandle& funcHandle);
    HcclResult PrepareDpuKernelResource(aclrtFuncHandle& funcHandle);
    HcclResult DestroyDpuKernelResource();
    HcclResult WaitDpuKernelThreadTerminate();
    void DestroyImpl();
    HcclResult InitAndLaunchDpuKernel();
    HcclResult InitAndLaunchAicpuKernel(); // 保存dpu taskexception共享内存到aicpu全局map中
    HcclResult DestroyDpuTaskexpShmemInDevice();

    HcclResult Init(const CommParams& commParams, std::unique_ptr<RankGraph>& inputRankGraph, DevId inputDevLogicId);
    HcclResult Init(
        const CommParams& commParams, std::unique_ptr<RankGraph>& inputRankGraph, HcclCommConfig& subConfig,
        DevId inputDevLogicId);
    HcclResult InitCommResource(const CommParams& commParams);

    void WaitReady() const;

    void InitMirrorTaskManager();
    HcclResult InitProfilingReporter();
    void UpdateProfStat();
    void InitTaskExceptionHandler() const;

    // 配置cachedReq属性：静态图模式：true, 动态图模式：false， 单算子模式：false
    void ReportProfInfo(uint64_t beginTime, bool cachedReq, bool opbased);

    void SelectCollService(); // 根据配置选择对应的collService

    CcuSFLMappingKey ccuParamsMappingKey{};
    std::unordered_map<const OpType, std::unordered_map<CcuSFLMappingKey, CachedCCUParams, ArrayHasher>>
        colCcuParamMapping{};
    std::unordered_map<const OpType, std::unordered_set<CcuSFLMappingKey, ArrayHasher>> colParamsNotCacheKey{};
    bool superFasterLoad{false};
    bool taskExceptionEnv{true}; // 默认HCCL_DFS_CONFIG="task_exception:on" 且默认on下不开启快速下发
    bool enableProfilingEnv{false};
    bool TryFastCcuLaunch(const CollOpParams& opParams, aclrtStream const stream);
    void FillAllToAllVArgs(const CollOpParams& opParams, rtCcuTaskInfo_t*& ccuParams) const;
    void ExecuteFastCcuLaunch(const CollOpParams& opParams, aclrtStream const stream, CachedCCUParams& params);

    void OpAcceleratorStateFallback(); // 算子粒度加速模式状态回退
    HcclResult ReLoadOpbasedOp();
    HcclResult ReLoadOffloadOp();

    void TryInitCcuFeature() const; // 根据通信域加速模式和rank信息，选择打开ccu功能，依赖hdc通道

    template <typename BufferType>
    static std::shared_ptr<BufferType> BarrierAllocBuffer(std::size_t size);

    void AppendLocalDieIdForLinks();
    HcclResult SetAivControlledCoreNum(bool isAiv);

    void CheckAcceleratorConsistency(AcceleratorState commAccelerator, AcceleratorState tilingAccelerator) const;
    HcclResult GetTilingAccelerator(void* mc2Tiling, AcceleratorState& acceleratorState) const;

    // AICPU场景aclgraph专用
    HcclResult OffloadResourcePre(std::string& opTag, const CollOpParams& opParams);

    RankIpPortMapPtr rankIpPortMap_;
};
} // namespace Hccl

#endif // HCCL_COMMUNICATOR_IMPL_H
