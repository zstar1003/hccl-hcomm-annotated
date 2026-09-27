/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_PUB_H
#define HCCL_COMM_PUB_H

#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include "hccl/base.h"
#include "hccl_common.h"
#include "common.h"
#include "mem_device_pub.h"
#include "topoinfo_struct.h"
#include "comm.h"
#include "topoinfo_struct.h"
#include "transport_heterog_def.h"
#include "hccl/hccl_res.h"
#include "comm_config_pub.h"
#include "transport_manager.h"
#include "independent_op.h"
#include "share_ccl_buffer_manager.h"
#ifndef HCCD
#include "coll_comm.h"
#endif

namespace hccl {
/* * 默认的rank_table, ranklist为空数组;  后续HCCL可以用于判断是否走新分支 */
extern RankTable_t g_hcclDefaultRankTable;

class HcclCommunicator;
class IHcclOneSidedService;

class hcclComm {
public:
    explicit hcclComm(
        u64 inCCLbufferSize = 0, u64 outCCLbufferSize = 0, std::string identifier = "", std::string cclBuffName = "");
    ~hcclComm();

    /**********************************************************************
     函 数 名  : hcclComm::init
     功能描述  : 集合通信域初始化
     输入参数  : HcclCommParams& params
             const RankTable_t &rankTable
     输出参数  : 无
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult
    init(HcclCommParams& params, const CommConfig& commConfig, const RankTable_t& rankTable = g_hcclDefaultRankTable);
    HcclResult init(
        HcclCommParams& params, const CommConfig& commConfig, const std::vector<RankInfo>& rankList,
        WorldGroupInfo& groupCommonData);

    /**********************************************************************
     功能描述  : 创建以group为名字的集合通信
     输入参数  : const std::string& group
             const u32& groupRank
             const std::vector<u32>& groupRanks
     输出参数  : std::shared_ptr<hcclComm>& groupComm
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult CreateGroup(
        const std::string& group, const u32& groupRank, const u32& userRank, const std::vector<u32>& groupRanks,
        std::shared_ptr<hcclComm>& groupComm);

    /**********************************************************************
     功能描述  : 销毁以group为名字的集合通信
     输入参数  : const std::string& group
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult DestroyGroup(const std::string& group) const;

    /**********************************************************************
     功能描述  : 查询当前的算法类型
     输出参数  : AlgType &algType
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult GetAlgType(AlgType& algType, HcclCMDType opType);

    /**********************************************************************
     功能描述  : AllGather功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 inputCount
                 HcclDataType datatype
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult AllGather(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType,
        rtStream_t stream, HcomCollOpInfo* opInfo = nullptr);
    HcclResult AllGatherOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType,
        rtStream_t stream);
    HcclResult AllGatherVOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, const void* outputCounts,
        const void* outputDispls, HcclDataType dataType, HcclRtStream stream);
    HcclResult AllGatherV(
        const std::string& tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts,
        const void* rdispls, HcclDataType dataType, HcclRtStream stream);

    /* *********************************************************************
     功能描述  : all reduce功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult AllReduce(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        rtStream_t stream, SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    HcclResult AllReduceOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        rtStream_t stream, SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    /* *********************************************************************
     功能描述  : broadcast功能实现
     输入参数  :const char *tag
                 void* ptr
                 s32 count
                 HcclDataType dataType
                 s32 root
                 rtStream_t stream
     输出参数  : void* ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult
    Broadcast(const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, rtStream_t stream);
    HcclResult
    BroadcastOutPlace(const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, rtStream_t stream);
    /* *********************************************************************
     功能描述  : scatter功能实现
     输入参数  : const char *tag
                const void* input_ptr
                void *outputPtr
                u64 recvCount
                HcclDataType dataType
                u32 root
                rtStream_t stream
     输出参数  : void* ptr
     返 回 值  : HcclResult
    ********************************************************************* */
    HcclResult Scatter(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
        rtStream_t stream);
    HcclResult ScatterOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
        rtStream_t stream);
    /**********************************************************************
     功能描述  : reduce功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 s32 root,
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult Reduce(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        u32 root, rtStream_t stream);
    HcclResult ReduceOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
        u32 root, rtStream_t stream);

    /**********************************************************************
     功能描述  : reduce-scatter功能实现
     输入参数  : const char *tag
                 const void* input_ptr
                 void *outputPtr
                 s32 count
                 HcclDataType data_type
                 HcclReduceOp op
                 rtStream_t stream
     输出参数  : void* output_ptr
     返 回 值  : HcclResult
    **********************************************************************/
    HcclResult ReduceScatter(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, HcclReduceOp op,
        rtStream_t stream);
    HcclResult ReduceScatterOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, HcclReduceOp op,
        rtStream_t stream);
    HcclResult ReduceScatterV(
        const std::string& tag, void* inputPtr, const void* inputCounts, const void* inputDispls, void* outputPtr,
        u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);
    HcclResult ReduceScatterVOutPlace(
        const std::string& tag, void* inputPtr, void* outputPtr, const void* inputCounts, const void* inputDispls,
        u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream);

    HcclResult
    BatchSendRecv(const std::string& tag, struct HcclSendRecvItemDef* sendRecvItemsPtr, u32 itemNum, rtStream_t stream);

    HcclResult send(
        const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream,
        u32 srTag, u32 localGroupRank);
    HcclResult SendOutPlace(
        const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream);

    HcclResult receive(
        const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream,
        u32 srTag, u32 localGroupRank);
    HcclResult ReceiveOutPlace(
        const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream);

    HcclResult AlltoAllV(
        const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
        const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag);
    HcclResult AlltoAllVOutPlace(
        const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
        const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag);

    HcclResult AlltoAllVC(
        const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);
    HcclResult AlltoAllVCOutPlace(
        const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);

    HcclResult AlltoAll(
        const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
        HcclDataType recvType, rtStream_t stream, const std::string& tag);

    /**********************************************************************
     功能描述  : 生成唯一的集合通信域标识
     输入参数  : 无
     输出参数  : HcclRootInfo* rootInfo
     返 回 值  : HcclResult
    **********************************************************************/
    static HcclResult GetUniqueId(HcclRootInfo* uniqueId);

    HcclResult GetInCCLbuffer(void*& buffer, u64& size);
    HcclResult GetOutCCLbuffer(void*& buffer, u64& size);
    HcclResult GetUserRank(u32& userRank);
    HcclResult GetGroupRank(u32& userRank);
    HcclResult GetRankSize(u32& rankSize);
    void ReleaseCommCCLbuffer() const;
    void RealeaseBarrierMemory();
    HcclResult RealeaseShareCCLbuffer();
    HcclResult CreateCommCCLbuffer() const;
    HcclResult CreateIndirectCCLbuf();
    void ReleaseIndirectCCLbuf();
    HcclResult SetAicpuCommEngine(bool isAicpuCommEngine);

    HcclResult GetOneSidedService(IHcclOneSidedService** service);                                   // host侧专用
    HcclResult InitOneSidedServiceNetDevCtx(u32 remoteRankId);                                       // host侧专用
    HcclResult OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx);                 // host侧专用
    HcclResult GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port); // host侧专用
    HcclResult DeinitOneSidedService();                                                              // host侧专用

    HcclResult GetIndirectInCCLbuf(void*& ptr, u64& size);
    HcclResult GetIndirectOutCCLbuf(void*& ptr, u64& size);
    HcclResult HcclSelectAlg(
        HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, int32_t aivCoreLimit,
        bool& ifAiv, std::string& algName);
    HcclResult HcclCalcNumBlocks(
        HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
        u32& numBlocks);

    HcclResult HcclGetAlgExecParam(
        const std::string& tag, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType, bool clearEnable,
        HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit);

    HcclResult GetWorkspaceSubStreamNum(
        u64 count, HcclDataType dataType, HcclReduceOp op, const std::string& algName, u64& streamNum, u64 dataSize = 0,
        bool ifAiv = false, HcclCMDType optype = HcclCMDType::HCCL_CMD_INVALID) const;
    HcclResult
    GetWorkspaceMemSize(const std::string& opType, u64 count, HcclDataType dataType, u32& rankSize, u64& size);
    HcclResult GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize) const;
    HcclResult SetWorkspaceResource(const std::string& tag, void* memPtr, u64 maxSize, std::vector<rtStream_t>& stream);
    HcclResult CreateOpBasedResources(const HcclCMDType& opType, const std::string& tag, const HcomCollOpInfo& opInfo);

    std::string GetIdentifier();
    std::string GetCCLbufferName();
    HcclResult CreateBarrierMemory();
    HcclResult ReleaseSubComms() const;
    HcclResult GetAlltoAllStagedWorkSpaceMemSize(
        u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls, HcclDataType recvType,
        u64& memSize) const;
    HcclResult
    GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize) const;
    // 目前支持按tag对资源释放、解绑定
    HcclResult ClearOpResource(const std::string& tag);
    HcclResult SetClearAivSyncBuf(bool aivClearEnable);
    HcclResult Isend(
        void* buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest& request,
        HcclUserRequire& userRequire) const;
    HcclResult Improbe(u32 peerRank, s32 tag, s32& flag, HcclMessage& msgHandle, HcclStatus& status) const;
    HcclResult Imrecv(void* buffer, s32 count, HcclDataType dataType, HcclMessage msg, HcclRequest& request) const;
    HcclResult HcclTest(HcclRequest hcclRequest, s32& flag, HcclStatus& compState) const;
    // 获取溢出Flag内存传给RTS
    HcclResult SetGlobalWorkSpace(std::vector<void*>& globalWorkSpaceAddr);
    HcclResult SetAttachedStream(u32 graphId, const std::vector<rtStream_t>& streams);
    // 获取rdma with reduce算子溢出的task信息，然后清除
    HcclResult GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo);
    HcclResult SupportDeterministicOptim(bool& isDeterministicOptim);
    HcclResult GetHccsLinkNum(u32& numHccsLink);
    HcclResult GetDeviceId(s32& deviceId);
    HcclResult GetDevType(DevType& devType);
    HcclResult IsStandardCard(bool& isStandardCard);
    HcclResult Is310PDuoCard(bool& is310PDuoCard);
    HcclResult AbortSelf(s32 tag);

    HcclResult RegistTaskAbortHandler() const;
    HcclResult UnRegistTaskAbortHandler() const;
    HcclResult RegTransportLinks(s32 linkNum, void* transportPara);
    HcclResult GetDeviceNumPerAggregation(u32& deviceNumPerAggregation);
    HcclResult GetBandWidthPerNPU(u32 level, float& bandWidth);
    bool IsNeedResetDevice() const;
    HcclResult ResetDeviceEnable();
    HcclResult CommCheckErrorCqe(HcclResult& result);
    HcclResult CommCheckOpInconsistentError(HcclResult& result);
    HcclResult SaveTraceInfo(std::string& logInfo);
    HcclResult AllocComResourceByTiling(const std::string& algConfig, void* param);
    HcclResult CreateCommResource(
        const std::string& tag, rtStream_t aiCpuStream, bool isOpbaseMode, void** commContext,
        const std::string& algConfig = "");
    bool GetCommResource(const std::string& tag, void** commContext);
    bool GetCommResource(void*& commContext);
    HcclResult SetStopFlag(bool value);
    HcclResult SetState(HcclCommState state);
    HcclCommState GetState();
    HcclResult GetAicpuOpStreamNotify(HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify);
    HcclResult Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t& aiCpuStream);
    HcclResult GetAiCpuNotifyData(HcclRtNotify notifyHandle, HcclSignalInfo& notifyInfo);
    HcclResult AddAiCpuNotify(HcclRtNotify* notifyHandle);
    HcclResult GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize);
    HcclResult GetCommUserMemSize(uint64_t& size);
    HcclResult SetDeterministicConfig(const u8 deterministic); // 设置确定性计算配置
    HcclResult SetAivModeConfig(const bool aivMode);           // 设置aiv模式配置
    HcclResult SetOnlyAivModeConfig(const bool isOnlyAiv);
    HcclResult GetOnlyAivModeConfig(bool& isOnlyAiv);
    HcclResult SetAicpuUnfoldConfig(const bool aicpuUnfold); // 设置aicpu配置
    HcclResult SetExecTimeOutConfig(const s32 execTimeOut);  // 设置HCCL执行超时时间
    HcclResult SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap); // 设置HCCL_ALGO
    u64 GetConfigInCCLbufferSize() const;  // 获取通信域配置的输入buffer大小
    u64 GetConfigOutCCLbufferSize() const; // 获取通信域配置的输出buffer大小
    u32 GetRankTableCrc();
    u32 GetServerNum();
    u32 GetModuleNum();
    u32 GetRealUserRank() const;
    HcclResult GetCommParams(HcclCommParams& params);            // 逆向解析获取HcclCommParams参数
    HcclResult GetCommRankTable(RankTable_t& rankTable);         // 逆向解析获取RankTable_t参数
    HcclResult SetQpQosAttr(u32 trafficClass, u32 serviceLevel); // 设置TC/SL配置
    HcclResult SetHcclQos(u32 hcclQos);
    u32 GetHcclQos() const;

    std::shared_ptr<struct hcclKernelPlanner> planner{nullptr}; // for group
    void* barrierSendBuf;
    void* barrierRecvBuf;
    std::mutex operatorlock_;
    HcclResult Suspend();
    HcclResult Resume();
    HcclResult GetCommStatus(HcclCommStatus& status);

    HcclResult InitZeroCopyMemoryAgent();
    HcclResult DeinitZeroCopyMemoryAgent();
    HcclResult SetMemoryRange(void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);
    HcclResult UnsetMemoryRange(void* baseVirPtr);
    HcclResult ActivateCommMemory(void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags);
    HcclResult DeactivateCommMemory(void* virPtr);
    HcclResult GetNumBlocks(u32& numBlocks);
    HcclResult SetAivCoreLimit(u32 aivCoreLimit);
    HcclResult SwitchNic(uint32_t nRanks, uint32_t* ranks, bool* useBackup);
    HcclResult InitHccpChannel();
    std::vector<RankInfo> GetRankLists();
    HcclResult RegisterCommUserMem(void* addr, u64 size, void** handle);
    HcclResult DeregisterCommUserMem(void* handle);
    HcclResult ExchangeCommUserMem(void* handle, std::vector<u32>& peerRanks);
    HcclResult SetCommDispatcherCtx();
    HcclResult ReleaseCommDispatcherCtx();
    // 独立算子专用
    HcclResult SetIndependentOpConfig(const CommConfig& commConfig, const RankTable_t& rankTable);
    HcclResult InitIndependentOp();
    void SetAicpuCommState(bool aicpuCommState);
    bool GetAicpuCommState() const;
    HcclResult KernelLaunchAicpuCommInit();
    HcclResult ReportProfilingKernel(uint64_t beginTime, std::string kernelName);
    bool IsCommunicatorV2();
#ifndef HCCD
    HcclResult InitCollComm(
        void* commV2, void* rankGraph, uint32_t userRank, HcclMem cclBuffer, const std::string& commName,
        const HcclCommConfig* config, CollCommInitMode initMode = CollCommInitMode::fullMode);
    HcclResult InitCollCommInner(uint32_t userRank);
#endif
    void* GetCommunicatorV2();
    HcclCommunicator* GetHcclCommunicator();
#ifndef CCL_KERNEL_AICPU
#ifndef HCCD
    CollComm* GetCollComm();
#endif
    IndependentOp& GetIndependentOp();
#endif
    // A5communicator相关

    HcclResult IndOpTransportAlloc(
        const std::string& tag, OpCommTransport& opCommTransport, bool isAicpuModeEn, const HcclMemHandle* memHandles,
        uint32_t memHandleNum);

    HcclResult PrepareChannelMem(
        const std::string& tag, TransportIOMem& transMem, const HcclMemHandle* memHandles, uint32_t memHandleNum);

    // Decouple for MC2
    HcclResult GetLocalCCLBuf(void** addr, uint64_t* size);
    HcclResult GetRemoteCCLBuf(uint32_t remoteRank, void** addr, uint64_t* size);
    HcclResult GetKFCWorkSpace(void** addr, uint64_t* size);
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);
    HcclResult CommGetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult CommGetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult CommGetInstTopoTypeByNetLayer(uint32_t netLayer, uint32_t* topoType);
    // rankgraph interface
    HcclResult GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum);
    HcclResult GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum);
    HcclResult GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType);
    HcclResult GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum);
    HcclResult GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
    HcclResult GetRankGraph(GraphType type, void** graph, uint32_t* len);
    HcclResult GetLinks(uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize);
    uint32_t GetConnectMode();
    HcclResult GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
    HcclResult GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
    HcclResult GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
    HcclResult GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num);
    HcclResult GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
    HcclResult GetEndpointInfo(
        uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);
    HcclResult GetHeterogMode(HcclHeterogMode* mode);
    // for group
    HcclResult SetGroupMode(bool isGroup);
    bool GetGroupMode() const;
    HcclResult RegisterWindow(void* ptr, size_t size, HcclCommSymWindow* winHandle);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcclResult GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset);
    aclrtBinHandle GetBinHandle();

protected:
    /* * 禁止用户对API类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
    hcclComm(const hcclComm&) = delete;
    hcclComm& operator=(const hcclComm&) = delete;

private:
    HcclResult InitImpl(DevType deviceType, const CommConfig& commConfig);
    void UpdateIsHaveCpuRank(const RankTable_t& rankTable);
    void UpdateIsHaveCpuRank(const std::vector<RankInfo>& rankList);
    void PrintSubmittedOpCnt(const std::string& tag, HcclResult ret);
    HcclResult ReleaseChannel();
    void BinaryUnLoad();
    HcclResult InitBinHandle();
    DeviceMem indirectInCCLbuffer_;  /* 保存inCCLbuffer指针的地址 */
    DeviceMem indirectOutCCLbuffer_; /* 保存outCCLbuffer_指针的地址 */
    u64 inCCLbufferSize_;
    u64 outCCLbufferSize_;
    DevType deviceType_;
    DeviceMem barrierInMemory_;
    DeviceMem barrierOutMemory_;
    bool isFirstBarrier_;
    const std::string identifier_;
    const std::string cclBuffName_;
    bool isHeterogComm_;
    bool isGroupMode_{false};
    bool isResetDevice_;
    bool isSpecialType_;
    bool isHaveCpuRank_{false};
    std::unique_ptr<HcclCommunicator> communicator_;

    bool isAicpuCommInit_ = false;
    CommAicpuParam commAicpuParam_;
    aclrtBinHandle binHandle_ = nullptr;
    DevType devType_ = DevType::DEV_TYPE_COUNT;
    u32 hcclQos_;
#ifndef CCL_KERNEL_AICPU
    // 独立算子专用成员变量
    IndependentOp independentOp_;
#ifndef HCCD
    // A5CollComm
    std::unique_ptr<CollComm> collComm_{nullptr};
#endif
#endif
};
} // namespace hccl

using HcclCommPtr = std::shared_ptr<hccl::hcclComm>;
#endif /* HCCL_COMM_PUB_H */
