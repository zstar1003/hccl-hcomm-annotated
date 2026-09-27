/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_COMMUNICATOR_H__
#define __AICPU_COMMUNICATOR_H__

#include <memory>
#include <vector>
#include <iterator>
#include <array>
#include <hccl/hccl_types.h>
#include "common/aicpu_hccl_def.h"
#include "log.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "local_notify.h"
#include "comm_factory_pub.h"
#include "coll_executor_base.h"
#include "dispatcher.h"
#include "coll_alg_param.h"
#include "transport_pub.h"
#include "hccl_common.h"
#include "aicpu_operator_pub.h"
#include "peterson_lock.h"
#include "aicpu_hdc.h"
#include "aicpu_zero_copy_exchanger.h"
#include "cann_error_reporter.h"
#include "hccl_trace_info.h"
#include "aicpu_share_data_manager.h"
#include <shared_mutex>
#include "hccl/hccl_res.h"
#include "channel_param.h"
#include "aicpu_launch_manager.h"
#include "aicpu_ts_thread.h"
#include "new/hccl_dispatcher_ctx.h"
#include "aicpu_init_param.h"
#include "task_exception.h"
#include "ub_transport_lite_impl.h"
#include "aicpu_cache_manager.h"

namespace hccl {

enum class CommTransportsType {
    INVALID,
    GENERAL, // 通信域通用，input和output的中转内存全部使用cllbuffer
    SPECIAL, // 通信域专用，input和output的中转内至少其中一个不是cllbuffer
};

enum class CommResourceCtrlType {
    INVALID,
    INIT,    // 资源初始化
    REFRESH, // 资源刷新
};

struct TagAddress {
    u64 addr0;
    u32 key0 = 0;
    u64 addr1;
    u32 key1 = 0;
    // 重载运算符==
    bool operator==(const TagAddress& other) const
    {
        return addr0 == other.addr0 && addr1 == other.addr1 && key0 == other.key0 && key1 == other.key1;
    }
};

struct RankData {
    u32 remoteWorldRank{INVALID_VALUE_RANKID};
    u32 remoteUsrRankId{INVALID_VALUE_RANKID};
};

enum class CqeExceptionStatus : uint32_t {
    kNone = 0,
    kSdmaErr, // 可重执行的ErrCqe
    kOther    // 其他ErrCqe
};
enum class AicpuKfcHandlerType : u32 {
    kSetStepSize,
    kNotifyRecord,
    kNotifyWait,
    kClearMsgArea,
    kClearCommitTurn,
    kSetProfTimeStart,
    kSetProfTimeOrch,
    kSetProfTimeEnd,
    kMax
};

struct AicpuStreamMontior {
    HcclUs historyTime;
    u32 historyHead;
    u32 historyTaskId;
    u32 historyType;
};
using AicpuKfcHandler = std::function<HcclResult(const std::vector<u64>&)>;
class HcclCommAicpu {
public:
    explicit HcclCommAicpu();
    ~HcclCommAicpu();
    HcclResult Init(const HcclOpResParam* commParam, bool isCustom);
    Stream& GetMainStream() { return mainStream_; }
    std::vector<Stream>& GetSlaveStream() { return slaveStreams_; }
    HcclDispatcher GetDispatcher() const { return dispatcher_; }
    const std::string& GetGroupName() const { return identifier_; }
    void SetAlgType(u64 algType);
    void SetDebugMode(u8 debugMode);
    void SetSendRecvInfoPtr(void* sendRecvInfoPtr);
    void SetDumpDebug(bool dumpDebug);
    void SetIsDeviceMode(bool isDeviceMode) { isDeviceMode_ = isDeviceMode; }
    void SetUserStreamId(s32 userStreamId) { userStreamId_ = userStreamId; }
    HcclResult StreamTaskMonitor(void);
    HcclResult UpdateNotifyWaitTimeOut(SyncMode syncMode, u64 notifyWaitTime);
    HcclResult GetStreamAll(std::vector<Stream>& streams);
    u32 GetDevId() const { return devId_; }
    DfxExtendInfo* GetDfxExtendInfo() { return &dfxExtendInfo_; }
    DevType GetDevType(void) const { return topoInfo_.deviceType; }
    uint32_t GetRankSize(void) const { return topoInfo_.userRankSize; }
    HcclResult
    ExecOp(const std::string& newTag, const std::string& algName, OpParam& opParam, const HcclOpResParam* commParam);
    HcclResult GetAlgResponseRes(
        const std::string& newTag, const std::string& algName, const OpParam& opParam, const HcclOpResParam* commParam,
        std::unique_ptr<CollExecutorBase>& executor, AlgResourceResponse*& algResResponse);
    void PrepareOpRetryHandler(
        u8 inplaceSupportRetry, u8 retryEnable, u8 inPlaceSupportRetryStatus, u8 isInplacePreSync, u8 isPostSync);
    void NsCommStop();
    void NsCommClean();
    void SetAicpuRpcServer(u64 rpc) { rpc_ = rpc; }
    HcclResult GetSuspendingFlag(HcclComSuspendingFlag& flag);
    HcclResult BackGroundGetCmd(KfcCommand& cmd);
    HcclResult BackGroundSetStatus(KfcStatus status);
    void SetNsOpStatus(bool status) { isOpLaunch = status; }
    bool BackGroundGetOpStatus() { return isOpLaunch; }
    void SetNsStopLaunchStatus(bool status) { endStopLaunch = status; }
    bool GetNsStopLaunchStatus() { return endStopLaunch; }
    void SetCommInfoStreamStatus(bool status) { groupNsCommStatus_ = status; }
    bool GetCommInfoStreamStatus() const { return groupNsCommStatus_; }
    bool GetCommInfoStatus() const { return commOpenStatus; }
    HcclResult GetBackGroundCommand(BackgroundCommand& bgCmd);
    HcclResult ResponseBackGroundStatus(KfcExecStatus& status);
    HcclResult GetKfcCommand(KfcCommand& cmd);
    void SetCommRecoveryFlag(bool status) { commNeedsRecovery = status; }
    bool GetCommRecoveryFlag() { return commNeedsRecovery; }
    void RecordReportStatus(dfx::ReportStatus status);
    void GetReportStatusQueue(std::queue<dfx::ReportStatus>& reportStatusQue);
    HcclResult Orchestrate(
        const std::string& newTag, const std::string& algName, OpParam& param,
        std::unique_ptr<CollExecutorBase>& executor, AlgResourceResponse& algResource, const HcclOpResParam* commParam);
    HcclResult SaveTraceInfo(std::string& logInfo);
    HcclResult FlushUtraceInfo();
    std::string GetExcuteOp();
    void HandleCqeException(hccl::Stream& stream, bool isReadClear);
    void HandleIndOpCqe();
    void ReportIndOpCqe(hccl::Stream& stream, const rtLogicCqReport_t& cqeException, CqeStatus cqeStatus);
    static void ResetErrMsgReport() { errMessageReport_ = true; };
    void PrintTaskExceptionAllComm();
    HcclResult PrintTaskExceptionAllThreads();
    bool GetOpRetryEnable();
    void SetZeroCopyEnable(bool enable);
    void SetSymmetricMemoryEnable(bool enable);
    bool IsTaskExceptionForHccs();
    u32 HcclGetWaitStopExecCmdTimeout();
    u32 HcclGetWaitRetryCmdTimeout(uint32_t retryCnt);
    HcclResult UpdateOpExecStatus(HcclOpExecFSM& fsmState, KfcStatus state, KfcError& errorCode, uint32_t retryCnt);
    HcclResult ResetOpRetryException(HcclCMDType opType);
    HcclResult CleanAllRoceResource();
    // aclgraph 销毁时通过 KFC 投递清理 aicpu 端单一 tag 的全部关联资源，配合 host 端 AclgraphCallback 使用
    HcclResult ClearOpResource(const std::string& tag);
    HcclResult SwitchNic();
    HcclResult ResumeChangeLink();
    HcclResult ParseHierarchicalAlgOption(u32* ahcConfInfo);
    void RegisterKfcHandler(AicpuKfcHandlerType type, AicpuKfcHandler cb)
    {
        kfcHandlers_[static_cast<size_t>(type)] = cb;
    }
    HcclResult RecordHostOrder(
        const HcclOpResParam* commParam, const std::string& tag, u8 orderLaunchMode); // kernel占到核后，通知host侧
    std::string GetTaskExceptionTaskInfo(
        u32 sqHead, SqeRingBuffer* sqeContextBuffer, uint8_t& type, uint16_t& taskId, uint32_t& remoteRank);
    HcclResult RegisterProfCallBack();
    // 独立算子专用
    HcclResult
    SetChannelP2pNotify(TransportDeviceP2pData& transDevP2pData, u64& p2pNotifyNum, HcclChannelP2p& channelP2p);
    HcclResult SetChannelRoceNotify(
        TransportDeviceIbverbsData& transDevIbverbsData, u64& roceNotifyNum, HcclChannelRoce& channelRoce);
    HcclResult InitP2pChannel(HcclIndOpChannelRemoteResV3* commParam, uint32_t channelIndex);
    HcclResult InitRoceChannel(HcclIndOpChannelRemoteResV3* commParam, uint32_t channelIndex);
    HcclResult AllocChannelResource(HcclIndOpChannelRemoteResV3* commParam);

    HcclResult InitAicpuIndOp(CommAicpuParam* commAicpuParam);
    bool GetIsInitIndOp() { return indOpCommInitialized_; };
    HcclResult InitThreads(ThreadMgrAicpuParam* param);
    HcclResult NotifyFree(NotifyMgrAicpuParam* param);
    HcclResult NotifyAlloc(NotifyMgrAicpuParam* param);

    HcclResult RegisterOpInfo(void* opInfo, u32 size);
    HcclResult RegOpTaskException(HcommGetOpInfoCallback callback);
    HcclResult InitProfthreadResource(u32 threadNum);

    HcclResult SetDispatcherCtxOnThread();

private:
    HcclResult SetHrtWorkMode(const HcclOpResParam* commParam);
    HcclResult SetHrtDeviceSatMode(const HcclOpResParam* commParam);
    HcclResult InitSlaveStreamObjs(const HcclOpResParam* commParam);
    HcclResult InitLocalNotifyObj(const HcclOpResParam* commParam);
    HcclResult InitOpNotifyObj(const HcclOpResParam* commParam);
    HcclResult StreamRestore(u32 streamId); // 将流资源映射到custom进程
    HcclResult
    ParseTlvToVector(u64 srcTlv, u64 srcTlvTotalLength, std::vector<std::vector<std::vector<u32>>>& vectorInfo);
    HcclResult ParseTlvToSubGroupVector(
        u64 srcTlv, u64 srcTlvTotalLength, std::vector<std::vector<std::vector<std::vector<u32>>>>& vectorInfo);
    HcclResult InitLocalTagRes(const ListCommon& head, bool reAllocFlag = false);
    HcclResult InitRemoteTagRes(
        u32& rankId, const ListCommon& head, const std::string& newTag, u32 notifyNum,
        TransportLinkType linkType = TransportLinkType::RDMA);
    template <typename T>
    HcclResult InitAndVerifySignal(const HcclSignalInfo& signalInfo, std::vector<std::shared_ptr<T>>& notifyVec);
    HcclResult InitTopoMatcher();
    HcclResult InitTopoInfo(const HcclOpResParam* commParam);
    HcclResult InitCclbuffer(const HcclOpResParam* commParam);
    HcclResult InitConfigInfo(const HcclOpResParam* commParam);
    HcclResult InitMainStreamObj(const HcclOpResParam* commParam);
    HcclResult InitOrderStreamObj(const HcclOpResParam* commParam);
    HcclResult InitStreamObj(const HcclStreamParam& streamParam, Stream& stream);
    HcclResult InitTimeOutConfig(const HcclOpResParam* commParam);
    HcclResult InitHostDeviceLock(const HcclOpResParam* commParam);
    HcclResult InitOpRetry(const HcclOpResParam* commParam);
    HcclResult InitZeroCopyExchanger(const HcclOpResParam* commParam);
    HcclResult
    PrepareZeroCopyExchanger(const std::string& newTag, OpParam& opParam, AlgResourceResponse* algResResponse);
    HcclResult RegisterDispatcherCallback();
    HcclResult RegisterProfilingCallback();
    HcclResult InitUtraceInfo(const HcclOpResParam* commParam);
    void InitSendRecvOpId(const OpParam& param, HcclOpIdentifier& opId);
    HcclResult
    GetStreamData(const HcclStreamInfo& streamInfo, HcclComStreamInfo& comStreamInfo, u32& sqHead, u32& sqTail);
    HcclResult RefreshTransportsResForRank(
        const HcclOpResParam* commParam, u32 rankId, const std::string& newTag, u32 notifyNum,
        TransportLinkType linkType = TransportLinkType::RDMA);
    HcclResult GetRdmaLinksByRankAndTag(
        const HcclOpResParam* commParam, CommTransportsType type, u32 rankId, const std::string& newTag, LINK& link,
        bool isBackup, u32 notifyNum, bool isSecond);
    HcclResult GetSdmaLinksByRankAndTag(
        const HcclOpResParam* commParam, CommTransportsType type, u32 rankId, const std::string& newTag, LINK& link,
        bool isBackup, u32 notifyNum, TransportLinkType linkType = TransportLinkType::RESERVED);
    HcclResult AllocTransportResource(
        const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam,
        AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse);
    HcclResult IncreAllocTransportResource(
        const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam,
        AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse);
    HcclResult CreateLink(
        const std::string& newTag, TransportRequest& transportRequest, const HcclOpResParam* commParam, LINK& link,
        u32 notifyNum, bool isBackup, bool isSecond = false);
    HcclResult AllocLocalNotifysResource(
        const std::string& newTag, const HcclOpResParam* commParam, const u32 notifyNum,
        std::vector<std::shared_ptr<LocalNotify>>& notifiesMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifiesAux);
    HcclResult AllocStreamsResource(
        const std::string& newTag, const HcclOpResParam* commParam, const u32 streamNum, std::vector<Stream>& streams);
    HcclResult AllocScratchMemResource(
        const std::string& newTag, const HcclOpResParam* commParam, const u64& scratchMemSize, DeviceMem& scratchMem,
        bool reAllocFlag = false);
    HcclResult AllocAlgResource(
        const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam,
        AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse);
    HcclResult CalcResRequest(
        const std::string& algName, const OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
        AlgResourceRequest& resourceRequest);
    HcclResult WaitFinishWhileLoop(
        Stream& mainStream, std::vector<Stream>& subStreams, std::string& tag, const uint32_t& beginSqePos,
        OpParam& param);
    HcclResult CheckOpExecStatusCallback();
    HcclResult CheckOpExecStatus();
    HcclResult
    UpdateSuspendStatus(const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt);
    HcclResult CheckTaskTimeout(const Stream& mainStream, const uint64_t startUsec);

    HcclResult HcclOpExecFsmInitProcess(
        const std::string& newTag, OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState,
        KfcError& errorCode);
    bool HcclOpCheckSupportRetry(HcclCMDType opType);
    HcclResult HcclOpExecChangeLinkProcess(
        const std::string& newTag, HcclOpExecFSM& state, KfcError& errorCode, uint32_t& retryCnt,
        AlgResourceResponse& algResource, const HcclOpResParam* commParam, const OpParam& param);
    HcclResult HcclOpExecFsmLaunchProcess(
        const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
        AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t& beginSqePos,
        uint32_t& endSqePos, uint32_t retryCnt);
    HcclResult HcclOpExecFsmWaitEndProcess(
        OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode,
        uint32_t retryCnt, std::string& tag, const uint32_t& beginSqePos);
    HcclResult
    HcclOpExecFsmStoppingProcess(const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt);
    HcclResult HcclOpExecFsmStoppedProcess(
        HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt, const std::string& algName, OpParam& param,
        uint32_t beginSqePos, uint32_t endSqePos);
    HcclResult HcclOpExecFsmWaitRetryProcess(
        const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, KfcCommand& lastCmd);
    HcclResult ResetSqBuff();
    HcclResult CleanStreamFunc();
    HcclResult UpdateSqStatus(Stream& stream);
    HcclResult HcclOpExecFsmRetryProcess(
        const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
        AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t& retryCnt,
        uint32_t& beginSqePos, uint32_t& endSqePos);
    HcclResult RetryOrchestrateHcclOp(
        const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
        AlgResourceResponse& algResource, uint32_t& beginSqePos, uint32_t& endSqePos);
    HcclResult HcclOpExecFsmEndProcess(uint32_t retryCnt);
    std::string PrintInplaceSupportRetryStatus(InplaceSupportRetryStatus inPlaceSupportRetryStatus);
    bool HcclOpSupportRetry(const std::string& algName, bool retryEnable, OpParam& param);
    std::string PrintInplaceStatus(u8 isInplaceStatus);
    HcclResult SupportRetryWithInplaceCheck(const std::string& algName, OpParam& param);
    bool isPollutedZeroCopyOp(OpParam& param);
    bool HcclOpCheckNsRecovery();
    HcclResult OrchestrateHcclOp(
        const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
        AlgResourceResponse& algResource, uint32_t& beginSqePos, uint32_t& endSqePos);
    HcclResult LaunchSlaveStreamTask(AlgResourceResponse& algResource);
    HcclResult GetAlltoAllvSendRecvInfo(const void* sendRecvInfoPtr, HcclDataType sendType, HcclDataType recvType);
    HcclResult GetAlltoAllvcSendRecvInfo(const void* sendCountMatrix, HcclDataType sendType, HcclDataType recvType);
    HcclResult CheckSendRecvParams(const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo);
    HcclResult SetAlltoAllInputAndOutPutMem(OpParam& param, AlgResourceResponse& algResource);
    HcclResult NotifyPost(void);
    HcclResult NotifyWait(void);
    HcclResult GetAlltoAllTotalCount(OpParam& param, u64& sendCount, u64& recvCount);
    HcclResult GetAlltoAllVTotalCount(OpParam& param, u64& sendCount, u64& recvCount);
    HcclResult GetAlltoAllVCTotalCount(OpParam& param, u64& sendCount, u64& recvCount);

    // taskException
    void
    PollCqeException(hccl::Stream& stream, bool isReadClear, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus);
    void ExchangeCqeContext(
        hccl::Stream& stream, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus, ErrCqeContext& cqeCtx);
    void ReportErrCqe(hccl::Stream& stream, ErrCqeContext& cqeCtx);
    HcclResult PrintTaskExceptionAllStreams();
    bool IsRepeatedOpTaskException(u32 idx, SqeRingBuffer* sqeContextBuffer); // 避免同一个算子重复打印taskException
    std::string GetTaskExceptionOpInfo(u32 idx, SqeRingBuffer* sqeContextBuffer); // 打印算子参数信息
    void PrintTaskExceptionTaskQue(
        u32 sqIdx, SqeRingBuffer* sqeContextBuffer, bool isMonitor = false); // 打印当前位置的前序task
    std::string GetTaskBriefsInfo(u32 idx, SqeRingBuffer* sqeContextBuffer); // 打印task简写
    void PrintAicpuCommExecStatus();

    HcclResult UpdateOpRingBufferIdx();
    HcclResult CombineReportOpInfo(OpParam& param, bool isRetry, bool isRelay);
    void UpdateBSRRetryCnt();
    void ResetBSRRetryCnt();
    HcclResult CommitBSRStoredException(HcclOpExecFSM& fsmState, KfcError& errorCode);
    HcclResult QueryBatchSendRecvPairBeginPos();
    HcclResult QueryBatchSendRecvPairEndPos();
    HcclResult UpdateOpExecStatus(
        HcclOpExecFSM& fsmState, HcclOpIdentifier& opId, KfcStatus state, KfcError& errorCode, uint32_t retryCnt);
    u32 HcclUpdateBatchSendRecvOpIndex(std::map<u32, u32>& bsrIndexMap, u32 peerRank);
    u32 HcclUpdateBatchSendRecvOpIndex(HcclSendRecvType opType, u32 srcRank, u32 dstRank);
    HcclResult InitBatchSendRecvOpId(
        const OpParam& param, const HcclSendRecvItem* sendrecvPair, HcclOpIdentifier& opId, u32 streamId,
        AlgResourceResponse& algResource);
    HcclResult InitBatchSendRecvOpId(const OpParam& param, AlgResourceResponse& algResource);
    HcclResult InitBsrSendRecvOpIdAndExcuteOpId(
        OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode);
    void SetBSRSendOpExecException();
    void SetBSRRecvOpExecException();
    bool GetBSRSendOpExecException();
    bool GetBSRRecvOpExecException();

    HcclResult CleanStream(Stream& stream);
    HcclResult ClearStreamCqeException(Stream& stream);
    HcclResult ResetBSRSendOpExecException();
    HcclResult ResetBSRRecvOpExecException();
    HcclResult ResetBSRException();
    HcclResult BSRStopedProcess(HcclOpExecFSM& fsmState, KfcError& errorCode);
    HcclResult GetBSRRetryOpId(const OpParam& param, HcclOpIdentifier& targetOpId);
    HcclResult InitExecLoop(OpParam& param, std::unique_ptr<CollExecutorBase>& executor, u32& loopNum);
    template <typename T>
    HcclResult InitAndVerifySingleSignal(const HcclSignalInfo& signalInfo, std::shared_ptr<T>& notify);
    HcclResult SetTransportMachinePara(
        MachinePara& machinePara, u32& rankId, const std::string& newTag,
        TransportLinkType linkType = TransportLinkType::RESERVED);
    HcclResult CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes);
    HcclResult SetTagRemoteRes(u32& rankId, const std::string& tag, HccltagRemoteResV2* tagRes);
    HcclResult SetTransportPtpNotify(
        TransportDeviceP2pData& transDevP2pData, u64& p2pNotifyNum, HcclLinkP2pV2& linkP2p, u32 notifyNum);
    HcclResult
    SetTransportRoceQP(TransportDeviceIbverbsData& transDevIbverbsData, u64& roceQpNum, HcclLinkRoceV2* linkRoce);
    HcclResult SetTransportRoceNotify(
        TransportDeviceIbverbsData& transDevIbverbsData, u64& roceNotifyNum, HcclLinkRoceV2* linkRoce, u32 notifyNum);
    HcclResult InitLinkP2p(
        HccltagRemoteResV2* tagRes, u32& rankId, const std::string& newTag, u32 notifyNum,
        TransportLinkType linkType = TransportLinkType::RESERVED);
    HcclResult InitLinkRoce(
        HccltagRemoteResV2* tagRes, u32& rankId, const std::string& newTag, u32 notifyNum, const bool isBackup = false);
    HcclResult InitLinkRoce(
        HccltagRemoteResV2* tagRes, HcclLinkRoceV2* linkRoce, u32& rankId, const std::string& newTag, u32 notifyNum,
        const bool isBackup = false, const bool isSecond = false);
    HcclResult GetBsrTransportQpn(const HcclSendRecvItem* sendrecvPair, AlgResourceResponse& algResource, u32& qpn);
    HcclResult ReAllocTransportResource(
        const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap,
        const HcclOpResParam* commParam, const OpParam& param);
    HcclResult CleanRoceResource(
        const std::string& newTag, AlgResourceResponse& algResResponse, const std::map<u32, bool>& remoteRankPortMap,
        const OpParam& param);
    HcclResult LoadChangeLinkInfo(ChangeLinkInfo& changeLinkInfo);

    HcclResult AddRetryExecFlipTask(AlgResourceResponse& algResource);
    HcclResult ReportHcclTaskInfo(Stream& mainStream, std::vector<Stream>& subStreams);
    HcclResult ClearLocalBuff(Stream& mainStream, std::vector<Stream>& subStreams);
    HcclResult UpdateProfReportStartSqeIdx();
    HcclResult TasktypeTransferD2H(const uint8_t sqeType, TaskType& taskType);
    void PrepareMc2Handler();
    HcclResult InitOpCounter(const OpCounterInfo& opCounterInfo);

    // rts调用接口,通过mailbox上报给tsfw
    HcclResult SendTaskExceptionByMBox(const uint16_t& rsErrorCode);

    HcclResult RefreshLinkForSwitchNic(
        const std::string& newTag, const TransportRequest& transportRequest,
        const std::map<u32, bool>& remoteRankPortMap, bool isSecondBuild, LINK& switchLink);
    HcclResult ReAllocTransportForSwitchNic(
        const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap);
    HcclResult RefreshRoceTransportsForSwitchNic(std::unordered_map<std::string, OpCommTransport>& reservedLinks);
    HcclResult RevertTransportsForSwitchNic(std::unordered_map<std::string, OpCommTransport>& reservedLinks);
    HcclResult SwitchNicWaitHandleCommand(std::unordered_map<std::string, OpCommTransport>& reservedLinks);
    HcclResult SwitchNicWaitResult(std::unordered_map<std::string, OpCommTransport>& reservedLinks);
    u32 CalculateOpExecIndex(const OpParam& opParam, u32 userRank); // 每次展开时计算

    HcclResult InitProfResource();
    void InitCommInfoStatus(bool commInfo);
    HcclResult InitTinyMem(const HcclOpResParam* commParam);
    HcclResult SetStreamEnable(Stream& stream);
    HcclResult RefreshAlgResponseTransportRes(
        const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap,
        bool isChangeLinkFlag, const HcclOpResParam* commParam, const OpParam& param);
    HcclResult RefreshCommResponseTransportRes(std::map<u32, bool>& remoteRankPortMap);
    HcclResult PrintTaskExceptionByTaskId(u8 sqeType, u16 taskId, hccl::Stream& stream, u32 tail);
    bool IsNoNeedWait(void);
    void SetStreamCqeExceptionStatus(const Stream& stream, CqeExceptionStatus cqeStatus);
    void ResetStreamCqeExceptionStatus(const Stream& stream);
    CqeExceptionStatus GetStreamCqeExceptionStatus(const Stream& stream);
    HcclResult GenTaskExceptionInfo(u8 sqeType, hccl::Stream& stream, u32 head);
    HcclResult InvokeKfcHandler(AicpuKfcHandlerType type, const std::vector<u64> args);

    bool IsNoNeedMonitor(void);
    void InsertMonitorData(Stream& stream, HcclUs& curTime, u32 sqHead, uint16_t taskId, uint8_t type);
    bool IsNeedRefreshMonitorData(
        AicpuStreamMontior& streamMontior, HcclUs& curTime, uint32_t remoteRank, uint16_t taskId, u32 sqHead,
        u32 sqTail, uint8_t type);
    // 对称内存
    HcclResult PrepareSymmetricMemory(const OpParam& param, OpCommTransport& opTransportResponse);
    HcclResult PrepareSymmetricMemRanges(
        const AlgResourceResponse& algResource, uint64_t inputSize, uint64_t outputSize,
        std::vector<OpUnfoldMemRange>& userInputMemRanges, std::vector<OpUnfoldMemRange>& userOutputMemRanges);

    HcclResult CalSendRecvInfoForAlltoall(const OpParam& param);
    HcclResult CalSendRecvInfoFor910B(
        const std::string& algName, const OpParam& param, std::unique_ptr<CollExecutorBase>& executor);
    void HandleExistTagReAlloc(
        HccltagLocalResV2* tagRes, const std::string& tag, bool reAllocFlag, ListCommon*& curList, bool& needSkip);

    // 获取指定索引的算子信息
    const AicpuOpInfo* GetOpInfoFromSqIdx(u32 sqIdx, SqeRingBuffer* sqeContextBuffer);

    // 打印算子数据信息
    void PrintOpDataInfo(u32 sqIdx, SqeRingBuffer* sqeContextBuffer, bool isMonitor);

    // 打印task序列行
    void PrintTaskLine(bool isMonitor, u32 lineNum, u32 totalPrinted, const std::string& taskLine) const;

    // 更新算子上下文
    void UpdateOpContext(
        u32& opIndex, std::string& opTag, u32& lineCount, std::vector<std::string>& currentOpTasks, u32 newOpIndex,
        const std::string& newOpTag) const;

    // 准备下一行数据
    void PrepareNextLine(u32 opIndex, u32& lineCount, std::vector<std::string>& currentOpTasks) const;

    // 拼接task列表为字符串
    std::string ConcatTaskLine(const std::vector<std::string>& tasks) const;

    // 打印剩余未满行的tasks
    void PrintRemainingTasks(
        bool isMonitor, u32 lineCount, u32 printedCount, const std::vector<std::string>& currentOpTasks) const;

    std::unordered_map<s32, u32> opExecIndexMap_;

    // 管理aicpu和custom进程共享的数据
    AicpuShareDataManager aicpuShareData_;
    bool isCustom_ = false;

    // local资源
    std::vector<Stream> slaveStreams_;
    Stream mainStream_;
    Stream orderStream_;
    std::unordered_set<u32> streamToObj_; // 从context的资源构造为Stream对象去重
    s32 userStreamId_;                    // 用户传入的stream的id

    std::vector<std::shared_ptr<LocalNotify>> localNotifies_; // 主从流之间同步的notify
    std::vector<std::shared_ptr<LocalNotify>> opNotifies_;    // host与device间同步的notify
    std::vector<std::shared_ptr<LocalNotify>> orderNotifies_{AICPU_ORDER_NOTIFY_MAX_NUM, nullptr}; // 按序下发的notify
    std::unordered_set<u32> notifysToObj_; // 从context的资源构造为LocalNotify对象去重

    std::unordered_map<std::string, std::shared_ptr<DeviceMem>> tagScratchMem_; // 本地scratchmem
    std::unordered_map<std::string, std::unordered_set<u64>> localTagResToObj_; // 从context的localtag资源构造为对象去重

    // 跨卡资源
    uint32_t notifySize_;
    const HcclOpResParam* commParam_ = nullptr;

    // 通信域内的link
    std::unordered_map<u32, std::unordered_map<std::string, std::shared_ptr<Transport>>>
        linkRes_; // 通信域内的SDMA hccs链路，包括通用和专用
    // 通信域内的SDMA sio链路
    std::unordered_map<u32, std::unordered_map<std::string, std::shared_ptr<Transport>>> linkResSio_;
    std::unordered_map<u32, std::unordered_map<std::string, std::vector<std::shared_ptr<Transport>>>>
        linkRdmaRes_; // (主链路) 通信域内的RDMA链路，包括通用和专用
    std::unordered_map<u32, std::unordered_map<std::string, std::vector<std::shared_ptr<Transport>>>>
        linkRdmaResBackUp_; // (备链路) 通信域内的RDMA链路，包括通用和专用

    std::unordered_map<u32, std::unordered_map<std::string, HccltagRemoteResV3>>
        rankTagRemoteRes_; // 以rankid&tag粒度保存HccltagRemoteResV3
    std::unordered_map<u32, std::unordered_map<std::string, u32>>
        usedGeneralLinkNum_; // 记录已经被使用的通信域内通用链路数量
    std::unordered_map<u32, std::unordered_map<std::string, u32>>
        usedSpecialLinkNum_; // 记录已经被使用的通信域内根据tag构造的链路数量
    std::unordered_map<u32, std::unordered_map<std::string, u32>>
        usedGeneralLinkRdmaNum_; // 记录已经被使用的通信域内通用RDMA链路数量
    std::unordered_map<u32, std::unordered_map<std::string, u32>>
        usedSpecialLinkRdmaNum_; // 记录已经被使用的通信域内根据tag构造的RDMA链路数量
    std::unordered_map<std::string, AlgResourceResponse> resMap_;

    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank_;
    std::vector<std::vector<std::vector<u32>>> commPlaneVector_;
    std::vector<bool> isBridgeVector_;
    std::shared_ptr<PetersonLock> hostDeviceLock_;
    u32 devId_ = 0;
    HcclTopoInfo topoInfo_;
    HcclAlgoInfo algoInfo_;
    std::unique_ptr<TopoMatcher> topoMatcher_;
    HcclDispatcher dispatcher_{nullptr};
    DeviceMem cclInputBuffer_;
    DeviceMem cclOutputBuffer_;
    DeviceMem tinySendRecvMem_;
    u8 deterministic_ = 0; // 确定性开关
    bool dumpDebug_ = false;
    bool fftsEnable_ = false;       // ffts使能开关
    bool inlineReducEnable_ = true; // inline reduce使能
    bool interHccsDisable_ = false; // 使能RDMA
    u32 multiQpThreshold_{HCCL_MULTI_QP_THRESHOLD_DEFAULT};
    u8 debugMode_ = 0;       // debug开关
    AlgType algType_;        // 算法类型
    std::string identifier_; // 通信域名称
    u64 cclbufferSize_ = 0;
    u32 localUserRank_ = 0;
    HcclExternalEnable externalEnable_;
    std::unordered_map<u32, RankData> rankData_;
    std::unordered_map<u32, bool> receivedAcks_;
    u64 rpc_;
    std::chrono::milliseconds linkTimeOut_; // 发送超时时间
    // 重执行参数
    bool retryEnable_ = false;
    u32 retryHoldTime_ = 0;
    u32 retryIntervalTime_ = 0;
    bool isDeviceMode_ = false; // 区分aicpu和mc2，true表示mc2
    u32 mc2OpIndex_ = 0;        // mc2算子计数
    u32 hcclOpExecIndex_ = 0;   // hccl算子执行计数，下沉场景执行计数和下发计数不相等

    std::queue<dfx::ReportStatus> reportStatusQueue_;
    std::mutex reportQueueMutex_;
    // N秒快恢
    bool needsResponseStopLaunch_ = false; // aicpu测试用例下，主线程是否实现停止算子展开
    bool isOpLaunch = false;               // 算子是否初始化
    bool endStopLaunch = false;            // 主线程/背景线程接收到命令字，是否需要进行处理
    bool commOpenStatus = false;           // 通信域是否可以使用
    bool commNeedsRecovery = false;        // 多通信域下，该通信域是否有故障
    bool groupNsCommStatus_ = false;       // N秒快恢场景下，流是否被激活
    // 通用的通道
    std::shared_ptr<hccl::HDCommunicate> kfcControlTransferH2D_{nullptr};
    std::shared_ptr<hccl::HDCommunicate> kfcStatusTransferD2H_{nullptr};
    std::shared_mutex threadAicpuMutex_;
    DfxExtendInfo dfxExtendInfo_;
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo_;
    std::shared_ptr<AicpuZeroCopyExchanger> ZeroCopyExchanger_{nullptr};
    AicpuHdc aicpuHdc_;
    uint64_t groupHashId_{0};
    std::map<u32, CqeExceptionStatus> streamCqeExceptionStatus_; // < sqid, status>
    HcclSendRecvType bsrRetryOp_{HCCL_SEND_RECV_RESERVED};
    HcclOpIdentifier excuteOpId_;
    HcclOpIdentifier bsrSendOpId_;
    HcclOpIdentifier bsrRecvOpId_;
    HcclOpIdentifier bsrTargetOpId_;
    u32 bsrSendOpBeginSqePos_ = 0xFFFFFFFF;
    u32 bsrRecvOpBeginSqePos_ = 0xFFFFFFFF;
    u32 bsrSendOpEndSqePos_ = 0xFFFFFFFF;
    u32 bsrRecvOpEndSqePos_ = 0xFFFFFFFF;
    u32 bsrSendRetryCnt_ = 0;
    u32 bsrRecvRetryCnt_ = 0;
    bool bsrSendOpExecException_ = false;
    bool bsrRecvOpExecException_ = false;
    Stream bsrSendStream_;
    Stream bsrRecvStream_;
    std::vector<std::vector<HcclSendRecvItem*>> bsrSendRecvPairs_;
    // aicpu和custom进程单独对bsr send/recv的index进行计数，用于在重执行过程中保证send/recv的index一致
    std::map<u32, u32> bsrSendIndexMap_;
    std::map<u32, u32> bsrRecvIndexMap_;
    std::map<u32, AicpuStreamMontior> streamTaskMonitor_;

    bool isZeroCopy_{false};
    bool isSymmetricMemory_{false};
    hccl::AlgOpContext algOpContext_;
    std::unique_ptr<HcclTraceInfo> UtraceInfo_;
    // taskException
    bool printTaskExceptionForErr_ = false;                // true表示算子执行异常，需要打印taskException
    std::unordered_map<std::string, u32> opTaskException_; // 记录已经打印过taskException的算子信息
    // alltoall pipeline
    void* sendRecvInfoPtr_ = nullptr;
    uint64_t sqeWaitTimeOut_ = dfx::kKfcTimeOut;
    uint32_t taskMonitorInterval_ = 0;

    OpCounterInfo opCounterInfo_;
    std::mutex queryCqeMutex_;
    std::mutex preemptMutexForResMap_;
    static bool errMessageReport_;
    AicpuKfcHandler kfcHandlers_[static_cast<size_t>(AicpuKfcHandlerType::kMax)]{};

    bool initialized_{false};

    // 独立算子
    bool indOpCommInitialized_{false}; // 独立算子流程通信域是否初始化
    DispatcherCtxPtr dispatcherCtx_{nullptr};
    std::unordered_map<std::string, ChannelHandle> channelHandleMap_;
    std::unordered_map<ChannelHandle, std::shared_ptr<Transport>> linkMap_;
    std::vector<std::shared_ptr<Thread>> threads_;
    std::vector<std::unique_ptr<LocalNotify>> notifys_;
    TaskException taskExecption_;

    // A3消息语义算子展开aicpu cache
    AicpuCacheManager aicpuCacheManager_;

    // 维护aicpu算子展开的索引, 方便定位当前展开的算子信息
    size_t opUnfoldIdx_ = 0;
};
} // namespace hccl
#endif // __AICPU_COMMUNICATOR_H__
