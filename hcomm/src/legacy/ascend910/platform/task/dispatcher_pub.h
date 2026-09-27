/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_DISPATCHER_PUB_H
#define HCCL_INC_DISPATCHER_PUB_H

#include "dispatcher.h"
#include "adapter_hccp.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "../platform/common/dlprof_func.h"
#include "externalinput_pub.h"
#include "hccl_common.h"

#ifdef CCL_LLT
constexpr s64 HCCL_SDMA_MAX_COUNT_4GB = 0xC800000; // llt模块编译时设置SDMA最大数据量为200M
#else
constexpr s64 HCCL_SDMA_MAX_COUNT_4GB = 0x100000000; // SDMA任务最大数据量4GB
#endif

#if T_DESC("DispatcherPub", true)
namespace hccl {
struct HostNicTaskInfo {
    u32 streamId = 0;
    u32 taskId = 0;
    u64 notifyID = 0;
    std::string tag;
};
struct RaSendWrParams {
    QpHandle qpHandle;
    SendWrlistDataExt wr;
    SendWrRsp opRsp;
    HostNicTaskInfo taskInfo;
    void* dispatcherPtr = nullptr;
    HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    LoadTaskCallBack callback = nullptr;
    void* callBackUserPtr = nullptr;

    RaSendWrParams(
        QpHandle& qpHandle, SendWrlistDataExt& wr, void* dispatcherPtr, u32& streamId, u32& taskId, u64& notifyID,
        HcclWorkflowMode& workMode, LoadTaskCallBack callback, void* callBackUserPtr)
        : qpHandle(qpHandle),
          wr(wr),
          dispatcherPtr(dispatcherPtr),
          workMode(workMode),
          callback(callback),
          callBackUserPtr(callBackUserPtr)
    {
        opRsp = {};
        taskInfo.streamId = streamId;
        taskInfo.taskId = taskId;
        taskInfo.notifyID = notifyID;
    }
};
struct RaSocketParams {
    FdHandle socketFdHandle;
    void* socketBufferPtr;
    u64 socketBufferLen;
    void* ptr;
    u64 len;
    HostNicTaskInfo taskInfo;
    void* dispatcherPtr = nullptr;
    HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    s32 deviceLogicId; // 当前设备的device id
    NICDeployment nicDeploy = NICDeployment::NIC_DEPLOYMENT_RESERVED;
    LoadTaskCallBack callback = nullptr;
    void* callBackUserPtr = nullptr;

    RaSocketParams(
        FdHandle& socketFdHandle, const void* constSocketBufferPtr, u64 socketBufferLen, const void* constPtr, u64 len,
        void* dispatcherPtr, u32& streamId, u32& taskId, HcclWorkflowMode& workMode, s32 deviceLogicId,
        NICDeployment nicDeploy, LoadTaskCallBack callback, void* callBackUserPtr)
        : socketFdHandle(socketFdHandle),
          socketBufferPtr(const_cast<void*>(constSocketBufferPtr)),
          socketBufferLen(socketBufferLen),
          ptr(const_cast<void*>(constPtr)),
          len(len),
          dispatcherPtr(dispatcherPtr),
          workMode(workMode),
          deviceLogicId(deviceLogicId),
          nicDeploy(nicDeploy),
          callback(callback),
          callBackUserPtr(callBackUserPtr)
    {
        taskInfo.streamId = streamId;
        taskInfo.taskId = taskId;
    }
    RaSocketParams(const RaSocketParams& that)
        : socketFdHandle(that.socketFdHandle),
          socketBufferPtr(that.socketBufferPtr),
          socketBufferLen(that.socketBufferLen),
          ptr(that.ptr),
          len(that.len),
          taskInfo(that.taskInfo),
          dispatcherPtr(that.dispatcherPtr),
          workMode(that.workMode),
          deviceLogicId(that.deviceLogicId),
          nicDeploy(that.nicDeploy),
          callback(that.callback),
          callBackUserPtr(that.callBackUserPtr)
    {}
    RaSocketParams(const RaSocketParams&& that)
        : socketFdHandle(that.socketFdHandle),
          socketBufferPtr(that.socketBufferPtr),
          socketBufferLen(that.socketBufferLen),
          ptr(that.ptr),
          len(that.len),
          taskInfo(that.taskInfo),
          dispatcherPtr(that.dispatcherPtr),
          workMode(that.workMode),
          deviceLogicId(that.deviceLogicId),
          nicDeploy(that.nicDeploy),
          callback(that.callback),
          callBackUserPtr(that.callBackUserPtr)
    {}
};

using WrInformation = struct TagWrInfo {
    struct WrInfo wrData {};
    u64 type; // 默认 WqeType::WQE_TYPE_DATA
    u64 wrDataAddr;
    u32 notifyId;
    TagWrInfo() : type(0), wrDataAddr(0), notifyId(INVALID_UINT) { wrData = {}; }
};

struct RdmaTaskInfo {
    u32 remoteRank = INVALID_UINT;
    RdmaType rdmaType = RdmaType::RDMA_TYPE_RESERVED;
    std::vector<WrInformation> wrInfos;
};

class DispatcherPub {
public:
    explicit DispatcherPub(const s32 deviceLogicId);
    virtual ~DispatcherPub();

    virtual HcclResult Init(); // 初始化必要信息
    virtual HcclResult AddRetryPreamble(Stream& stream);
    virtual HcclResult StreamSync(Stream& stream);
    HcclResult SetNotifyWaitMode(SyncMode notifyWaitMode);
    SyncMode GetNotifyWaitMode();

    // 算法下发task时，不要使用HcclRtStream参数类型接口，需要改为hccl::Stream参数类型的接口
    HcclResult MemcpySync(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind);
    HcclResult MemcpyAsync(
        void* dst, uint64_t destMax, const void* src, u64 count, HcclRtMemcpyKind kind, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);
    HcclResult MemcpyAsync(hccl::HostMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream);
    HcclResult MemcpyAsync(hccl::HostMem& dst, const hccl::HostMem& src, hccl::Stream& stream);
    HcclResult MemcpyAsync(hccl::DeviceMem& dst, const hccl::HostMem& src, hccl::Stream& stream);
    HcclResult MemcpyAsyncWithoutCheckKind(
        void* dst, uint64_t destMax, const void* src, u64 count, HcclRtMemcpyKind kind, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);
    virtual HcclResult WaitValue(hccl::Stream& stream, u64 waitAddr, u64 valueAddr, bool reset);
    virtual HcclResult WriteValue(hccl::Stream& stream, u64 writeAddr, u64 valueAddr);
    virtual HcclResult MemcpyAsync(
        hccl::DeviceMem& dst, const hccl::DeviceMem& src, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);
    virtual HcclResult InlineReduceAsync(
        const void* src, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream, void* dst,
        u32 remoteUserRank = INVALID_VALUE_RANKID, hccl::LinkType inLinkType = hccl::LinkType::LINK_ONCHIP);
    virtual HcclResult ReduceAsync(
        const void* src, void* dst, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        HcclReduceType reduceType = HcclReduceType::HCCL_TBE_REDUCE);
    HcclResult ReduceAsync(
        const void* src, u64 dataCount, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream, void* dst,
        const u32 remoteUserRank, const hccl::LinkType linkType, const u64 reduceAttr)
    {
        return (INLINE_REDUCE_BITMASK & reduceAttr) ?
                   InlineReduceAsync(src, dataCount, datatype, redOp, stream, dst, remoteUserRank, linkType) :
                   ReduceAsync(src, dst, dataCount, datatype, redOp, stream);
    }

    virtual HcclResult SignalRecord(
        hccl::DeviceMem& dst, hccl::DeviceMem& src, hccl::Stream& stream, u32 remoteUserRank, hccl::LinkType inLinkType,
        u32 notifyId);
    virtual HcclResult RdmaRecord(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, RdmaType rdmaType, u32 userRank,
        u64 offset, u32 notifyId);

    // 下沉模式下的发送接口
    HcclResult
    RdmaSend(u32 qpn, u32 wqeIndex, const struct SendWr& wr, hccl::Stream& stream, u32 userRank = INVALID_VALUE_RANKID);
    HcclResult RdmaSend(u32 qpn, u32 wqeIndex, const struct SendWr& wr, hccl::Stream& stream, u32 userRank, u64 offset);

    // op base 模式下的发送接口
    virtual HcclResult RdmaSend(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream,
        u32 remoteUserRank = INVALID_VALUE_RANKID, bool isCapture = false);
    virtual HcclResult RdmaSend(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, hccl::Stream& stream, u32 userRank, u64 offset,
        bool isCapture = false);

    virtual HcclResult RdmaSend(u32 dbindex, u64 dbinfo, hccl::Stream& stream, RdmaTaskInfo& taskInfo);

    // host网卡模式下的rdma send
    HcclResult HostNicRdmaSend(
        QpHandle qpHandle, SendWrlistDataExt& wr, SendWrRsp& opRsp, hccl::Stream& stream,
        u32 userRank = INVALID_VALUE_RANKID, u64 offset = 0xFFFFFFFFFFFFFFFF);
    // host网卡模式下的tcp send
    HcclResult HostNicTcpSend(
        SocketHandle socketFdHandle, const void* socketBufferPtr, u64 socketBufferLen, const void* src, u64 len,
        hccl::Stream& stream, const NICDeployment nicDeploy);
    // host网卡模式下的tcp recv
    HcclResult HostNicTcpRecv(
        SocketHandle socketFdHandle, const void* socketBufferPtr, u64 socketBufferLen, const void* src, u64 len,
        hccl::Stream& stream, const NICDeployment nicDeploy);

    // host网卡模式下的tcp send处理线程
    void HostNicTcpSendThreadTask();
    // 下callback task：阻塞入队列，等待send线程将当前队列中send task执行完毕
    HcclResult HostNicTcpWaitSendCompletion(hccl::Stream& stream);
    // host网卡模式下的tcp send参数入队列
    HcclResult SetHostNicTcpSendThreadPara(void* fnData);
    void JudgeOpBaseTcpSendComplete(bool& closeSendThreadFlag);
    void WaitHostNicTcpSendThreadComplete();
    void WaitHostNicTcpSendTaskDone();
    void ClearHostNicRdmaParamsVec();
    void ClearHostNicTcpSendParamsVec();
    void ClearHostNicTcpRecvParamsVec();
    HcclResult DelHostNICRdmaTask(u32 streamID, u32 taskID);
    HcclResult DelHostNICTcpSendTask(u32 streamID, u32 taskID);
    HcclResult DelHostNICTcpRecvTask(u32 streamID, u32 taskID);
    HcclResult GetCallbackResult();
    HcclResult SetGlobalWorkSpace(std::vector<void*>& globalWorkSpaceAddr);
    HcclResult GetNotifyMaxWaitTime();
    HcclResult SetHcclExecTimeOut(s32 execTimeOut = NOTIFY_DEFAULT_WAIT_TIME);
    s32 GetExecTimeOut();
    bool GetExecTimeOutSet();
    virtual HcclResult SignalRecord(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool inchip = false, u64 signalAddr = INVALID_U64,
        u32 notifyId = INVALID_UINT);
    virtual HcclResult SignalWait(
        HcclRtNotify signal, hccl::Stream& stream, u32 userRank, u32 remoteUserRank, s32 stage = INVALID_VALUE_STAGE,
        bool inchip = false, u32 notifyId = INVALID_UINT, u32 timeOut = NOTIFY_INVALID_WAIT_TIME);

    virtual HcclResult SignalRecord(Stream& stream, u64 notifyId)
    {
        return SignalRecord(
            reinterpret_cast<HcclRtNotify>(notifyId), stream, INVALID_VALUE_RANKID, INVALID_U64, INVALID_VALUE_STAGE,
            true, INVALID_U64, INVALID_UINT);
    }
    virtual HcclResult SignalWait(Stream& stream, u32 notifyId, u32 timeOut)
    {
        return SignalWait(
            reinterpret_cast<HcclRtNotify>(notifyId), stream, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID,
            INVALID_VALUE_STAGE, true, INVALID_UINT, timeOut);
    }
    virtual HcclResult LaunchTasksEx([[maybe_unused]] Stream& stream, [[maybe_unused]] std::vector<Stream>& subStreams)
    {
        return HCCL_SUCCESS;
    }
    virtual HcclResult LaunchAllTasks() { return HCCL_SUCCESS; }
    virtual HcclResult ResetGraphCtx(
        [[maybe_unused]] bool enableCache, [[maybe_unused]] const std::string& key,
        [[maybe_unused]] bool useGraphConstructorV2)
    {
        return HCCL_SUCCESS;
    }
    virtual void SetNormalMode() { return; }

    virtual void RegLoadTaskCallBack(void* userPtr, LoadTaskCallBack callback)
    {
        callback_ = callback;
        callBackUserPtr_ = userPtr;
    }

    uint64_t GetMsprofSysCycleTime(void)
    {
        if (!GetIfProfile()) {
            return 0;
        }
        u64 ret = DlProfFunc::GetInstance().dlMsprofSysCycleTime();
        return ret;
    }

    static void ForceProf(bool isForce) { isForce_ = isForce; }

    static bool IsProfSubscribeAdditionInfo();

    virtual HcclResult SetMultiQpMode([[maybe_unused]] bool multiQpMode) { return HCCL_SUCCESS; }

    void SetHcclQos(u32 hcclQos);
    void SetMpamid(u32 mPamid);

    uint32_t GetHcclQos() { return hcclQos_; }

    inline bool IsPlaceholder() const { return isPlaceholder_; }

    inline void SetPlaceholder(const bool isPlaceholder)
    {
        isPlaceholder_ = isPlaceholder;
        return;
    }

protected:
    HcclResult RdmaSend(
        u32 qpn, u32 wqeIndex, const struct SendWr& wr, HcclRtStream stream, hccl::RdmaType rdmaType,
        u64 notifyID = INVALID_U64, bool isMainStream = false);
    HcclResult RdmaSend(
        u32 dbindex, u64 dbinfo, const struct SendWr& wr, HcclRtStream stream, hccl::RdmaType rdmaType,
        u64 notifyID = INVALID_U64, u64 offset = 0, bool isMainStream = false);
    HcclResult SignalRecord(
        HcclRtNotify signal, HcclRtStream stream, u32 userRank, u64 offset = INVALID_U64,
        s32 stage = INVALID_VALUE_STAGE, bool isMainStream = false);
    HcclResult SignalWait(
        HcclRtNotify signal, HcclRtStream stream, u32 userRank, u32 remoteUserRank, s32 stage = INVALID_VALUE_STAGE,
        u32 timeOut = NOTIFY_INVALID_WAIT_TIME, bool isMainStream = false);
    HcclResult TbeReduceAsync(
        const void* src1, const void* src2, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        const void* dst);
    u32 GetNotifyWaitTime(u32 timeOut);
    HcclResult DevMemMalloc(void* stream, void*& devMem1, void*& devMem2);
    HcclResult JudgeIsTail(
        const void* src1, const void* src2, const void* dst, u64 count, const HcclDataType dataType, u64& headCount,
        u64& tailCount, void*& tailSrc1, void*& tailSrc2, void*& tailDst);

    s32 deviceLogicId_; // 当前设备的device id
    std::mutex mutex_;

    SyncMode notifyWaitMode_;
    std::map<u32, std::queue<std::unique_ptr<RaSendWrParams>>> hostNicRdmaParamsVec_;
    std::map<u32, std::queue<std::unique_ptr<RaSocketParams>>> hostNicTcpSendParamsVec_; // host网卡tcp模式下存放发task
    std::map<u32, std::queue<std::unique_ptr<RaSocketParams>>> hostNicTcpRecvParamsVec_; // host网卡tcp模式下存放收task
    std::unique_ptr<RaSocketParams> hostNicTcpSendThreadParam_;
    std::unique_ptr<std::thread> hostNicTcpSendThread_;
    bool hostNicTcpSendThreadState_;
    std::mutex hostNicMutex_;
    void* overflowAddr_;
    void* fftsPubInfo_{nullptr};
    bool setDeviceFlag_;
    uint32_t notifyMaxWaitTime_ = NOTIFY_DEFAULT_WAIT_TIME;
    LoadTaskCallBack callback_{nullptr};
    void* callBackUserPtr_{nullptr};
    std::map<int32_t, void*> devMemMap_; // streamId和device内存的map
    std::mutex devMemMutex_;
    static bool isForce_; // 强制profiling上报或缓存
    s32 execTimeOut_;
    bool execTimeOutByConfig_;
    uint32_t hcclQos_ = SDMA_QOS_DEFAULT;
    uint32_t mPamid_ = 0;
    bool isPlaceholder_ = false; // 用于区分是否生成placeholder SQE还是正常SQE

private:
    void SetupTaskParaDma(
        hccl::TaskPara& taskPara, hccl::TaskParaDMA& para, TaskType taskType, ProfilerType profilerType,
        hccl::Stream& stream, u64 beginTime, bool isMainStream) const;
    void SetupTaskParaDma(
        hccl::TaskPara& taskPara, hccl::TaskParaDMA& para, TaskType taskType, HcclRtStream stream, u64 beginTime,
        bool isMainStream) const;
    HcclResult DealTbeReduce(
        const void* src1, const void* src2, u64 count, const HcclDataType datatype, HcclReduceOp redOp, Stream& stream,
        const void* dst);
};
} // namespace hccl
#endif
#endif //  HCCL_INC_DISPATCHER_PUB_H
