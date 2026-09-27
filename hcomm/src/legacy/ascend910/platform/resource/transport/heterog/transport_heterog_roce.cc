/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_heterog_roce.h"
#include "log.h"
#include "externalinput_pub.h"
#include "mr_manager.h"
#include "adapter_hccp.h"
#include "adapter_rts.h"
#include "network_manager_pub.h"
#include "transport_ibverbs_pub.h"
#include "hccl_socket.h"

using namespace std;
namespace hccl {
constexpr u32 MAX_COSTTIME_COUNT = 1000;
constexpr u32 MAX_TOTALCOST_COUNT = 10000; // 总耗时预警门槛 10ms
constexpr u32 BLOCK_ALLOCATOR_POOL_SIZE = 4096;
constexpr s32 PROTOCOL_TYPE = 0;
constexpr s32 LINK_NUM = 3;
constexpr u32 SOCKET_FOR_TAG_QP = 0;
constexpr u32 SOCKET_FOR_DATA_QP = 1;
constexpr u32 SOCKET_FOR_SENDRECV_QP = 2;

constexpr u32 RECV_WQE_HDC_BATCH_NUM = 128;
constexpr u32 RECV_WQE_HDC_BATCH_SUPPLEMENT = 1;
constexpr u32 RECV_WQE_NUM_THRESHOLD = 96;
constexpr u32 RECV_WQE_BATCH_SUPPLEMENT = 96;
constexpr u32 SMALL_PAGE_SIZE = 4096;
constexpr u32 EIGHE_BIT = 8;
constexpr u32 WAIT_SLEEP_TIME_US = 50;

enum class RdmaOp { OP_WRITE = 0, OP_SEND = 2, OP_READ = 4 };

TransportHeterogRoce::TransportHeterogRoce(
    const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
    const TransportResourceInfo& transportResourceInfo)
    : TransportHeterog(transTag, selfIp, peerIp, peerPort, selfPort, transportResourceInfo),
      nicRdmaHandle_(nullptr),
      mrManager_(transportResourceInfo.mrManager.get()),
      blockMemLkey_(transportResourceInfo.lkey),
      recvWqeBatchNum_(RECV_WQE_BATCH_NUM),
      recvWqeBatchThreshold_(RECV_WQE_NUM_THRESHOLD),
      recvWqeBatchSupplement_(RECV_WQE_BATCH_SUPPLEMENT),
      access_(RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE),
      tagRecvWqeNum_(0),
      dataRecvWqeNum_(0),
      dataRecvWqeExpNum_(0),
      memBlocksManager_(transportResourceInfo.memBlocksManager),
      pRecvWrInfosMem_(transportResourceInfo.pRecvWrInfosMem),
      deviceEvePtr_(nullptr),
      deviceEveLkey_(0),
      useDevMem_(true),
      isRawConn_(transportResourceInfo.isRawConn)
{
    GetTransportResourceInfo(transportResourceInfo);
}

TransportHeterogRoce::TransportHeterogRoce(const TransportResourceInfo& transportResourceInfo)
    : TransportHeterog(transportResourceInfo),
      nicRdmaHandle_(nullptr),
      mrManager_(transportResourceInfo.mrManager.get()),
      blockMemLkey_(transportResourceInfo.lkey),
      recvWqeBatchNum_(RECV_WQE_BATCH_NUM),
      recvWqeBatchThreshold_(RECV_WQE_NUM_THRESHOLD),
      recvWqeBatchSupplement_(RECV_WQE_BATCH_SUPPLEMENT),
      access_(RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE),
      tagRecvWqeNum_(0),
      dataRecvWqeNum_(0),
      dataRecvWqeExpNum_(0),
      memBlocksManager_(transportResourceInfo.memBlocksManager),
      pRecvWrInfosMem_(transportResourceInfo.pRecvWrInfosMem),
      deviceEvePtr_(nullptr),
      deviceEveLkey_(0),
      useDevMem_(true),
      isRawConn_(transportResourceInfo.isRawConn)
{
    GetTransportResourceInfo(transportResourceInfo);
}

TransportHeterogRoce::~TransportHeterogRoce() {}

u64 HostAddrToDev(const u64& hostAddr, u64 hostAddrBegin, u64 devAddrBegin)
{
    u64 devAddr = hostAddr - hostAddrBegin + devAddrBegin;
    return devAddr;
}

HcclResult TransportHeterogRoce::Init()
{
    if (!isHdcMode_ && (remoteIsHdc_ && (deviceLogicId_ == HOST_DEVICE_ID))) {
        HCCL_INFO("TransportHeterogRoce no useDevMem_");
        useDevMem_ = false;
    }

    CHK_RET(CheckRecvMsgAndRequestBuffer());

    CHK_RET(GetNetworkResource());

    CHK_RET(PreQpConnect());

    CHK_RET(InitTransportConnect(PROTOCOL_TYPE, LINK_NUM));

    CHK_RET(ConnectAsync());

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::Deinit()
{
    if (isDeinited_ == true) {
        return HCCL_SUCCESS;
    }

    if (isHdcMode_) {
        if (deviceLogicId_ == HOST_DEVICE_ID) {
            CHK_PRT(MemBlocksManagerDeInit());
        }
        CHK_PRT(MrManagerDeInit());
        if (deviceEvePtr_ != nullptr) {
#ifndef CCL_KERNEL
            CHK_RET(HrtDevFree(deviceEvePtr_));
#endif
        }
    }

    CHK_RET(DeleteNotifyValueBuffer());

    CHK_RET(DestroyCqAndQp());

    CHK_RET(SocketClose());

    isDeinited_ = true;
    return HCCL_SUCCESS;
}

HcclResult
TransportHeterogRoce::Isend(const TransData& sendData, const TransportEndPointParam& epParam, HcclRequestInfo*& request)
{
    CHK_RET(GenerateSendRequest(sendData, epParam, request));

    u32 lkey = 0;
    CHK_RET(RegMr(
        reinterpret_cast<void*>(sendData.srcBuf), static_cast<u64>(sendData.count * SIZE_TABLE[sendData.dataType]),
        lkey));
    HCCL_DEBUG(
        "addr[%llu] count[%d] datatype[%s]", sendData.srcBuf, sendData.count,
        GetDataTypeEnumStr(sendData.dataType).c_str());

    HcclEnvelope envelope(
        request->transportRequest.protocol, request->transportRequest.transData, request->transportRequest.epParam,
        lkey, request->transportRequest.msn);

    // 如果建链未完成，或者积压的信封未发送完成，则Isend不进行信封发送。
    // Test接口中推动积压信封发送完成后，Isend接口才启动信封发送。
    std::unique_lock<std::mutex> lock(envelopeBacklogQueueLock_);
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        envelopeBacklogQueue_.push(envelope);
        return HCCL_SUCCESS;
    }

    return SendEnvelope(envelope);
}

HcclResult TransportHeterogRoce::Send(
    [[maybe_unused]] const TransData& sendData, [[maybe_unused]] const TransportEndPointParam& epParam)
{
    HCCL_ERROR("TransportHeterogRoce::Send is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

HcclResult TransportHeterogRoce::Improbe(
    const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status,
    [[maybe_unused]] bool& flag)
{
    return Improbe(epParam, matched, msg, status);
}

HcclResult TransportHeterogRoce::Improbe(
    [[maybe_unused]] const TransportEndPointParam& epParam, s32& matched, HcclMessageInfo*& msg, HcclStatus& status)
{
    // 建链未完成时，返回未匹配到
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(ConnectAsync());
        return ProbeNothing(matched, msg, status);
    }

    // 先检查本地能否匹配
    HcclEnvelopeSummary envelopInfo;
    bool envelopeExist = GetSavedEnvelope(envelopInfo);

    auto probeSomething = [&]() -> HcclResult {
        CHK_RET(GenerateRecvMessage(envelopInfo, msg, status));
        matched = HCCL_IMPROBE_COMPLETED;
        return HCCL_SUCCESS;
    };

    if (envelopeExist) {
        return probeSomething();
    }
    // 再拉取roce cqe，检查是否能匹配
    CHK_RET(PullRecvRequestStatus());

    envelopeExist = GetSavedEnvelope(envelopInfo);
    if (envelopeExist) {
        return probeSomething();
    } else {
        return ProbeNothing(matched, msg, status);
    }
}

HcclResult
TransportHeterogRoce::Iwrite(const TransData& sendData, const HcclEnvelope& envelope, HcclRequestInfo*& request)
{
    if (isHdcMode_ && dataQpInfo_.qpMode != NORMAL_QP_MODE) {
        CHK_RET(TransportHeterog::WaitBuildLinkComplete());
    }

    TransportEndPointParam epParam{};
    CHK_RET(GenerateSendRequest(sendData, epParam, request));
    request->transportRequest.requestType = HcclRequestType::HCCL_REQUEST_RECV;

    u32 lkey = 0;
    CHK_RET(RegMr(
        reinterpret_cast<void*>(sendData.srcBuf), static_cast<u64>(sendData.count * SIZE_TABLE[sendData.dataType]),
        lkey, false));

    bool tmp = true;
    CHK_RET(GetQpStatus(tmp));

    if (!isHdcMode_ || dataQpInfo_.qpMode == NORMAL_QP_MODE) {
        dataWriteSge_.addr = static_cast<uint64_t>(sendData.srcBuf);
        dataWriteSge_.length = envelope.transData.count * SIZE_TABLE[envelope.transData.dataType];
        dataWriteSge_.lkey = lkey;
        dataWriteWr_.wr_id = reinterpret_cast<uint64_t>(request);
        dataWriteWr_.wr.rdma.remote_addr = static_cast<uint64_t>(envelope.transData.dstBuf);
        dataWriteWr_.wr.rdma.rkey = envelope.key;

        struct ibv_send_wr* badWr = nullptr;
        HCCL_INFO(
            "rdma write: remote addr[%llu] count[%d] datatype[%s] wrId[%llu]",
            reinterpret_cast<u64>(envelope.transData.dstBuf), envelope.transData.count,
            GetDataTypeEnumStr(envelope.transData.dataType).c_str(), dataWriteWr_.wr_id);
        CHK_RET(hrtIbvPostSend(dataQpInfo_.qp, &dataWriteWr_, &badWr));
        // 写notify
        if (deviceLogicId_ == HOST_DEVICE_ID) {
            Stream tmpStream(nullptr);
            CHK_RET(RecordNotify(tmpStream, RdmaNotifyOp::SEND_NOTIFY, dataWriteWr_.wr_id));
        }
    } else {
        struct SgList list = {};
        u64 srcBufDevAddr = 0;
        CHK_RET(dataQpMrManager_->GetDevVirAddr(
            reinterpret_cast<void*>(sendData.srcBuf),
            static_cast<u64>(envelope.transData.count * SIZE_TABLE[envelope.transData.dataType]), srcBufDevAddr));

        list.addr = srcBufDevAddr;
        list.len = envelope.transData.count * SIZE_TABLE[envelope.transData.dataType];
        list.lkey = lkey;

        struct SendWrV2 wr {};
        wr.wrId = reinterpret_cast<uint64_t>(request);
        HCCL_INFO("iwrite wr.wrId[%llu]", wr.wrId);
        wr.bufList = &list;
        wr.bufNum = 1;
        wr.dstAddr = static_cast<uint64_t>(envelope.transData.dstBuf);
        wr.rkey = envelope.key;
        wr.op = static_cast<u32>(RdmaOp::OP_WRITE);
        wr.sendFlag = RA_SEND_FENCE;
        struct SendWrRsp opRsp = {};
        CHK_RET(HrtRaSendWrV2(dataQpInfo_.qpHandle, &wr, &opRsp, GetWorkflowMode()));
        CHK_RET(DoorBellSend(dataQpInfo_.qpMode, opRsp));

        // 写notify
        if (deviceLogicId_ == HOST_DEVICE_ID) {
            Stream tmpStream(nullptr);
            CHK_RET(RecordNotify(tmpStream, RdmaNotifyOp::SEND_NOTIFY, wr.wrId));
        }

        s32 writeAndNotifyFlag = HCCL_TEST_INCOMPLETED;
        TIME_PRINT(CHK_RET(this->Wait(*request, writeAndNotifyFlag)));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::Imrecv(
    const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request, [[maybe_unused]] bool flag,
    [[maybe_unused]] bool needRecordFlag)
{
    HcclResult ret = Imrecv(recvData, msg, request);
    return ret;
}

HcclResult TransportHeterogRoce::Imrecv(const TransData& recvData, HcclMessageInfo& msg, HcclRequestInfo*& request)
{
    CHK_RET(GenerateRecvRequest(recvData, msg, request));

    u32 lkey = 0;
    CHK_RET(RegMr(
        reinterpret_cast<void*>(recvData.dstBuf), static_cast<u64>(recvData.count * SIZE_TABLE[recvData.dataType]),
        lkey, false));

    HcclEnvelope& envelope = msg.envelope.envelope;

    if (!isHdcMode_ || dataQpInfo_.qpMode == NORMAL_QP_MODE) {
        dataReadSge_.addr = static_cast<uint64_t>(recvData.dstBuf);
        dataReadSge_.length = envelope.transData.count * SIZE_TABLE[envelope.transData.dataType];
        dataReadSge_.lkey = lkey;
        dataReadWr_.wr_id = reinterpret_cast<uint64_t>(request);
        dataReadWr_.wr.rdma.remote_addr = static_cast<uint64_t>(envelope.transData.srcBuf);
        dataReadWr_.wr.rdma.rkey = envelope.key;
        dataReadWr_.next = nullptr;

        if (!(remoteIsHdc_ && (deviceLogicId_ == HOST_DEVICE_ID))) {
            HCCL_INFO("general server not load ack ");
            dataReadWr_.next = &dataAckWr_;
            dataAckSge_.addr = reinterpret_cast<uint64_t>(&envelope.msn);
            dataAckSge_.length = sizeof(uint64_t);
            dataAckSge_.lkey = 0;
            dataAckWr_.wr_id = 0;
        }

        struct ibv_send_wr* badWr = nullptr;
        HCCL_INFO(
            "rdma read: remote addr[%llx] count[%d] datatype[%s] wrId[%llu]",
            reinterpret_cast<u64>(envelope.transData.srcBuf), envelope.transData.count,
            GetDataTypeEnumStr(envelope.transData.dataType).c_str(), dataReadWr_.wr_id);
        CHK_RET(hrtIbvPostSend(dataQpInfo_.qp, &dataReadWr_, &badWr));
    } else {
        if (envelope.transData.count == 0) {
            request->transportRequest.transData.count = 0;
            CHK_RET(FreeRecvMessage(msg));
            return HCCL_SUCCESS;
        }

        struct SgList list = {};
        u64 devAddr = 0;
        CHK_RET(dataQpMrManager_->GetDevVirAddr(
            reinterpret_cast<void*>(recvData.dstBuf), static_cast<u64>(recvData.count * SIZE_TABLE[recvData.dataType]),
            devAddr));
        list.addr = static_cast<uint64_t>(devAddr);
        list.len = envelope.transData.count * SIZE_TABLE[envelope.transData.dataType];
        list.lkey = lkey;

        struct SendWrV2 wr = {};
        wr.wrId = reinterpret_cast<uint64_t>(request);

        HCCL_INFO("Imrecv wr.wrId[%llu]", wr.wrId);
        wr.bufList = &list;
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = static_cast<uint64_t>(envelope.transData.srcBuf);
        wr.rkey = envelope.key;
        wr.op = static_cast<u32>(RdmaOp::OP_READ);
        wr.sendFlag = RA_SEND_SIGNALED | RA_SEND_FENCE;
        struct SendWrRsp opRsp = {};
        CHK_RET(HrtRaSendWrV2(dataQpInfo_.qpHandle, &wr, &opRsp, GetWorkflowMode()));
        CHK_RET(DoorBellSend(dataQpInfo_.qpMode, opRsp));

        s32 imrecvFlag = HCCL_TEST_INCOMPLETED;
        TIME_PRINT(CHK_RET(this->Wait(*request, imrecvFlag)));
    }

    CHK_RET(FreeRecvMessage(msg));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::Test(HcclRequestInfo& request, s32& flag, HcclStatus& compState)
{
    if (isHdcMode_ && dataQpInfo_.qpMode != NORMAL_QP_MODE) {
        flag = HCCL_TEST_COMPLETED;
        HCCL_INFO("TransportHeterogRoce QueryRequestStatus: flag [%d]", flag);
        compState.error = 0;
        CHK_RET(FreeRequest(request));
        return HCCL_SUCCESS;
    }

    // 建链未完成时，继续推进建链流程；
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(ConnectAsync());
    }

    CHK_RET(PullSendOrRecvStatus(request));

    return QueryRequestStatus(request, flag, compState);
}

HcclResult TransportHeterogRoce::PullSendOrRecvStatus(const HcclRequestInfo& request)
{
    if ((GetState() != ConnState::CONN_STATE_COMPLETE) && (GetState() != ConnState::CONN_STATE_FLUSH_QUEUE)) {
        return HCCL_SUCCESS;
    }

    if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_SEND) {
        CHK_RET(PullSendStatus());
    } else if (request.transportRequest.requestType == HcclRequestType::HCCL_REQUEST_RECV) {
        CHK_RET(PullRecvStatus());
    } else {
        HCCL_ERROR("[HcclTest] requestType[%u] is invalid", request.transportRequest.requestType);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::Wait(HcclRequestInfo& request, s32& flag)
{
    // 建链未完成时，继续推进建链流程；
    if (GetState() != ConnState::CONN_STATE_COMPLETE) {
        CHK_RET(ConnectAsync());
    }

    auto startTime = chrono::steady_clock::now();
    auto timeout = chrono::seconds(GetExternalInputHcclLinkTimeOut());

    while ((flag != HCCL_TEST_COMPLETED) && ((chrono::steady_clock::now() - startTime) < timeout)) {
        CHK_RET(PullSendOrRecvStatus(request));

        if (request.transportRequest.status >= 0) {
            flag = HCCL_TEST_COMPLETED;
            HCCL_INFO("QueryRequestStatus: flag[%d]", flag);
            return HCCL_SUCCESS;
        }

        SaluSleep(WAIT_SLEEP_TIME_US);
    }

    HCCL_ERROR("Wait Cqe timeOut[%d] s, State[%d]", GetExternalInputHcclLinkTimeOut(), GetState());

    return HCCL_E_TIMEOUT;
}

HcclResult TransportHeterogRoce::QueryRequestStatus(HcclRequestInfo& request, s32& flag, HcclStatus& compState)
{
    if (request.transportRequest.status >= 0) {
        // 该request已完成
        flag = HCCL_TEST_COMPLETED;
        HCCL_INFO("QueryRequestStatus: flag [%d]", flag);
        compState.tag = request.transportRequest.epParam.src.tag;
        compState.srcRank = request.transportRequest.epParam.src.rank;
        compState.error = request.transportRequest.status;
        CHK_RET(FreeRequest(request));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PullSendStatus(bool allowNotify)
{
    if (isHdcMode_ && (tagQpInfo_.qpMode == OFFLINE_QP_MODE || tagQpInfo_.qpMode == OFFLINE_QP_MODE_EXT)) {
        return HCCL_SUCCESS;
    }

    struct ibv_wc wcTagCq[HCCL_POLL_CQ_DEPTH];
    s32 tagCqNum = 0;
    CHK_RET(PollCq(tagQpInfo_, true, tagCqNum, wcTagCq));
    for (int i = 0; i < tagCqNum; i++) {
        if (wcTagCq[i].status != 0) {
            CHK_RET(ParseErrorTagSqe(wcTagCq, i));
            HCCL_ERROR("rdma poll tag sq failed, cqe status[%u]", wcTagCq[i].status);
            return HCCL_E_NETWORK;
        }
    }
    struct ibv_wc wcDataRq[HCCL_POLL_CQ_DEPTH];
    s32 dataRqNum = 0;

    CHK_RET(PollCq(dataQpInfo_, false, dataRqNum, wcDataRq));
    if ((dataRqNum == 0) && allowNotify) {
        CHK_RET(hrtIbvReqNotifyCq(dataQpInfo_.recvCq, 0));
    } else {
        HCCL_DEBUG("data rq: poll cq num:%d", dataRqNum);
        CHK_RET(ParseDataRqes(wcDataRq, dataRqNum));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PullRecvRequestStatus(bool allowNotify)
{
    struct ibv_wc wc[HCCL_POLL_CQ_DEPTH];
    s32 num = 0;
    CHK_RET(PollCq(tagQpInfo_, false, num, wc));
    if ((num == 0) && allowNotify) {
        CHK_RET(hrtIbvReqNotifyCq(tagQpInfo_.recvCq, 0));
    } else {
        HCCL_DEBUG("tag rq: poll cq num:%d", num);
        CHK_RET(ParseTagRqes(wc, num));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PullRecvStatus(bool allowNotify)
{
    HCCL_INFO("Pull dataQp RecvStatus");
    struct ibv_wc wc[HCCL_POLL_CQ_DEPTH];
    s32 num = 0;
    CHK_RET(PollCq(dataQpInfo_, true, num, wc));
    if ((num == 0) && allowNotify) {
        CHK_RET(hrtIbvReqNotifyCq(dataQpInfo_.sendCq, 0));
    } else {
        HCCL_DEBUG("data sq: poll cq num:%d", num);
        CHK_RET(ParseDataSqes(wc, num));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::ParseTagRqes(const struct ibv_wc* wc, int num)
{
    for (int i = 0; i < num; i++) {
        HCCL_INFO("rq cqe info: wrId[%llu] status[%u] opcode[%u]", wc[i].wr_id, wc[i].status, wc[i].opcode);
        CHK_PRT_RET(
            wc[i].status != 0,
            HCCL_ERROR(
                "rdma send failed, cqe status[%u] wrId[%llu] opcode[%u]", wc[i].status, wc[i].wr_id, wc[i].opcode),
            HCCL_E_INTERNAL);
        RecvWrInfo* info = reinterpret_cast<RecvWrInfo*>(wc[i].wr_id);
        CHK_PTR_NULL(info);

        TransportHeterogRoce* transportPtr = reinterpret_cast<TransportHeterogRoce*>(info->transportHandle);
        CHK_PTR_NULL(transportPtr);
        CHK_RET(transportPtr->SupplyTagRecvWqe());
        HcclEnvelope* envelope = nullptr;
        if (useDevMem_ && (deviceLogicId_ == HOST_DEVICE_ID)) {
#ifndef CCL_KERNEL
            // 根据device内存求host内存
            CHK_RET(hrtSetDevice(index_));
            u64 uDevPtr = reinterpret_cast<u64>(info->buf);
            void* devPtr = reinterpret_cast<void*>(uDevPtr);
            u64 uHostPtr = uDevPtr - reinterpret_cast<uint64_t>(deviceEvePtr_) + hostAddrBegin_;
            void* hostPtr = reinterpret_cast<void*>(uHostPtr);
            HCCL_DEBUG("ParseTagRqes devPtr[%p][%llu] hostPtr[%p][%llu]", devPtr, uDevPtr, hostPtr, uHostPtr);
            CHK_RET(hrtMemcpy(
                hostPtr, MEM_BLOCK_SIZE, devPtr, MEM_BLOCK_SIZE, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
            envelope = reinterpret_cast<HcclEnvelope*>(hostPtr);
            // ps侧DbSend对当前线程SetDevice后，改变了原GE通信域初始化时setdevice 0
            // 若后面使用本线程save会导致getctx失败获取不到通信域句柄，所以需要在此处重新set回默认
            CHK_RET(hrtSetDevice(0));
#endif
        } else {
            envelope = reinterpret_cast<HcclEnvelope*>(info->buf);
        }
        CHK_PTR_NULL(envelope);

        HCCL_INFO(
            "recv request: tag:%d srcRank:%u dstRank:%u status:%u msn:0x%016llx count:%d", envelope->epParam.src.tag,
            envelope->epParam.src.rank, envelope->epParam.dst.rank, wc[i].status, envelope->msn,
            envelope->transData.count);
        HcclEnvelopeSummary envelopSummary(*envelope, wc[i].status);
        transportPtr->SaveEnvelope(envelopSummary);
        CHK_RET(transportPtr->FreeMemBlock(envelope));
        CHK_RET(transportPtr->FreeRecvWrId(wc[i].wr_id));
    }
    return HCCL_SUCCESS;
}

void TransportHeterogRoce::SaveEnvelope(HcclEnvelopeSummary& envelope)
{
    unique_lock<mutex> lock(envelopeQueMutex_);
    envelopeQue_.push(envelope);
}

bool TransportHeterogRoce::GetSavedEnvelope(HcclEnvelopeSummary& envelope)
{
    unique_lock<mutex> lock(envelopeQueMutex_);
    if (envelopeQue_.empty()) {
        return false;
    }
    envelope = envelopeQue_.front();
    envelopeQue_.pop();
    return true;
}

HcclResult TransportHeterogRoce::ParseErrorTagSqe(const struct ibv_wc* wc, int index)
{
    // wr_id内容即信封中的msn
    HcclRequestInfo* wrPtr = reinterpret_cast<HcclRequestInfo*>(wc[index].wr_id);
    CHK_PTR_NULL(wrPtr);
    wrPtr->transportRequest.status = wc[index].status;
    TransportHeterogRoce* transportPtr = reinterpret_cast<TransportHeterogRoce*>(wrPtr->transportHandle);
    CHK_PTR_NULL(transportPtr);

    HCCL_INFO(
        "exception send msg: tag:%d srcRank:%u dstRank:%u status:%d msn:0x%016llx request:%p",
        wrPtr->transportRequest.epParam.src.tag, wrPtr->transportRequest.epParam.src.rank,
        wrPtr->transportRequest.epParam.dst.rank, wrPtr->transportRequest.status, wrPtr->transportRequest.msn, wrPtr);

    CHK_RET(transportPtr->DeregMr(
        reinterpret_cast<void*>(wrPtr->transportRequest.transData.srcBuf),
        static_cast<u64>(
            wrPtr->transportRequest.transData.count * SIZE_TABLE[wrPtr->transportRequest.transData.dataType])));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::ParseDataRqes(const struct ibv_wc* wc, int num)
{
    for (int i = 0; i < num; i++) {
        HCCL_INFO("rq cqe info: wrId[%llu] status[%u] opcode[%u]", wc[i].wr_id, wc[i].status, wc[i].opcode);
        CHK_PRT_RET(
            wc[i].status != 0,
            HCCL_ERROR(
                "rdma poll data rq failed, cqe status[%u] wrId[%llu] opcode[%u]", wc[i].status, wc[i].wr_id,
                wc[i].opcode),
            HCCL_E_NETWORK);
        RecvWrInfo* info = reinterpret_cast<RecvWrInfo*>(wc[i].wr_id);
        CHK_PTR_NULL(info);
        HcclRequestInfo* wrPtr = reinterpret_cast<HcclRequestInfo*>(*reinterpret_cast<u64*>(info->buf));
        CHK_PRT_RET(
            wrPtr == nullptr, HCCL_ERROR("wrId[%llu] status[%u] opcode[%u]", wc[i].wr_id, wc[i].status, wc[i].opcode),
            HCCL_E_PTR);
        TransportHeterogRoce* transportPtr = reinterpret_cast<TransportHeterogRoce*>(wrPtr->transportHandle);
        CHK_PRT_RET(
            transportPtr == nullptr,
            HCCL_ERROR(
                "wrId[%llu] opcode[%u] tag:%d peerRank:%u status:%d msn:0x%016llx request:%p", wc[i].wr_id,
                wc[i].opcode, wrPtr->transportRequest.epParam.src.tag, wrPtr->transportRequest.epParam.src.rank,
                wrPtr->transportRequest.status, wrPtr->transportRequest.msn, wrPtr),
            HCCL_E_PTR);
        CHK_RET(transportPtr->SupplyDataRecvWqe());
        wrPtr->transportRequest.status = wc[i].status;

        CHK_RET(transportPtr->DeregMr(
            reinterpret_cast<void*>(wrPtr->transportRequest.transData.srcBuf),
            static_cast<u64>(
                wrPtr->transportRequest.transData.count * SIZE_TABLE[wrPtr->transportRequest.transData.dataType]),
            false));
        CHK_RET(transportPtr->FreeMemBlock(info->buf));
        CHK_RET(transportPtr->FreeRecvWrId(wc[i].wr_id));
        HCCL_INFO(
            "send completion: tag:%d peerRank:%u status:%d msn:0x%016llx request:%p",
            wrPtr->transportRequest.epParam.src.tag, wrPtr->transportRequest.epParam.src.rank,
            wrPtr->transportRequest.status, wrPtr->transportRequest.msn, wrPtr);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::ParseDataSqes(const struct ibv_wc* wc, int num)
{
    for (int i = 0; i < num; i++) {
        HCCL_INFO("sq cqe info: wrId[%llu] status[%u] opcode[%u]", wc[i].wr_id, wc[i].status, wc[i].opcode);
        CHK_PRT_RET(
            wc[i].status != 0,
            HCCL_ERROR(
                "rdma poll data sq failed, cqe status[%u] wrId[%llu] opcode[%u]", wc[i].status, wc[i].wr_id,
                wc[i].opcode),
            HCCL_E_NETWORK);
        HcclRequestInfo* wrPtr = reinterpret_cast<HcclRequestInfo*>(wc[i].wr_id);

        CHK_PRT_RET(
            wrPtr == nullptr, HCCL_ERROR("wrId[%llu] status[%u] opcode[%u]", wc[i].wr_id, wc[i].status, wc[i].opcode),
            HCCL_E_PTR);
        TransportHeterogRoce* transportPtr = reinterpret_cast<TransportHeterogRoce*>(wrPtr->transportHandle);
        CHK_PRT_RET(
            transportPtr == nullptr,
            HCCL_ERROR(
                "wrId[%llu] opcode[%u] tag:%d peerRank:%u status:%d msn:0x%016llx request:%p", wc[i].wr_id,
                wc[i].opcode, wrPtr->transportRequest.epParam.src.tag, wrPtr->transportRequest.epParam.src.rank,
                wrPtr->transportRequest.status, wrPtr->transportRequest.msn, wrPtr),
            HCCL_E_PTR);
        wrPtr->transportRequest.status = wc[i].status;

        if (!isHdcMode_ && !(remoteIsHdc_ && (deviceLogicId_ == HOST_DEVICE_ID))) {
            CHK_RET(transportPtr->DeregMr(
                reinterpret_cast<void*>(wrPtr->transportRequest.transData.dstBuf),
                static_cast<u64>(
                    wrPtr->transportRequest.transData.count * SIZE_TABLE[wrPtr->transportRequest.transData.dataType]),
                false));
        }
        HCCL_INFO(
            "recv completion: tag:%d peerRank:%u status:%d msn:0x%016llx", wrPtr->transportRequest.epParam.src.tag,
            wrPtr->transportRequest.epParam.src.rank, wrPtr->transportRequest.status, wrPtr->transportRequest.msn);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::SendEnvelope(HcclEnvelope& envelopInfo, void* stream)
{
    if (isHdcMode_ && tagQpInfo_.qpMode != NORMAL_QP_MODE) {
        CHK_RET(TransportHeterog::WaitBuildLinkComplete());
    }

    if (!isHdcMode_ || tagQpInfo_.qpMode == NORMAL_QP_MODE) {
        CHK_RET(SendFlowControl());
    }

    if (!isHdcMode_ || tagQpInfo_.qpMode == NORMAL_QP_MODE) {
        envelopeSge_.addr = reinterpret_cast<uint64_t>(&envelopInfo);
        envelopeSge_.length = sizeof(envelopInfo);
        envelopeSge_.lkey = 0;
        envelopeWr_.wr_id = envelopInfo.msn;

        struct ibv_send_wr* badWr = nullptr;
        HCCL_INFO(
            "rdma send: srcRank[%u] dstRank[%u] tag[%d]: addr[%llu] count[%d] dtype[%s] msn[%llu]",
            envelopInfo.epParam.src.rank, envelopInfo.epParam.dst.rank, envelopInfo.epParam.src.tag,
            reinterpret_cast<u64>(envelopInfo.transData.srcBuf), envelopInfo.transData.count,
            GetDataTypeEnumStr(envelopInfo.transData.dataType).c_str(), envelopInfo.msn);
        CHK_RET(hrtIbvPostSend(tagQpInfo_.qp, &envelopeWr_, &badWr));
    } else {
        struct SgList list = {};
        list.addr = reinterpret_cast<uint64_t>(&envelopInfo);
        list.len = sizeof(envelopInfo);
        list.lkey = 0;

        struct SendWr wr = {};
        wr.bufList = &list;
        wr.bufNum = 1;
        wr.op = static_cast<u32>(RdmaOp::OP_SEND);
        wr.sendFlag = RA_SEND_SIGNALED;
        struct SendWrRsp opRsp = {};
        CHK_RET(HrtRaSendWr(tagQpInfo_.qpHandle, &wr, &opRsp));
        CHK_RET(DoorBellSend(tagQpInfo_.qpMode, opRsp, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::InitTagRecvWqe()
{
    CHK_RET(IssueRecvWqe(tagQpInfo_.qp, recvWqeBatchNum_));
    tagRecvWqeNum_ = recvWqeBatchNum_;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::InitDataRecvWqe()
{
    CHK_RET(IssueRecvWqe(dataQpInfo_.qp, recvWqeBatchNum_));
    dataRecvWqeNum_ = recvWqeBatchNum_;
    dataRecvWqeExpNum_ = recvWqeBatchNum_;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::SendFlowControl()
{
    if (dataRecvWqeNum_ <= recvWqeBatchThreshold_) {
        CHK_RET(IssueRecvWqe(dataQpInfo_.qp, recvWqeBatchSupplement_));
        dataRecvWqeNum_ += recvWqeBatchSupplement_;
        dataRecvWqeExpNum_ += recvWqeBatchSupplement_;
    }

    u32 dataRecvWqeNum = dataRecvWqeNum_.load();
    u32 dataRecvWqeExpNum = dataRecvWqeExpNum_.load();
    if (dataRecvWqeNum - dataRecvWqeExpNum >= recvWqeBatchSupplement_) {
        CHK_RET(PullSendStatus());
        HCCL_RUN_INFO(
            "Flow control is activated, because dataRecvWqeNum[%u] - dataRecvWqeExpNum[%u] >="
            " recvWqeBatchSupplement[%u]",
            dataRecvWqeNum, dataRecvWqeExpNum, recvWqeBatchSupplement_);

        return HCCL_E_AGAIN;
    }

    dataRecvWqeExpNum_--;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::SupplyTagRecvWqe()
{
    tagRecvWqeNum_--;
    if (tagRecvWqeNum_ <= recvWqeBatchThreshold_) {
        CHK_RET(IssueRecvWqe(tagQpInfo_.qp, recvWqeBatchSupplement_));
        tagRecvWqeNum_ += recvWqeBatchSupplement_;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::SupplyDataRecvWqe()
{
    dataRecvWqeNum_--;
    if (dataRecvWqeNum_ <= recvWqeBatchThreshold_) {
        CHK_RET(IssueRecvWqe(dataQpInfo_.qp, recvWqeBatchSupplement_));
        dataRecvWqeNum_ += recvWqeBatchSupplement_;
        dataRecvWqeExpNum_ += recvWqeBatchSupplement_;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::IssueRecvWqe(struct ibv_qp* qp, u32 num)
{
    if (isHdcMode_ && (tagQpInfo_.qpMode == OFFLINE_QP_MODE || tagQpInfo_.qpMode == OFFLINE_QP_MODE_EXT)) {
        return HCCL_SUCCESS;
    }

    list<void*> blockList(num, nullptr);
    CHK_RET(AllocMemBlocks(blockList));

    auto iter = blockList.begin();
    struct ibv_recv_wr* nextRqWr = nullptr;
    struct ibv_recv_wr rqWr[num];
    struct ibv_sge sgeList[num];

    std::vector<struct RecvWrlistData> recvWrVec(num);
    struct RecvWrlistData* recvWr = recvWrVec.data();

    if (!isHdcMode_ || tagQpInfo_.qpMode == NORMAL_QP_MODE) {
        for (int i = num - 1; i >= 0; i--) {
            CHK_PTR_NULL(*iter);
            u64 wrId = 0;
            CHK_RET(GenerateRecvWrId(*iter, wrId));
            rqWr[i].wr_id = wrId;
            rqWr[i].next = nextRqWr;
            rqWr[i].sg_list = &sgeList[i];
            rqWr[i].num_sge = 1;
            sgeList[i].addr = reinterpret_cast<uint64_t>(*iter);
            sgeList[i].length = MEM_BLOCK_SIZE;
            sgeList[i].lkey = blockMemLkey_;
            nextRqWr = &rqWr[i];
            iter++;
        }

        struct ibv_recv_wr* badRqWr = nullptr;
        CHK_RET(hrtIbvPostRecv(qp, &rqWr[0], &badRqWr));
        return HCCL_SUCCESS;
    } else {
        for (int i = num - 1; i >= 0; i--) {
            CHK_PTR_NULL(*iter);
            u64 wrId = 0;
            if (useDevMem_) {
                // 根据host内存地址计算出device内存地址
                u64 uDevPtr
                    = reinterpret_cast<uint64_t>(*iter) - hostAddrBegin_ + reinterpret_cast<uint64_t>(deviceEvePtr_);
                CHK_RET(GenerateRecvWrId(reinterpret_cast<void*>(uDevPtr), wrId));
                recvWr[i].memList.addr = uDevPtr;
                recvWr[i].memList.lkey = deviceEveLkey_;
            } else {
                CHK_RET(GenerateRecvWrId(*iter, wrId));
                recvWr[i].memList.addr
                    = HostAddrToDev(reinterpret_cast<uint64_t>(*iter), hostAddrBegin_, devAddrBegin_);
                recvWr[i].memList.lkey = blockMemLkey_;
            }
            recvWr[i].wrId = wrId;
            recvWr[i].memList.len = MEM_BLOCK_SIZE;
            iter++;
        }
    }

    u32 completeNum = 0;
    s32 ret = hrtRaRecvWrlist(tagQpInfo_.qpHandle, recvWr, num, &completeNum);
    if (ret == HCCL_SUCCESS && completeNum == num) {
        HCCL_INFO("hrtRaRecvWrlist success ");
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[Transport][RdmaData]In RdmaDataTransport, hrtRaRecvWrlist failed. ret[%d]", ret);
        return HCCL_E_NETWORK;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetQpStatus(bool& completed)
{
    int qpStatus = 0;
    s32 ret = 0;

    ret = hrtGetRaQpStatus(tagQpInfo_.qpHandle, &qpStatus);
    if (ret != 0) {
        HCCL_ERROR("get tag qp status fail. qpStatus[%d] ret[%d]", qpStatus, ret);
        return HCCL_E_INTERNAL;
    } else if (ret == 0 && qpStatus != 1) { // 为1时，qp 建链成功
        return HCCL_E_AGAIN;
    }

    ret = hrtGetRaQpStatus(dataQpInfo_.qpHandle, &qpStatus);
    if (ret != 0) {
        HCCL_ERROR("get data qp status fail. qpStatus[%d] ret[%d]", qpStatus, ret);
        return HCCL_E_INTERNAL;
    } else if (ret == 0 && qpStatus != 1) { // 为1时，qp 建链成功
        return HCCL_E_AGAIN;
    }

    completed = true;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::AllocMemBlocks(list<void*>& blockList)
{
    const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManagerPtr
        = (IsRamdHandleLevelMr()) ? memBlocksManager_ : tagMemBlocksManager_;
    CHK_PTR_NULL(memBlocksManagerPtr);
    CHK_RET(memBlocksManagerPtr->Alloc(blockList));
    if (isHdcMode_) {
        for (auto iter : blockList) {
            wqeBlockLists_.push_back(iter);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::FreeMemBlock(void* block)
{
    const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManagerPtr
        = ((IsRamdHandleLevelMr())) ? memBlocksManager_ : tagMemBlocksManager_;
    CHK_PTR_NULL(memBlocksManagerPtr);
    CHK_RET(memBlocksManagerPtr->Free(block));
    if (isHdcMode_) {
        auto iter = std::find(wqeBlockLists_.begin(), wqeBlockLists_.end(), block);
        if (iter != wqeBlockLists_.end()) {
            wqeBlockLists_.erase(iter);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::FreeRecvWrId(u64 wrId)
{
    pRecvWrInfosMem_->Free(reinterpret_cast<RecvWrInfo*>(wrId));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GenerateRecvWrId(void* recvBuf, u64& wrId)
{
    RecvWrInfo* data = pRecvWrInfosMem_->Alloc();
    CHK_PTR_NULL(data);
    data->buf = recvBuf;
    data->transportHandle = reinterpret_cast<void*>(this);
    CHK_PTR_NULL(data->transportHandle);
    wrId = reinterpret_cast<uint64_t>(data);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetNetworkResource()
{
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(index_).GetRaResourceInfo(raResourceInfo));
    auto it = raResourceInfo.nicSocketMap.find(selfIp_);
    if (it == raResourceInfo.nicSocketMap.end()) {
        HCCL_ERROR("[TransportHeterogRoce][Init]nic socket handle was not found");
        return HCCL_E_PARA;
    }
    nicSocketHandle_ = it->second.nicSocketHandle;
    CHK_PTR_NULL(nicSocketHandle_);
    nicRdmaHandle_ = it->second.nicRdmaHandle;
    CHK_PTR_NULL(nicRdmaHandle_);
    HCCL_INFO(
        "TransportHeterogRoce GetNetworkResource index_[%d] nicSocketHandle_[%p] nicRdmaHandle_[%p]", index_,
        nicSocketHandle_, nicRdmaHandle_);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PreQpConnect()
{
    // 创建QP及CQ，多个QP可共享CQ
    CHK_RET(CreateCqAndQp());

    if (isHdcMode_) { // 不是HDC模式，走的peer，但是训练时，wqe下发情况和hdc相同
        CHK_RET(PreHdcResource());
    } else {
        // 下发post recv, 注：HCCP完成QP建链后需要两端握手确认QP状态OK后才能发起通信
        CHK_RET(InitTagRecvWqe());
        if (!(remoteIsHdc_ && (deviceLogicId_ == HOST_DEVICE_ID))) {
            CHK_RET(InitDataRecvWqe());
        }
    }

    // 为提高收发处理速度，提前准备post send需要的wr模板
    CHK_SAFETY_FUNC_RET(memset_s(&envelopeWr_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    envelopeWr_.sg_list = &envelopeSge_;
    envelopeWr_.next = nullptr;
    envelopeWr_.num_sge = 1;
    envelopeWr_.opcode = IBV_WR_SEND;
    envelopeWr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

    CHK_SAFETY_FUNC_RET(memset_s(&dataReadWr_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    dataReadWr_.sg_list = &dataReadSge_;
    dataReadWr_.next = nullptr;
    dataReadWr_.num_sge = 1;
    dataReadWr_.opcode = IBV_WR_RDMA_READ;
    dataReadWr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;

    CHK_SAFETY_FUNC_RET(memset_s(&dataWriteWr_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    dataWriteWr_.sg_list = &dataWriteSge_;
    dataWriteWr_.next = nullptr;
    dataWriteWr_.num_sge = 1;
    dataWriteWr_.opcode = IBV_WR_RDMA_WRITE;
    dataWriteWr_.send_flags = IBV_SEND_FENCE;

    CHK_SAFETY_FUNC_RET(memset_s(&notifyWriteWr_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    notifyWriteWr_.sg_list = &notifyWriteSge_;
    notifyWriteWr_.next = nullptr;
    notifyWriteWr_.num_sge = 1;
    notifyWriteWr_.opcode = IBV_WR_RDMA_WRITE;
    notifyWriteWr_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;

    CHK_SAFETY_FUNC_RET(memset_s(&dataAckWr_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    dataAckWr_.sg_list = &dataAckSge_;
    dataAckWr_.next = nullptr;
    dataAckWr_.num_sge = 1;
    dataAckWr_.opcode = IBV_WR_SEND_WITH_IMM;
    dataAckWr_.send_flags = IBV_SEND_FENCE | IBV_SEND_INLINE;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreateCqAndQp()
{
    HCCL_INFO("TransportHeterogRoce CreateCqAndQp");
    CHK_RET(CreateQpWithCq(nicRdmaHandle_, -1, -1, nullptr, nullptr, tagQpInfo_, isHdcMode_, isESMode_));
    CHK_RET(CreateQpWithCq(nicRdmaHandle_, -1, -1, nullptr, nullptr, dataQpInfo_, isHdcMode_, isESMode_));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::DestroyCqAndQp()
{
    HCCL_INFO("TransportHeterogRoce DestroyCqAndQp");
    CHK_RET(DestroyQpWithCq(tagQpInfo_, isHdcMode_));
    tagQpInfo_ = QpInfo();
    CHK_RET(DestroyQpWithCq(dataQpInfo_, isHdcMode_));
    dataQpInfo_ = QpInfo();
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::QpConnect(bool& completed)
{
    CHK_RET(HrtRaQpNonBlockConnectAsync(tagQpInfo_.qpHandle, initSM_.locInitInfo.socketInfo[0].fdHandle));
    CHK_RET(HrtRaQpNonBlockConnectAsync(dataQpInfo_.qpHandle, initSM_.locInitInfo.socketInfo[1].fdHandle));

    completed = true;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::RegMr(void* mem, u64 size, u32& lkey, bool isTagQpHandle)
{
    HCCL_DEBUG("reg mr mem[%p] size[%llu Byte]", mem, size);
    if (size == 0) {
        lkey = 0;
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(mem);

    if (isTagQpHandle || IsRamdHandleLevelMr()) {
        CHK_RET(mrManager_->GetKey(mem, size, lkey));
    } else {
        CHK_RET(dataQpMrManager_->GetKey(mem, size, lkey));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::DeregMr(void* mem, u64 size, bool isTagQpHandle)
{
    HCCL_DEBUG("dereg mr mem[%p] size[%llu Byte]", mem, size);
    if (size == 0) {
        return HCCL_SUCCESS;
    }

    if (isTagQpHandle || IsRamdHandleLevelMr()) {
        CHK_RET(mrManager_->ReleaseKey(mem, size));
    } else {
        CHK_RET(dataQpMrManager_->ReleaseKey(mem, size));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::RoceConnectSocket(SocketConnectInfoT conn[], u32 num, bool& completed)
{
    if (initSM_.locInitInfo.role == CLIENT_ROLE_SOCKET) {
        return ConnectSocket(conn, num, completed);
    } else {
        completed = true;
        return HCCL_SUCCESS;
    }
}

HcclResult TransportHeterogRoce::FlushSendQueue(bool& completed)
{
    if (envelopeBacklogQueue_.size() > 0) {
        HcclEnvelope tmpEnvelopeInfo;
        while (!envelopeBacklogQueue_.empty()) {
            tmpEnvelopeInfo = envelopeBacklogQueue_.front();
            CHK_RET(SendEnvelope(tmpEnvelopeInfo));
            envelopeBacklogQueue_.pop();
        }
    }
    completed = true;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::EnterStateProcess(ConnState nextState)
{
    switch (nextState) {
        case ConnState::CONN_STATE_CONNECT_CHECK_SOCKET:
            initSM_.socketNum = 1;
            break;
        case ConnState::CONN_STATE_GET_CHECK_SOCKET:
            initSM_.socketNum = 1;
            initSM_.completeNum = 0;
            break;
        case ConnState::CONN_STATE_SEND_CF:
        case ConnState::CONN_STATE_RECV_CF:
            initSM_.size = HETEROG_MAX_FRAME_LEN;
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_CHECK_CF:
            CHK_RET(CheckConsistentFrame());
            CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_CONNECT_ALL_SOCKET));
            break;
        case ConnState::CONN_STATE_CONNECT_ALL_SOCKET:
            initSM_.socketNum = initSM_.locInitInfo.socketConnInfo.size() - 1;
            break;
        case ConnState::CONN_STATE_GET_ALL_SOCKET:
            initSM_.socketNum = initSM_.locInitInfo.socketInfo.size() - 1;
            initSM_.completeNum = 0;
            break;
        case ConnState::CONN_STATE_SEND_STATUS:
            initSM_.size = sizeof(initSM_.locInitInfo.signal);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_RECV_STATUS:
            initSM_.size = sizeof(initSM_.remInitInfo.signal);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_COMPLETE:
            HCCL_INFO("link[%s]: connect complete", initSM_.locInitInfo.socketInfo[0].tag);
            break;
        default:
            HCCL_INFO("link[%s]: state[%u] no need to do anything", initSM_.locInitInfo.socketInfo[0].tag, nextState);
    }

    return HCCL_SUCCESS;
}
// 需要循环检查的状态
HcclResult TransportHeterogRoce::LoopStateProcess()
{
    HcclResult testRet = HCCL_SUCCESS;
    bool completed = false;
    switch (GetState()) {
        case ConnState::CONN_STATE_CONNECT_CHECK_SOCKET:
            testRet = RoceConnectSocket(initSM_.locInitInfo.socketConnInfo.data(), initSM_.socketNum, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_GET_CHECK_SOCKET));
            break;
        case ConnState::CONN_STATE_GET_CHECK_SOCKET:
            testRet = GetSocket(
                initSM_.locInitInfo.role, initSM_.locInitInfo.socketInfo.data(), initSM_.socketNum, initSM_.completeNum,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_SEND_CF));
            break;
        case ConnState::CONN_STATE_SEND_CF:
            testRet = SocketSend(
                initSM_.locInitInfo.socketInfo[0].fdHandle, initSM_.locInitInfo.checkFrame, initSM_.size,
                initSM_.completeSize, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_RECV_CF));
            break;
        case ConnState::CONN_STATE_RECV_CF:
            testRet = SocketRecv(
                initSM_.locInitInfo.socketInfo[0].fdHandle, initSM_.remInitInfo.checkFrame, initSM_.size,
                initSM_.completeSize, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_CHECK_CF));
            break;
        case ConnState::CONN_STATE_CONNECT_ALL_SOCKET:
            testRet = RoceConnectSocket(
                reinterpret_cast<SocketConnectInfoT*>(initSM_.locInitInfo.socketConnInfo.data()) + 1, initSM_.socketNum,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_GET_ALL_SOCKET));
            break;
        case ConnState::CONN_STATE_GET_ALL_SOCKET:
            testRet = GetSocket(
                initSM_.locInitInfo.role,
                reinterpret_cast<struct SocketInfoT*>(initSM_.locInitInfo.socketInfo.data()) + 1, initSM_.socketNum,
                initSM_.completeNum, completed);
            testRet = ((testRet == HCCL_SUCCESS) && completed) ? CreatSignalMesg() : testRet;
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_CONNECT_QP));
            break;
        case ConnState::CONN_STATE_CONNECT_QP:
            testRet = QpConnect(completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_GET_QP));
            break;
        case ConnState::CONN_STATE_GET_QP:
            testRet = GetQpStatus(completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_SEND_STATUS));
            break;
        case ConnState::CONN_STATE_SEND_STATUS:
            testRet = SocketSend(
                initSM_.locInitInfo.socketInfo[SOCKET_FOR_SENDRECV_QP].fdHandle, &(initSM_.locInitInfo.signal),
                initSM_.size, initSM_.completeSize, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_RECV_STATUS));
            break;
        case ConnState::CONN_STATE_RECV_STATUS:
            testRet = SocketRecv(
                initSM_.locInitInfo.socketInfo[SOCKET_FOR_SENDRECV_QP].fdHandle, &(initSM_.remInitInfo.signal),
                initSM_.size, initSM_.completeSize, completed);
            fdHandle_ = initSM_.locInitInfo.socketInfo[SOCKET_FOR_SENDRECV_QP].fdHandle;
            testRet = ((testRet == HCCL_SUCCESS) && completed && !isRawConn_) ? ExchangeSignalMesg() : testRet;
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_FLUSH_QUEUE));
            break;
        case ConnState::CONN_STATE_FLUSH_QUEUE: {
            // 为防止Isend中积压信封入队和TestSome中flush积压信封队列并发问题，
            // 该处flush积压信封队列并状态迁移完成后，再解锁。
            std::unique_lock<std::mutex> lock(envelopeBacklogQueueLock_);
            testRet = FlushSendQueue(completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_COMPLETE));
            break;
        }
        default:
            HCCL_ERROR(
                "Establish communication connection failed[%s]: state[%u]", initSM_.locInitInfo.socketInfo[0].tag,
                GetState());
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetSocketInfos(std::vector<std::vector<HcclSocketInfo>>& socketInfos)
{
    std::vector<HcclSocketInfo> hcclSocketInfo;
    for (SocketInfoT raSocketInfo : initSM_.locInitInfo.socketInfo) {
        hcclSocketInfo.push_back({raSocketInfo.socketHandle, raSocketInfo.fdHandle});
    }
    socketInfos.push_back(hcclSocketInfo);
    return HCCL_SUCCESS;
}

void TransportHeterogRoce::GetTransportResourceInfo(const TransportResourceInfo& transportResourceInfo)
{
    tagQpInfo_.flag = transportResourceInfo.flag;
    tagQpInfo_.qpMode = transportResourceInfo.qpMode;
    dataQpInfo_.flag = transportResourceInfo.flag;
    dataQpInfo_.qpMode = transportResourceInfo.qpMode;
    isHdcMode_ = transportResourceInfo.isHdcMode;
    deviceLogicId_ = transportResourceInfo.deviceLogicId;
    memBlockNum_ = transportResourceInfo.memBlockNum;
    remoteIsHdc_ = transportResourceInfo.remoteIsHdc;
    isESMode_ = transportResourceInfo.isESMode;
    isGlobalMrmanagerInit_ = transportResourceInfo.isGlobalMrmanagerInit;
    hdcHostWqeBatchNum_ = transportResourceInfo.hdcHostWqeBatchNum;
    HCCL_INFO(
        "tagQpInfo_.flag[%d] tagQpInfo_.qpMode[%d] dataQpInfo_.flag[%d] dataQpInfo_.qpMode[%d] isHdcMode_[%d] "
        "deviceLogicId_[%d] memBlockNum_[%u] remoteIsHdc_[%d] isESMode_[%d] isGlobalMrmanagerInit_[%d] "
        "hdcHostWqeBatchNum_[%u]",
        tagQpInfo_.flag, tagQpInfo_.qpMode, dataQpInfo_.flag, dataQpInfo_.qpMode, isHdcMode_, deviceLogicId_,
        memBlockNum_, remoteIsHdc_, isESMode_, isGlobalMrmanagerInit_, hdcHostWqeBatchNum_);
}

HcclResult TransportHeterogRoce::PollCq(QpInfo& qpInfo, bool isSend, s32& num, struct ibv_wc* wc)
{
    if (!isHdcMode_ || tagQpInfo_.qpMode == NORMAL_QP_MODE) {
        if (isSend) {
            CHK_RET(hrtIbvPollCq(qpInfo.sendCq, HCCL_POLL_CQ_DEPTH, wc, num));
        } else {
            CHK_RET(hrtIbvPollCq(qpInfo.recvCq, HCCL_POLL_CQ_DEPTH, wc, num));
        }
    } else {
        s32 ret = hrtRaPollCq(qpInfo.qpHandle, isSend, HCCL_POLL_CQ_ONETIME, wc);
        if (ret >= 0 && static_cast<u32>(ret) <= HCCL_POLL_CQ_ONETIME) {
            num = ret;
        } else {
            HCCL_ERROR("call trace: hcclRet -> %d", ret);
            return HCCL_E_REMOTE;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetRemoteIsendDoneSignal(std::shared_ptr<LocalIpcNotify>& signal)
{
    signal = remoteIsendDoneSignal_;
    CHK_SMART_PTR_NULL(signal);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetRemoteImrecvDoneSignal(std::shared_ptr<LocalIpcNotify>& signal)
{
    signal = remoteImrecvDoneSignal_;
    CHK_SMART_PTR_NULL(signal);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::GetNotifySize()
{
    DevType devType;
    CHK_RET(hrtHalGetDeviceType(index_, devType));

    if (devType == DevType::DEV_TYPE_910) {
        notifySize_ = NOTIFY_SIZE_BYTES_910A;
    } else if ((devType == DevType::DEV_TYPE_910B) || (devType == DevType::DEV_TYPE_910_93)) {
        notifySize_ = NOTIFY_SIZE_BYTES_910B;
    } else {
        notifySize_ = NOTIFY_SIZE_BYTES_910A;
    }
    HCCL_INFO("devType[%d] notifySize[%d]", devType, notifySize_);
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreateRdmaSignal(
    std::shared_ptr<LocalIpcNotify>& localNotify, HcclRdmaSignalInfo& rdmaSignalInfo, MemType notifyType)
{
    EXCEPTION_CATCH((localNotify = std::make_shared<LocalIpcNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(localNotify);
    s32 pid = 0;
    CHK_RET(SalGetBareTgid(&pid)); // 当前进程id
    CHK_RET(localNotify->Init(deviceLogicId_, deviceLogicId_));
    s64 recvId = 0xFFFFFFFF00000000 | (static_cast<s64>(pid) & 0xFFFFFFFF);
    CHK_RET(localNotify->Grant(recvId));

    u64 notifyOffset = 0;
    u64 notifyBaseVa = 0; // notify寄存器虚拟地址
    u64 notifyTotalSize = 0;
    CHK_RET(HrtRaGetNotifyBaseAddr(nicRdmaHandle_, &notifyBaseVa, &notifyTotalSize));
    CHK_RET(localNotify->GetNotifyOffset(notifyOffset));
    u64 notifyVa = notifyBaseVa + notifyOffset;

    rdmaSignalInfo.mrRegFlag = 0;
    rdmaSignalInfo.notifyAddr = reinterpret_cast<void*>(notifyVa);
    rdmaSignalInfo.len = notifySize_;
    rdmaSignalInfo.type = notifyType;

    struct MrInfoT mrInfo = {};
    mrInfo.addr = rdmaSignalInfo.notifyAddr;
    mrInfo.size = rdmaSignalInfo.len;
    mrInfo.access = access_;
    CHK_RET(HrtRaMrReg(dataQpInfo_.qpHandle, &mrInfo));
    rdmaSignalInfo.lkey = mrInfo.lkey;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PsRdmaDbSend(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream)
{
    CHK_RET(hrtSetDevice(index_));
    s32 ret = hrtRDMADBSend(dbindex, dbinfo, stream);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rtRDMADBSend]errNo[0x%016llx] rt rdma send fail, "
            "return[%d]. para: dbindex[%u] dbinfo[%llu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dbindex, dbinfo),
        HCCL_E_RUNTIME);
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        // ps侧DbSend对当前线程SetDevice后，改变了原GE通信域初始化时setdevice 0
        // 若后面使用本线程save会导致getctx失败获取不到通信域句柄，所以需要在此处重新set回默认
        CHK_RET(hrtSetDevice(0));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreateDevMemForNotify(DeviceMem& devMem, u64 size, u32 value)
{
    HCCL_INFO("Use dev mem for notify value");
    void* devMemAddr{nullptr};
    CHK_RET(hrtSetDevice(index_));
    CHK_RET(HrtDevMalloc(&devMemAddr, size));

    devMemPtrs_.emplace_back(devMemAddr);

    devMem = DeviceMem::create(devMemAddr, size);
    CHK_RET(hrtMemcpy(devMemAddr, size, &value, sizeof(u32), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreateHostMemForNotify(DeviceMem& devMem, u64 size, u32 value, bool needMap)
{
    HCCL_INFO("PS use host mem for notify value");
    u64 memLen = size + SMALL_PAGE_SIZE;
    s8* ptr = new (std::nothrow) s8[memLen];
    CHK_PTR_NULL(ptr);
    hostMemPtr_.emplace_back(ptr);
    u64 pageSizeNum = reinterpret_cast<u64>(ptr) / SMALL_PAGE_SIZE;
    void* ptrVoid = reinterpret_cast<void*>((pageSizeNum + 1) * SMALL_PAGE_SIZE);
    void* devVirAddr = ptrVoid;
    if (needMap) {
        s32 ret = dataQpMrManager_->MapMem(ptrVoid, size, devVirAddr);
        if (ret != 0 || devVirAddr == nullptr) {
            HCCL_ERROR("PS malloc device mem fail[%d]", HCCL_E_MEMORY);
            return HCCL_E_MEMORY;
        }

        devMem = DeviceMem::create(devVirAddr, size);
        CHK_RET(hrtMemcpy(
            ptrVoid, notifyMem_.size(), &value, sizeof(u32), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST));
        return HCCL_SUCCESS;
    }

    devMem = DeviceMem::create(devVirAddr, size);
    s32 ret = memcpy_s(ptrVoid, notifyMem_.size(), &value, sizeof(u32));
    if (ret < 0) {
        HCCL_ERROR("memcpy_s fail[%d]", ret);
        return HCCL_E_MEMORY;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreateNotifyValueBuffer()
{
    if (notifyMem_.ptr() == nullptr) {
        u32 notifyVaule = 1;

        if (deviceLogicId_ == HOST_DEVICE_ID && isHdcMode_ && (dataQpInfo_.qpMode != NORMAL_QP_MODE)) {
            // ES多机AI server的PS，申请device内存
            CHK_RET(CreateDevMemForNotify(notifyMem_, notifyValueSize_, notifyVaule));
        } else {
            CHK_RET(CreateHostMemForNotify(notifyMem_, notifyValueSize_, notifyVaule, isHdcMode_));
        }

        CHK_PRT_RET(!notifyMem_.ptr(), HCCL_ERROR("CreateNotifyValueBuffer malloc failed."), HCCL_E_MEMORY);
    }

    struct MrInfoT mrInfo = {};
    mrInfo.addr = notifyMem_.ptr();
    mrInfo.size = notifyValueSize_;
    mrInfo.access = access_;
    CHK_RET(HrtRaMrReg(dataQpInfo_.qpHandle, &mrInfo));

    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].mrRegFlag = REG_VALID;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].addr = notifyMem_.ptr();
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].len = notifyValueSize_;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].memType = MemType::NOTIFY_VALUE_MEM;
    notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].lkey = mrInfo.lkey;
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::DeleteNotifyValueBuffer()
{
    for (u64 i = 0; i < devMemPtrs_.size(); i++) {
        if (devMemPtrs_[i] != nullptr) {
            CHK_RET(HrtDevFree(devMemPtrs_[i]));
            devMemPtrs_[i] = nullptr;
        }
    }

    devMemPtrs_.clear();

    for (u64 i = 0; i < hostMemPtr_.size(); i++) {
        if (hostMemPtr_[i] != nullptr) {
            delete[] hostMemPtr_[i];
            hostMemPtr_[i] = nullptr;
        }
    }

    hostMemPtr_.clear();

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::RecoverNotifyMsg(HcclRdmaSignalInfo* remoteRdmaSignal, u64 signalNum)
{
    if (signalNum <= 0) {
        return HCCL_E_NOT_FOUND;
    }

    for (u64 i = 0; i < signalNum; i++) {
        u32 tmpMemType = (remoteRdmaSignal + i)->type;
        notifyMemMsg_[tmpMemType].mrRegFlag = (remoteRdmaSignal + i)->mrRegFlag;
        notifyMemMsg_[tmpMemType].addr = (remoteRdmaSignal + i)->notifyAddr;
        notifyMemMsg_[tmpMemType].len = (remoteRdmaSignal + i)->len;
        notifyMemMsg_[tmpMemType].memType = (remoteRdmaSignal + i)->memType;
        notifyMemMsg_[tmpMemType].rkey = (remoteRdmaSignal + i)->lkey;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::CreatSignalMesg()
{
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        // ps
        // 310soc的ps不需要申请notify value，直接返回
        if (!isHdcMode_ && !remoteIsHdc_) {
            return HCCL_SUCCESS;
        }
        // Notify start
        if (isHdcMode_) {
            CHK_RET(hrtSetDevice(index_));
        }
        CHK_RET(CreateNotifyValueBuffer());
    } else {
        // worker
        // 310soc的worker不需要在这里申请notify，直接返回
        if (!isHdcMode_) {
            return HCCL_SUCCESS;
        }
        // Notify start
        CHK_RET(GetNotifySize());
        CHK_RET(CreateRdmaSignal(remoteIsendDoneSignal_, rdmaSignal_[0], MemType::SEND_NOTIFY_MEM));
        CHK_RET(CreateRdmaSignal(remoteImrecvDoneSignal_, rdmaSignal_[1], MemType::RECV_NOTIFY_MEM));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::ExchangeSignalMesg()
{
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        // ps
        // Notify start
        HcclRdmaSignalInfo remoteRdmaSignal[REMOTE_RDMA_SIGNAL_SIZE];
        CHK_RET(
            hrtRaSocketBlockRecv(fdHandle_, remoteRdmaSignal, sizeof(HcclRdmaSignalInfo) * REMOTE_RDMA_SIGNAL_SIZE));
        CHK_RET(RecoverNotifyMsg(remoteRdmaSignal, REMOTE_RDMA_SIGNAL_SIZE));
    } else {
        // worker
        // Notify start
        CHK_RET(hrtRaSocketBlockSend(fdHandle_, rdmaSignal_, sizeof(HcclRdmaSignalInfo) * REMOTE_RDMA_SIGNAL_SIZE));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::RecordNotifyWithReq(Stream& stream, RdmaNotifyOp type, HcclRequestInfo*& request)
{
    TransData sendData{};
    TransportEndPointParam epParam{};

    CHK_RET(GenerateSendRequest(sendData, epParam, request));
    request->transportRequest.requestType = HcclRequestType::HCCL_REQUEST_RECV;
    u64 wrId = reinterpret_cast<uint64_t>(request);
    CHK_RET(RecordNotify(stream, type, wrId));

    s32 notifyFlag = HCCL_TEST_INCOMPLETED;
    TIME_PRINT(CHK_RET(this->Wait(*request, notifyFlag)));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::RecordNotify([[maybe_unused]] Stream& stream, RdmaNotifyOp type, u64 wrId)
{
    HCCL_INFO(
        "RecordNotify notifyType[%u], wrId[%llu] isHdcMode_[%d] qpMode[%d]", type, wrId, isHdcMode_,
        dataQpInfo_.qpMode);
    MemType opType = MemType::MEM_TYPE_RESERVED;
    if (type == RdmaNotifyOp::SEND_NOTIFY) {
        opType = MemType::SEND_NOTIFY_MEM;
    } else if (type == RdmaNotifyOp::RECV_NOTIFY) {
        opType = MemType::RECV_NOTIFY_MEM;
    } else {
        HCCL_ERROR("TransportHeterogRoce::TYPE is not supported.");
        return HCCL_E_PARA;
    }

    if (!isHdcMode_ || dataQpInfo_.qpMode == NORMAL_QP_MODE) {
        notifyWriteSge_.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(notifyMem_.ptr()));
        notifyWriteSge_.length = notifyMemMsg_[static_cast<u32>(opType)].len;
        notifyWriteSge_.lkey = notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].lkey;

        notifyWriteWr_.wr_id = wrId;
        notifyWriteWr_.wr.rdma.remote_addr
            = static_cast<u64>(reinterpret_cast<uintptr_t>(notifyMemMsg_[static_cast<u32>(opType)].addr));
        notifyWriteWr_.wr.rdma.rkey = notifyMemMsg_[static_cast<u32>(opType)].rkey;

        struct ibv_send_wr* badWr = nullptr;
        HCCL_INFO(
            "notify write: remote addr[%llu] length[%d] wrId[%llu]", notifyWriteWr_.wr.rdma.remote_addr,
            notifyWriteSge_.length, notifyWriteWr_.wr_id);
        CHK_RET(hrtIbvPostSend(dataQpInfo_.qp, &notifyWriteWr_, &badWr));
    } else {
        struct SgList list = {};
        list.addr = static_cast<u64>(reinterpret_cast<uintptr_t>(notifyMem_.ptr()));
        list.len = notifyMemMsg_[static_cast<u32>(opType)].len;
        list.lkey = notifyMemMsg_[static_cast<u32>(MemType::NOTIFY_VALUE_MEM)].lkey;

        struct SendWrV2 wr {};
        wr.wrId = wrId;
        wr.bufList = &list;
        wr.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr.dstAddr = static_cast<u64>(reinterpret_cast<uintptr_t>(notifyMemMsg_[static_cast<u32>(opType)].addr));
        wr.rkey = notifyMemMsg_[static_cast<u32>(opType)].rkey;
        wr.op = static_cast<u32>(RdmaOp::OP_WRITE); /* RDMA_WRITE: 0 */
        wr.sendFlag = RA_SEND_SIGNALED | RA_SEND_FENCE;
        struct SendWrRsp opRsp = {};
        CHK_RET(HrtRaSendWrV2(dataQpInfo_.qpHandle, &wr, &opRsp, GetWorkflowMode()));
        CHK_RET(DoorBellSend(dataQpInfo_.qpMode, opRsp));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::DoorBellSend(const s32 qpMode, const SendWrRsp& opRsp, void* stream)
{
    if (qpMode == OPBASE_QP_MODE || qpMode == OPBASE_QP_MODE_EXT || qpMode == OFFLINE_QP_MODE_EXT) {
        HCCL_DEBUG("entry PsRdmaDbSend");
        u32 dbIndex = static_cast<u32>(opRsp.db.dbIndex);
        u64 dbInfo = static_cast<u64>(opRsp.db.dbInfo);
        CHK_RET(PsRdmaDbSend(dbIndex, dbInfo, stream));
    } else {
        HCCL_DEBUG("entry hrtRDMASend");
        u32 qpn = opRsp.wqeTmp.sqIndex;
        u32 wqe_index = opRsp.wqeTmp.wqeIndex;
        CHK_RET(hrtRDMASend(qpn, wqe_index, stream));
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::MrManagerInit()
{
    // mrManager_管理信封内存
    if (IsRamdHandleLevelMr()) {
        // 通信域初始化时外部还未传入全局内存，需要在这里面手动去初始化需要的全局内存
        CHK_PTR_NULL(mrManager_);
        std::map<MrMapKey, MrInfo> unRegMrMap = MrManager::GetInstance().GetUnregMap();
        mrManager_->InitUnRegMrMap(unRegMrMap);
        // 使用全局的MrManager时dataQp也使用全局的MrManager
        dataQpMrManager_ = mrManager_;
        return HCCL_SUCCESS;
    }
    mrManager_ = new (nothrow) MrManager();
    CHK_PTR_NULL(mrManager_);
    std::map<MrMapKey, MrInfo> unRegMrMap = MrManager::GetInstance().GetUnregMap();
    CHK_PRT(mrManager_->Init(tagQpInfo_.qpHandle, index_, deviceLogicId_ == HOST_DEVICE_ID, unRegMrMap));

    // dataQpManager_管理数据收发内存
    dataQpMrManager_ = new (nothrow) MrManager();
    CHK_PTR_NULL(dataQpMrManager_);
    unRegMrMap = MrManager::GetInstance().GetUnregMap();
    CHK_PRT(dataQpMrManager_->Init(dataQpInfo_.qpHandle, index_, deviceLogicId_ == HOST_DEVICE_ID, unRegMrMap));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::MrManagerDeInit()
{
    if (IsRamdHandleLevelMr()) {
        // 若mrManager是全局的，那么就在通信类外部释放
        dataQpMrManager_ = nullptr;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("entry MrManagerDeInit");
    CHK_PTR_NULL(mrManager_);
    CHK_PRT(mrManager_->DeInit(tagQpInfo_.qpHandle));
    delete mrManager_;

    CHK_PTR_NULL(dataQpMrManager_);
    CHK_PRT(dataQpMrManager_->DeInit(dataQpInfo_.qpHandle));
    delete dataQpMrManager_;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::PreHdcResource()
{
    // hdc模式下在通信类内部注册内存
    CHK_PRT(MrManagerInit());
    // worker侧不做信封内存注册
    if (deviceLogicId_ == HOST_DEVICE_ID) {
        if (!IsRamdHandleLevelMr()) {
            CHK_PRT(MemBlocksManagerInit());
            CHK_RET(mrManager_->GetKey(
                tagMemBlocksManager_->GetMemAddr(), tagMemBlocksManager_->GetMemSize(), blockMemLkey_));
        }

        const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManagerPtr
            = (IsRamdHandleLevelMr()) ? memBlocksManager_ : tagMemBlocksManager_;
        // mrmanager是全局的时使用的信封内存管理类也是外部传入的全局的管理类
        hostAddrBegin_ = (u64)memBlocksManagerPtr->GetMemAddr();

        devAddrBegin_ = MrManager::g_devAddr;
        recvWqeBatchNum_ = hdcHostWqeBatchNum_;
        recvWqeBatchThreshold_ = hdcHostWqeBatchNum_;
        recvWqeBatchSupplement_ = RECV_WQE_HDC_BATCH_SUPPLEMENT;
        HCCL_INFO(
            "PreHdcResource IsRamdHandleLevelMr[%d] recvWqeBatchNum_[%u] recvWqeBatchThreshold_[%u]",
            IsRamdHandleLevelMr(), recvWqeBatchNum_, recvWqeBatchThreshold_);
        if (useDevMem_) {
#ifndef CCL_KERNEL
            CHK_RET(hrtSetDevice(index_));
            // device内存申请跟host内存一样大的内存
            u64 memSize = memBlocksManagerPtr->GetMemSize();
            s32 ret = HrtDevMalloc(&deviceEvePtr_, memSize);
            if (ret != 0 || deviceEvePtr_ == nullptr) {
                HCCL_ERROR("PS HrtDevMalloc device mem fail ret=[%d]", ret);
                return HCCL_E_MEMORY;
            }

            struct MrInfoT mrInfo = {};
            mrInfo.addr = deviceEvePtr_;
            mrInfo.size = memSize;
            mrInfo.access = RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE | RA_ACCESS_REMOTE_READ;
            CHK_RET(HrtRaMrReg(dataQpInfo_.qpHandle, &mrInfo));
            deviceEveLkey_ = mrInfo.lkey;
            HCCL_INFO("index_[%u] deviceEvePtr_[%p] memSize[%llu]", index_, deviceEvePtr_, memSize);
#endif
        }
        CHK_RET(InitTagRecvWqe());
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::MemBlocksManagerInit()
{
    // 初始化信封内存
    tagMemBlocksManager_.reset(new (std::nothrow) HeterogMemBlocksManager());
    CHK_SMART_PTR_NULL(tagMemBlocksManager_);
    CHK_RET(tagMemBlocksManager_->Init(memBlockNum_));

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRoce::MemBlocksManagerDeInit()
{
    while (wqeBlockLists_.size() > 0) {
        CHK_RET(FreeMemBlock(wqeBlockLists_.front()));
    }

    if (IsRamdHandleLevelMr()) {
        // MrManager是全局的时候在通信类外部统一释放
        return HCCL_SUCCESS;
    }
    CHK_SMART_PTR_NULL(tagMemBlocksManager_);
    CHK_PTR_NULL(mrManager_);
    CHK_RET(mrManager_->ReleaseKey(tagMemBlocksManager_->GetMemAddr(), tagMemBlocksManager_->GetMemSize()));
    tagMemBlocksManager_ = nullptr;

    return HCCL_SUCCESS;
}

void TransportHeterogRoce::GetLinkTag(std::string& tag)
{
    tag = initSM_.locInitInfo.socketInfo[0].tag;
    return;
}

bool TransportHeterogRoce::IsRamdHandleLevelMr()
{
    // 非hdc模式或者AI-Server910B场景ps下不分平面时以RdmaHandle粒度注册MR
    return (!isHdcMode_ || (isHdcMode_ && isGlobalMrmanagerInit_));
}

} // namespace hccl
