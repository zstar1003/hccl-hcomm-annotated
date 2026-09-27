/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_heterog_raw_roce.h"
#include "adapter_hccp.h"
#include "adapter_verbs.h"
#include "externalinput.h"
#include "network/hccp_common.h"

using namespace std;
namespace hccl {

constexpr s32 TAG_QP_APPEND = 1;
constexpr s32 DATA_QP_APPEND = 2;
constexpr u32 RECV_WQE_BATCH_NUM = MEM_BLOCK_RECV_WQE_BATCH_NUM;
constexpr u32 RECV_WQE_NUM_THRESHOLD = 96;
constexpr u32 RECV_WQE_BATCH_SUPPLEMENT = 96;

TransportHeterogRawRoce::TransportHeterogRawRoce(
    const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
    const TransportResourceInfo& transportResourceInfo)
    : TransportHeterogRoce(transTag, selfIp, peerIp, peerPort, selfPort, transportResourceInfo)
{}

TransportHeterogRawRoce::~TransportHeterogRawRoce() {}

HcclResult TransportHeterogRawRoce::Init() { return HCCL_SUCCESS; }

HcclResult TransportHeterogRawRoce::Init(SocketInfoT& socketInfo, RdmaHandle rdmaHandle, MrHandle mrHandle)
{
    CHK_PTR_NULL(socketInfo.socketHandle);
    CHK_PTR_NULL(socketInfo.fdHandle);
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(mrHandle);
    HCCL_INFO(
        "TransportHeterogRawRoce Init start socketHandle[%p] fdHandle[%p] rdmaHandle[%p] mrHandle[%p]",
        socketInfo.socketHandle, socketInfo.fdHandle, rdmaHandle, mrHandle);

    mrManager_ = static_cast<MrManager*>(mrHandle);
    nicRdmaHandle_ = rdmaHandle;
    nicSocketHandle_ = socketInfo.socketHandle;

    SocketConnectInfoT tmpConnInfo{};
    tmpConnInfo.port = HETEROG_CCL_PORT;
    initSM_.locInitInfo.socketConnInfo.emplace_back(tmpConnInfo);
    initSM_.locInitInfo.socketInfo.emplace_back(socketInfo);

    CHK_RET(CheckRecvMsgAndRequestBuffer());

    CHK_RET(PreQpConnect());

    CHK_RET(TryTransition(HCCL_SUCCESS, true, ConnState::CONN_STATE_GET_TAG_QP_ATTR));

    CHK_RET(ConnectAsync());

    HCCL_INFO("TransportHeterogRoce Init success");
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::CreateCqAndQp()
{
    CHK_RET(
        CreateQpWithSharedCq(nicRdmaHandle_, selfIp_, peerIp_, -1, -1, tagQpInfo_, TAG_QP_APPEND, MAX_SCATTER_BUF_NUM));
    CHK_RET(CreateQpWithSharedCq(
        nicRdmaHandle_, selfIp_, peerIp_, -1, -1, dataQpInfo_, DATA_QP_APPEND, MAX_SCATTER_BUF_NUM));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::DestroyCqAndQp()
{
    HCCL_INFO("TransportHeterogRawRoce DestroyCqAndQp");
    CHK_RET(DestroyQpWithSharedCq(tagQpInfo_, TAG_QP_APPEND));
    tagQpInfo_ = QpInfo();
    CHK_RET(DestroyQpWithSharedCq(dataQpInfo_, DATA_QP_APPEND));
    dataQpInfo_ = QpInfo();
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::PreQpConnect()
{
    CHK_RET(TransportHeterogRoce::PreQpConnect());

    CHK_SAFETY_FUNC_RET(memset_s(&dataReadWrScatter_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    dataReadWrScatter_.next = &dataAckWrScatter_;
    dataReadWrScatter_.opcode = IBV_WR_RDMA_READ;
    dataReadWrScatter_.send_flags = IBV_SEND_SIGNALED | IBV_SEND_FENCE;

    CHK_SAFETY_FUNC_RET(memset_s(&dataAckWrScatter_, sizeof(struct ibv_send_wr), 0, sizeof(struct ibv_send_wr)));
    dataAckWrScatter_.sg_list = &dataAckSge_;
    dataAckWrScatter_.next = nullptr;
    dataAckWrScatter_.num_sge = 1;
    dataAckWrScatter_.opcode = IBV_WR_SEND_WITH_IMM;
    dataAckWrScatter_.send_flags = IBV_SEND_FENCE | IBV_SEND_INLINE;

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::EnterStateProcess(ConnState nextState)
{
    switch (nextState) {
        case ConnState::CONN_STATE_GET_TAG_QP_ATTR:
            break;
        case ConnState::CONN_STATE_SEND_TAG_QP_INFO:
            CHK_RET(PrepareModifyInfo(localTagQpAttr_, localTagModifyInfo_));
            initSM_.size = sizeof(localTagModifyInfo_);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_RECV_TAG_QP_INFO:
            initSM_.size = sizeof(remoteTagModifyInfo_);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_MODIFY_TAG_QP:
            break;
        case ConnState::CONN_STATE_GET_DATA_QP_ATTR:
            break;
        case ConnState::CONN_STATE_SEND_DATA_QP_INFO:
            CHK_RET(PrepareModifyInfo(localDataQpAttr_, localDataModifyInfo_));
            initSM_.size = sizeof(localDataModifyInfo_);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_RECV_DATA_QP_INFO:
            initSM_.size = sizeof(remoteDataModifyInfo_);
            initSM_.completeSize = 0;
            break;
        case ConnState::CONN_STATE_MODIFY_DATA_QP:
            break;
        case ConnState::CONN_STATE_SEND_STATUS:
            initSM_.locInitInfo.signal = SYNC_SIGNAL;
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

HcclResult TransportHeterogRawRoce::LoopStateProcess()
{
    HcclResult testRet = HCCL_SUCCESS;
    bool completed = false;
    switch (GetState()) {
        case ConnState::CONN_STATE_GET_TAG_QP_ATTR:
            testRet = GetQpAttr(tagQpInfo_.qpHandle, &localTagQpAttr_, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_SEND_TAG_QP_INFO));
            break;
        case ConnState::CONN_STATE_SEND_TAG_QP_INFO:
            testRet = SocketSend(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &localTagModifyInfo_, initSM_.size, initSM_.completeSize,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_RECV_TAG_QP_INFO));
            break;
        case ConnState::CONN_STATE_RECV_TAG_QP_INFO:
            testRet = SocketRecv(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &remoteTagModifyInfo_, initSM_.size, initSM_.completeSize,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_MODIFY_TAG_QP));
            break;
        case ConnState::CONN_STATE_MODIFY_TAG_QP:
            testRet = TypicalQpModify(tagQpInfo_.qpHandle, &localTagModifyInfo_, &remoteTagModifyInfo_, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_GET_DATA_QP_ATTR));
            break;
        case ConnState::CONN_STATE_GET_DATA_QP_ATTR:
            testRet = GetQpAttr(dataQpInfo_.qpHandle, &localDataQpAttr_, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_SEND_DATA_QP_INFO));
            break;
        case ConnState::CONN_STATE_SEND_DATA_QP_INFO:
            testRet = SocketSend(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &localDataModifyInfo_, initSM_.size, initSM_.completeSize,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_RECV_DATA_QP_INFO));
            break;
        case ConnState::CONN_STATE_RECV_DATA_QP_INFO:
            testRet = SocketRecv(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &remoteDataModifyInfo_, initSM_.size, initSM_.completeSize,
                completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_MODIFY_DATA_QP));
            break;
        case ConnState::CONN_STATE_MODIFY_DATA_QP:
            testRet = TypicalQpModify(dataQpInfo_.qpHandle, &localDataModifyInfo_, &remoteDataModifyInfo_, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_SEND_STATUS));
            break;
        case ConnState::CONN_STATE_SEND_STATUS:
            testRet = SocketSend(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &(initSM_.locInitInfo.signal), initSM_.size,
                initSM_.completeSize, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_RECV_STATUS));
            break;
        case ConnState::CONN_STATE_RECV_STATUS:
            testRet = SocketRecv(
                initSM_.locInitInfo.socketInfo[0].fdHandle, &(initSM_.remInitInfo.signal), initSM_.size,
                initSM_.completeSize, completed);
            CHK_RET(TryTransition(testRet, completed, ConnState::CONN_STATE_COMPLETE));
            break;
        default:
            HCCL_ERROR(
                "Establish communication connection failed[%s]: state[%u]", initSM_.locInitInfo.socketInfo[0].tag,
                GetState());
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::PrepareModifyInfo(struct QpAttr& qpAttr, struct TypicalQp& typicalQpInfo)
{
    typicalQpInfo.qpn = qpAttr.qpn;
    typicalQpInfo.psn = qpAttr.psn;
    typicalQpInfo.gidIdx = qpAttr.gidIdx;
    typicalQpInfo.tc = GetExternalInputRdmaTrafficClass();
    typicalQpInfo.sl = GetExternalInputRdmaServerLevel();
    typicalQpInfo.retryCnt = GetExternalInputRdmaRetryCnt();
    typicalQpInfo.retryTime = GetExternalInputRdmaTimeOut();
    CHK_SAFETY_FUNC_RET(memcpy_s(typicalQpInfo.gid, HCCP_GID_RAW_LEN, qpAttr.gid, HCCP_GID_RAW_LEN));
    HCCL_INFO(
        "TransportHeterogRawRoce ModifyInfo qpn[%u] psn[%u] gid_idxp[%u] gid[%p] tc[%u] sl[%u] retryCnt[%u] "
        "retryTime[%u]",
        typicalQpInfo.qpn, typicalQpInfo.psn, typicalQpInfo.gidIdx, typicalQpInfo.gid, typicalQpInfo.tc,
        typicalQpInfo.sl, typicalQpInfo.retryCnt, typicalQpInfo.retryTime);

    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::ImrecvScatter(
    void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessageInfo& msg, HcclRequestInfo*& request)
{
    CHK_RET(CheckRecvScatterEnvelope(buf, count, bufCount, datatype, msg.envelope));
    CHK_RET(GenerateRecvScatterRequest(msg, request));

    HcclEnvelope& envelope = msg.envelope.envelope;
    u32 dataSize = SIZE_TABLE[datatype];
    for (s32 i = 0; i < bufCount; i++) {
        u32 lkey = 0;
        u64 dstBuf = reinterpret_cast<u64>(buf[i]);
        CHK_RET(RegMr(reinterpret_cast<void*>(dstBuf), static_cast<u64>(count[i] * dataSize), lkey));
        dataReadSgeArry_[i].lkey = lkey;
    }

    for (s32 i = 0; i < bufCount; i++) {
        dataReadSgeArry_[i].addr = reinterpret_cast<uint64_t>(buf[i]);
    }

    for (s32 i = 0; i < bufCount; i++) {
        dataReadSgeArry_[i].length = count[i] * dataSize;
    }

    dataReadWrScatter_.sg_list = dataReadSgeArry_;
    dataReadWrScatter_.num_sge = bufCount;
    dataReadWrScatter_.wr_id = reinterpret_cast<uint64_t>(request);
    dataReadWrScatter_.wr.rdma.remote_addr = static_cast<uint64_t>(envelope.transData.srcBuf);
    dataReadWrScatter_.wr.rdma.rkey = envelope.key;

    dataAckSge_.addr = reinterpret_cast<uint64_t>(&envelope.msn);
    dataAckSge_.length = sizeof(uint64_t);
    dataAckSge_.lkey = 0;
    dataAckWrScatter_.wr_id = 0;

    struct ibv_send_wr* badWr = nullptr;
    HCCL_INFO(
        "rdma read: remote addr[%llu] count[%d] datatype[%s] wrId[%llu] num_sge[%d] qpHandle[%p]",
        reinterpret_cast<u64>(envelope.transData.srcBuf), envelope.transData.count,
        GetDataTypeEnumStr(envelope.transData.dataType).c_str(), dataReadWr_.wr_id, bufCount, dataQpInfo_.qp);
    HcclResult ret = hrtIbvPostSend(dataQpInfo_.qp, &dataReadWrScatter_, &badWr);
    if (ret != HCCL_SUCCESS) {
        if (ret == HCCL_E_AGAIN) {
            HCCL_WARNING("rdma read post send wqe overflow.[%d]", ret);
        } else {
            HCCL_ERROR(
                "rdma read fail: remote addr[%llx] count[%d] datatype[%s] wrId[%llu] bufCount[%d]",
                reinterpret_cast<u64>(envelope.transData.srcBuf), envelope.transData.count,
                GetDataTypeEnumStr(envelope.transData.dataType).c_str(), dataReadWr_.wr_id, bufCount);
        }
        return ret;
    }
    HCCL_INFO("rdma send ack: msn:0x%016llx request:%p", envelope.msn, &request);

    CHK_RET(FreeRecvMessage(msg));
    return HCCL_SUCCESS;
}

HcclResult TransportHeterogRawRoce::GetQpAttr(QpHandle& qpHandle, struct QpAttr* attr, bool& completed)
{
    HcclResult ret = hrtRaGetQpAttr(qpHandle, attr);
    if (ret == HCCL_SUCCESS) {
        completed = true;
    } else {
        HCCL_ERROR(
            "GetQpAttr fail qpHandle[%p] qpn[%u] udpSport[%u] psn[%u] gidIdx[%u] gid[%p] completed[%u]", qpHandle,
            attr->qpn, attr->udpSport, attr->psn, attr->gidIdx, attr->gid, completed);
    }
    return ret;
}

HcclResult TransportHeterogRawRoce::TypicalQpModify(
    QpHandle& qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo, bool& completed)
{
    HcclResult ret = hrtRaTypicalQpModify(qpHandle, localQpInfo, remoteQpInfo);
    if (ret == HCCL_SUCCESS) {
        completed = true;
    } else if (ret != HCCL_E_AGAIN) {
        HCCL_ERROR(
            "hrtRaTypicalQpModify fail qpHandle[%p] completed[%u] "
            "local: qpn[%u] psn[%u] gidIdx[%u] gid[%p] tc[%u] sl[%u] retryCnt[%u] retryTime[%u]"
            "remote: qpn[%u] psn[%u] gidIdx[%u] gid[%p] tc[%u] sl[%u] retryCnt[%u] retryTime[%u] ",
            qpHandle, completed, localQpInfo->qpn, localQpInfo->psn, localQpInfo->gidIdx, localQpInfo->gid,
            localQpInfo->tc, localQpInfo->sl, localQpInfo->retryCnt, localQpInfo->retryTime, remoteQpInfo->qpn,
            remoteQpInfo->psn, remoteQpInfo->gidIdx, remoteQpInfo->gid, remoteQpInfo->tc, remoteQpInfo->sl,
            remoteQpInfo->retryCnt, remoteQpInfo->retryTime);
    }
    return ret;
}
} // namespace hccl
