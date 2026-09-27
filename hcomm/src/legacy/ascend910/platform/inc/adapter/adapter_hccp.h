/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HCCL_INC_ADAPTER_HCCP_H
#define HCOMM_HCCL_INC_ADAPTER_HCCP_H

#include <functional>

#include "hccl/base.h"
// ltm路径指定
#include "network/hccp.h"
#include "hccl_common.h"
#include "workflow_pub.h"
#include "network/hccp_tlv.h"
#include "adapter_hccp_common.h"

constexpr u64 SOCKET_SEND_MAX_SIZE = 0x7FFFFFFFFFFFFFFF;
constexpr u32 MAX_VALUE_U32 = 0xFFFFFFFF; // u32 数据类型最大值
constexpr u32 MAX_SRQ_DEPTH = 16 * 1024 - 1;
constexpr u32 DEFAULT_INIT_PHY_ID = 0;
constexpr u32 DEFAULT_INIT_NIC_POS = 0;
constexpr u32 DEFAULT_HDC_TYPE = 6;
constexpr u32 PID_HDC_TYPE = 18;
constexpr u32 DEFAULT_INIT_RDMA_CONFIG = 0; // 初始化rdma rdev_init_info默认初始化参数
constexpr u32 MAX_PORT_ID = 65535;
constexpr u32 MIN_PORT_ID = 1024;
constexpr u32 AUTO_LISTEN_PORT = 0;
constexpr u32 MAX_SEND_SGE_NUM = 8;
constexpr u32 HOST_SQ_CQ_DEPTH = 8192;  // 8K
constexpr u32 AICPU_SQ_CQ_DEPTH = 2048; // 2K

// QP CQ default attr
constexpr u32 DEFAULT_OPBASE_MAX_SEND_WR = 32768;
constexpr u32 DEFAULT_OFFLINE_MAX_SEND_WR = 128;
constexpr u32 DEFAULT_MAX_RECV_WR = 64;
constexpr u32 DEFAULT_MAX_SEND_SGE = 1;
constexpr u32 DEFAULT_MAX_RECV_SGE = 1;
constexpr u32 DEFAULT_MAX_SEND_CQ_DEPTH = 32768;
constexpr u32 DEFAULT_MAX_ONE_SIDED_SEND_CQ_DEPTH = 512;
constexpr u32 DEFAULT_MAX_RECV_CQ_DEPTH = 64;
constexpr u32 DEFAULT_MAX_INLINE_DATA = 32;
constexpr u32 HETEROG_OFFLINE_EXT_MAX_SEND_WR = 512;

constexpr u32 AI_QP_CREATE = 68;                   // RA_RS_AI_QP_CREATE 的 opcode为68
constexpr u32 AI_QP_CREATE_VERSION = 2;            // 当前支持的版本号为2
constexpr u32 AI_NORMAL_QP_CREATE_VERSION = 3;     // 支持创建NormalQP的版本号为3
constexpr u32 AI_QP_CREATE_WITH_ATTRS = 86;        // RA_RS_AI_QP_CREATE_WITH_ATTRS  的 opcode为86
constexpr u32 AI_QP_CREATE_WITH_ATTRS_VERSION = 1; // 当前支持的版本号为1

constexpr u32 GET_HCCH_CFG = 100;
constexpr u32 GET_HCCH_CFG_VERSION = 1; // 当前支持的版本号为1

const std::string SOC_NAME_910B = "Ascend910B";

constexpr u32 RA_RS_GET_ROCE_API = 96;        // RA_RS_GET_ROCE_API 的 opcode为96
constexpr u32 RA_RS_ATOMIC_WRITE_VERSION = 1; // 支持使用atomic write的版本号为1

constexpr u32 QP_DEPTH_MAX = 32768;
constexpr u32 QP_DEPTH_MIN = 64;

using QpConfig = struct QpConfigDef {
    hccl::HcclIpAddress selfIp;
    hccl::HcclIpAddress peerIp;
    u32 maxWr;
    u32 maxSendSge;
    u32 maxRecvSge;
    s32 sqEvent;
    s32 rqEvent;

    QpConfigDef(
        hccl::HcclIpAddress& selfIp, hccl::HcclIpAddress& peerIp, u32 maxWr, u32 maxSendSge, u32 maxRecvSge,
        s32 sqEvent, s32 rqEvent)
        : selfIp(selfIp),
          peerIp(peerIp),
          maxWr(maxWr),
          maxSendSge(maxSendSge),
          maxRecvSge(maxRecvSge),
          sqEvent(sqEvent),
          rqEvent(rqEvent)
    {}
    QpConfigDef(u32 maxWr, u32 maxSendSge, u32 maxRecvSge, s32 sqEvent, s32 rqEvent)
        : maxWr(maxWr),
          maxSendSge(maxSendSge),
          maxRecvSge(maxRecvSge),
          sqEvent(sqEvent),
          rqEvent(rqEvent)
    {}
    QpConfigDef() : maxWr(0), maxSendSge(0), maxRecvSge(0), sqEvent(0), rqEvent(0) {}
};

using QpInfo = struct QpInfoDef {
    QpConfig attr;
    RdmaHandle rdmaHandle;
    QpHandle qpHandle;
    struct ibv_qp* qp;
    void* context;
    struct ibv_cq* sendCq;
    struct ibv_cq* recvCq;
    struct ibv_srq* srq;
    struct ibv_cq* srqCq;
    void* srqContext;
    struct ibv_comp_channel* sendChannel;
    struct ibv_comp_channel* recvChannel;
    s32 flag = 0;
    s32 qpMode = 0;
    u32 trafficClass;
    u32 serviceLevel;
    QpInfoDef()
        : rdmaHandle(nullptr),
          qpHandle(nullptr),
          qp(nullptr),
          context(nullptr),
          sendCq(nullptr),
          recvCq(nullptr),
          srq(nullptr),
          srqCq(nullptr),
          srqContext(nullptr),
          sendChannel(nullptr),
          recvChannel(nullptr),
          trafficClass(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
          serviceLevel(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
    {}
    QpInfoDef(
        QpConfig attr, RdmaHandle rdmaHandle, QpHandle qpHandle, struct ibv_qp* qp, void* context,
        struct ibv_cq* sendCq, struct ibv_cq* recvCq, struct ibv_srq* srq, struct ibv_cq* srqCq,
        void* srqContext = nullptr, struct ibv_comp_channel* sendChannel = nullptr,
        struct ibv_comp_channel* recvChannel = nullptr, u32 tc = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET,
        u32 sl = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET)
        : attr(attr),
          rdmaHandle(rdmaHandle),
          qpHandle(qpHandle),
          qp(qp),
          context(context),
          sendCq(sendCq),
          recvCq(recvCq),
          srq(srq),
          srqCq(srqCq),
          srqContext(srqContext),
          sendChannel(sendChannel),
          recvChannel(recvChannel),
          trafficClass(tc),
          serviceLevel(sl)
    {}
};

using QpConfigInfo = struct QpConfigInfoDef {
    uint32_t sq_depth;
    uint32_t rq_depth;
    uint32_t scq_depth;
    uint32_t rcq_depth;
    uint32_t use_resv_mem;
    uint32_t resv_mem_pool_id;
};

using QpConfigWithCQInfo = struct QpConfigWithCQInfoDef {
    uint32_t sq_depth;
    uint32_t rq_depth;
    uint32_t scq_depth;
    uint32_t rcq_depth;
    uint32_t sendCqn;
    uint32_t recvCqn;
    uint32_t use_resv_mem;
    uint32_t resv_mem_pool_id;
    int sq_sig_all;
    uint32_t max_send_sge;
    uint32_t max_recv_sge;
    uint32_t max_inline_data;
};

using CqInfo = struct CqInfoDef {
    struct ibv_cq* sq;
    struct ibv_cq* rq;
    void* context;
    u32 depth;
    u32 used;
    s32 sqEvent;
    s32 rqEvent;
    void* srqContext;
    struct ibv_comp_channel* sendChannel;
    struct ibv_comp_channel* recvChannel;
    std::vector<QpInfo> qps;
    CqInfoDef()
        : sq(nullptr),
          rq(nullptr),
          context(nullptr),
          depth(0),
          used(0),
          sqEvent(-1),
          rqEvent(-1),
          srqContext(nullptr),
          sendChannel(nullptr),
          recvChannel(nullptr)
    {}
    CqInfoDef(
        struct ibv_cq* sq, struct ibv_cq* rq, void* context, u32 depth, s32 sqEvent, s32 rqEvent,
        void* srqContext = nullptr, struct ibv_comp_channel* sendChannel = nullptr,
        struct ibv_comp_channel* recvChannel = nullptr)
        : sq(sq),
          rq(rq),
          context(context),
          depth(depth),
          used(0),
          sqEvent(sqEvent),
          rqEvent(rqEvent),
          srqContext(srqContext),
          sendChannel(sendChannel),
          recvChannel(recvChannel)
    {}
};

using SrqInfo = struct SrqInfoDef {
    struct ibv_srq* srq;
    struct ibv_cq* srqCq;
    void* context;
    s32 srqDepth;
    s32 srqEvent;
    SrqInfoDef() : srq(nullptr), srqCq(nullptr), context(nullptr), srqDepth(0), srqEvent(-1) {}
};

template <typename connStruct>
void CheckConnPort(connStruct& conn, u32 num)
{
    for (u32 i = 0; i < num; i++) {
        if (conn[i].port > MAX_PORT_ID) {
            // 未定义或者已定义但不合法的情况下，port默认赋值16666
            HCCL_WARNING("Port is invalid, set to 16666!");
            conn[i].port = HETEROG_CCL_PORT;
        }
    }
}

inline HcclResult CheckQpDepth(uint32_t depth)
{
    if ((depth == INVALID_UINT) || (depth >= QP_DEPTH_MIN && ((depth & (depth - 1)) == 0) && depth <= QP_DEPTH_MAX)) {
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}

HcclResult
ConstructQpAttrs(s32 qpMode, struct QpExtAttrs& attrs, const QueueDepthAttr& qpDepth, bool isWorkFlowLib = false);

HcclResult HrtRaGetQpDepth(RdmaHandle rdmaHandle, unsigned int* tempDepth, unsigned int* qpNum);
HcclResult HrtRaSetQpDepth(RdmaHandle rdmaHandle, unsigned int tempDepth, unsigned int* qpNum);
HcclResult HrtRaQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode, QpHandle& qpHandle);
HcclResult HrtRaQpDestroy(QpHandle handle);
HcclResult HrtRaQpDestroyWithoutCQ(QpHandle handle);
HcclResult HrtRaQpNonBlockConnectAsync(QpHandle handle, const SocketHandle sockHandle);
HcclResult HrtRaQpConnectAsync(
    QpHandle handle, const SocketHandle sockHandle,
    std::function<bool()> needStop =
        []() {
            return false;
        },
    u32 timeout = 0);
s32 hrtGetRaQpStatus(QpHandle handle, int* status);
HcclResult HrtRaMrReg(QpHandle handle, struct MrInfoT* mrInfo);
HcclResult HrtRaGetNotifyMrInfo(u32 phyId, RdmaHandle handle, struct MrInfoT* mrInfo);
HcclResult HrtRaMrDereg(QpHandle handle, struct MrInfoT* mrInfo);
HcclResult HrtRaSendWr(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp);
HcclResult HrtRaSendWrV2(
    QpHandle handle, struct SendWrV2* wr, struct SendWrRsp* opRsp,
    HcclWorkflowMode workflowMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
HcclResult HrtRaSendWrVerbs(QpHandle handle, struct SendWrVerbs* wr, struct SendWrRsp* opRsp);
HcclResult HrtRaRecvWrVerbs(QpHandle handle, struct RecvWrVerbs* wr);
s32 hrtRaPollCq(QpHandle handle, bool is_send_cq, unsigned int num, void* wc);
s32 HrtRaPollTypicalCq(void* cqHandle, u32 num, void* wc);
HcclResult HrtRaSendWrlist(
    QpHandle handle, struct SendWrlistData wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum);
HcclResult HrtRaSendWrlistExt(
    QpHandle handle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
    unsigned int* completeNum);
HcclResult HrtRaSendNormalWrlist(
    QpHandle handle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum);
HcclResult HrtRaGetNotifyBaseAddr(
    RdmaHandle handle, u64* va, u64* size, std::function<bool()> needStop = []() {
        return false;
    });
HcclResult HrtRaInit(struct RaInitConfig* config);
HcclResult HrtRaDeInit(struct RaInitConfig* config);

HcclResult HrtRaRdmaInit(int mode, u32 notifyType, struct rdev rdevInfo, RdmaHandle& rdmaHandle);
HcclResult HrtRaRdmaInitRef(int mode, u32 notifyType, const struct rdev& rdevInfo, RdmaHandle& rdmaHandle);
HcclResult HrtRaRdmaInitWithAttr(struct RdevInitInfo& init_info, const struct rdev& rdevInfo, RdmaHandle& rdmaHandle);
HcclResult HrtRdmaInitWithBackupAttr(
    struct RdevInitInfo& init_info, struct rdev& rdevInfo, struct rdev& backupRdevInfo, RdmaHandle& rdmaHandle);
HcclResult HrtRaRdmaGetHandle(unsigned int phyId, RdmaHandle& rdmaHandle);
HcclResult HrtRaRdmaDeInit(RdmaHandle& rdmaHandle, u32 notifyType);
HcclResult HrtRaRdmaDeInitRef(RdmaHandle& rdmaHandle, u32 notifyType);
HcclResult HrtGetRdmaLiteStatus(RdmaHandle rdmaHandle, int* supportLite);
HcclResult hrtRaSocketInit(int mode, struct rdev rdevInfo, SocketHandle& socketHandle);
HcclResult hrtRaSocketInitRef(int mode, const struct rdev& rdevInfo, SocketHandle& socketHandle);
HcclResult hrtRaSocketInitV1(int mode, struct SocketInitInfoT socket_init, SocketHandle& socketHandle);
HcclResult hrtRaSocketDeInit(SocketHandle& socketHandle);
HcclResult hrtRaSocketDeInitRef(SocketHandle& socketHandle);

HcclResult hrtRaSocketNonBlockListenStart(struct SocketListenInfoT conn[], u32 num);
HcclResult hrtRaSocketListenStart(struct SocketListenInfoT conn[], u32 num);
HcclResult hrtRaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], u32 num, u32 creditLimit);
HcclResult hrtRaSocketListenStop(struct SocketListenInfoT conn[], u32 num);
HcclResult hrtRaSocketNonBlockBatchConnect(struct SocketConnectInfoT conn[], u32 num);
HcclResult hrtRaSocketBatchConnect(
    struct SocketConnectInfoT conn[], u32 num, u32 maxLen = MAX_VALUE_U32, std::function<bool()> needStop = []() {
        return false;
    });
HcclResult hrtRaSocketBatchClose(struct SocketCloseInfoT conn[], u32 num, u32 maxLen = MAX_VALUE_U32);
s32 hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum);
HcclResult hrtRaNonBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum);
HcclResult hrtRaBlockGetSockets(u32 role, struct SocketInfoT conn[], u32 num);
HcclResult hrtRaSocketNonBlockSendHeterog(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize);
s32 hrtRaSocketNonBlockSend(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize);
HcclResult hrtRaSocketNonBlockSendHeart(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize);
HcclResult hrtRaSocketBlockSend(
    const FdHandle fdHandle, const void* data, u64 sendSize, std::function<bool()> needStop = []() {
        return false;
    });
HcclResult hrtRaSocketNonBlockRecvHeterog(const FdHandle fdHandle, void* data, u64 size, u64* recvSize);
s32 hrtRaSocketNonBlockRecv(const FdHandle fdHandle, void* data, u64 size, u64* recvSize);
HcclResult hrtRaSocketNonBlockRecvHeart(const FdHandle fdHandle, void* data, u64 size, u64* recvSize);
HcclResult hrtRaSocketBlockRecv(
    const FdHandle fdHandle, void* data, u64 size,
    std::function<bool()> needStop =
        []() {
            return false;
        },
    u32 timeout = 0);

s32 hrtRaSocketRecv(const FdHandle fdHandle, void* data, u64 size, u64* recvSize);

HcclResult IsSupportHdcAsync(bool& isSupportHdcAsync);
s32 hrtRaSocketSendAsync(const FdHandle fdHandle, const void* data, u64 size, u64* sentSize, void** reqHandle);
s32 hrtRaSocketRecvAsync(const FdHandle fdHandle, void* data, u64 size, u64* receivedSize, void** reqHandle);
s32 hrtRaSocketGetAsyncReqResult(void* reqHandle, s32* reqResult);

HcclResult hrtEpollCtlAdd(const FdHandle fdHandle, RaEpollEvent event);
HcclResult hrtEpollCtlMod(const FdHandle fdHandle, RaEpollEvent event);
HcclResult hrtEpollCtlDel(const FdHandle fdHandle);
HcclResult hrtSetRecvDataCallback(const SocketHandle socketHandle, const void* callback);

HcclResult hrtGetServerId(std::string& serverId);
HcclResult hrtGetIfNum(struct RaGetIfattr& config, u32& num);
HcclResult hrtGetIfAddress(struct RaGetIfattr& config, struct InterfaceInfo ifaddrInfos[], u32& num);
HcclResult hrtRaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion);
HcclResult hrtRaSocketSetWhiteListStatus(u32 enable);
HcclResult hrtRaSocketGetWhiteListStatus(u32& enable);
HcclResult hrtRaSocketWhiteListAdd(SocketHandle socketHandle, struct SocketWlistInfoT whiteList[], u32 num);
HcclResult hrtRaSocketWhiteListDel(SocketHandle socketHandle, struct SocketWlistInfoT whiteList[], u32 num);
HcclResult hrtRaRegGlobalMr(const RdmaHandle rdmaHandle, struct MrInfoT& mrInfo, MrHandle& mrHandle);
HcclResult hrtRaDeRegGlobalMr(const RdmaHandle rdmaHandle, MrHandle mrHandle);
HcclResult
hrtRaNormalQpCreate(RdmaHandle handle, struct ibv_qp_init_attr* initAttr, QpHandle& qpHandle, struct ibv_qp*& qp);
HcclResult hrtRaNormalQpDestroy(QpHandle qpHandle);
HcclResult hrtRaCreateCq(RdmaHandle handle, struct CqAttr* attr);
HcclResult hrtRaDestroyCq(RdmaHandle handle, struct CqAttr* attr);
HcclResult CreateNormalQp(RdmaHandle rdmaHandle, QpInfo& qp);
HcclResult CreateQp(RdmaHandle rdmaHandle, int& flag, s32& qpMode, QpInfo& qp, bool isESMode = false);
HcclResult CreateCqAndQp(RdmaHandle& rdmaHandle, std::string& label, QpConfig& config, QpInfo& info);
HcclResult CreateQpWithSharedCq(
    RdmaHandle rdmaHandle, hccl::HcclIpAddress& selfIp, hccl::HcclIpAddress& peerIp, s32 sqEvent, s32 rqEvent,
    QpInfo& info, s32 qpAppend = 0, u32 maxSegNum = MAX_SEND_SGE_NUM);
HcclResult DestroyQpWithSharedCq(const QpInfo& info, s32 qpAppend);
HcclResult CreateQpWithCq(
    RdmaHandle rdmaHandle, s32 sqEvent, s32 rqEvent, RdmaHandle sendChannel, RdmaHandle recvChannel, QpInfo& info,
    bool isHdcMode = false, bool isESMode = false);
HcclResult DestroyQpWithCq(const QpInfo& info, bool isHdcMode = false);
HcclResult CreateAiQp(RdmaHandle rdmaHandle, struct AiQpInfo& aiQpInfo, QpInfo& info, u32 devicePhyId);
HcclResult DestroyAiQp(const QpInfo& info);
HcclResult hrtRaSetQpAttrQos(QpHandle qpHandle, struct QosAttr& attr);
HcclResult hrtRaSetQpAttrTimeOut(QpHandle qpHandle, u32& timeOut);
HcclResult hrtRaSetQpAttrRetryCnt(QpHandle qpHandle, u32& retryCnt);
HcclResult SetQpAttrQos(
    QpHandle qpHandle, u32 tc = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET,
    u32 sl = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET);
HcclResult SetQpAttrTimeOut(QpHandle qpHandle, u32 timeout = INVALID_UINT);
HcclResult SetQpAttrRetryCnt(QpHandle qpHandle, u32 retryCnt = INVALID_UINT);
HcclResult hrtRaCreateCompChannel(RdmaHandle rdmaHandle, void** compChannel);
HcclResult hrtRaDestroyCompChannel(RdmaHandle rdmaHandle, void* compChannel);
HcclResult hrtRaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo* info);
HcclResult hrtRaGetCqeErrInfoList(RdmaHandle rdmaHandle, struct CqeErrInfo* infolist, u32* num);
HcclResult IsSuppCqeErrInfoListConfig(bool& supCqeErrInfoListConfig);
HcclResult IsSupportRaSendNormalWrlist(bool& isSupportRaSendNormalWrlist);
HcclResult hrtRaGetQpAttr(QpHandle qpHandle, struct QpAttr* attr);
HcclResult hrtRaCreateSrq(RdmaHandle rdmaHandle, SrqInfo& srqInfo);
HcclResult hrtRaDestroySrq(RdmaHandle rdmaHandle, SrqInfo& srqInfo);
HcclResult hrtRaRecvWrlist(QpHandle handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum);

HcclResult hrtRaQpCreateWithAttrs(RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, QpHandle& qpHandle);
HcclResult hrtRaQpCreateWithCQWithAttrs(
    RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, unsigned int sendCqn, unsigned int recvCqn, QpHandle& qpHandle);
HcclResult
hrtRaAiQpCreate(u32 phyId, RdmaHandle rdmaHandle, struct QpExtAttrs* attrs, struct AiQpInfo* info, QpHandle& qpHandle);

HcclResult IsSuppportRaGetSocketVnicIps(bool& supportGetSocketVnicIp);
HcclResult hrtRaGetSocketVnicIpInfos(
    u32 phyId, enum IdType type, const std::vector<u32>& deviceIds, std::vector<hccl::HcclIpAddress>& vnicIps);

HcclResult hrtRaPingInit(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, void** pingHandle);
HcclResult hrtRaPingDeinit(void* pingHandle);
HcclResult hrtRaPingTargetAdd(void* pingHandle, struct PingTargetInfo target[], uint32_t num);
HcclResult hrtRaPingTargetDel(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num);
HcclResult hrtRaPingTaskStart(void* pingHandle, struct PingTaskAttr* attr);
HcclResult hrtRaPingTaskStop(void* pingHandle);
HcclResult hrtRaPingGetResults(void* pingHandle, struct PingTargetResult target[], uint32_t* num);

HcclResult hrtRaIsFirstUsed(s32 insId, bool& used);
HcclResult hrtRaIsLastUsed(s32 insId, bool& used);
HcclResult hrtRaQpBatchModify(RdmaHandle rdmaHandle, QpHandle qpHandle[], unsigned int num, int expectStatus);
HcclResult
hrtRaTypicalQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode, struct TypicalQp* qpInfo, QpHandle& qpHandle);
HcclResult hrtRaTypicalQpModify(QpHandle qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo);
HcclResult hrtRaTypicalSendWr(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp);
HcclResult hrtRaRdevGetPortStatus(RdmaHandle rdmaHandle, enum PortStatus* status);
HcclResult HrtRaRemapMr(RdmaHandle rdmaHandle, struct MemRemapInfo info[], unsigned int num);
HcclResult HrtRaGetTlsEnable(struct RaInfo* info, bool* tlsEnable);
// 目前该接口只支持peer模式，且只适用于终止未建链成功的链路，即未get_socket成功的链路
HcclResult hrtRaSocketNonBlockBatchAbort(SocketConnectInfoT conn[], u32 num);

HcclResult CreateQpWithDepthConfig(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigInfo& qpConfig, QpHandle& qpHandle, struct TypicalQp& qpInfo);
HcclResult CreateQpWithCQConfig(
    RdmaHandle rdmaHandle, s32 qpMode, const QpConfigWithCQInfo& qpConfig, QpHandle& qpHandle,
    struct TypicalQp& qpInfo);
HcclResult CreateTypicalCq(RdmaHandle rdmaHandle, u32 cqDepth, u32& cqn, void** cqHandle);
HcclResult DestroyTypicalCq(RdmaHandle rdmaHandle, u32 cqn, void* cqHandle);
HcclResult IsSupportRaSocketAbort(bool& isSupportRaSocketAbort);
HcclResult hrtRaGetSecRandom(struct RaInfo* info, unsigned int* token);
HcclResult hrtRaGetDevEidInfoNum(RaInfo info, unsigned int* num);
HcclResult hrtRaGetDevEidInfoList(RaInfo info, struct HccpDevEidInfo* eid_info, unsigned int* num);
#endif
