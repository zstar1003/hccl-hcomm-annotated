/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLRAFUNCTION_H
#define HCCL_SRC_DLRAFUNCTION_H

#include <functional>
#include <atomic>
#include <mutex>
#include <hccl/hccl_types.h>
#include <infiniband/verbs.h>
#include "hccl/base.h"
#include "network/hccp.h"
#include "network/hccp_ping.h"
#include "private_types.h"
#include "hccl_common.h"

namespace hccl {
class DlRaFunction {
public:
    class Init;
    virtual ~DlRaFunction();
    static DlRaFunction& GetInstance();
    HcclResult DlRaFunctionInit();
    std::function<int(RdmaHandle rdmaHandle, unsigned int* tempDepth, unsigned int* qpNum)> dlRaGetQpDepth;
    std::function<int(RdmaHandle rdmaHandle, unsigned int tempDepth, unsigned int* qpNum)> dlRaSetQpDepth;
    std::function<int(RdmaHandle rdmaHandle, int flag, int qpMode, QpHandle* qpHandle)> dlRaQpCreate;
    std::function<int(void* qpHandle, void** qp, void** sendCq, void** recvCq)> dlRaGetQpContext;
    std::function<int(QpHandle handle)> dlRaQpDestroy;
    std::function<int(QpHandle handle)> dlRaQpDestroyWithoutCQ;
    std::function<int(QpHandle handle, const SocketHandle sockHandle)> dlRaQpConnectAsync;
    std::function<int(QpHandle handle, int* status)> dlRaGetQpStatus;
    std::function<int(QpHandle handle, struct MrInfoT* info)> dlRaMrReg;
    std::function<int(RdmaHandle rdmaHandle, struct MrInfoT* info)> dlRaGetNotifyMrInfo;
    std::function<int(QpHandle handle, struct MrInfoT* info)> dlRaMrDereg;
    std::function<int(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp)> dlRaSendWr;
    std::function<int(QpHandle handle, struct SendWrV2* wr, struct SendWrRsp* opRsp)> dlRaSendWrV2;
    std::function<int(QpHandle handle, struct SendWrVerbs* wr, struct SendWrRsp* opRsp)> dlRaSendWrVerbs;
    std::function<int(QpHandle handle, struct RecvWrVerbs* wr)> dlRaRecvWrVerbs;
    std::function<int(QpHandle handle, bool is_send_cq, unsigned int num, void* wc)> dlRaPollCq;
    std::function<int(void* cqHandle, unsigned int num, void* wc)> dlRaPollTypicalCq;
    std::function<int(
        QpHandle handle, struct SendWrlistData wr[], struct SendWrRsp op_rsp[], unsigned int sendNum,
        unsigned int* completeNum)>
        dlRaSendWrlist;
    std::function<int(
        QpHandle handle, struct SendWrlistDataExt wr[], struct SendWrRsp op_rsp[], unsigned int sendNum,
        unsigned int* completeNum)>
        dlRaSendWrlistExt;
    std::function<int(
        QpHandle handle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum, unsigned int* completeNum)>
        dlRaSendNormalWrlist;
    std::function<int(RdmaHandle handle, unsigned long long* va, unsigned long long* size)> dlRaGetNotifyBaseAddr;
    std::function<int(struct RaInitConfig* config)> dlRaInit;
    std::function<int(struct RaInitConfig* config)> dlRaDeInit;
    std::function<int(int mode, u32 notifyType, struct rdev rdevInfo, RdmaHandle* rdmaHandle)> dlRaRdmaInit;
    std::function<int(struct RdevInitInfo init_info, struct rdev rdevInfo, RdmaHandle* rdmaHandle)>
        dlRaRdmaInitWithAttr;
    std::function<int(
        struct RdevInitInfo* init_info, struct rdev* rdevInfo, struct rdev* backupRdevInfo, RdmaHandle* rdmaHandle)>
        dlRaRdmaInitWithBackupAttr;
    std::function<int(unsigned int phyId, RdmaHandle* rdmaHandle)> dlRaRdmaGetHandle;
    std::function<int(RdmaHandle handle, u32 notifyType)> dlRaRdmaDeInit;
    std::function<int(int mode, struct rdev rdevInfo, SocketHandle* socketHandle)> dlRaSocketInit;
    std::function<int(int mode, struct SocketInitInfoT rdevInfo, SocketHandle* socketHandle)> dlRaSocketInitV1;
    std::function<int(SocketHandle handle)> dlRaSocketDeInit;
    std::function<int(struct SocketListenInfoT conn[], unsigned int num)> dlRaSocketListenStart;
    std::function<int(struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit)>
        dlRaSocketAcceptCreditAdd;
    std::function<int(struct SocketListenInfoT conn[], unsigned int num)> dlRaSocketListenStop;
    std::function<int(struct SocketConnectInfoT conn[], unsigned int num)> dlRaSocketBatchConnect;
    std::function<int(struct SocketConnectInfoT conn[], unsigned int num)> dlRaSocketBatchAbort;
    std::function<int(struct SocketCloseInfoT conn[], unsigned int num)> dlRaSocketBatchClose;
    std::function<int(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connected_num)>
        dlRaGetSockets;
    std::function<int(const FdHandle, const void*, unsigned long long, unsigned long long* sent_size)> dlRaSocketSend;
    std::function<int(const FdHandle fdHandle, void* data, unsigned long long size, unsigned long long* received_size)>
        dlRaSocketRecv;

    std::function<int(
        const FdHandle fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize,
        void** reqHandle)>
        dlRaSocketSendAsync;
    std::function<int(
        const FdHandle fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize,
        void** reqHandle)>
        dlRaSocketRecvAsync;
    std::function<int(void* reqHandle, int* reqResult)> dlRaGetAsyncReqResult;

    std::function<int(unsigned int enable)> dlRaSocketSetWhiteListStatus;
    std::function<int(unsigned int* enable)> dlRaSocketGetWhiteListStatus;
    std::function<int(SocketHandle handle, struct SocketWlistInfoT whiteList[], unsigned int num)>
        dlRaSocketWhiteListAdd;
    std::function<int(SocketHandle handle, struct SocketWlistInfoT whiteList[], unsigned int num)>
        dlRaSocketWhiteListDel;
    std::function<int(struct RaGetIfattr* config, unsigned int* num)> dlRaGetIfNum;
    std::function<int(struct RaGetIfattr* config, struct InterfaceInfo interface_infos[], unsigned int* num)>
        dlRaGetIfAddress;
    std::function<int(unsigned int phyId, unsigned int interface_opcode, unsigned int* interface_version)>
        dlRaGetInterfaceVersion;
    std::function<int(const RdmaHandle rdmaHandle, struct MrInfoT* mrInfo, MrHandle* mrHandle)> dlRaRegGlobalMr;
    std::function<int(const RdmaHandle rdmaHandle, MrHandle mrHandle)> dlRaDeRegGlobalMr;
    std::function<int(const FdHandle fdHandle, RaEpollEvent event)> dlRaEpollCtlAdd;
    std::function<int(const FdHandle fdHandle, RaEpollEvent event)> dlRaEpollCtlMod;
    std::function<int(const FdHandle fdHandle)> dlRaEpollCtlDel;
    std::function<int(const SocketHandle socketHandle, const void* callback)> dlRaSetRecvDataCallback;
    std::function<int(RdmaHandle handle, struct CqAttr* attr)> dlRaCreateCq;
    std::function<int(RdmaHandle handle, struct CqAttr* attr)> dlRaDestroyCq;
    std::function<int(RdmaHandle handle, struct ibv_qp_init_attr* initAttr, QpHandle* qpHandle, void** qp)>
        dlRaNormalQpCreate;
    std::function<int(QpHandle handle)> dlRaNormalQpDestroy;
    std::function<int(QpHandle qpHandle, struct QosAttr* attr)> dlRaSetQpAttrQos;
    std::function<int(QpHandle qpHandle, unsigned int* timeout)> dlRaSetQpAttrTimeOut;
    std::function<int(QpHandle qpHandle, unsigned int* retryCnt)> dlRaSetQpAttrRetryCnt;
    std::function<int(RdmaHandle rdmaHandle, void** compChannel)> dlRaCreateCompChannel;
    std::function<int(RdmaHandle rdmaHandle, void* compChannel)> dlRaDestroyCompChannel;
    std::function<int(unsigned int phyId, struct CqeErrInfo* info)> dlRaGetCqeErrInfo;
    std::function<int(RdmaHandle rdmaHandle, struct CqeErrInfo* info_list, u32* number)> dlRaGetCqeErrInfoList;
    std::function<int(QpHandle qpHandle, struct QpAttr* attr)> dlRaGetQpAttr;
    std::function<int(RdmaHandle rdmaHandle, struct SrqAttr* attr)> dlRaCreateSrq;
    std::function<int(RdmaHandle rdmaHandle, struct SrqAttr* attr)> dlRaDestroyeSrq;
    std::function<int(int* event_handle)> dlRaCreateEventHandle;
    std::function<int(int event_handle, const void* fdHandle, int opcode, RaEpollEvent event)> dlRaCtlEventHandle;

    std::function<int(
        int event_handle, struct SocketEventInfoT* event_infos, int timeout, unsigned int maxevents,
        unsigned int* events_num)>
        dlRaWaitEventHandle;

    std::function<int(int* event_handle)> dlRaDestroyEventHandle;
    std::function<int(QpHandle handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)>
        dlRaRecvWrlist;

    std::function<int(RdmaHandle rdmaHandle, struct QpExtAttrs* ext_attrs, QpHandle* qpHandle)> dlRaQpCreateWithAttrs;
    std::function<int(RdmaHandle rdmaHandle, int flag, int qpMode, struct TypicalQp* qpInfo, QpHandle* qpHandle)>
        dlRaTypicalQpCreate;
    std::function<int(RdmaHandle rdmaHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo)>
        dlRaTypicalQpModify;
    std::function<int(QpHandle handle, struct SendWr* wr, struct SendWrRsp* opRsp)> dlRaTypicalSendWr;
    std::function<int(RdmaHandle rdmaHandle, unsigned int cqDepth, unsigned int* cqn, void** cqHandle)>
        dlRaTypicalCqCreate;
    std::function<int(RdmaHandle rdmaHandle, unsigned int cqn, void* cqHandle)> dlRaTypicalCqDestroy;
    std::function<int(
        RdmaHandle rdmaHandle, struct QpExtAttrs* ext_attrs, unsigned int sendCqn, unsigned int recvCqn,
        QpHandle* qpHandle)>
        dlRaQpCreateWithCQWithAttrs;

    std::function<int(RdmaHandle rdmaHandle, struct QpExtAttrs* ext_attrs, struct AiQpInfo* info, QpHandle* qpHandle)>
        dlRaAiQpCreate;

    std::function<int(
        unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])>
        dlRaGetSocketVnicIpInfos;
    std::function<int(RdmaHandle rdmaHandle, int* supportlite)> dlRaGetRdmaLiteStatus;
    std::function<int(RdmaHandle rdmaHandle, void** qpHandle, unsigned int num, int expectStatus)> dlRaQpBatchModify;

    std::function<int(struct PingInitAttr* init_attr, struct PingInitInfo* init_info, void** pingHandle)> dlRaPingInit;
    std::function<int(void* pingHandle)> dlRaPingDeinit;
    std::function<int(void* pingHandle, struct PingTargetInfo target[], uint32_t num)> dlRaPingTargetAdd;
    std::function<int(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num)> dlRaPingTargetDel;
    std::function<int(void* pingHandle, struct PingTaskAttr* attr)> dlRaPingTaskStart;
    std::function<int(void* pingHandle)> dlRaPingTaskStop;
    std::function<int(void* pingHandle, struct PingTargetResult target[], uint32_t* num)> dlRaPingGetResults;

    std::function<int(int ins_id)> dlRaIsFirstUsed{};
    std::function<int(int ins_id)> dlRaIsLastUsed{};
    std::function<int(RdmaHandle rdmaHandle, enum PortStatus* status)> dlRaRdevGetPortStatus{};
    std::function<int(RdmaHandle rdmaHandle, struct MemRemapInfo info[], unsigned int num)> dlRaRemapMr{};
    std::function<int(struct TlvInitInfo* init_info, uint32_t* buffer_size, void** tlv_handle)> dlH2DTlvInit;
    std::function<int(void* tlv_handle)> dlH2DTlvDeinit;
    std::function<int(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg)>
        dlH2DTlvRequest;
    std::function<int(struct RaInfo* info, bool* tls_enable)> dlRaRaGetTlsEnable;
    std::function<int(struct RaInfo* info, enum SaveSnapshotAction action)> dlRaSaveSnapShot;
    std::function<int(struct RaInfo* info)> dlRaRestoreSnapShot;
    std::function<int(struct RaInfo* info, enum HccnCfgKey ext_attrs, char* value, int* value_len)> dlRaGetHccnCfg;
    std::function<int(struct RaInfo* info, unsigned int*)> dlRaGetSecRandom;
    std::function<int(RaInfo info, unsigned int*)> dlRaGetDevEidInfoNum;
    std::function<int(RaInfo info, struct HccpDevEidInfo* eid_info, unsigned int*)> dlRaGetDevEidInfoList;

protected:
private:
    friend Init;
    void* handle_;
    std::mutex handleMutex_;
    static DlRaFunction* hcclDlRaFunction;
    DlRaFunction(const DlRaFunction&);
    DlRaFunction& operator=(const DlRaFunction&);
    DlRaFunction();
    HcclResult DlRaFunctionSocketInit();
    HcclResult DlRaFunctionRdmaInit();
};

class DlRaFunction::Init {
public:
    Init();
    ~Init();

private:
    static std::atomic<unsigned> initCount;
};

static DlRaFunction::Init g_dlraInit; // 关键变量

} // namespace hccl

#endif // HCCL_SRC_DLRAFUNCTION_H
