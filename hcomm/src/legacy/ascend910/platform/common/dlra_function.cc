/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlra_function.h"

#include <string>
#include <map>
#include "hccl_dl.h"
#include "log.h"

namespace hccl {

std::atomic<unsigned> DlRaFunction::Init::initCount(0);
DlRaFunction* DlRaFunction::hcclDlRaFunction = nullptr;

DlRaFunction::Init::Init()
{
    if (initCount.fetch_add(1) == 0) {
        DlRaFunction::hcclDlRaFunction = new (std::nothrow) DlRaFunction;
        CHK_PRT_CONT(
            DlRaFunction::hcclDlRaFunction == nullptr, HCCL_ERROR("[DlRaFunction][Init] hcclDlRaFunction new failed"));
    }
}

DlRaFunction::Init::~Init()
{
    if (initCount.fetch_sub(1) == 0) {
        if (DlRaFunction::GetInstance().handle_ != nullptr) {
            (void)HcclDlclose(DlRaFunction::GetInstance().handle_);
            DlRaFunction::GetInstance().handle_ = nullptr;
        }
        delete DlRaFunction::hcclDlRaFunction;
    }
}

DlRaFunction& DlRaFunction::GetInstance() { return *hcclDlRaFunction; }

DlRaFunction::DlRaFunction() : handle_(nullptr) {}

DlRaFunction::~DlRaFunction() {}

HcclResult DlRaFunction::DlRaFunctionRdmaInit()
{
    dlRaGetQpDepth = (int (*)(RdmaHandle, unsigned int*, unsigned int*))HcclDlsym(handle_, "RaGetTsqpDepth");
    CHK_SMART_PTR_NULL(dlRaGetQpDepth);
    dlRaSetQpDepth = (int (*)(RdmaHandle, unsigned int, unsigned int*))HcclDlsym(handle_, "RaSetTsqpDepth");
    CHK_SMART_PTR_NULL(dlRaSetQpDepth);
    dlRaQpCreate = (int (*)(RdmaHandle, int, int, QpHandle*))HcclDlsym(handle_, "RaQpCreate");
    CHK_SMART_PTR_NULL(dlRaQpCreate);
    dlRaQpDestroy = (int (*)(QpHandle))HcclDlsym(handle_, "RaQpDestroy");
    CHK_SMART_PTR_NULL(dlRaQpDestroy);
    dlRaQpDestroyWithoutCQ = (int (*)(QpHandle))HcclDlsym(handle_, "RaQpDestroyWithoutCQ");
    CHK_SMART_PTR_NULL(dlRaQpDestroyWithoutCQ);
    dlRaGetQpContext = (int (*)(void*, void**, void**, void**))HcclDlsym(handle_, "RaGetQpContext");
    CHK_SMART_PTR_NULL(dlRaGetQpContext);
    dlRaQpConnectAsync = (int (*)(QpHandle, const SocketHandle))HcclDlsym(handle_, "RaQpConnectAsync");
    CHK_SMART_PTR_NULL(dlRaQpConnectAsync);
    dlRaGetQpStatus = (int (*)(QpHandle, int*))HcclDlsym(handle_, "RaGetQpStatus");
    CHK_SMART_PTR_NULL(dlRaGetQpStatus);
    dlRaMrDereg = (int (*)(QpHandle, struct MrInfoT*))HcclDlsym(handle_, "RaMrDereg");
    CHK_SMART_PTR_NULL(dlRaMrDereg);
    dlRaMrReg = (int (*)(QpHandle, struct MrInfoT*))HcclDlsym(handle_, "RaMrReg");
    CHK_SMART_PTR_NULL(dlRaMrReg);
    dlRaGetNotifyMrInfo = (int (*)(RdmaHandle, struct MrInfoT*))HcclDlsym(handle_, "RaGetNotifyMrInfo");
    CHK_SMART_PTR_NULL(dlRaGetNotifyMrInfo);
    dlRaRdmaDeInit = (int (*)(RdmaHandle, u32))HcclDlsym(handle_, "RaRdevDeinit");
    CHK_SMART_PTR_NULL(dlRaRdmaDeInit);
    dlRaRdmaInitWithAttr = (int (*)(struct RdevInitInfo, struct rdev, RdmaHandle*))HcclDlsym(handle_, "RaRdevInitV2");
    CHK_SMART_PTR_NULL(dlRaRdmaInitWithAttr);
    dlRaRdmaInitWithBackupAttr = (int (*)(struct RdevInitInfo*, struct rdev*, struct rdev*, RdmaHandle*))HcclDlsym(
        handle_, "RaRdevInitWithBackup");
    CHK_SMART_PTR_NULL(dlRaRdmaInitWithBackupAttr);
    dlRaRdmaGetHandle = (int (*)(unsigned int, RdmaHandle*))HcclDlsym(handle_, "RaRdevGetHandle");
    dlRaRdmaInit = (int (*)(int, u32, struct rdev, RdmaHandle*))HcclDlsym(handle_, "RaRdevInit");
    CHK_SMART_PTR_NULL(dlRaRdmaInit);
    dlRaSendWr = (int (*)(QpHandle, struct SendWr*, struct SendWrRsp*))HcclDlsym(handle_, "RaSendWr");
    CHK_SMART_PTR_NULL(dlRaSendWr);
    dlRaSendWrV2 = (int (*)(QpHandle, struct SendWrV2*, struct SendWrRsp*))HcclDlsym(handle_, "RaSendWrV2");
    CHK_SMART_PTR_NULL(dlRaSendWrV2);
    dlRaSendWrVerbs = (int (*)(QpHandle, struct SendWrVerbs*, struct SendWrRsp*))HcclDlsym(handle_, "RaSendWrVerbs");
    CHK_SMART_PTR_NULL(dlRaSendWrVerbs);
    dlRaRecvWrVerbs = (int (*)(QpHandle, struct RecvWrVerbs*))HcclDlsym(handle_, "RaRecvWrVerbs");
    CHK_SMART_PTR_NULL(dlRaRecvWrVerbs);
    dlRaPollCq = (int (*)(QpHandle, bool, unsigned int, void*))HcclDlsym(handle_, "RaPollCq");
    CHK_SMART_PTR_NULL(dlRaPollCq);
    dlRaPollTypicalCq = (int (*)(void*, unsigned int, void*))HcclDlsym(handle_, "RaPollTypicalCq");
    CHK_SMART_PTR_NULL(dlRaPollTypicalCq);
    dlRaSendWrlist = (int (*)(
        QpHandle handle, struct SendWrlistData wr[], struct SendWrRsp op_rsp[], unsigned int sendNum,
        unsigned int* completeNum))HcclDlsym(handle_, "RaSendWrlist");
    if (dlRaSendWrlist == nullptr) {
        HCCL_WARNING("dlRaSendWrlist is nullptr, can not use RaSendWrlist");
    }
    dlRaSendWrlistExt = (int (*)(
        QpHandle handle, struct SendWrlistDataExt wr[], struct SendWrRsp op_rsp[], unsigned int sendNum,
        unsigned int* completeNum))HcclDlsym(handle_, "RaSendWrlistExt");
    if (dlRaSendWrlistExt == nullptr) {
        HCCL_WARNING("dlRaSendWrlistExt is nullptr, can not use ra_send_wrlist_ext");
    }
    dlRaSendNormalWrlist = (int (*)(
        QpHandle handle, struct WrInfo wr[], struct SendWrRsp opRsp[], unsigned int sendNum,
        unsigned int* completeNum))HcclDlsym(handle_, "RaSendNormalWrlist");
    CHK_SMART_PTR_NULL(dlRaSendNormalWrlist);
    dlRaRegGlobalMr
        = (int (*)(const RdmaHandle, struct MrInfoT* info, MrHandle* mrHandle))HcclDlsym(handle_, "RaRegisterMr");
    CHK_SMART_PTR_NULL(dlRaRegGlobalMr);
    dlRaDeRegGlobalMr = (int (*)(const RdmaHandle, MrHandle mrHandle))HcclDlsym(handle_, "RaDeregisterMr");
    CHK_SMART_PTR_NULL(dlRaDeRegGlobalMr);
    dlRaCreateCq = (int (*)(RdmaHandle, struct CqAttr*))HcclDlsym(handle_, "RaCqCreate");
    CHK_SMART_PTR_NULL(dlRaCreateCq);
    dlRaDestroyCq = (int (*)(RdmaHandle, struct CqAttr*))HcclDlsym(handle_, "RaCqDestroy");
    CHK_SMART_PTR_NULL(dlRaDestroyCq);
    dlRaNormalQpCreate
        = (int (*)(RdmaHandle, struct ibv_qp_init_attr*, void**, void**))HcclDlsym(handle_, "RaNormalQpCreate");
    CHK_SMART_PTR_NULL(dlRaNormalQpCreate);
    dlRaNormalQpDestroy = (int (*)(QpHandle))HcclDlsym(handle_, "RaNormalQpDestroy");
    CHK_SMART_PTR_NULL(dlRaNormalQpDestroy);
    dlRaSetQpAttrQos = (int (*)(QpHandle, struct QosAttr*))HcclDlsym(handle_, "RaSetQpAttrQos");
    CHK_SMART_PTR_NULL(dlRaSetQpAttrQos);
    dlRaSetQpAttrTimeOut = (int (*)(QpHandle, u32*))HcclDlsym(handle_, "RaSetQpAttrTimeout");
    CHK_SMART_PTR_NULL(dlRaSetQpAttrTimeOut);
    dlRaSetQpAttrRetryCnt = (int (*)(QpHandle, u32*))HcclDlsym(handle_, "RaSetQpAttrRetryCnt");
    CHK_SMART_PTR_NULL(dlRaSetQpAttrRetryCnt);
    dlRaCreateCompChannel = (int (*)(const void*, void**))HcclDlsym(handle_, "RaCreateCompChannel");
    CHK_SMART_PTR_NULL(dlRaCreateCompChannel);
    dlRaDestroyCompChannel = (int (*)(const void*, void*))HcclDlsym(handle_, "RaDestroyCompChannel");
    CHK_SMART_PTR_NULL(dlRaDestroyCompChannel);
    dlRaGetCqeErrInfo = (int (*)(unsigned int phyId, struct CqeErrInfo*))HcclDlsym(handle_, "RaGetCqeErrInfo");
    CHK_SMART_PTR_NULL(dlRaGetCqeErrInfo);
    dlRaGetCqeErrInfoList
        = (int (*)(RdmaHandle, struct CqeErrInfo*, u32*))HcclDlsym(handle_, "RaRdevGetCqeErrInfoList");
    CHK_SMART_PTR_NULL(dlRaGetCqeErrInfoList);
    dlRaGetQpAttr = (int (*)(QpHandle, struct QpAttr*))HcclDlsym(handle_, "RaGetQpAttr");
    CHK_SMART_PTR_NULL(dlRaGetQpAttr);
    dlRaCreateSrq = (int (*)(const void*, struct SrqAttr*))HcclDlsym(handle_, "RaCreateSrq");
    CHK_SMART_PTR_NULL(dlRaCreateSrq);
    dlRaDestroyeSrq = (int (*)(const void*, struct SrqAttr*))HcclDlsym(handle_, "RaDestroySrq");
    CHK_SMART_PTR_NULL(dlRaDestroyeSrq);
    dlRaQpCreateWithAttrs
        = (int (*)(RdmaHandle, struct QpExtAttrs*, QpHandle*))HcclDlsym(handle_, "RaQpCreateWithAttrs");
    CHK_SMART_PTR_NULL(dlRaQpCreateWithAttrs);
    dlRaTypicalQpCreate
        = (int (*)(RdmaHandle, int, int, struct TypicalQp*, QpHandle*))HcclDlsym(handle_, "RaTypicalQpCreate");
    CHK_SMART_PTR_NULL(dlRaTypicalQpCreate);
    dlRaTypicalQpModify
        = (int (*)(QpHandle, struct TypicalQp*, struct TypicalQp*))HcclDlsym(handle_, "RaTypicalQpModify");
    CHK_SMART_PTR_NULL(dlRaTypicalQpModify);
    dlRaTypicalSendWr = (int (*)(QpHandle, struct SendWr*, struct SendWrRsp*))HcclDlsym(handle_, "RaTypicalSendWr");
    CHK_SMART_PTR_NULL(dlRaTypicalSendWr);
    dlRaTypicalCqCreate
        = (int (*)(RdmaHandle, unsigned int, unsigned int*, void**))HcclDlsym(handle_, "RaTypicalCqCreate");
    CHK_SMART_PTR_NULL(dlRaTypicalCqCreate);
    dlRaTypicalCqDestroy = (int (*)(RdmaHandle, unsigned int, void*))HcclDlsym(handle_, "RaTypicalCqDestroy");
    CHK_SMART_PTR_NULL(dlRaTypicalCqDestroy);
    dlRaQpCreateWithCQWithAttrs
        = (int (*)(RdmaHandle, struct QpExtAttrs*, unsigned int, unsigned int, QpHandle*))HcclDlsym(
            handle_, "RaQpCreateWithCQWithAttrs");
    CHK_SMART_PTR_NULL(dlRaQpCreateWithCQWithAttrs);
    dlRaAiQpCreate
        = (int (*)(RdmaHandle, struct QpExtAttrs*, struct AiQpInfo*, QpHandle*))HcclDlsym(handle_, "RaAiQpCreate");
    CHK_SMART_PTR_NULL(dlRaAiQpCreate);
    dlRaRecvWrlist
        = (int (*)(QpHandle handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum))
            HcclDlsym(handle_, "RaRecvWrlist");
    if (dlRaRecvWrlist == nullptr) {
        HCCL_WARNING("dlRaRecvWrlist is nullptr, can not use ra_recv_wrlist");
    }
    dlRaGetRdmaLiteStatus = (int (*)(RdmaHandle, int*))HcclDlsym(handle_, "RaRdevGetSupportLite");
    CHK_SMART_PTR_NULL(dlRaGetRdmaLiteStatus);

    dlRaQpBatchModify = (int (*)(RdmaHandle rdmaHandle, void** qpHandle, unsigned int num, int expectStatus))HcclDlsym(
        handle_, "RaQpBatchModify");
    if (dlRaQpBatchModify == nullptr) {
        HCCL_ERROR("dlRaQpBatchModify is nullptr, can not use ra_qp_batch_modify");
    }
    CHK_SMART_PTR_NULL(dlRaQpBatchModify);

    dlRaPingInit = (int (*)(struct PingInitAttr*, struct PingInitInfo*, void**))HcclDlsym(handle_, "RaPingInit");
    if (dlRaPingInit == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingInit, please check!");
    }
    dlRaPingDeinit = (int (*)(void*))HcclDlsym(handle_, "RaPingDeinit");
    if (dlRaPingDeinit == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingDeinit, please check!");
    }
    dlRaPingTargetAdd = (int (*)(void*, struct PingTargetInfo target[], uint32_t))HcclDlsym(handle_, "RaPingTargetAdd");
    if (dlRaPingTargetAdd == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingTargetAdd, please check!");
    }
    dlRaPingTargetDel
        = (int (*)(void*, struct PingTargetCommInfo target[], uint32_t))HcclDlsym(handle_, "RaPingTargetDel");
    if (dlRaPingTargetDel == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingTargetDel, please check!");
    }
    dlRaPingTaskStart = (int (*)(void*, struct PingTaskAttr*))HcclDlsym(handle_, "RaPingTaskStart");
    if (dlRaPingTaskStart == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingTaskStart, please check!");
    }
    dlRaPingTaskStop = (int (*)(void*))HcclDlsym(handle_, "RaPingTaskStop");
    if (dlRaPingTaskStop == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingTaskStop, please check!");
    }
    dlRaPingGetResults
        = (int (*)(void*, struct PingTargetResult target[], uint32_t*))HcclDlsym(handle_, "RaPingGetResults");
    if (dlRaPingGetResults == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaPingGetResults, please check!");
    }

    dlRaIsFirstUsed = (int (*)(int))HcclDlsym(handle_, "RaIsFirstUsed");
    CHK_SMART_PTR_NULL(dlRaIsFirstUsed);

    dlRaIsLastUsed = (int (*)(int))HcclDlsym(handle_, "RaIsLastUsed");
    CHK_SMART_PTR_NULL(dlRaIsLastUsed);

    dlRaRdevGetPortStatus = (int (*)(RdmaHandle, enum PortStatus*))HcclDlsym(handle_, "RaRdevGetPortStatus");
    CHK_SMART_PTR_NULL(dlRaRdevGetPortStatus);

    dlRaRemapMr = (int (*)(RdmaHandle, struct MemRemapInfo info[], unsigned int num))HcclDlsym(handle_, "RaRemapMr");
    if (dlRaRemapMr == nullptr) {
        HCCL_WARNING("Current package doesn't have dlRaRemapMr, please check!");
    }

    dlH2DTlvInit = (int (*)(struct TlvInitInfo*, uint32_t*, void**))HcclDlsym(handle_, "RaTlvInit");
    if (dlH2DTlvInit == nullptr) {
        HCCL_WARNING("Current package doesn't have dlH2DTlvInit, please check!");
    }
    dlH2DTlvDeinit = (int (*)(void*))HcclDlsym(handle_, "RaTlvDeinit");
    if (dlH2DTlvDeinit == nullptr) {
        HCCL_WARNING("Current package doesn't have dlH2DTlvDeinit, please check!");
    }
    dlH2DTlvRequest
        = (int (*)(void*, unsigned int, struct TlvMsg[], struct TlvMsg[]))HcclDlsym(handle_, "RaTlvRequest");
    if (dlH2DTlvRequest == nullptr) {
        HCCL_WARNING("Current package doesn't have H2DTlvRequest, please check!");
    }
    return HCCL_SUCCESS;
}

HcclResult DlRaFunction::DlRaFunctionSocketInit()
{
    dlRaGetNotifyBaseAddr
        = (int (*)(RdmaHandle, unsigned long long*, unsigned long long*))HcclDlsym(handle_, "RaGetNotifyBaseAddr");
    CHK_SMART_PTR_NULL(dlRaGetNotifyBaseAddr);
    dlRaGetSockets
        = (int (*)(unsigned int, struct SocketInfoT[], unsigned int, unsigned int*))HcclDlsym(handle_, "RaGetSockets");
    CHK_SMART_PTR_NULL(dlRaGetSockets);
    dlRaSocketBatchClose = (int (*)(struct SocketCloseInfoT[], unsigned int))HcclDlsym(handle_, "RaSocketBatchClose");
    CHK_SMART_PTR_NULL(dlRaSocketBatchClose);
    dlRaSocketBatchConnect
        = (int (*)(struct SocketConnectInfoT[], unsigned int num))HcclDlsym(handle_, "RaSocketBatchConnect");
    CHK_SMART_PTR_NULL(dlRaSocketBatchConnect);
    dlRaSocketBatchAbort
        = (int (*)(struct SocketConnectInfoT[], unsigned int num))HcclDlsym(handle_, "RaSocketBatchAbort");
    CHK_SMART_PTR_NULL(dlRaSocketBatchAbort);
    dlRaSocketDeInit = (int (*)(SocketHandle))HcclDlsym(handle_, "RaSocketDeinit");
    CHK_SMART_PTR_NULL(dlRaSocketDeInit);
    dlRaSocketInit = (int (*)(int, struct rdev, SocketHandle*))HcclDlsym(handle_, "RaSocketInit");
    CHK_SMART_PTR_NULL(dlRaSocketInit);
    dlRaSocketInitV1 = (int (*)(int, struct SocketInitInfoT, SocketHandle*))HcclDlsym(handle_, "RaSocketInitV1");
    CHK_SMART_PTR_NULL(dlRaSocketInitV1);
    dlRaSocketListenStart
        = (int (*)(struct SocketListenInfoT[], unsigned int))HcclDlsym(handle_, "RaSocketListenStart");
    CHK_SMART_PTR_NULL(dlRaSocketListenStart);
    dlRaSocketAcceptCreditAdd = (int (*)(struct SocketListenInfoT[], unsigned int, unsigned int))HcclDlsym(
        handle_, "RaSocketAcceptCreditAdd");
    CHK_SMART_PTR_NULL(dlRaSocketAcceptCreditAdd);
    dlRaSocketListenStop = (int (*)(struct SocketListenInfoT[], unsigned int))HcclDlsym(handle_, "RaSocketListenStop");
    CHK_SMART_PTR_NULL(dlRaSocketListenStop);
    dlRaSocketRecv = (int (*)(const FdHandle, const void*, unsigned long long, unsigned long long*))HcclDlsym(
        handle_, "RaSocketRecv");
    CHK_SMART_PTR_NULL(dlRaSocketRecv);
    dlRaSocketSend = (int (*)(const FdHandle, const void*, unsigned long long, unsigned long long*))HcclDlsym(
        handle_, "RaSocketSend");
    CHK_SMART_PTR_NULL(dlRaSocketSend);
    dlRaSocketSendAsync
        = (int (*)(const FdHandle, const void*, unsigned long long, unsigned long long*, void**))HcclDlsym(
            handle_, "RaSocketSendAsync");
    if (dlRaSocketSendAsync == nullptr) {
        HCCL_WARNING("dlRaSocketSendAsync is nullptr, can not use RaSocketSendAsync");
    }
    dlRaSocketRecvAsync = (int (*)(const FdHandle, void*, unsigned long long, unsigned long long*, void**))HcclDlsym(
        handle_, "RaSocketRecvAsync");
    if (dlRaSocketRecvAsync == nullptr) {
        HCCL_WARNING("dlRaSocketRecvAsync is nullptr, can not use RaSocketRecvAsync");
    }
    dlRaGetAsyncReqResult = (int (*)(void*, int*))HcclDlsym(handle_, "RaGetAsyncReqResult");
    if (dlRaGetAsyncReqResult == nullptr) {
        HCCL_WARNING("dlRaGetAsyncReqResult is nullptr, can not use RaGetAsyncReqResult");
    }
    dlRaSocketSetWhiteListStatus = (int (*)(unsigned int))HcclDlsym(handle_, "RaSocketSetWhiteListStatus");
    CHK_SMART_PTR_NULL(dlRaSocketSetWhiteListStatus);
    dlRaSocketGetWhiteListStatus = (int (*)(unsigned int*))HcclDlsym(handle_, "RaSocketGetWhiteListStatus");
    CHK_SMART_PTR_NULL(dlRaSocketGetWhiteListStatus);
    dlRaSocketWhiteListAdd
        = (int (*)(SocketHandle, struct SocketWlistInfoT[], unsigned int))HcclDlsym(handle_, "RaSocketWhiteListAdd");
    CHK_SMART_PTR_NULL(dlRaSocketWhiteListAdd);
    dlRaSocketWhiteListDel
        = (int (*)(SocketHandle, struct SocketWlistInfoT[], unsigned int))HcclDlsym(handle_, "RaSocketWhiteListDel");
    CHK_SMART_PTR_NULL(dlRaSocketWhiteListDel);
    /* 考虑兼容性问题，这里不校验dlRaGetIfNum是否为空，在使用处校验 */
    dlRaGetIfNum = (int (*)(struct RaGetIfattr* config, unsigned int* num))HcclDlsym(handle_, "RaGetIfnum");
    if (dlRaGetIfNum == nullptr) {
        HCCL_WARNING("dlRaGetIfNum is nullptr, can not use ra_get_ifnum");
    }

    dlRaGetIfAddress
        = (int (*)(struct RaGetIfattr* config, struct InterfaceInfo interface_infos[], unsigned int* num))HcclDlsym(
            handle_, "RaGetIfaddrs");
    CHK_SMART_PTR_NULL(dlRaGetIfAddress);
    dlRaGetInterfaceVersion
        = (int (*)(unsigned int phyId, unsigned int interface_opcode, unsigned int* interface_version))HcclDlsym(
            handle_, "RaGetInterfaceVersion");
    if (dlRaGetInterfaceVersion == nullptr) {
        HCCL_WARNING("dlRaGetInterfaceVersion is nullptr, can not use ra_get_interface_version");
    }
    dlRaEpollCtlAdd = (int (*)(const FdHandle fdHandle, RaEpollEvent event))HcclDlsym(handle_, "RaEpollCtlAdd");
    CHK_SMART_PTR_NULL(dlRaEpollCtlAdd);
    dlRaEpollCtlMod = (int (*)(const FdHandle fdHandle, RaEpollEvent event))HcclDlsym(handle_, "RaEpollCtlMod");
    CHK_SMART_PTR_NULL(dlRaEpollCtlMod);
    dlRaEpollCtlDel = (int (*)(const FdHandle fdHandle))HcclDlsym(handle_, "RaEpollCtlDel");
    CHK_SMART_PTR_NULL(dlRaEpollCtlDel);
    dlRaSetRecvDataCallback
        = (int (*)(const SocketHandle socketHandle, const void* callback))HcclDlsym(handle_, "RaSetTcpRecvCallback");
    CHK_SMART_PTR_NULL(dlRaSetRecvDataCallback);

    dlRaCreateEventHandle = (int (*)(int* event_handle))HcclDlsym(handle_, "RaCreateEventHandle");
    if (dlRaCreateEventHandle == nullptr) {
        HCCL_WARNING("dlRaCreateEventHandle is nullptr, can not use ra_create_event_handle");
    }
    dlRaCtlEventHandle = (int (*)(int event_handle, const void* fdHandle, int opcode, RaEpollEvent event))HcclDlsym(
        handle_, "RaCtlEventHandle");
    if (dlRaCtlEventHandle == nullptr) {
        HCCL_WARNING("dlRaCtlEventHandle is nullptr, can not use ra_ctl_event_handle");
    }
    dlRaWaitEventHandle = (int (*)(
        int event_handle, struct SocketEventInfoT* event_infos, int timeout, unsigned int maxevents,
        unsigned int* events_num))HcclDlsym(handle_, "RaWaitEventHandle");
    if (dlRaWaitEventHandle == nullptr) {
        HCCL_WARNING("dlRaWaitEventHandle is nullptr, can not use ra_wait_event_handle");
    }
    dlRaDestroyEventHandle = (int (*)(int* event_handle))HcclDlsym(handle_, "RaDestroyEventHandle");
    if (dlRaDestroyEventHandle == nullptr) {
        HCCL_WARNING("dlRaDestroyEventHandle is nullptr, can not use ra_destroy_event_handle");
    }

    dlRaGetSocketVnicIpInfos
        = (int (*)(unsigned int, enum IdType, unsigned int*, unsigned int, struct IpInfo infos[]))HcclDlsym(
            handle_, "RaSocketGetVnicIpInfos");
    CHK_SMART_PTR_NULL(dlRaGetSocketVnicIpInfos);

    dlRaRaGetTlsEnable = (int (*)(struct RaInfo*, bool*))HcclDlsym(handle_, "RaGetTlsEnable");
    if (dlRaRaGetTlsEnable == nullptr) {
        HCCL_WARNING("dlRaRaGetTlsEnable is nullptr, can not use ra_get_tls_enable");
    }

    dlRaSaveSnapShot = (int (*)(struct RaInfo*, enum SaveSnapshotAction))HcclDlsym(handle_, "RaSaveSnapshot");
    CHK_SMART_PTR_NULL(dlRaSaveSnapShot);

    dlRaRestoreSnapShot = (int (*)(struct RaInfo*))HcclDlsym(handle_, "RaRestoreSnapshot");
    CHK_SMART_PTR_NULL(dlRaRestoreSnapShot);

    dlRaGetHccnCfg = (int (*)(struct RaInfo*, enum HccnCfgKey, char*, int*))HcclDlsym(handle_, "RaGetHccnCfg");
    if (dlRaGetHccnCfg == nullptr) {
        HCCL_WARNING("dlRaGetHccnCfg is nullptr, can not use RaGetHccnCfg");
    }
    dlRaGetSecRandom = (int (*)(struct RaInfo* info, unsigned int*))HcclDlsym(handle_, "RaGetSecRandom");
    if (dlRaGetSecRandom == nullptr) {
        HCCL_WARNING("dlRaGetSecRandom is nullptr, can not use RaGetSecRandom");
    }
    dlRaGetDevEidInfoNum = (int (*)(RaInfo info, unsigned int*))HcclDlsym(handle_, "RaGetDevEidInfoNum");
    if (dlRaGetDevEidInfoNum == nullptr) {
        HCCL_WARNING("dlRaGetDevEidInfoNum is nullptr, can not use RaGetDevEidInfoNum");
    }
    dlRaGetDevEidInfoList = (int (*)(RaInfo info, struct HccpDevEidInfo* eid_info, unsigned int*))HcclDlsym(
        handle_, "RaGetDevEidInfoList");
    if (dlRaGetDevEidInfoList == nullptr) {
        HCCL_WARNING("dlRaGetDevEidInfoList is nullptr, can not use RaGetDevEidInfoList");
    }
    return HCCL_SUCCESS;
}

HcclResult DlRaFunction::DlRaFunctionInit()
{
    std::lock_guard<std::mutex> lock(handleMutex_);
    if (handle_ == nullptr) {
        handle_ = HcclDlopen("libra.so", RTLD_NOW);
        const char* errMsg = dlerror();
        CHK_PRT_RET(
            handle_ == nullptr,
            HCCL_ERROR(
                "dlopen [%s] failed, %s", "libra.so",
                (errMsg == nullptr) ? "please check the file exist or permission denied." : errMsg),
            HCCL_E_OPEN_FILE_FAILURE);
    }
    dlRaInit = (int (*)(struct RaInitConfig*))HcclDlsym(handle_, "RaInit");
    CHK_SMART_PTR_NULL(dlRaInit);
    dlRaDeInit = (int (*)(struct RaInitConfig*))HcclDlsym(handle_, "RaDeinit");
    CHK_SMART_PTR_NULL(dlRaDeInit);
    CHK_RET(DlRaFunctionRdmaInit());
    CHK_RET(DlRaFunctionSocketInit());

    return HCCL_SUCCESS;
}
} // namespace hccl
