/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_stub.h"
#include "fake_socket.h"
#include <unistd.h>

#ifdef __cplusplus
extern "C" {

static fake_socket fakeSocket;

int RaSocketInit(int mode, struct rdev rdev_info, void** socket_handle)
{
    *socket_handle = fakeSocket.GetSocketHandle();
    return 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], unsigned int num) { return 0; }

int RaSocketDeinit(void* socket_handle) { return 0; }

int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connected_num)
{
    conn[0].status = 1; // 打桩已连接
    *connected_num = 1; // 打桩连接数量
    return 0;
}

int RaSocketWhiteListDel(void* socket_handle, struct SocketWlistInfoT white_list[], unsigned int num) { return 0; }

int RaSocketSend(const void* fd_handle, const void* data, unsigned long long size, unsigned long long* sent_size)
{
    *sent_size = size;
    auto ret = fakeSocket.Send((int*)fd_handle, data, size);
    if (!ret) {
        return 1;
    }
    return 0;
}

int RaSocketRecv(const void* fd_handle, void* data, unsigned long long size, unsigned long long* received_size)
{
    *received_size = size;
    auto ret = fakeSocket.Recv((int*)fd_handle, data, size);

    // 可能先发起收，需要重试多次

    int retryCnt = 0;
    while (retryCnt++ <= 2) {
        if (!ret) {
            sleep(1);
            ret = fakeSocket.Recv((int*)fd_handle, data, size);
        } else {
            return 0;
        }
    }

    if (!ret) {
        return 1;
    }
    return 0;
}

int RaSocketGetWhiteListStatus(unsigned int* enable) { return 0; }

int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num) { return 0; }

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num)
{
    // TODO 当前只打桩num=1情况
    if (num != 1) {
        return 1;
    }

    auto ret = fakeSocket.Connect(conn[0]);

    if (!ret) {
        return 1;
    }

    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num) { return 0; }

int RaSocketWhiteListAdd(void* socket_handle, struct SocketWlistInfoT white_list[], unsigned int num) { return 0; }

int RaTlvInit(struct TlvInitInfo* init_info, unsigned int* buffer_size, void** tlv_handle)
{
    static char fakeTlvHandle;
    if (tlv_handle != nullptr) {
        *tlv_handle = &fakeTlvHandle;
    }
    return 0;
}
int RaTlvRequest(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg)
{
    return 0;
}
int RaTlvDeinit(void* tlv_handle) { return 0; }

int RaInit(struct RaInitConfig* config) { return 0; }

int RaDeinit(struct RaInitConfig* config) { return 0; }

int RaSocketSetWhiteListStatus(unsigned int enable) { return 0; }

int RaRdevInit(int mode, unsigned int NotifyTypeT, struct rdev rdev_info, void** rdma_handle) { return 0; }

int RaGetQpStatus(void* qp_handle, int* status) { return 0; }

int RaGetIfnum(struct RaGetIfattr* config, unsigned int* num) { return 0; }

int RaGetIfaddrs(struct RaGetIfattr* config, struct InterfaceInfo interface_infos[], unsigned int* num) { return 0; }

int RaQpConnectAsync(void* qp_handle, const void* fd_handle) { return 0; }

int RaQpDestroy(void* qp_handle) { return 0; }

int RaQpCreate(void* rdev_handle, int flag, int qp_mode, void** qp_handle) { return 0; }

int RaRdevDeinit(void* rdma_handle, unsigned int NotifyTypeT) { return 0; }

int RaMrReg(void* qp_handle, struct MrInfoT* info) { return 0; }

int RaMrDereg(void* qp_handle, struct MrInfoT* info) { return 0; }

int RaGetNotifyBaseAddr(void* rdev_handle, unsigned long long* va, unsigned long long* size) { return 0; }

int RaSocketInitV1(int mode, struct SocketInitInfoT socket_init, void** socket_handle) { return 0; }

int RaSendWr(void* qp_handle, struct SendWr* wr, struct SendWrRsp* op_rsp) { return 0; }

int RaCtxInit(struct CtxInitCfg* cfg, struct CtxInitAttr* info, void** ctx_handle) { return 0; }

int RaCtxDeinit(void* ctx_handle) { return 0; }

int RaCtxTokenIdAlloc(void* ctx_handle, struct HccpTokenId* info, void** token_id_handle) { return 0; }

int RaCtxTokenIdFree(void* ctx_handle, void* token_id_handle) { return 0; }

int RaCtxLmemRegister(void* ctx_handle, struct MrRegInfoT* lmem_info, void** lmem_handle) { return 0; }

int RaCtxLmemUnregister(void* ctx_handle, void* lmem_handle) { return 0; }

int RaCtxRmemImport(void* ctx_handle, struct MrImportInfoT* rmem_info, void** rmem_handle) { return 0; }

int RaCtxRmemUnimport(void* ctx_handle, void* rmem_handle) { return 0; }

int RaCtxCqCreate(void* ctx_handle, struct CqInfoT* info, void** cq_handle) { return 0; }

int RaCtxCqDestroy(void* ctx_handle, void* cq_handle) { return 0; }

int RaCtxQpCreate(void* ctx_handle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qp_handle)
{
    return 0;
}

int RaCtxQpDestroy(void* qp_handle) { return 0; }

int RaCtxQpImport(void* ctx_handle, struct QpImportInfoT* qp_info, void** rem_qp_handle) { return 0; }

int RaCtxQpUnimport(void* ctx_handle, void* rem_qp_handle) { return 0; }

int RaCtxQpBind(void* qp_handle, void* rem_qp_handle) { return 0; }

int RaCtxQpUnbind(void* qp_handle) { return 0; }

int RaBatchSendWr(
    void* qp_handle, struct SendWrData wr_list[], struct SendWrResp op_resp[], unsigned int num,
    unsigned int* complete_num)
{
    return 0;
}

int RaGetDevEidInfoNum(struct RaInfo info, unsigned int* num) { return 0; }

int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo info_list[], unsigned int* num) { return 0; }

int RaGetDevBaseAttr(void* ctx_handle, struct DevBaseAttr* attr) { return 0; }

int RaCtxUpdateCi(void* qp_handle, uint16_t ci) { return 0; }

int RaGetAsyncReqResult(void* req_handle, int* req_result) { return 0; }

int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaSocketSendAsync(
    const void* fd_handle, const void* data, unsigned long long size, unsigned long long* sent_size, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    *sent_size = size;
    return 0;
}

int RaSocketRecvAsync(
    const void* fd_handle, void* data, unsigned long long size, unsigned long long* received_size, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    *received_size = size;
    return 0;
}

int RaCtxLmemRegisterAsync(void* ctx_handle, struct MrRegInfoT* lmem_info, void** lmem_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxLmemUnregisterAsync(void* ctx_handle, void* lmem_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxQpCreateAsync(
    void* ctx_handle, struct QpCreateAttr* attr, struct QpCreateInfo* info, void** qp_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxQpDestroyAsync(void* qp_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxQpImportAsync(void* ctx_handle, struct QpImportInfoT* info, void** rem_qp_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaCtxQpUnimportAsync(void* rem_qp_handle, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

int RaGetTpInfoListAsync(
    void* ctx_handle, struct GetTpCfg* cfg, struct HccpTpInfo info_list[], unsigned int* num, void** req_handle)
{
    *req_handle = reinterpret_cast<void*>(0x12345678);
    *num = 1;
    return 0;
}

int RaCreateEventHandle(int* event_handle) { return 0; }

int RaCtlEventHandle(int event_handle, const void* fd_handle, int opcode, enum RaEpollEvent event) { return 0; }

int RaWaitEventHandle(
    int event_handle, struct SocketEventInfoT* event_infos, int timeout, unsigned int maxevents,
    unsigned int* events_num)
{
    return 0;
}

int RaDestroyEventHandle(int* event_handle) { return 0; }

int RaEpollCtlAdd(const void* fd_handle, RaEpollEvent event) { return 0; }

int RaEpollCtlMod(const void* fd_handle, RaEpollEvent event) { return 0; }

int RaEpollCtlDel(const void* fd_handle) { return 0; }

int RaGetSecRandom(struct RaInfo* info, uint32_t* value) { return 0; }

int RaCtxQpDestroyBatchAsync(void* ctx_handle, void* qp_handle[], unsigned int* num, void** req_handle) { return 0; }

int RaGetEidByIp(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num) { return 0; }
}
#endif
