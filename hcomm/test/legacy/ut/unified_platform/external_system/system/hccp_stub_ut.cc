/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "hccp.h"
#include "hccp_async.h"
#include "hccp_ctx.h"
#include "hccp_async_ctx.h"
#include "hccp_tlv.h"
#include "hccp_nda.h"

/**
 * @brief Set hccp network debug config mask
 * @param info [IN] see struct RaInfo
 * @param debugConfig [IN] debug config mask
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSetDebugConfig(struct RaInfo* info, uint64_t debugConfig) { return 0; }

/**
 * @ingroup libsocket
 * @brief Client sockets batch connect to server sockets(async)
 * @param conn [IN] client sockets array
 * @param num [IN] num of conn
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num) { return 0; }

/**
 * @ingroup libsocket
 * @brief Sockets batch close
 * @param conn [IN] sockets array
 * @param num [IN] num of conn
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num) { return 0; }

/**
 * @ingroup libsocket
 * @brief Sockets batch listen
 * @param conn [IN] server info array
 * @param num [IN] num of conn
 * @attention check if IP exist when SOCK_EADDRNOTAVAIL is returned
 * check if IP has been listened when SOCK_EADDRINUSE is returned
 * one IP address can only be listened once
 * @see RaSocketListenStop
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketListenStart(struct SocketListenInfoT conn[], unsigned int num) { return 0; }

/**
 * @ingroup libsocket
 * @brief Sockets batch stop
 * @param conn [IN sockets info array
 * @param num [IN] num of conn
 * @see RaSocketListenStart
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num) { return 0; }

/**
 * @ingroup libsocket
 * @brief Get socket info of connected socket
 * @param role [IN] 0:server 1:client
 * @param conn [IN/OUT] connection info of sockets
 * @param connected_num [OUT] num of connected sockets
 * @attention if connected_num is zero or greater than zero but less than num when return value is zero,
   the function needed to be revoked again until the total connected_num is equal to num
 * @retval #zero Success
 * @retval #non-zero Failure
*/
int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num, unsigned int* connected_num)
{
    conn[0].status = 1; // 打桩已连接
    *connected_num = 1; // 打桩连接数量
    return 0;
}

/**
 * @ingroup libsocket
 * @brief get socket num
 * @param role [IN] 0:server 1:client
 * @param socket_handle [IN] socket handle
 * @param socket_num [OUT] socket num
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int ra_get_socket_num(unsigned int role, void* socket_handle, unsigned int* socket_num) { return 0; }

/**
 * @ingroup libsocket
 * @brief get all socket status
 * @param role [IN] 0:server 1:client
 * @param conn [IN/OUT] connect info
 * @param num [IN] conn num
 * @param socket_num [OUT] real get socket num
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int ra_get_all_sockets(unsigned int role, struct SocketInfoT* conn, unsigned int num, unsigned int* socket_num)
{
    return 0;
}

/**
 * @ingroup libsocket
 * @brief Send data by fd handle
 * @param fd_handle [IN] fd handle
 * @param data [IN] send storage buff
 * @param size [IN] size of data you want to send unit(Byte)
 * @param sent_size [OUT] number of sent bytes
 * @see RaSocketRecv
 * @attention if sent_size is greate than zero but less than size,
 * the function needed to be revoked again,
 * the param of size is original size minus sent_size,
 *the param of data should also offset by sent_size
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketSend(const void* fd_handle, const void* data, unsigned long long size, unsigned long long* sent_size)
{
    return 0;
}

/**
 * @ingroup libsocket
 * @brief Receive data by fd handle
 * @param fd_handle [IN] fd handle
 * @param data [IN/OUT] receive storage buff
 * @param size [IN] size of data you want to receive unit(Byte)
 * @param received_size [OUT] number of received bytes
 * @see RaSocketSend
 * @attention if return value is SOCK_EAGAIN which means no data right now,
 * you need to revoke the funcion again
 * @retval #zero Success
 * @retval #SOCK_EAGAIN Success(no data received by socket)
 * @retval #non-zero Failure(exclude SOCK_EAGAIN)
 */
int RaSocketRecv(const void* fd_handle, void* data, unsigned long long size, unsigned long long* received_size)
{
    return 0;
}

/**
 * @ingroup librdma
 * @brief Get notify base address and size
 * @param rdev_handle [IN] rdev handle
 * @param va [IN/OUT] address of notify
 * @param size [IN/OUT] size of notify
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaGetNotifyBaseAddr(void* rdev_handle, unsigned long long* va, unsigned long long* size) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent initialization
 * @param config [IN] initialization configuration of rdma_agent
 * @see RaDeinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaInit(struct RaInitConfig* config) { return 0; }

int RaTlvInit(struct TlvInitInfo* init_info, unsigned int* buffer_size, void** tlv_handle) { return 0; }
int RaTlvRequest(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg)
{
    return 0;
}
int RaTlvDeinit(void* tlv_handle) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent deinitialization
 * @param config [IN] deinitialization configuration of rdma_agent
 * @see RaInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaDeinit(struct RaInitConfig* config) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent socket initialization
 * @param mode [IN] network mode
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param socket_handle [OUT] socket handle info
 * @see RaSocketDeinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketInit(int mode, struct rdev rdev_info, void** socket_handle) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent socket initialization
 * @param mode [IN] network mode
 * @param SocketInitInfoT [IN] socket_init including dev_id and ip info
 * @param socket_handle [OUT] socket handle info
 * @see RaSocketDeinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketInitV1(int mode, struct SocketInitInfoT socket_init, void** socket_handle) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent socket deinitialization
 * @param socket_handle [IN] deinitialization handle of socket
 * @see RaSocketInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketDeinit(void* socket_handle) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev initialization
 * @param mode [IN] network mode
 * @param NotifyTypeT [IN] notify type
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param rdev_info [OUT] rdma_handle rdma handle info
 * @see RaRdevDeinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaRdevInit(int mode, unsigned int NotifyTypeT, struct rdev rdev_info, void** rdma_handle) { return 0; }

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev deinitialization
 * @param rdma_handle [IN] deinitialization handle of rdev
 * @param NotifyTypeT [IN] notify type
 * @see RaRdevInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaRdevDeinit(void* rdma_handle, unsigned int NotifyTypeT) { return 0; }

/**
 * @ingroup libsocket
 * @brief set socket whitelist status
 * @param enable [IN] enable or disable
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketSetWhiteListStatus(unsigned int enable) { return 0; }

/**
 * @ingroup libsocket
 * @brief get socket whitelist status
 * @param enable [OUT] enable or disable
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketGetWhiteListStatus(unsigned int* enable) { return 0; }

/**
 * @ingroup libsocket
 * @brief Add server socket whitelist
 * @param socket_handle [IN] socket handle
 * @param white_list [IN] server's whitelist
 * @param num [IN] num of whitelist
 * @see RaSocketWhiteListDel
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketWhiteListAdd(void* socket_handle, struct SocketWlistInfoT white_list[], unsigned int num) { return 0; }

/**
 * @ingroup libsocket
 * @brief Remove server socket whitelist
 * @param socket_handle [IN] socket handle
 * @param white_list [IN] server's whitelist
 * @param num [IN] num of whitelist
 * @see RaSocketWhiteListAdd
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaSocketWhiteListDel(void* socket_handle, struct SocketWlistInfoT white_list[], unsigned int num) { return 0; }

/**
 * @ingroup libinit
 * @brief get the number of interfaces
 * @param config [IN] config
 * @param num [OUT] num of interfaces
 * @see RaGetIfnum
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaGetIfnum(struct RaGetIfattr* config, unsigned int* num) { return 0; }

/**
 * @ingroup libinit
 * @brief get interface ips by device id
 * @param config [IN] config
 * @param interface_infos [OUT] ip result
 * @param num [IN/OUT] num of param and num of param found
 * @see RaGetIfaddrs
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaGetIfaddrs(struct RaGetIfattr* config, struct InterfaceInfo interface_infos[], unsigned int* num) { return 0; }

/**
 * @ingroup libcommon
 * @brief get interface version
 * @param interface_opcode [IN] interface opcode
 * @param phy_id [IN] phy id
 * @param interface_version [OUT] interface version
 * @retval #zero Success
 * @retval #non-zero Failure
 */
int RaGetInterfaceVersion(unsigned int phy_id, unsigned int interface_opcode, unsigned int* interface_version)
{
    return 0;
}

int RaGetTlsEnable(struct RaInfo* info, bool* tlsEnable)
{
    if (tlsEnable != nullptr) {
        *tlsEnable = false;
    }
    return 0;
}

int RaQpCreate(void* rdev_handle, int flag, int qp_mode, void** qp_handle) { return 0; }

int RaQpDestroy(void* qp_handle) { return 0; }

int RaQpConnectAsync(void* qp_handle, const void* fd_handle) { return 0; }

int RaGetQpStatus(void* qp_handle, int* status) { return 0; }

int RaMrReg(void* qp_handle, struct MrInfoT* info) { return 0; }

int RaMrDereg(void* qp_handle, struct MrInfoT* info) { return 0; }

int RaRegisterMr(const void* handle, struct MrInfoT* mrInfo, void** mrHandle)
{
    *mrHandle = mrInfo;
    return 0;
}

int RaDeregisterMr(const void* handle, void* mrHandle) { return 0; }

int RaCqCreate(void* rdev_handle, struct CqAttr* attr) { return 0; }

int RaCqDestroy(void* rdev_handle, struct CqAttr* attr) { return 0; }

int RaNormalQpCreate(void* rdev_handle, struct ibv_qp_init_attr* qp_init_attr, void** qp_handle, void** qp)
{
    return 0;
}

int RaNormalQpDestroy(void* qp_handle) { return 0; }

int RaLoopbackQpCreate(void* rdevHandle, struct LoopbackQpPair* qpPair, void** qpHandle) { return 0; }

int RaSendWr(void* qp_handle, struct SendWr* wr, struct SendWrRsp* op_rsp) { return 0; }

int RaCtxInit(struct CtxInitCfg* cfg, struct CtxInitAttr* info, void** ctx_handle)
{
    *ctx_handle = reinterpret_cast<void*>(0x12345678);
    return 0;
}

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
    info->key.size = DEV_QP_KEY_SIZE;
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
    info_list[0] = HccpTpInfo{0x12345, 0};
    *num = 1; // hccl预期调用1，随意调整可能越界
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

int RaCtxGetAuxInfo(void* ctx_handle, struct HccpAuxInfoIn* in, struct HccpAuxInfoOut* out) { return 0; }

int RaCtxQpQueryBatch(void* qp_handle[], struct JettyAttr attr[], unsigned int* num) { return 0; }

int RaGetLbMax(void* rdevHandle, int* lbMax) { return 0; }

int RaCtxQpDestroyBatchAsync(void* ctx_handle, void* qp_handle[], unsigned int* num, void** req_handle) { return 0; }

int RaSocketGetVnicIpInfos(
    unsigned int phyId, enum IdType type, unsigned int ids[], unsigned int num, struct IpInfo infos[])
{
    return 0;
}

int RaGetEidByIp(void* ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int* num) { return 0; }

int RaGetHccnCfg(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    (void)info;
    (void)key;
    if (value != nullptr && valueLen != nullptr && *valueLen > 0U) {
        value[0] = '\0';
    }
    if (valueLen != nullptr) {
        *valueLen = 0U;
    }
    // 未配置 qos_dscp：返回成功空串，由 GetDscpByQos 走默认 DSCP。
    return 0;
}

int RaSetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kStubRaSetTpAttrReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    if (reqHandle != nullptr) {
        *reqHandle = &kStubRaSetTpAttrReq;
    }
    return 0;
}

int RaGetTpAttrAsync(void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr, void** reqHandle)
{
    static char kStubRaTpAttrReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kStubRaTpAttrReq;
    }
    return 0;
}

int RaCtxGetTpInfoList(void* ctxHandle, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num)
{
    return 0;
}

int RaCtxGetTpAttr(void* ctxHandle, uint64_t tpHandle, uint32_t* attrBitmap, struct TpAttr* attr)
{
    (void)ctxHandle;
    (void)tpHandle;
    if (attrBitmap != nullptr) {
        *attrBitmap = 0x3U;
    }
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 1U;
    }
    return 0;
}

int RaCtxSetTpAttr(void* ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr* attr)
{
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    return 0;
}

int RaNdaGetDirectFlag(void* rdmaHandle, int* directFlag)
{
    if (directFlag != nullptr) {
        *directFlag = 1;
    }
    return 0;
}

int RaNdaCqCreate(void* rdmaHandle, struct NdaCqInitAttr* attr, struct NdaCqInfo* info, void** cqHandle) { return 0; }

int RaNdaCqDestroy(void* rdmaHandle, void* cqHandle) { return 0; }

int RaNdaQpCreate(void* rdmaHandle, struct NdaQpInitAttr* attr, struct NdaQpInfo* info, void** qpHandle) { return 0; }

int RaGetQpAttr(void* qpHandle, struct QpAttr* attr)
{
    if (attr != nullptr) {
        attr->qpn = 1;
        attr->psn = 1;
        attr->gidIdx = 0;
    }
    return 0;
}
