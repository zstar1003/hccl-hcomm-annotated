/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_H
#define HCCP_H

#include "hccp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup libsocket
 * @brief Client sockets batch connect to server sockets(async)
 * @param conn [IN] client sockets array
 * @param num [IN] num of conn
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Sockets batch close
 * @param conn [IN] sockets array, use disuse_linger of the fist conn as the common attr for all
 * @param num [IN] num of conn
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Client sockets batch abort connect to server sockets
 * @param conn [IN] client sockets array
 * @param num [IN] num of conn
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Sockets batch listen
 * @param conn [IN/OUT] server info array
 * @param num [IN] num of conn
 * @attention check if IP exist when SOCK_EADDRNOTAVAIL is returned
 * check if IP has been listened when SOCK_EADDRINUSE is returned
 * one IP address can only be listened once
 * @see ra_socket_listen_stop
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketListenStart(struct SocketListenInfoT conn[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Sockets batch stop
 * @param conn [IN] sockets info array
 * @param num [IN] num of conn
 * @see ra_socket_listen_start
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num);

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
HCCP_ATTRI_VISI_DEF int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num,
    unsigned int *connectedNum);

/**
 * @ingroup libsocket
 * @brief Send data by fd handle
 * @param fd_handle [IN] fd handle
 * @param data [IN] send storage buff
 * @param size [IN] size of data you want to send unit(Byte)
 * @param sent_size [OUT] number of sent bytes
 * @see ra_socket_recv
 * @attention if sent_size is greater than zero but less than size,
 * the function needed to be revoked again,
 * the param of size is original size minus sent_size,
 *the param of data should also offset by sent_size
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketSend(const void *fdHandle, const void *data, unsigned long long size,
    unsigned long long *sentSize);

/**
 * @ingroup libsocket
 * @brief Receive data by fd handle
 * @param fd_handle [IN] fd handle
 * @param data [IN/OUT] receive storage buff
 * @param size [IN] size of data you want to receive unit(Byte)
 * @param received_size [OUT] number of received bytes
 * @see ra_socket_send
 * @attention if return value is SOCK_EAGAIN which means no data right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #SOCK_EAGAIN Success(no data received by socket)
 * @retval #non-zero Failure(exclude SOCK_EAGAIN)
 */
HCCP_ATTRI_VISI_DEF int RaSocketRecv(const void *fdHandle, void *data, unsigned long long size,
    unsigned long long *receivedSize);

/**
 * @ingroup libsocket
 * @brief Get client sockets error info
 * @param conn [IN] client sockets array
 * @param err [OUT] sockets error info array
 * @param num [IN] num of conn and err
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetClientSocketErrInfo(struct SocketConnectInfoT conn[], struct SocketErrInfo err[],
    unsigned int num);

/**
 * @ingroup libsocket
 * @brief Get server sockets error info
 * @param conn [IN] server info array
 * @param err [OUT] sockets error info array
 * @param num [IN] num of conn and err
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetServerSocketErrInfo(struct SocketListenInfoT conn[], struct ServerSocketErrInfo err[],
    unsigned int num);

/**
 * @ingroup libsocket
 * @brief Add epoll listening event
 * @param fd_handle [IN] fd handle
 * @param event [IN] epoll event
 * @see ra_epoll_ctl_del
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event);

/**
 * @ingroup libsocket
 * @brief Modify epoll listening event
 * @param fd_handle [IN] fd handle
 * @param event [IN] epoll event
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaEpollCtlMod(const void *fdHandle, enum RaEpollEvent event);

/**
 * @ingroup libsocket
 * @brief Delete epoll listening event
 * @param fd_handle [IN] fd handle
 * @param event [IN] epoll event
 * @see ra_epoll_ctl_add
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaEpollCtlDel(const void *fdHandle);

/**
 * @ingroup libsocket
 * @brief Set hook function for epoll listening events
 * @param socket_handle [IN] socket handle
 * @param callback [IN] hook function
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetTcpRecvCallback(const void *socketHandle, const void *callback);

/**
 * @ingroup librdma
 * @brief Create qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev_handle
 * @param flag [IN] type of qp(reserved)
 * @param qp_mode [IN] mode to create qp
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpCreate(void *rdevHandle, int flag, int qpMode, void **qpHandle);

/**
 * @ingroup librdma
 * @brief Create qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev_handle
 * @param ext_attrs [IN] qp ext attrs
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpCreateWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, void **qpHandle);

/**
 * @ingroup librdma
 * @brief Create qp with pre-existing CQs
 * @param rdev_handle [IN] rdev_handle
 * @param ext_attrs [IN] qp ext attrs
 * @param send_cqn [IN] send cq number
 * @param recv_cqn [IN] recv cq number
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpCreateWithCQWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, unsigned int sendCqn,
    unsigned int recvCqn, void **qpHandle);

/**
 * @ingroup librdma
 * @brief Create qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev_handle
 * @param attrs [IN] ai qp attrs
 * @param info [OUT] ai qp info
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaAiQpCreate(void *rdevHandle, struct QpExtAttrs *attrs, struct AiQpInfo *info,
    void **qpHandle);

/**
 * @ingroup librdma
 * @brief Create loopback qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev_handle
 * @param qp_pair [OUT] loopback qp pair
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaLoopbackQpCreate(void *rdevHandle, struct LoopbackQpPair *qpPair, void **qpHandle);

/**
 * @ingroup librdma
 * @brief Destroy qp handle
 * @param qp_handle [IN] QP handle
 * @see ra_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpDestroy(void *qpHandle);
HCCP_ATTRI_VISI_DEF int RaQpDestroyWithoutCQ(void *qpHandle);

/**
 * @ingroup librdma
 * @brief set qp load balance value
 * @param qpHandle [IN] qp handle
 * @param lbValue [IN] load balance value
 * @see RaQpCreate
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetQpLbValue(void *qpHandle, int lbValue);

/**
 * @ingroup librdma
 * @brief get qp load balance value
 * @param qpHandle [IN] qp handle
 * @param lbValue [IN/OUT] load balance value
 * @see RaSetQpLbValue
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetQpLbValue(void *qpHandle, int *lbValue);

/**
 * @ingroup librdma
 * @brief Build QP chain by socket(exchange qp and mr info by socket)
 * revoke ra_get_qp_status to get qp async status
 * @param qp_handle [IN] QP handle
 * @param fd_handle [IN] fd handle
 * @see ra_get_qp_status
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpConnectAsync(void *qpHandle, const void *fdHandle);

/**
 * @ingroup librdma
 * @brief Create qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev_handle
 * @param flag [IN] type of qp(reserved)
 * @param qp_mode [IN] mode to create qp
 * @param qp_info [OUT] qp info
 * @param qp_handle [OUT] qp handle
 * @see ra_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTypicalQpCreate(void *rdevHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
    void **qpHandle);

/**
 * @ingroup librdma
 * @brief Modify qp handle step by step from init to rts
 * @param qp_handle [IN] local qp handle
 * @param local_qp_info [IN] local qp info
 * @param remote_qp_info [IN] remote qp info
 * @see ra_typical_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTypicalQpModify(void *qpHandle, struct TypicalQp *localQpInfo,
    struct TypicalQp *remoteQpInfo);

/**
 * @ingroup librdma
 * @brief Get qp async stats after revoking ra_qp_connect_async function
 * @param qp_handle [IN] qp handle
 * @param status [IN/OUT] qp connection status
 * 0:not connected, 1:connected, 2:timeout, 3:connecting
 * @see ra_qp_connect_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetQpStatus(void *qpHandle, int *status);

/**
 * @ingroup librdma
 * @brief Create comp channel
 * @param rdma_handle [IN] rdma handle
 * @param comp_channel [OUT] comp channel
 * @see ra_destroy_comp_channel
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCreateCompChannel(const void *rdmaHandle, void **compChannel);

/**
 * @ingroup librdma
 * @brief Destroy comp channel
 * @param rdma_handle [IN] rdma handle
 * @param comp_channel [IN] comp channel
 * @see ra_create_comp_channel
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaDestroyCompChannel(const void *rdmaHandle, void *compChannel);

/**
 * @ingroup librdma
 * @brief Create cq
 * @param rdev_handle [IN] rdev handle
 * @param attr [OUT] cq attr
 * @see ra_cq_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCqCreate(void *rdevHandle, struct CqAttr *attr);

/**
 * @ingroup librdma
 * @brief Destroy cq
 * @param rdev_handle [IN] rdev handle
 * @param attr [IN] cq attr
 * @see ra_cq_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCqDestroy(void *rdevHandle, struct CqAttr *attr);

/**
 * @ingroup librdma
 * @brief Create qp handle(only one qp handle)
 * @param rdev_handle [IN] rdev handle
 * @param qp_init_attr [IN] qp attr
 * @param qp_handle [OUT] qp handle
 * @param qp [OUT] qp resource
 * @see ra_normal_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNormalQpCreate(void *rdevHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle,
    void **qp);

/**
 * @ingroup librdma
 * @brief Destroy qp handle
 * @param qp_handle [IN] qp handle
 * @see ra_normal_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaNormalQpDestroy(void *qpHandle);

/**
 * @ingroup librdma
 * @brief Register mr
 * @param qp_handle [IN] qp handle
 * @param info [IN/OUT] mr info
 * @see ra_mr_dereg
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaMrReg(void *qpHandle, struct MrInfoT *info);

/**
 * @ingroup librdma
 * @brief Deregister mr
 * @param qp_handle [IN] qp handle
 * @param info [IN] mr info
 * @see ra_mr_reg
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaMrDereg(void *qpHandle, struct MrInfoT *info);

/**
 * @ingroup librdma
 * @brief Register mr
 * @param rdma_handle [IN] rdma handle
 * @param info [IN/OUT] mr info
 * @param mr_handle [OUT] mr handle
 * @see ra_deregister_mr
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRegisterMr(const void *rdmaHandle, struct MrInfoT *info, void **mrHandle);

/**
 * @ingroup librdma
 * @brief Remap mr
 * @param rdma_handle [IN] rdma handle
 * @param info [IN] mem info list of mr
 * @param num [IN] num of info, max num of input is REMAP_MR_MAX_NUM
 * @see ra_register_mr
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRemapMr(const void *rdmaHandle, struct MemRemapInfo info[], unsigned int num);

/**
 * @ingroup librdma
 * @brief Deregister mr
 * @param rdma_handle [IN] rdma handle
 * @param mr_handle [IN] handle
 * @see ra_register_mr
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaDeregisterMr(const void *rdmaHandle, void *mrHandle);

/**
 * @ingroup librdma
 * @brief Send RDMA packet async
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request
 * @param op_rsp [IN/OUT] respond of sending work request
 * @attention if return value is SOCK_ENOENT which means mr async not success right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #SOCK_ENOENT Success
 * @retval #non-zero Failure(exclude SOCK_ENOENT)
 */
HCCP_ATTRI_VISI_DEF int RaSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp);

/**
 * @ingroup librdma
 * @brief Send RDMA packet async v2
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request
 * @param op_rsp [IN/OUT] respond of sending work request
 * @attention if return value is SOCK_ENOENT which means mr async not success right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #SOCK_ENOENT Success
 * @retval #non-zero Failure(exclude SOCK_ENOENT)
 */
HCCP_ATTRI_VISI_DEF int RaSendWrV2(void *qpHandle, struct SendWrV2 *wr, struct SendWrRsp *opRsp);
HCCP_ATTRI_VISI_DEF int RaSendWrVerbs(void *qpHandle, struct SendWrVerbs *wr, struct SendWrRsp *opRsp);
HCCP_ATTRI_VISI_DEF int RaRecvWrVerbs(void *qpHandle, struct RecvWrVerbs *wr);

/**
 * @ingroup librdma
 * @brief Send RDMA packet async with typical qp
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request
 * @param op_rsp [IN/OUT] respond of sending work request
 * @retval #zero Success
 * @retval #SOCK_ENOENT Success
 * @retval #non-zero Failure(exclude SOCK_ENOENT)
 */
HCCP_ATTRI_VISI_DEF int RaTypicalSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp);

/**
 * @ingroup librdma
 * @brief Send RDMA request async
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request list
 * @param op_rsp [IN/OUT] respond list of sending work request
 * @param send_num [IN] size of wr list
 * @param complete_num [OUT] number of wr been sent successfully
 * @attention if return value is SOCK_ENOENT which means mr async not success right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #SOCK_ENOENT Success
 * @retval #non-zero Failure(exclude SOCK_ENOENT)
 */
HCCP_ATTRI_VISI_DEF int RaSendWrlist(void *qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum);

/**
 * @ingroup librdma
 * @brief Send RDMA request async
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request list (ext)
 * @param op_rsp [IN/OUT] respond list of sending work request
 * @param send_num [IN] size of wr list
 * @param complete_num [OUT] number of wr been sent successfully
 * @attention if return value is SOCK_ENOENT which means mr async not success right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #SOCK_ENOENT Success
 * @retval #non-zero Failure(exclude SOCK_ENOENT)
 */
HCCP_ATTRI_VISI_DEF int RaSendWrlistExt(void *qpHandle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum);

/**
 * @ingroup librdma
 * @brief Send RDMA request async
 * @param qp_handle [IN] qp handle
 * @param wr [IN] work request list
 * @param op_rsp [IN/OUT] respond list of sending work request
 * @param send_num [IN] size of wr list
 * @param complete_num [OUT] number of wr been sent successfully
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSendNormalWrlist(void *qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum);

/**
 * @ingroup librdma
 * @brief Get notify base address and size
 * @param rdev_handle [IN] rdev handle
 * @param va [IN/OUT] address of notify
 * @param size [IN/OUT] size of notify
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetNotifyBaseAddr(void *rdevHandle, unsigned long long *va, unsigned long long *size);

/**
 * @ingroup librdma
 * @brief Get notify mr info
 * @param rdev_handle [IN] rdev handle
 * @param info [OUT] mr info
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetNotifyMrInfo(void *rdevHandle, struct MrInfoT *info);

/**
 * @ingroup libinit
 * @brief Rdma_agent initialization
 * @param config [IN] initialization configuration of rdma_agent
 * @see ra_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaInit(struct RaInitConfig *config);

/**
 * @ingroup libcommon
 * @brief get tls enable info
 * @param info [IN] see ra_info
 * @param tls_enable [OUT] tls enable. true: enable, false: disable
 * @see ra_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetTlsEnable(struct RaInfo *info, bool *tlsEnable);

/**
 * @ingroup libcommon
 * @brief get hccn cfg value
 * @param info [IN] see ra_info
 * @param key [IN] see enum hccn_cfg_key
 * @param value [IN/OUT] corresponding key value
 * @param value_len [IN/OUT] value len, in_value_len should >= 2K
 * @see ra_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen);

/**
 * @ingroup libinit
 * @brief Rdma_agent deinitialization
 * @param config [IN] deinitialization configuration of rdma_agent
 * @see ra_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaDeinit(struct RaInitConfig *config);

/**
 * @ingroup libinit
 * @brief Rdma_agent socket initialization
 * @param mode [IN] network mode
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param socket_handle [OUT] socket handle info
 * @see ra_socket_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketInit(int mode, struct rdev rdevInfo, void **socketHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent socket initialization
 * @param mode [IN] network mode
 * @param socket_init_info_t [IN] socket_init including dev_id and ip info
 * @param socket_handle [OUT] socket handle info
 * @see ra_socket_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit, void **socketHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent socket deinitialization
 * @param socket_handle [IN] deinitialization handle of socket
 * @see ra_socket_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketDeinit(void *socketHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev initialization will start lite thread by default
 * @param mode [IN] network mode
 * @param notify_type [IN] notify type
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param rdma_handle [OUT] rdma_handle rdma handle info
 * @see ra_rdev_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevInit(int mode, unsigned int notifyType, struct rdev rdevInfo, void **rdmaHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev initialization
 * @param init_info [IN] init info, can customize to start lite thread or not
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param rdma_handle [OUT] rdma_handle rdma handle info
 * @see ra_rdev_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevInitV2(struct RdevInitInfo initInfo, struct rdev rdevInfo, void **rdmaHandle);

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev initialization with backup
 * @param init_info [IN] init info, can customize to start lite thread or not
 * @param rdev_info [IN] rdev_info including dev_id and ip info
 * @param backup_rdev_info [IN] backup_rdev_info including dev_id and ip info
 * @param rdma_handle [OUT] rdma_handle rdma handle info related to rdev_info
 * @see ra_rdev_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevInitWithBackup(struct RdevInitInfo *initInfo, struct rdev *rdevInfo,
    struct rdev *backupRdevInfo, void **rdmaHandle);

/**
 * @ingroup librdma
 * @brief get max load balance value
 * @param rdevHandle [IN] rdev handle
 * @param lbMax [IN/OUT] load balance max value
 * @see RaRdevInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetLbMax(void *rdevHandle, int *lbMax);

/**
 * @ingroup libinit
 * @brief Rdma_agent rdev deinitialization
 * @param rdma_handle [IN] deinitialization handle of rdev
 * @param notify_type [IN] notify type
 * @see ra_rdev_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevDeinit(void *rdmaHandle, unsigned int notifyType);

/**
 * @ingroup libinit
 * @brief Rdma_agent get support lite
 * @param rdma_handle [IN] rdma handle
 * @param support_lite [OUT] rdma lite support(0: not support lite, 1: 4KB page align lite, 2: 2MB page align lite)
 * @see ra_rdev_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevGetSupportLite(void *rdmaHandle, int *supportLite);

/**
 * @ingroup libsocket
 * @brief set socket whitelist status
 * @param enable [IN] enable or disable
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketSetWhiteListStatus(unsigned int enable);

/**
 * @ingroup libsocket
 * @brief get socket whitelist status
 * @param enable [OUT] enable or disable
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketGetWhiteListStatus(unsigned int *enable);

/**
 * @ingroup libsocket
 * @brief Add server socket whitelist
 * @param socket_handle [IN] socket handle
 * @param white_list [IN] server's whitelist
 * @param num [IN] num of whitelist
 * @see ra_socket_white_list_del
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketWhiteListAdd(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Remove server socket whitelist
 * @param socket_handle [IN] socket handle
 * @param white_list [IN] server's whitelist
 * @param num [IN] num of whitelist
 * @see ra_socket_white_list_add
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketWhiteListDel(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num);

/**
 * @ingroup libsocket
 * @brief Sockets batch listen
 * @param conn [IN] server info array
 * @param num [IN] num of conn
 * @param credit_limit [IN] credit limit
 * @attention once set accept credit, once it exhausted listen_fd will be ctl_del.
 * credit need to be add again to ctl_add listen_fd to handle accept events
 * @see ra_socket_listen_start
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num,
    unsigned int creditLimit);

/**
 * @ingroup libinit
 * @brief get the number of interfaces
 * @param config [IN] config
 * @param num [OUT] num of interfaces
 * @see ra_get_ifnum
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetIfnum(struct RaGetIfattr *config, unsigned int *num);

/**
 * @ingroup libinit
 * @brief get interface ips by device id
 * @param config [IN] config
 * @param interface_infos [OUT] ip result
 * @param num [IN/OUT] num of param and num of param found
 * @see ra_get_ifaddrs
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetIfaddrs(struct RaGetIfattr *config, struct InterfaceInfo interfaceInfos[],
    unsigned int *num);

/**
 * @ingroup libinit
 * @brief get vnic ip infos by corresponding id_type
 * @param phy_id [IN] phy id
 * @param type [IN] id type
 * @param ids [IN] id info, see id_type
 * @param num [IN] ids and infos array size
 * @param infos [OUT] ip infos
 * @see ra_socket_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSocketGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[],
    unsigned int num, struct IpInfo infos[]);

/**
 * @ingroup libcommon
 * @brief get interface version
 * @param interface_opcode [IN] interface opcode
 * @param phy_id [IN] phy id
 * @param interface_version [OUT] interface version
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode,
    unsigned int *interfaceVersion);

/**
 * @ingroup libcommon
 * @brief check whether the RA instance has the specified capability
 * @param info [IN] see struct RaInfo
 * @param capability [IN] Capability flag to check (RA_CAP_xxx).
 * @retval true  if the capability is supported,
 *         false otherwise.
 */
HCCP_ATTRI_VISI_DEF bool RaHasCapability(struct RaInfo *info, unsigned int capability);

/**
 * @ingroup librdma
 * @brief This function only invoked in asynchronous GDR scenario for more flexible shared memory partition,
 * it will set template depth and obtain max supported qp numbers. The template is a mechanism designed for
 * implementing asynchronous GDR. RoCE produce packets to template, and TS consumer packets from template.
 * @param rdev_handle [IN] rdev_handle
 * @param temp_depth [IN] template depth which decided by service requirements
 * @param qp_num [OUT] max supported numbers of qp
 * @see ra_get_tsqp_depth
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetTsqpDepth(void *rdevHandle, unsigned int tempDepth, unsigned int *qpNum);

/**
 * @ingroup librdma
 * @brief This function only invoked in asynchronous GDR scenario to get the template depth
 * and max supported qp numbers
 * @param rdev_handle [IN] rdev_handle
 * @param temp_depth [OUT] template depth
 * @param qp_num [OUT] max supported numbers of qp
 * @see ra_set_tsqp_depth
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetTsqpDepth(void *rdevHandle, unsigned int *tempDepth, unsigned int *qpNum);

/**
 * @ingroup librdma
 * @brief Rdma_agent get port status
 * @param rdma_handle [IN] rdma_handle
 * @param status [OUT] port status, see enum port_status
 * @see ra_rdev_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevGetPortStatus(void *rdmaHandle, enum PortStatus *status);

/**
 * @ingroup librdma
 * @brief Post RDMA recv request async
 * @param qp_handle [IN] qp handle
 * @param wr [IN] recv work request list
 * @param recv_num [IN] size of wr list
 * @param complete_num [OUT] number of wr been post recv request successfully
 * @attention if return value is SOCK_ENOENT which means mr async not success right now,
 * you need to revoke the function again
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRecvWrlist(void *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum);

/**
 * @ingroup librdma
 * @brief poll cq
 * @param qp_handle [IN] qp handle
 * @param is_send_cq [IN] true(send cq) or false(recv cq)
 * @param num_entries [IN] maximum number of completions to return
 * @param wc [OUT] array of at least @num_entries of &struct wc where completions
 * will be returned
 * @retval #non-negative Success it is the number of completions returned.
 * @retval #negative Failure
 */
HCCP_ATTRI_VISI_DEF int RaPollCq(void *qpHandle, bool isSendCq, unsigned int numEntries, void *wc);

HCCP_ATTRI_VISI_DEF int RaPollTypicalCq(void *cqHandle, unsigned int numEntries, void *wc);

/**
 * @ingroup librdma
 * @brief create typical cq
 * @param rdevHandle [IN] rdma device handle
 * @param cqDepth [IN] cq depth
 * @param cqn [OUT] cq number
 * @param cqHandle [OUT] cq handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTypicalCqCreate(void *rdevHandle, unsigned int cqDepth, unsigned int *cqn, void **cqHandle);

/**
 * @ingroup librdma
 * @brief destroy typical cq
 * @param rdevHandle [IN] rdma device handle
 * @param cqn [IN] cq number
 * @param cqHandle [IN] cq handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaTypicalCqDestroy(void *rdevHandle, unsigned int cqn, void *cqHandle);

/**
 * @ingroup librdma
 * @brief get rdma qp context
 * @param rdev_handle [IN] qp_handle
 * @param qp [OUT] rdma qp
 * @param send_cq [OUT] rdma send cq
 * @param recv_cq [OUT] rdma recv cq
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetQpContext(void *qpHandle, void **qp, void **sendCq, void **recvCq);

/**
 * @ingroup librdma
 * @brief set qos attr in qp
 * @param qp_handle [IN] qp handle
 * @param attr [IN] qp qos attr
 * @see ra_mr_dereg
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetQpAttrQos(void *qpHandle, struct QosAttr *attr);

/**
 * @ingroup librdma
 * @brief set timeout attr in qp
 * @param qp_handle [IN] qp handle
 * @param attr [IN] qp timeout attr
 * @see ra_mr_dereg
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetQpAttrTimeout(void *qpHandle, unsigned int *timeout);

/**
 * @ingroup librdma
 * @brief set retry cnt attr in qp
 * @param qp_handle [IN] qp handle
 * @param attr [IN] qp qretry cntos attr
 * @see ra_mr_dereg
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetQpAttrRetryCnt(void *qpHandle, unsigned int *retryCnt);

/**
 * @ingroup librdma
 * @brief get cqe err info
 * @param phy_id [IN] phy id
 * @param info [IN/OUT] cqe err info
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info);

/**
 * @ingroup librdma
 * @brief Rdma_agent get cqe err info by rdma_handle
 * @param rdma_handle [IN] rdma handle
 * @param info_list [IN/OUT] cqe err info
 * @param num [IN/OUT] num of cqe err info, max num of input is CQE_ERR_INFO_MAX_NUM
 * @see ra_rdev_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevGetCqeErrInfoList(void *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num);

/**
 * @ingroup librdma
 * @brief get qp attr from qp handle
 * @param qp_handle [IN] qp handle
 * @param attr [IN/OUT] qp attr, see qp_attr
 * @see ra_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetQpAttr(void *qpHandle, struct QpAttr *attr);

/**
 * @ingroup librdma
 * @brief create srq by rdma handle
 * @param rdma_handle [IN] rdma handle,
 * @param rx_depth [IN] srq depth
 * @param srq_handle [IN/OUT] srq handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCreateSrq(const void *rdmaHandle, struct SrqAttr *attr);

/**
 * @ingroup librdma
 * @brief create destroy by srq handle
 * @param srq_handle [IN] srq handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaDestroySrq(const void *rdmaHandle, struct SrqAttr *attr);

/**
 * @ingroup libsocket
 * @brief Create event handle
 * @param event_handle [OUT] event handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCreateEventHandle(int *eventHandle);

/**
 * @ingroup libsocket
 * @brief Control event handle
 * @param event_handle [IN] event handle
 * @param fd_handle [IN] fd handle
 * @param opcode [IN] valid opcodes to issue to sys_epoll_ctl
 * @param event [IN] epoll event
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event);

/**
 * @ingroup libsocket
 * @brief Wait event handle
 * @param event_handle [IN] event handle
 * @param event_infos [OUT] socket events
 * @param timeout [IN] epoll timeout
 * @param maxevents [IN] max socket events num
 * @param events_num [OUT] socket events num
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout,
    unsigned int maxevents, unsigned int *eventsNum);

/**
 * @ingroup libsocket
 * @brief Destroy event handle
 * @param event_handle [OUT] event handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaDestroyEventHandle(int *eventHandle);

/**
 * @ingroup librdma
 * @brief unimport remote mem
 * @param rdma_handle [IN] rdma handle
 * @param qp_handle [IN] qp handle
 * @param num [IN] qp handle num
 * @param expect_status [IN] expect_status
 * @see ra_get_qp_status
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaQpBatchModify(void *rdmaHandle, void *qpHandle[], unsigned int num, int expectStatus);

/**
 * @ingroup librdma
 * @brief get rdma handle
 * @param phy_id [IN] phy id
 * @param rdma_handle [IN] rdma handle
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRdevGetHandle(unsigned int phyId, void **rdmaHandle);

/**
 * @ingroup libcommon
 * @brief CRIU(Checkpoint/Restore in Userspace) save snapshot
 * @param info [IN] see ra_info
 * @param action [IN] see enum save_snapshot_action
 * @see ra_restore_snapshot
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSaveSnapshot(struct RaInfo *info, enum SaveSnapshotAction action);

/**
 * @ingroup libcommon
 * @brief CRIU(Checkpoint/Restore in Userspace) restore snapshot
 * @param info [IN] see ra_info
 * @see ra_save_snapshot
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaRestoreSnapshot(struct RaInfo *info);

/**
 * @ingroup libcommon
 * @brief ra get sec random
 * @param info [IN] see ra_info
 * @param value [OUT] sec random value
 * @see ra_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetSecRandom(struct RaInfo *info, uint32_t *value);

/**
 * @ingroup libcommon
 * @brief ra set debug config
 * @param info [IN] see struct RaInfo
 * @param debugConfig [IN] debug config mask
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetDebugConfig(struct RaInfo *info, uint64_t debugConfig);

#ifdef __cplusplus
}
#endif
#endif // HCCP_H
