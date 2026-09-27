/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ra_rs_comm.h"
#include "dl_ibverbs_function.h"
#include "dl_net_function.h"
#include "rs.h"

struct OpcodeInterfaceInfo gInterfaceInfoList[] = {
    // outer opcode version: 1.0
    {RA_RS_SOCKET_CONN, 2},
    {RA_RS_SOCKET_CLOSE, 2},
    {RA_RS_SOCKET_ABORT, 1},
    {RA_RS_SOCKET_LISTEN_START, 2},
    {RA_RS_SOCKET_LISTEN_STOP, 2},
    {RA_RS_GET_SOCKET, 3},
    {RA_RS_SOCKET_SEND, 1},
    {RA_RS_SOCKET_RECV, 1},
    {RA_RS_QP_CREATE, 2},
    {RA_RS_QP_CREATE_WITH_ATTRS, 1},
    {RA_RS_AI_QP_CREATE, 3},
    {RA_RS_AI_QP_CREATE_WITH_ATTRS, 1},
    {RA_RS_TYPICAL_QP_CREATE, 1},
    {RA_RS_QP_DESTROY, 1},
    {RA_RS_QP_DESTROY_WITHOUT_CQ, 1},
    {RA_RS_QP_CONNECT, 2},
    {RA_RS_TYPICAL_QP_MODIFY, 3},
    {RA_RS_QP_STATUS, 1},
    {RA_RS_QP_INFO, 1},
    {RA_RS_QP_BATCH_MODIFY, 2},
    {RA_RS_MR_REG, 2},
    {RA_RS_MR_DEREG, 1},
    {RA_RS_TYPICAL_MR_REG_V1, 2},
    {RA_RS_TYPICAL_MR_REG, 1},
    {RA_RS_REMAP_MR, 1},
    {RA_RS_TYPICAL_MR_DEREG, 1},
    {RA_RS_SEND_WR, 1},
    {RA_RS_GET_NOTIFY_BA, 2},
    {RA_RS_INIT, 2},
    {RA_RS_DEINIT, 1},
    {RA_RS_SOCKET_INIT, 1},
    {RA_RS_SOCKET_DEINIT, 1},
    {RA_RS_RDEV_INIT, 2},
    {RA_RS_RDEV_INIT_WITH_BACKUP, 1},
    {RA_RS_RDEV_GET_PORT_STATUS, 1},
    {RA_RS_RDEV_DEINIT, 1},
    {RA_RS_WLIST_ADD, 1},
    {RA_RS_WLIST_ADD_V2, 1},
    {RA_RS_WLIST_DEL, 1},
    {RA_RS_WLIST_DEL_V2, 1},
    {RA_RS_ACCEPT_CREDIT_ADD, 1},
    {RA_RS_GET_IFADDRS, 2},
    {RA_RS_GET_IFADDRS_V2, 3},
    {RA_RS_GET_INTERFACE_VERSION, 1},
    {RA_RS_SEND_WRLIST, 1},
    {RA_RS_SEND_WRLIST_V2, 1},
    {RA_RS_SEND_WRLIST_EXT, 1},
    {RA_RS_SEND_WRLIST_EXT_V2, 1},
    {RA_RS_SEND_NORMAL_WRLIST, 1},
    {RA_RS_SET_TSQP_DEPTH, 1},
    {RA_RS_GET_TSQP_DEPTH, 1},
    {RA_RS_SET_QP_ATTR_QOS, 1},
    {RA_RS_SET_QP_ATTR_TIMEOUT, 1},
    {RA_RS_SET_QP_ATTR_RETRY_CNT, 1},
    {RA_RS_GET_CQE_ERR_INFO, 1},
    {RA_RS_GET_CQE_ERR_INFO_NUM, 1},
    {RA_RS_GET_CQE_ERR_INFO_LIST, 1},
    {RA_RS_GET_LITE_SUPPORT, 2},
    {RA_RS_GET_LITE_RDEV_CAP, 1},
    {RA_RS_GET_LITE_QP_CQ_ATTR, 1},
    {RA_RS_GET_LITE_QP_ATTR, 1},
    {RA_RS_GET_LITE_CONNECTED_INFO, 1},
    {RA_RS_GET_LITE_MEM_ATTR, 1},
    {RA_RS_PING_INIT, 1},
    {RA_RS_PING_ADD, 1},
    {RA_RS_PING_START, 1},
    {RA_RS_PING_GET_RESULTS, 1},
    {RA_RS_PING_STOP, 1},
    {RA_RS_PING_DEL, 1},
    {RA_RS_PING_DEINIT, 1},
    {RA_RS_GET_VNIC_IP_INFOS_V1, 1},
    {RA_RS_GET_VNIC_IP_INFOS, 1},
    {RA_RS_TLV_INIT_V1, 2},
    {RA_RS_TLV_INIT, 1},
    {RA_RS_TLV_DEINIT, 1},
    {RA_RS_TLV_REQUEST, 1},
    {RA_RS_TLV_REQUEST_V2, 1},
    {RA_RS_CUSTOM_CHANNEL_DEPRECATED, 1},
    {RA_RS_GET_TLS_ENABLE, 1},
    {RA_RS_GET_SEC_RANDOM, 1},
    {RA_RS_GET_HCCN_CFG, 3},
    {RA_RS_GET_ROCE_API_VERSION, 0},
    {RA_RS_TYPICAL_CQ_CREATE, 1},
    {RA_RS_GET_LITE_CQ_ATTR, 1},
    {RA_RS_QP_CREATE_WITH_CQ_WITH_ATTRS, 1},
    {RA_RS_TYPICAL_CQ_DESTROY, 1},

    // outer opcode version: 2.0
    {RA_RS_GET_DEV_EID_INFO_NUM, 1},
    {RA_RS_GET_DEV_EID_INFO_LIST, 2},
    {RA_RS_CTX_INIT, 2},
    {RA_RS_CTX_GET_ASYNC_EVENTS, 2},
    {RA_RS_CTX_DEINIT, 1},
    {RA_RS_GET_EID_BY_IP, 1},
    {RA_RS_GET_IP_BY_EID, 1},
    {RA_RS_GET_TP_INFO_LIST, 2},
    {RA_RS_GET_TP_ATTR, 3},
    {RA_RS_SET_TP_ATTR, 3},
    {RA_RS_CTX_TOKEN_ID_ALLOC, 1},
    {RA_RS_CTX_TOKEN_ID_FREE, 1},
    {RA_RS_LMEM_REG, 1},
    {RA_RS_LMEM_UNREG, 1},
    {RA_RS_RMEM_IMPORT, 1},
    {RA_RS_RMEM_UNIMPORT, 1},
    {RA_RS_CTX_CHAN_CREATE, 1},
    {RA_RS_CTX_CHAN_DESTROY, 1},
    {RA_RS_CTX_CQ_CREATE, 1},
    {RA_RS_CTX_QUERY_QP_BATCH, 1},
    {RA_RS_CTX_CQ_DESTROY, 1},
    {RA_RS_CTX_QP_DESTROY_BATCH, 1},
    {RA_RS_CTX_QP_CREATE, 1},
    {RA_RS_CTX_QP_DESTROY, 1},
    {RA_RS_CTX_QP_IMPORT, 1},
    {RA_RS_CTX_QP_UNIMPORT_DEPRECATED, 1},
    {RA_RS_CTX_QP_UNIMPORT, 1},
    {RA_RS_CTX_QP_BIND, 1},
    {RA_RS_CTX_QP_UNBIND, 1},
    {RA_RS_CTX_BATCH_SEND_WR, 1},
    {RA_RS_CTX_UPDATE_CI, 1},
    {RA_RS_CTX_GET_AUX_INFO, 1},
    {RA_RS_CTX_GET_CR_ERR_INFO_LIST, 1},
    {RA_RS_CTX_GET_UB_CONTEXT, 1},
    {RA_RS_CTX_NOTIFY_EVENT, 1},
    {RA_RS_GET_NET_API_VERSION, 0},

    // inner opcode version
    {RA_RS_HDC_SESSION_CLOSE, 1},
    {RA_RS_GET_VNIC_IP, 1},
    {RA_RS_NOTIFY_CFG_SET, 1},
    {RA_RS_NOTIFY_CFG_GET, 1},
    {RA_RS_SET_PID, 1},
    {RA_RS_ASYNC_HDC_SESSION_CONNECT, 2},
    {RA_RS_ASYNC_HDC_SESSION_CLOSE, 1},
    {RA_RS_SET_DEBUG_CONFIG, 1},
};

RS_ATTRI_VISI_DEF int RsGetInterfaceVersion(unsigned int opcode, unsigned int *version)
{
    int i;
    unsigned int interfaceVersion = 0; // default interface is 0 (0: not support this interface opcode)
    int num = sizeof(gInterfaceInfoList) / sizeof(gInterfaceInfoList[0]);

    CHK_PRT_RETURN(version == NULL, hccp_err("rs_get_interface_version failed! version is null"), -EINVAL);

    for (i = 0; i < num; i++) {
        if (opcode == gInterfaceInfoList[i].opcode && opcode != RA_RS_GET_ROCE_API_VERSION &&
            opcode != RA_RS_GET_NET_API_VERSION) {
            interfaceVersion = gInterfaceInfoList[i].version;
            break;
        } else if (opcode == RA_RS_GET_ROCE_API_VERSION) {
            interfaceVersion = RsRoceGetApiVersion();
            break;
        } else if (opcode == RA_RS_GET_NET_API_VERSION) {
            interfaceVersion = RsNetGetApiVersion();
            break;
        }
    }

    *version = interfaceVersion;
    return 0;
}
