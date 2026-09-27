/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RS_OPCODE_H
#define RA_RS_OPCODE_H

#include <stdbool.h>
#include "hccp_opcode.h"

enum OpType {
    RA_RS_SOCKET_CONN = 0,
    RA_RS_SOCKET_CLOSE = 1,
    RA_RS_SOCKET_LISTEN_START = 2,
    RA_RS_SOCKET_LISTEN_STOP = 3,
    RA_RS_GET_SOCKET = 4,
    RA_RS_SOCKET_SEND = 5,
    RA_RS_SOCKET_RECV = 6,
    RA_RS_QP_CREATE = 7,
    RA_RS_QP_DESTROY = 8,
    RA_RS_QP_CONNECT = 9,
    RA_RS_QP_STATUS = 10,
    RA_RS_MR_REG = 11,
    RA_RS_MR_DEREG = 12,
    RA_RS_SEND_WR = 13,
    RA_RS_GET_NOTIFY_BA = 14,
    RA_RS_INIT = 15,
    RA_RS_DEINIT = 16,
    RA_RS_SOCKET_INIT = 17,
    RA_RS_SOCKET_DEINIT = 18,
    RA_RS_RDEV_INIT = 19,
    RA_RS_RDEV_DEINIT = 20,
    RA_RS_WLIST_ADD = 21,
    RA_RS_WLIST_DEL = 22,
    RA_RS_GET_IFADDRS = 23,
    RA_RS_GET_INTERFACE_VERSION = 24,
    RA_RS_SEND_WRLIST = 25,
    RA_RS_SET_TSQP_DEPTH = 26,
    RA_RS_GET_TSQP_DEPTH = 27,
    RA_RS_SEND_WRLIST_EXT = 28,
    RA_RS_SET_QP_ATTR_QOS = 29,
    RA_RS_SET_QP_ATTR_TIMEOUT = 30,
    RA_RS_SET_QP_ATTR_RETRY_CNT = 31,
    RA_RS_GET_CQE_ERR_INFO = 32,
    RA_RS_GET_IFNUM = 33,
    RA_RS_GET_LITE_SUPPORT = 34,
    RA_RS_GET_LITE_RDEV_CAP = 35,
    RA_RS_GET_LITE_QP_CQ_ATTR = 36,
    RA_RS_GET_LITE_CONNECTED_INFO = 37,
    RA_RS_GET_IFADDRS_V2 = 38,
    RA_RS_QP_CREATE_WITH_ATTRS = 39,
    RA_RS_WLIST_ADD_V2 = 40,
    RA_RS_WLIST_DEL_V2 = 41,
    RA_RS_SEND_WRLIST_V2 = 42,
    RA_RS_SEND_WRLIST_EXT_V2 = 43,
    RA_RS_GET_LITE_MEM_ATTR = 44,
    RA_RS_TYPICAL_QP_CREATE = 45,
    RA_RS_TYPICAL_QP_MODIFY = 46,
    RA_RS_TYPICAL_MR_REG_V1 = 47,
    RA_RS_TYPICAL_MR_DEREG = 48,
    RA_RS_CTX_INIT = 49,
    RA_RS_CTX_DEINIT = 50,
    RA_RS_LMEM_REG = 51,
    RA_RS_LMEM_UNREG = 52,
    RA_RS_RMEM_IMPORT = 53,
    RA_RS_RMEM_UNIMPORT = 54,
    RA_RS_GET_VNIC_IP_INFOS_V1 = 55,
    RA_RS_GET_DEV_EID_INFO_NUM = 56,
    RA_RS_GET_DEV_EID_INFO_LIST = 57,
    RA_RS_CTX_CQ_CREATE = 58,
    RA_RS_CTX_CQ_DESTROY = 59,
    RA_RS_CTX_QP_CREATE = 60,
    RA_RS_CTX_QP_DESTROY = 61,
    RA_RS_CTX_QP_IMPORT = 62,
    RA_RS_CTX_QP_UNIMPORT_DEPRECATED = 63,
    RA_RS_CTX_QP_BIND = 64,
    RA_RS_CTX_QP_UNBIND = 65,
    RA_RS_CTX_BATCH_SEND_WR = 66,
    RA_RS_QP_BATCH_MODIFY = 67,
    RA_RS_AI_QP_CREATE = 68,
    RA_RS_CUSTOM_CHANNEL_DEPRECATED = 69,
    RA_RS_RDEV_GET_PORT_STATUS = 70,
    RA_RS_PING_INIT = 71,
    RA_RS_PING_ADD = 72,
    RA_RS_PING_START = 73,
    RA_RS_PING_GET_RESULTS = 74,
    RA_RS_PING_STOP = 75,
    RA_RS_PING_DEL = 76,
    RA_RS_PING_DEINIT = 77,
    RA_RS_CTX_UPDATE_CI = 78,
    RA_RS_GET_CQE_ERR_INFO_NUM = 79,
    RA_RS_GET_CQE_ERR_INFO_LIST = 80,
    RA_RS_RDEV_INIT_WITH_BACKUP = 81,
    RA_RS_QP_INFO = 82,
    RA_RS_SEND_NORMAL_WRLIST = 83,
    RA_RS_CTX_CHAN_CREATE = 84,
    RA_RS_CTX_CHAN_DESTROY = 85,
    RA_RS_AI_QP_CREATE_WITH_ATTRS = 86,
    RA_RS_TLV_INIT_V1 = 87,
    RA_RS_TLV_DEINIT = 88,
    RA_RS_TLV_REQUEST = 89,
    RA_RS_CTX_TOKEN_ID_ALLOC = 90,
    RA_RS_CTX_TOKEN_ID_FREE = 91,
    RA_RS_REMAP_MR = 92,
    RA_RS_GET_TP_INFO_LIST = 93,
    RA_RS_GET_VNIC_IP_INFOS = 94,
    RA_RS_GET_TLS_ENABLE = 95,
    RA_RS_GET_ROCE_API_VERSION = 96,
    RA_RS_SOCKET_ABORT = 97,
    RA_RS_ACCEPT_CREDIT_ADD = 98,
    RA_RS_GET_SEC_RANDOM = 99,
    RA_RS_GET_HCCN_CFG = 100,
    RA_RS_TYPICAL_MR_REG = 101,
    RA_RS_CTX_QP_DESTROY_BATCH = 102,
    RA_RS_CTX_QUERY_QP_BATCH = 103,
    RA_RS_GET_EID_BY_IP = 104,
    RA_RS_CTX_GET_AUX_INFO = 105,
    RA_RS_GET_TP_ATTR = 106,
    RA_RS_SET_TP_ATTR = 107,
    RA_RS_CTX_GET_CR_ERR_INFO_LIST = 108,
    RA_RS_CTX_GET_ASYNC_EVENTS = 109,
    RA_RS_TLV_INIT = 110,
    RA_RS_CTX_GET_UB_CONTEXT = 111,
    RA_RS_TYPICAL_CQ_CREATE = 112,
    RA_RS_GET_LITE_CQ_ATTR = 113,
    RA_RS_QP_CREATE_WITH_CQ_WITH_ATTRS = 114,
    RA_RS_TYPICAL_CQ_DESTROY = 115,
    RA_RS_QP_DESTROY_WITHOUT_CQ = 116,
    RA_RS_GET_LITE_QP_ATTR = 117,
    RA_RS_GET_NET_API_VERSION = 118,
    RA_RS_TLV_REQUEST_V2 = 119,
    RA_RS_CTX_QP_UNIMPORT = 120,
    RA_RS_GET_IP_BY_EID = 121,
    RA_RS_CTX_NOTIFY_EVENT = 122,
    RA_RS_EXTER_OP_MAX_NUM,

    // 上面opcode是对部opcode,下面是内部opcode
    RA_RS_HDC_SESSION_CLOSE = 1000,
    RA_RS_GET_VNIC_IP = 1001,
    RA_RS_NOTIFY_CFG_SET = 1002,
    RA_RS_NOTIFY_CFG_GET = 1003,
    RA_RS_SET_PID = 1004,
    RA_RS_ASYNC_HDC_SESSION_CONNECT = 1005,
    RA_RS_ASYNC_HDC_SESSION_CLOSE = 1006,
    RA_RS_SET_DEBUG_CONFIG = 1007,
    RA_RS_OP_MAX_NUM,
};

enum ModuleType {
    HCCP_INIT = 0,
    RDMA_OP = 1,
    SOCKET_OP = 2,
    OTHERS = 3,
};

int ConverReturnCode(enum ModuleType module, int erroCode);

static inline bool RaRsHasCapability(unsigned int capability, unsigned int roceVersion, unsigned int netVersion)
{
    static struct {
        unsigned int capability;
        unsigned int roceVersion;
        unsigned int netVersion;
    } capInfoList[] = {
        {RA_CAP_DRV_SHAREPOOL_NON_PIN, 0, 1},
        {RA_CAP_UDMA_NOTIFY_EVENT, 0, 2},
    };

    if (capability >= RA_CAP_INVALID) {
        return false;
    }
    return ((roceVersion >= capInfoList[capability].roceVersion) && (netVersion >= capInfoList[capability].netVersion));
}
#endif // RA_RS_OPCODE_H
