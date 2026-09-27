/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_OPCODE_H
#define HCCP_OPCODE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup libinit
 * init module error code conversion
 */
#define HCCP_EAGAIN 128001      /* EAGAIN: try again */
#define HCCP_EINVALIDIPS 328008 /* EINVALIDIPS: ranktable中ip和物理网卡的ip不一致 */
#define HCCP_ELINKDOWN 328004   /* ELINKDOWN: 网口down */

/**
 * @ingroup librdma
 * @ingroup libudma
 * rdma/udma module error code conversion
 */
#define ROCE_EAGAIN 128101   /* EAGAIN: try again */
#define ROCE_EINVAL 128103   /* EINVAL: invalid argument */
#define ROCE_ENOENT 228100   /* ENOENT: means mr async not success right now, revoke the function again */
#define ROCE_ENOMEM 328100   /* ENOMEM: roce module has ENOMEM error */
#define ROCE_EOPENSRC 528101 /* EOPENSRC: open source verbs error */
#define ROCE_ENOTSUPP 528102 /* ENOTSUPP: operation not supported */

/**
 * @ingroup libsocket
 * socket module error code conversion
 */
#define SOCK_EAGAIN 128201        /* EAGAIN: no data received by socket */
#define SOCK_CLOSE 128203         /* EINVAL: device异常关闭时作为心跳返回值返回给hccl*/
#define SOCK_EADDRINUSE 128205    /* EADDRINUSE：check if IP has been listened when SOCK_EADDRINUSE is returned */
#define SOCK_EADDRNOTAVAIL 128206 /* EADDRNOTAVAIL：check if IP exist when SOCK_EADDRNOTAVAIL is returned */
#define SOCK_ESOCKCLOSED 128207   /* ESOCKCLOSED：socket has been closed */
#define SOCK_ENOENT 228200        /* ENOENT: means socket not success right now, revoke the function again */
#define SOCK_ENODEV 228202        /* ENODEV: socket 设备不存在 */

/**
 * @ingroup libcommon
 * others module error code conversion
 */
#define OTHERS_EAGAIN 128301 /* EAGAIN: try again */
#define OTHERS_EUSERS 128308
#define OTHERS_ENOTSUPP 528302 /* ENOTSUPP: operation not supported */

enum HccnCfgKey {
    HCCN_CFG_UDP_PORT_MODE = 0,
    HCCN_CFG_MULTI_QP_COUNT = 1,
    HCCN_CFG_MULTI_QP_UDP_PORTS = 2,
    HCCN_CFG_RESV_MEM_INFO = 3,
    HCCN_CFG_QOS_DSCP = 4,
    HCCN_CFG_KEY_INVALID
};

enum {
    RA_CAP_DRV_SHAREPOOL_NON_PIN = 0,
    RA_CAP_UDMA_NOTIFY_EVENT = 1,
    RA_CAP_INVALID,
};
#ifdef __cplusplus
}
#endif

#endif // HCCP_OPCODE_H
